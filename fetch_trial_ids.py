#!/usr/bin/env python3
"""
Fetch Trial IDs from International Registries (Step 1 only).

Searches 6 registries in parallel using Gemini + Google Search:
  - ChiCTR       (China)
  - CTRI         (India)
  - JRCT         (Japan)
  - ANZCTR       (Australia / New Zealand)
  - CRIS         (Korea)
  - ReBEC        (Brazil)

ClinicalTrials.gov and EU Clinical Trials Register are intentionally excluded.

Usage:
    python fetch_trial_ids.py --molecule Semaglutide
    python fetch_trial_ids.py --molecule Tirzepatide --output results.json
"""

import re
import json
import asyncio
import argparse
from google import genai
from google.genai import types
from json_repair import repair_json
from .utils import MODEL, SEARCH_MODEL, get_gemini_client, trial_registries_per_call

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------
# Registry search needs a model with strong Google Search grounding.
# Default to gemini-3-flash-preview; override via GEMINI_SEARCH_MODEL env var.
import os


MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds

# Dedicated thread pool so all 6 registry searches can run truly in parallel
# even when this module is called from inside another ThreadPoolExecutor thread.
from concurrent.futures import ThreadPoolExecutor
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="intl-reg")

# ---------------------------------------------------------------------------
# Registries — ClinicalTrials.gov and EU Clinical Trials Register excluded
# ---------------------------------------------------------------------------
REGISTRIES = [
    {
        "name": "ChiCTR",
        "id_prefix": "ChiCTR",
        "url": "https://www.chictr.org.cn/",
        "region": "China",
        "description": "Chinese Clinical Trial Registry",
    },
    {
        "name": "CTRI",
        "id_prefix": "CTRI",
        "url": "https://ctri.nic.in/",
        "region": "India",
        "description": "Clinical Trials Registry - India",
    },
    {
        "name": "JRCT",
        "id_prefix": "jRCT",
        "url": "https://rctportal.niph.go.jp/en/",
        "region": "Japan",
        "description": "Japan Registry of Clinical Trials",
    },
    {
        "name": "ANZCTR",
        "id_prefix": "ACTRN",
        "url": "https://www.anzctr.org.au/",
        "region": "Australia/New Zealand",
        "description": "Australian New Zealand Clinical Trials Registry",
    },
    {
        "name": "CRIS",
        "id_prefix": "KCT",
        "url": "https://cris.nih.go.kr/",
        "region": "Korea",
        "description": "Clinical Research Information Service - Korea",
    },
    {
        "name": "ReBEC",
        "id_prefix": "RBR",
        "url": "https://ensaiosclinicos.gov.br/",
        "region": "Brazil",
        "description": "Brazilian Clinical Trials Registry",
    },
]


# ---------------------------------------------------------------------------
# JSON helpers (ported from gemini_extractor.py)
# ---------------------------------------------------------------------------

def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object or array from text."""
    start = text.find("{")
    if start == -1:
        start = text.find("[")
        if start == -1:
            return ""
        opening, closing = "[", "]"
    else:
        opening, closing = "{", "}"

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def _repair_truncated_json(response: str) -> str:
    """Attempt to repair truncated JSON by closing the last complete trial object."""
    if not response:
        return response

    trials_start = response.find('"trials"')
    if trials_start == -1:
        return response

    array_start = response.find("[", trials_start)
    if array_start == -1:
        return response

    last_complete_pos = -1
    depth = 0
    in_string = False
    escape_next = False

    for i in range(array_start + 1, len(response)):
        char = response[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                next_idx = i + 1
                while next_idx < len(response) and response[next_idx] in " \n\r\t":
                    next_idx += 1
                if next_idx < len(response) and response[next_idx] in ",]":
                    last_complete_pos = i

    if last_complete_pos != -1:
        response = response[: last_complete_pos + 1]
        response += "\n  ]\n}"
        return response

    last_brace = response.rfind("}")
    if last_brace != -1 and last_brace > array_start:
        response = response[: last_brace + 1]
        response += "\n  ]\n}"
        return response

    open_braces = response.count("{") - response.count("}")
    open_brackets = response.count("[") - response.count("]")
    response += "]" * open_brackets
    response += "}" * open_braces
    return response


def _parse_trial_json(response: str) -> list:
    """Parse JSON response from a registry search with multi-level repair fallbacks."""
    if not response or not response.strip():
        return []

    response = response.strip()

    # Strip markdown code fences
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    json_str = _extract_first_json_object(response)
    if not json_str:
        return []

    # Attempt 1: direct parse
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("trials", [])
        return []
    except json.JSONDecodeError as e:
        error_msg = str(e)
        print(f"  ⚠️  JSON parse error: {e}")

    # Attempt 2: handle "Extra data" — decode only the first valid object
    if "Extra data" in error_msg:
        print("  Attempting to extract first valid JSON object...")
        try:
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(json_str)
            if isinstance(data, list):
                print(f"  ✓ Recovered {len(data)} trials (ignored extra data)")
                return data
            elif isinstance(data, dict):
                trials = data.get("trials", [])
                print(f"  ✓ Recovered {len(trials)} trials (ignored extra data)")
                return trials
        except Exception as e2:
            print(f"  ❌ Extra-data extraction failed: {e2}")

    # Attempt 3: repair truncated JSON
    print("  Attempting to repair truncated JSON...")
    try:
        repaired = _repair_truncated_json(json_str)
        data = json.loads(repaired)
        if isinstance(data, list):
            print(f"  ✓ Repaired JSON — recovered {len(data)} trials")
            return data
        elif isinstance(data, dict):
            trials = data.get("trials", [])
            print(f"  ✓ Repaired JSON — recovered {len(trials)} trials")
            return trials
    except json.JSONDecodeError as e2:
        print(f"  ❌ Repair failed: {e2}")

    # Attempt 4: json_repair library
    try:
        fixed = repair_json(json_str, return_objects=True)
        if isinstance(fixed, list):
            print(f"  ✓ json_repair — recovered {len(fixed)} trials")
            return fixed
        elif isinstance(fixed, dict):
            trials = fixed.get("trials", [])
            print(f"  ✓ json_repair — recovered {len(trials)} trials")
            return trials
    except Exception:
        pass

    # Attempt 5: regex last resort — match any JSON object with NCT_ID
    trial_pattern = r'\{[^{}]*"NCT_ID"\s*:\s*"[^"]+?"[^{}]*\}'
    matches = re.finditer(trial_pattern, json_str, re.DOTALL)
    trials = []
    for match in matches:
        try:
            trials.append(json.loads(match.group(0)))
        except Exception:
            continue
    if trials:
        print(f"  ✓ Last-resort regex — recovered {len(trials)} trials")
        return trials

    print("  ❌ All recovery attempts failed.")
    return []


# ---------------------------------------------------------------------------
# Deduplication (ported from gemini_extractor.py)
# ---------------------------------------------------------------------------

def _phase_to_number(phase: str) -> float:
    """Convert a phase string to a numeric value for comparison."""
    if not phase or phase.upper() == "N/A":
        return 0.0
    phase = str(phase).strip().upper().replace("PHASE", "").strip()
    return {
        "1": 1.0,
        "1A": 1.3,
        "1B": 1.5,
        "2": 2.0,
        "2A": 2.3,
        "2B": 2.5,
        "3": 3.0,
        "3A": 3.3,
        "3B": 3.5,
        "4": 4.0,
    }.get(phase, 0.0)


def _deduplicate_by_phase(trials: list) -> list:
    """Deduplicate trials by ID, keeping the entry with the highest phase."""
    if not trials:
        return []

    trial_groups: dict[str, list] = {}
    for trial in trials:
        trial_id = trial.get("NCT_ID", "").strip()
        if not trial_id:
            continue
        trial_groups.setdefault(trial_id, []).append(trial)

    deduplicated = []
    duplicates_found = 0

    for trial_id, group in trial_groups.items():
        if len(group) > 1:
            duplicates_found += len(group) - 1
            group_sorted = sorted(
                group,
                key=lambda t: _phase_to_number(t.get("Phase", "0")),
                reverse=True,
            )
            selected = group_sorted[0]
            phases = [t.get("Phase", "N/A") for t in group]
            print(
                f"  ℹ️  {trial_id}: Multiple phases {phases}, "
                f"selected Phase {selected.get('Phase', 'N/A')}"
            )
        else:
            selected = group[0]
        deduplicated.append(selected)

    if duplicates_found > 0:
        print(f"\n  Resolved {duplicates_found} duplicate(s) by selecting highest phase")

    return deduplicated


# ---------------------------------------------------------------------------
# Gemini API call with Google Search + exponential backoff
# ---------------------------------------------------------------------------

def _sync_gemini_call(model: str, contents: list, config) -> str:
    """Synchronous Gemini streaming call.

    Creates a **fresh** genai.Client per call so the underlying httpx session
    lives on the same thread where it is used — avoids "client has been closed"
    errors when called from ``asyncio.to_thread`` or a ``ThreadPoolExecutor``.
    """
    client = get_gemini_client()
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            response_text += chunk.text
    return response_text.strip()


async def gemini_call_with_search(
    prompt: str,
    model: str,
) -> str:
    """Async Gemini call with Google Search grounding and exponential backoff.

    Each call runs in its own thread (via to_thread) with a fresh genai.Client
    so all 6 registry searches can proceed in parallel without sharing httpx
    sessions or event-loop resources.
    """
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
    ]
    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(tools=tools)

    retry_count = 0
    backoff_delay = INITIAL_BACKOFF

    while retry_count <= MAX_RETRIES:
        try:
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                _SEARCH_EXECUTOR,
                _sync_gemini_call, model, contents, config,
            )
            return response_text
        except Exception as e:
            error_str = str(e).lower()
            if any(
                err in error_str
                for err in ["429", "rate limit", "quota", "resource exhausted"]
            ):
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print(f"\n  ❌ Max retries ({MAX_RETRIES}) exceeded. Error: {e}")
                    raise
                print(
                    f"\n  ⚠️  Rate limit — waiting {backoff_delay:.1f}s "
                    f"before retry {retry_count}/{MAX_RETRIES}..."
                )
                await asyncio.sleep(backoff_delay)
                backoff_delay *= 2
            else:
                print(f"  ❌ Gemini API error: {e}")
                raise

    return ""


# ---------------------------------------------------------------------------
# Per-registry search (mirrors search_single_registry in gemini_extractor.py)
# ---------------------------------------------------------------------------

async def search_single_registry(molecule: str, registry: dict) -> list:
    """Search one registry for all trials of a molecule via Gemini + Google Search."""
    return await search_registries_batch(molecule, [registry])


async def search_registries_batch(molecule: str, registries: list) -> list:
    """Search one or more registries in a single Gemini call.
    
    When multiple registries are passed, they are combined into a single
    prompt so the LLM searches all of them in one call.
    """
    if len(registries) == 1:
        reg = registries[0]
        registry_block = f"""REGISTRY: {reg['name']}
URL: {reg['url']}
REGION: {reg['region']}
ID PREFIX: {reg['id_prefix']}"""
        search_instructions = f"""SEARCH STRATEGY:
1. Search the registry website ({reg['url']}) directly for "{molecule}"
2. Search Google for: site:{reg['url'].replace('https://', '').rstrip('/')} {molecule}
3. Search Google for: {reg['id_prefix']} {molecule} clinical trial {reg['region']}
4. Search for known brand names and synonyms of {molecule}
5. Also try searching with the molecule name in the local language if applicable"""
        id_format = f"{reg['id_prefix']}XXXXXXXX"
    else:
        reg_lines = []
        search_lines = []
        id_prefixes = []
        for reg in registries:
            reg_lines.append(f"  - {reg['name']} ({reg['region']}): {reg['url']} — ID prefix: {reg['id_prefix']}")
            search_lines.append(f"  - Search {reg['url']} for \"{molecule}\"")
            search_lines.append(f"  - Search Google for: site:{reg['url'].replace('https://', '').rstrip('/')} {molecule}")
            search_lines.append(f"  - Search Google for: {reg['id_prefix']} {molecule} clinical trial {reg['region']}")
            id_prefixes.append(reg['id_prefix'])
        registry_block = "REGISTRIES TO SEARCH:\n" + "\n".join(reg_lines)
        search_instructions = "SEARCH STRATEGY (for each registry above):\n" + "\n".join(search_lines) + f"""
  - Search for known brand names and synonyms of {molecule}
  - Also try searching with the molecule name in local languages if applicable"""
        id_format = " or ".join(f"{p}XXXXXXXX" for p in id_prefixes)

    prompt = f"""Search the following clinical trial registries for trials related to {molecule}.

{registry_block}

YOUR TASK: Find as many {molecule} trials as possible registered on {"this registry" if len(registries) == 1 else "these registries"}.

{search_instructions}

INCLUDE ALL of these:
- All phases (1, 1a, 1b, 2, 2a, 2b, 3, 3a, 3b, 4, post-marketing)
- All indications: Type 2 Diabetes (T2DM), Obesity, Weight Management, MASH/NASH, 
  Cardiovascular (CV), CKD, Heart Failure, Alzheimer's, Liver disease, PCOS, etc.
- All dosage forms: subcutaneous injection, oral tablets, etc.
- Innovator trials (Novo Nordisk, Eli Lilly, etc.) AND generic/biosimilar trials
- All statuses: Completed, Active, Recruiting, Not yet recruiting, Terminated, Withdrawn
- Combination trials where {molecule} is one of the study drugs
- Head-to-head comparison trials involving {molecule}
- Extension studies and long-term follow-up studies of {molecule}
- Bioequivalence and pharmacokinetic studies of {molecule}

ONLY EXCLUDE:
- Trials that have absolutely NO connection to {molecule}
- Trials where {molecule} is mentioned only in passing or as background context

For each trial found, provide:
- NCT_ID: Registry ID (should start with {id_format}, 
  but include any valid trial ID found even if it has a different format)
- Program_Name: Official program name (e.g., "STEP 1", "SUSTAIN 6") or "N/A" if not named
- Indication: Primary indication (e.g., "Obesity", "Type 2 Diabetes", "MASH")
- Phase: Trial phase (e.g., "1", "2", "3", "3b", "4") — use "N/A" if not clear
- Title: Brief trial title or description

IMPORTANT:
- Cast a WIDE NET — include trials even if you are not 100% certain they involve {molecule},
  as long as there is reasonable evidence. We will verify later.
- Return as many trials as you can find. It is MUCH better to include too many than too few.
- Some registries may list trials in local language — include those too.
- If you find cross-referenced IDs from other registries (e.g. NCT IDs mentioned on a 
  registry page), include them with their original registry prefix.

Return as JSON:

{{
  "trials": [
    {{
      "NCT_ID": "{id_format}",
      "Program_Name": "PROGRAM NAME",
      "Indication": "INDICATION",
      "Phase": "3",
      "Title": "Brief trial title"
    }}
  ]
}}

If no trials are found, return: {{"trials": []}}
"""

    response = await gemini_call_with_search(
        prompt, model=SEARCH_MODEL,
    )
    trials = _parse_trial_json(response)

    # Tag each trial with its source registry
    # For multi-registry batches, try to infer source from ID prefix
    for trial in trials:
        if "registry_source" not in trial or not trial["registry_source"]:
            trial_id = trial.get("NCT_ID", "").upper()
            matched = False
            for reg in registries:
                if trial_id.startswith(reg["id_prefix"].upper()):
                    trial["registry_source"] = reg["name"]
                    matched = True
                    break
            if not matched:
                trial["registry_source"] = registries[0]["name"] if len(registries) == 1 else "International"

    return trials


# ---------------------------------------------------------------------------
# Parallel Step 1 — mirrors get_all_trials_parallel in gemini_extractor.py
# ---------------------------------------------------------------------------

async def fetch_trial_ids(molecule: str) -> list:
    """
    Step 1: Search all configured registries in parallel and return
    a deduplicated list of trial ID records.

    Each record contains:
        NCT_ID          - Registry-specific trial identifier
        Program_Name    - Named trial programme (e.g. "STEP 1") or "N/A"
        Indication      - Primary indication
        Phase           - Trial phase
        registry_source - Name of the registry that returned this trial
    """

    print(f"\n{'='*80}")
    print(f"STEP 1: Searching registries for {molecule}")
    print(f"Model : {SEARCH_MODEL}")
    print(f"Registries searched: {len(REGISTRIES)} (batched {trial_registries_per_call} per call)")
    print(f"(ClinicalTrials.gov and EU Clinical Trials Register handled separately)")
    print(f"{'='*80}\n")

    # Batch registries according to trial_registries_per_call
    registry_batches = [
        REGISTRIES[i:i + trial_registries_per_call]
        for i in range(0, len(REGISTRIES), trial_registries_per_call)
    ]

    tasks = []
    for batch in registry_batches:
        names = ", ".join(r["name"] for r in batch)
        print(f"  📋 Queuing batch: {names}")
        tasks.append(search_registries_batch(molecule, batch))

    print(f"\n🚀 Launching {len(tasks)} batched search(es) "
          f"({len(REGISTRIES)} registries, {trial_registries_per_call} per call)...\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_trials = []
    registry_counts: dict[str, int] = {}

    for batch, result in zip(registry_batches, results):
        batch_names = ", ".join(r["name"] for r in batch)
        if isinstance(result, Exception):
            print(f"  ❌ {batch_names}: Error — {result}")
            for reg in batch:
                registry_counts[reg["name"]] = 0
        else:
            count = len(result)
            # Attribute counts per registry from the batch
            for reg in batch:
                reg_count = sum(
                    1 for t in result
                    if isinstance(t, dict) and t.get("registry_source") == reg["name"]
                )
                registry_counts[reg["name"]] = reg_count
            if count > 0:
                print(f"  ✓ [{batch_names}]: Found {count} trials total")
                all_trials.extend([t for t in result if isinstance(t, dict)])
            else:
                print(f"  ○ [{batch_names}]: No trials found")

    # Deduplicate — keep highest phase if same ID appears in multiple registries
    unique_trials = _deduplicate_by_phase(all_trials)

    # Summary
    print(f"\n{'='*80}")
    print(f"STEP 1 SUMMARY — {molecule}")
    print(f"{'='*80}")
    print(f"  Total trials (raw)       : {len(all_trials)}")
    print(f"  Unique trials (deduped)  : {len(unique_trials)}")
    print(f"\n  Breakdown by registry:")
    for reg_name, count in registry_counts.items():
        marker = "✓" if count > 0 else "○"
        print(f"    {marker} {reg_name:<35} {count}")

    print(f"\n  Trial IDs found:")
    for trial in unique_trials:
        trial_id  = trial.get("NCT_ID", "N/A")
        program   = trial.get("Program_Name", "N/A")
        phase     = trial.get("Phase", "N/A")
        source    = trial.get("registry_source", "")
        indication = trial.get("Indication", "N/A")
        print(f"    {trial_id:<28} Phase {phase:<4} [{source}]  {program}  ({indication})")

    return unique_trials


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch trial IDs from 6 international registries (Step 1 only)."
    )
    parser.add_argument(
        "--molecule",
        nargs="+",
        required=True,
        help="Molecule name, e.g. --molecule Semaglutide",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save results as JSON, e.g. --output results.json",
    )
    args = parser.parse_args()

    molecule_name = " ".join(args.molecule)

    # Run async fetch
    trials = asyncio.run(fetch_trial_ids(molecule_name))

    # Optionally save to JSON
    if args.output:
        output_data = {
            "molecule": molecule_name,
            "total_trials": len(trials),
            "registries_searched": [r["name"] for r in REGISTRIES],
            "trials": trials,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output}")

    print(f"\n✅ Done — {len(trials)} unique trial IDs found for {molecule_name}")
    return trials


if __name__ == "__main__":
    main()