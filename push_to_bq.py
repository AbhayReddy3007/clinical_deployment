
"""
trade&conference.py – Extract clinical trial IDs from:
  • Data Source 15: Medical & Scientific Conferences (ASCO, ESMO, AHA, ADA, EASD, etc.)
  • Data Source 16: Pharma Trade Publications (Endpoints, STAT, FierceBiotech, etc.)

Strategy (Gemini + Google Search grounding, mirroring 12_13.py):
  1. Build dynamic targeted queries per connector using molecule name + synonyms.
  2. Call Gemini with googleSearch tool for each query.
  3. Parse structured JSON + regex-scan raw response for registry IDs.
  4. Deduplicate and return a flat list of dicts for trial_fetcher.py ingestion.

Public API
----------
    results = scan_trade_and_conferences(molecule: str) -> list[dict]

Each dict has keys:
    trial_id, registry_source, trial_title, phase, company_name,
    source_url, indication, source_type  ("Conference" | "Trade Publication")
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

# ── optional dotenv ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Gemini client ─────────────────────────────────────────────────────────────
from .utils import MODEL as _MODEL, get_gemini_client, trade_calls as _TRADE_CALLS, conference_calls as _CONFERENCE_CALLS

try:
    from google import genai
    from google.genai import types as _gtypes
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ── Registry ID regex patterns ────────────────────────────────────────────────
_TRIAL_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bNCT\d{8}\b',                          re.IGNORECASE),
    re.compile(r'\bEUCTR\d{4}-\d{6}-\d{2}\b',           re.IGNORECASE),
    re.compile(r'\bCTRI/\d{4}/\d{2}/\d+\b',             re.IGNORECASE),
    re.compile(r'\bISRCTN\d{8}\b',                       re.IGNORECASE),
    re.compile(r'\bACTRN\d{14}\b',                       re.IGNORECASE),
    re.compile(r'\bChiCTR-?[A-Z]{0,5}-?\d{8,10}\b',     re.IGNORECASE),
    re.compile(r'\bJRCT\d{6,12}\b',                      re.IGNORECASE),
    re.compile(r'\bKCT\d{7}\b',                          re.IGNORECASE),
    re.compile(r'\bPACTR\d{15}\b',                       re.IGNORECASE),
    re.compile(r'\bDRKS\d{8}\b',                         re.IGNORECASE),
    re.compile(r'\bIRCT\d{10,20}[A-Z0-9]{0,10}\b',      re.IGNORECASE),
]

_SOURCE_MAP: Dict[str, str] = {
    "NCT":    "ClinicalTrials.gov",
    "EUCTR":  "EudraCT",
    "CTRI":   "CTRI (India)",
    "ISRCTN": "ISRCTN",
    "ACTRN":  "ANZCTR",
    "CHICTR": "ChiCTR",
    "JRCT":   "JRCT (Japan)",
    "KCT":    "CRIS (Korea)",
    "PACTR":  "PACTR",
    "DRKS":   "DRKS (Germany)",
    "IRCT":   "IRCT (Iran)",
}

_MAX_WORKERS = 1  # concurrent Gemini calls


def _registry_source(trial_id: str) -> str:
    prefix = trial_id.upper().split("/")[0].split("-")[0]
    for key, source in _SOURCE_MAP.items():
        if prefix.startswith(key):
            return source
    return "Conference / Trade Publication"


def _regex_scan(text: str) -> List[str]:
    found, seen = [], set()
    for pat in _TRIAL_ID_PATTERNS:
        for m in pat.findall(text):
            norm = m.upper()
            if norm not in seen:
                seen.add(norm)
                found.append(norm)
    return found


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _gemini_call(prompt: str) -> str:
    """Call Gemini with Google Search grounding. Returns text or '' on error."""
    if not _HAS_GENAI:
        return ""
    try:
        client   = get_gemini_client()
        contents = [_gtypes.Content(role="user",
                                    parts=[_gtypes.Part.from_text(text=prompt)])]
        config   = _gtypes.GenerateContentConfig(
            tools=[_gtypes.Tool(googleSearch=_gtypes.GoogleSearch())]
        )
        out = ""
        for chunk in client.models.generate_content_stream(
            model=_MODEL, contents=contents, config=config
        ):
            if chunk.text:
                out += chunk.text
        return out.strip()
    except Exception as exc:
        print(f"  [TRADE&CONF] Gemini call failed: {exc}", file=sys.stderr)
        return ""


def _safe_json(text: str) -> Any:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    for i, ch in enumerate(text):
        if ch in "{[":
            text = text[i:]
            break
    try:
        from json_repair import repair_json
        return repair_json(text, return_objects=True)
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return None


# ── Connector definitions (mirrored from 12_13.py) ───────────────────────────

_CONNECTOR_15 = {
    "connector_id": "data_source-15",
    "source_name": "Medical & Scientific Conferences (ASCO, ESMO, AHA, ADA, EASD, ACR, etc.)",
    "source_type": "Conference",
    "source_description": (
        "Conference proceedings, presentation slides, program agendas, posters, abstracts, oral "
        "presentations, late-breaking sessions, and session summaries from major medical/scientific "
        "conferences such as ASCO, ESMO, AHA, ACC, ADA, EASD, ACR, AASLD, EHA, ASH, AAD, DDW, "
        "AAN, SNO, WCLC, and similar. Must be the ACTUAL conference organizer website, proceedings, "
        "abstract, poster, or presentation material."
    ),
    "what_to_find": (
        "Clinical trial data presentations (oral, poster, late-breaker), efficacy and safety results, "
        "primary/secondary endpoint data, subgroup analyses, long-term follow-up data, "
        "new trial designs presented, Phase 1/2/3 results, overall survival updates, "
        "progression-free survival data, response rates, biomarker analyses, "
        "head-to-head comparisons, patient-reported outcomes, and combination trial results."
    ),
    "not_available": (
        "Do NOT report news articles about conferences. Find ACTUAL proceedings, posters, abstracts, "
        "presentations, programs, or organizer pages. Avoid non-clinical content (policy, pricing) "
        "unless tied to a clinical data presentation."
    ),
    "acceptable_examples": (
        "- asco.org/... (ASCO abstract/presentation)\n"
        "- abstracts.esmo.org/... (ESMO conference abstract)\n"
        "- conference organizer abstract/poster page with clinical trial data"
    ),
    "unacceptable_examples": (
        "- news articles about a conference instead of the actual conference page/material\n"
        "- social posts about a conference\n"
        "- wikipedia.org, reddit.com, youtube.com (unless official conference channel)"
    ),
}

_CONNECTOR_16 = {
    "connector_id": "data_source-16",
    "source_name": "Pharma Trade Publications (Clinical Trial focused)",
    "source_type": "Trade Publication",
    "source_description": (
        "Clinical-trial-focused articles from pharma/biotech trade publications such as "
        "Endpoints News, STAT News, FierceBiotech, FiercePharma, Scrip, Evaluate, "
        "BioPharma Dive, Drug Discovery & Development, and similar industry outlets. "
        "Articles must specifically discuss clinical trial events: data readouts, enrollment, "
        "phase transitions, trial failures, clinical holds, or development program updates."
    ),
    "what_to_find": (
        "Trial data readout coverage, enrollment updates, phase transition announcements, "
        "clinical hold reports, trial failure/discontinuation coverage, pivotal trial launches, "
        "competitive clinical landscape commentary, trial design analysis, "
        "interim analysis reports, adaptive design changes, and development timelines."
    ),
    "not_available": (
        "Do NOT report general pharma news unrelated to clinical trials or development programs. "
        "Do NOT focus on pricing, commercial launches, or patent litigation unless tied to a clinical event."
    ),
    "acceptable_examples": (
        "- endpointsnews.com/... (trial data readout coverage)\n"
        "- fiercebiotech.com/...trial... (trade publication clinical trial article)\n"
        "- statnews.com/... (STAT News clinical trial report)"
    ),
    "unacceptable_examples": (
        "- patent-only article with no clinical trial event\n"
        "- general business article unrelated to clinical development\n"
        "- wikipedia.org, reddit.com"
    ),
}

_CONNECTORS = [_CONNECTOR_15, _CONNECTOR_16]


# ── Query builder (mirrors 12_13.py build_queries_for_connector) ──────────────

def _build_queries(connector: Dict, molecule: str) -> List[str]:
    cid = connector["connector_id"]
    queries: List[str] = []

    if cid == "data_source-15":
        # Conference queries — combine into at most _CONFERENCE_CALLS queries
        all_conference_queries = [
            f'"{molecule}" ASCO ESMO AHA ADA EASD clinical trial data presentation oral poster abstract',
            f'"{molecule}" late-breaking abstract Phase 3 results conference presentation',
            f'"{molecule}" clinical trial data conference 2024 2025 2026 poster oral session',
            f'"{molecule}" AHA ACC ASH AASLD EHA ACR DDW trial results abstract symposium',
        ]
        max_calls = _CONFERENCE_CALLS
        if max_calls >= len(all_conference_queries):
            queries = all_conference_queries[:max_calls]
        else:
            # Combine queries to fit within max_calls
            batch_size = len(all_conference_queries) // max_calls
            remainder = len(all_conference_queries) % max_calls
            idx = 0
            for i in range(max_calls):
                count = batch_size + (1 if i < remainder else 0)
                combined = " | ".join(all_conference_queries[idx:idx + count])
                queries.append(combined)
                idx += count

    elif cid == "data_source-16":
        # Trade publication queries — combine into at most _TRADE_CALLS queries
        all_trade_queries = [
            f'"{molecule}" clinical trial data readout Phase 3 results Endpoints STAT FierceBiotech',
            f'"{molecule}" enrollment topline results trial failed discontinued trade publication',
            f'"{molecule}" pivotal trial clinical hold interim analysis BioPharma Dive Evaluate Scrip',
            f'"{molecule}" trial initiation first patient dosed phase transition pharma trade news',
        ]
        max_calls = _TRADE_CALLS
        if max_calls >= len(all_trade_queries):
            queries = all_trade_queries[:max_calls]
        else:
            # Combine queries to fit within max_calls
            batch_size = len(all_trade_queries) // max_calls
            remainder = len(all_trade_queries) % max_calls
            idx = 0
            for i in range(max_calls):
                count = batch_size + (1 if i < remainder else 0)
                combined = " | ".join(all_trade_queries[idx:idx + count])
                queries.append(combined)
                idx += count

    return queries


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are a clinical trial intelligence assistant with access to Google Search.

MOLECULE: {molecule}
DATA SOURCE: {source_name}
SEARCH QUERY: {query}

SOURCE DESCRIPTION:
{source_description}

WHAT TO FIND:
{what_to_find}

IMPORTANT — DO NOT INCLUDE:
{not_available}

ACCEPTABLE SOURCE EXAMPLES:
{acceptable_examples}

UNACCEPTABLE SOURCE EXAMPLES:
{unacceptable_examples}

Search for clinical trial IDs and events related to {molecule} from the source type above.
For each item found, extract:
  - trial_id:    Clinical trial registry ID (NCT, EudraCT, CTRI, ISRCTN, etc.)
                 Use the named trial program (e.g. "SURPASS-2") if no registry ID is visible.
  - trial_title: Title of the trial or abstract/article
  - phase:       Trial phase (Phase 1 / 2 / 3 / 4) or "N/A"
  - indication:  Disease/condition being studied
  - company_name: Sponsor / drug company
  - source_url:  Direct URL where this was found
  - event_type:  One of: Data Presentation | Trial Initiation | Enrollment Milestone |
                 Topline Results | Phase Transition | Trial Discontinuation | Other

Return ONLY valid JSON — no markdown, no preamble:

{{
  "events": [
    {{
      "trial_id":    "NCT01234567",
      "trial_title": "SURPASS-2: A Phase 3 trial of tirzepatide vs semaglutide in T2D",
      "phase":       "Phase 3",
      "indication":  "Type 2 Diabetes",
      "company_name": "Eli Lilly",
      "source_url":  "https://asco.org/abstract/...",
      "event_type":  "Data Presentation"
    }}
  ]
}}

Rules:
- Only include events from the described source type.
- trial_id must be a real registry ID (NCT…, EUCTR…, etc.) or a named program ID.
- Do not invent data. Only report what you find via search right now.
- Each event must have its own unique source_url where possible.
"""


# ── Per-query worker ──────────────────────────────────────────────────────────

def _run_query(connector: Dict, molecule: str, query: str, q_idx: int) -> List[Dict[str, str]]:
    """Run one Gemini search query and return extracted hits."""
    label = f"{connector['connector_id']}|Q{q_idx}"
    print(f"  [TRADE&CONF] {label}: {query[:80]}", file=sys.stderr)

    prompt = _EXTRACTION_PROMPT.format(
        molecule=molecule,
        source_name=connector["source_name"],
        query=query,
        source_description=connector["source_description"],
        what_to_find=connector["what_to_find"],
        not_available=connector["not_available"],
        acceptable_examples=connector["acceptable_examples"],
        unacceptable_examples=connector["unacceptable_examples"],
    )

    raw = _gemini_call(prompt)
    if not raw:
        return []

    hits: List[Dict[str, str]] = []
    seen: set = set()

    # ── Structured extraction ──────────────────────────────────────────────
    parsed = _safe_json(raw)
    events: list = []
    if isinstance(parsed, dict):
        events = parsed.get("events") or []
    elif isinstance(parsed, list):
        events = parsed

    for item in (events or []):
        if not isinstance(item, dict):
            continue
        tid = str(item.get("trial_id") or "").strip().upper()
        if not tid:
            # Try pulling any registry ID from the item dict text
            for extra_id in _regex_scan(json.dumps(item)):
                if extra_id not in seen:
                    seen.add(extra_id)
                    hits.append({
                        "trial_id":    extra_id,
                        "trial_title": str(item.get("trial_title") or "")[:300],
                        "phase":       str(item.get("phase") or ""),
                        "indication":  str(item.get("indication") or ""),
                        "company_name": str(item.get("company_name") or ""),
                        "source_url":  str(item.get("source_url") or ""),
                        "event_type":  str(item.get("event_type") or ""),
                        "source_type": connector["source_type"],
                    })
            continue
        if tid not in seen:
            seen.add(tid)
            hits.append({
                "trial_id":    tid,
                "trial_title": str(item.get("trial_title") or "")[:300],
                "phase":       str(item.get("phase") or ""),
                "indication":  str(item.get("indication") or ""),
                "company_name": str(item.get("company_name") or ""),
                "source_url":  str(item.get("source_url") or ""),
                "event_type":  str(item.get("event_type") or ""),
                "source_type": connector["source_type"],
            })

    # ── Regex sweep of full raw response for any IDs not in JSON ──────────
    for extra_id in _regex_scan(raw):
        if extra_id not in seen:
            seen.add(extra_id)
            hits.append({
                "trial_id":    extra_id,
                "trial_title": "",
                "phase":       "",
                "indication":  "",
                "company_name": "",
                "source_url":  "",
                "event_type":  "",
                "source_type": connector["source_type"],
            })

    return hits


# ── Main connector ────────────────────────────────────────────────────────────

def scan_trade_and_conferences(molecule: str) -> List[Dict[str, str]]:
    """
    Main entry point. Returns a list of dicts with keys:
        trial_id, registry_source, trial_title, phase, company_name,
        source_url, indication, source_type
    """
    print(f"  [TRADE&CONF] Scanning conferences + trade publications for '{molecule}' …",
          file=sys.stderr)

    all_hits: List[Dict[str, str]] = []
    seen_ids: set = set()

    # Build (connector, query, q_idx) work units
    work_units = []
    for connector in _CONNECTORS:
        queries = _build_queries(connector, molecule)
        print(f"  [TRADE&CONF] {connector['connector_id']} ({connector['source_type']}): "
              f"{len(queries)} queries", file=sys.stderr)
        for q_idx, query in enumerate(queries, 1):
            work_units.append((connector, query, q_idx))

    # Run all queries in parallel
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_query, conn, molecule, query, q_idx): (conn, q_idx)
            for conn, query, q_idx in work_units
        }
        for fut in as_completed(futures):
            try:
                hits = fut.result()
                for hit in hits:
                    tid = hit.get("trial_id", "").strip()
                    if tid and tid not in seen_ids:
                        seen_ids.add(tid)
                        all_hits.append(hit)
            except Exception as exc:
                conn, q_idx = futures[fut]
                print(f"  [TRADE&CONF] Worker error [{conn['connector_id']}|Q{q_idx}]: {exc}",
                      file=sys.stderr)

    # Build final output dicts
    final: List[Dict[str, str]] = []
    conf_count  = 0
    trade_count = 0

    for item in all_hits:
        tid = item.get("trial_id", "").strip()
        if not tid:
            continue
        src_type = item.get("source_type", "")
        registry_src = _registry_source(tid)
        final.append({
            "trial_id":        tid,
            "registry_source": registry_src,
            "trial_title":     item.get("trial_title", ""),
            "phase":           item.get("phase", ""),
            "company_name":    item.get("company_name", ""),
            "source_url":      item.get("source_url", ""),
            "indication":      item.get("indication", ""),
            "source_type":     src_type,
        })
        if src_type == "Conference":
            conf_count += 1
        elif src_type == "Trade Publication":
            trade_count += 1

    print(f"  [TRADE&CONF] Done — {conf_count} conference ID(s), "
          f"{trade_count} trade publication ID(s), {len(final)} total unique.",
          file=sys.stderr)
    return final


# ── CLI (quick test) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    mol = _sys.argv[1] if len(_sys.argv) > 1 else "Semaglutide"
    results = scan_trade_and_conferences(mol)
    print(json.dumps(results, indent=2))
