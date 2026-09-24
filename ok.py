"""
nice_trials.py – Extract clinical trial IDs from NICE Technology Appraisals.

Strategy (Gemini + Google Search grounding):
  1. Build targeted queries against nice.org.uk for the molecule.
  2. Call Gemini with googleSearch tool to find TA numbers, trial IDs,
     recommendation status, indication, and phase.
  3. Regex-scan the Gemini response text for any additional registry IDs.
  4. Return a deduplicated list of dicts ready for trial_fetcher.py ingestion.

Public API
----------
    results = scan_nice_appraisals(molecule: str) -> list[dict]

Each dict has keys:
    trial_id, registry_source, trial_title, phase, phase_status,
    trial_location, company_name, source_url, indication
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# ── optional dotenv ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Gemini client ─────────────────────────────────────────────────────────────
from .utils import MODEL as _MODEL, get_gemini_client

try:
    from google import genai
    from google.genai import types as _gtypes
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ── regex patterns for any embedded trial registry IDs ───────────────────────
_TRIAL_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bNCT\d{8}\b',                          re.IGNORECASE),
    re.compile(r'\bEUCTR\d{4}-\d{6}-\d{2}\b',           re.IGNORECASE),
    re.compile(r'\bCTRI/\d{4}/\d{2}/\d+\b',             re.IGNORECASE),
    re.compile(r'\bISRCTN\d{8}\b',                       re.IGNORECASE),
    re.compile(r'\bACTRN\d{14}\b',                       re.IGNORECASE),
    re.compile(r'\bChiCTR-?[A-Z]{0,5}-?\d{8,10}\b',     re.IGNORECASE),
    re.compile(r'\bJRCT\d{6,12}\b',                      re.IGNORECASE),
    re.compile(r'\bKCT\d{7}\b',                          re.IGNORECASE),
]

# NICE TA / HST number pattern
_NICE_TA_PATTERN = re.compile(r'\b(?:TA|HST|MTA|STA)\s*\d{2,4}\b', re.IGNORECASE)

_SOURCE_MAP: Dict[str, str] = {
    "NCT":    "ClinicalTrials.gov",
    "EUCTR":  "EudraCT",
    "CTRI":   "CTRI (India)",
    "ISRCTN": "ISRCTN",
    "ACTRN":  "ANZCTR",
    "CHICTR": "ChiCTR",
    "JRCT":   "JRCT (Japan)",
    "KCT":    "CRIS (Korea)",
}


def _registry_source(trial_id: str) -> str:
    prefix = trial_id.upper().split("/")[0].split("-")[0]
    for key, source in _SOURCE_MAP.items():
        if prefix.startswith(key):
            return source
    return "NICE Technology Appraisals"


def _regex_scan_ids(text: str) -> List[str]:
    """Return all unique registry IDs found by regex in text."""
    found, seen = [], set()
    for pat in _TRIAL_ID_PATTERNS:
        for m in pat.findall(text):
            norm = m.upper()
            if norm not in seen:
                seen.add(norm)
                found.append(norm)
    return found


def _regex_scan_nice_tas(text: str) -> List[str]:
    """Return all unique NICE TA/HST numbers found in text."""
    found, seen = [], set()
    for m in _NICE_TA_PATTERN.findall(text):
        norm = re.sub(r'\s+', '', m).upper()  # e.g. "TA123"
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
        print(f"  [NICE] Gemini call failed: {exc}", file=sys.stderr)
        return ""


def _safe_json(text: str) -> Any:
    """Extract and parse the first JSON object from Gemini output."""
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


# ── Query builder ─────────────────────────────────────────────────────────────

def _build_queries(molecule: str) -> List[str]:
    # If the molecule is a combined OR term (e.g. "LY3437943 OR LY-3437943"),
    # split into individual aliases and build queries for each.
    # This avoids wrapping the whole OR expression in double quotes,
    # which would turn it into a useless literal phrase search.
    if " OR " in molecule:
        aliases = [a.strip() for a in molecule.split(" OR ") if a.strip()]
    else:
        aliases = [molecule]

    queries = []
    for alias in aliases:
        queries.extend([
            f'"{alias}" technology appraisal NICE recommended site:nice.org.uk',
            f'"{alias}" NICE TA guidance ACD FAD appraisal site:nice.org.uk',
            f'"{alias}" NICE highly specialised technology HST site:nice.org.uk',
            f'"{alias}" NICE appraisal scope evidence review 2024 2025 site:nice.org.uk',
            f'"{alias}" NICE clinical trial evidence submission site:nice.org.uk',
        ])
    return queries


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """You are a pharmaceutical regulatory intelligence assistant with access to Google Search.

MOLECULE: {molecule}
SEARCH QUERY: {query}

Search NICE (nice.org.uk) for any Technology Appraisals (TA), Highly Specialised Technologies (HST),
or related NICE guidance documents concerning {molecule}.

For each NICE appraisal or related clinical trial found, extract:
  - ta_number:        NICE TA/HST number (e.g. TA123, HST14)
  - trial_id:         Clinical trial registry ID cited in the appraisal (NCT, EudraCT, CTRI, etc.) — or the TA number if no registry ID
  - trial_title:      Full NICE appraisal title or linked trial title
  - recommendation:   NICE recommendation status (Recommended / Not recommended / Optimised / In appraisal)
  - indication:       Disease/condition being appraised
  - phase:            Trial phase cited (Phase 1/2/3/4) or "Technology Appraisal"
  - company_name:     Marketing authorisation holder / drug company
  - source_url:       Direct URL on nice.org.uk
  - date_published:   Publication date (YYYY-MM-DD or YYYY-MM)

Return ONLY valid JSON — no markdown, no preamble:

{{
  "appraisals": [
    {{
      "ta_number":      "TA123",
      "trial_id":       "NCT01234567",
      "trial_title":    "...",
      "recommendation": "Recommended",
      "indication":     "Type 2 diabetes",
      "phase":          "Phase 3",
      "company_name":   "Novo Nordisk",
      "source_url":     "https://www.nice.org.uk/guidance/ta123",
      "date_published": "2024-03-15"
    }}
  ]
}}

Rules:
- Only include entries from nice.org.uk.
- If a TA has multiple linked trial IDs, create one entry per trial ID.
- Use the TA number as trial_id when no registry ID is available.
- Do not invent data; only report what you find via search now.
"""


# ── Fallback: regex-only sweep of Gemini free-text ───────────────────────────

def _sweep_free_text(molecule: str) -> List[Dict[str, str]]:
    """Ask Gemini for NICE info as free text, then regex-scan for IDs."""
    prompt = (
        f"Search nice.org.uk for all NICE Technology Appraisals and Highly Specialised "
        f"Technologies that assess {molecule}. List every TA number, linked trial registry "
        f"ID (NCT, EudraCT, etc.), recommendation status, and indication you find."
    )
    raw = _gemini_call(prompt)
    if not raw:
        return []

    results: List[Dict[str, str]] = []
    seen: set = set()

    # Registry IDs embedded in the text
    for tid in _regex_scan_ids(raw):
        if tid not in seen:
            seen.add(tid)
            results.append({
                "trial_id":     tid,
                "trial_title":  "",
                "indication":   "",
                "phase":        "",
                "recommendation": "",
                "company_name": "",
                "source_url":   "https://www.nice.org.uk",
                "ta_number":    "",
            })

    # NICE TA numbers with no linked registry ID
    for ta in _regex_scan_nice_tas(raw):
        if ta not in seen:
            seen.add(ta)
            results.append({
                "trial_id":     ta,
                "trial_title":  "",
                "indication":   "",
                "phase":        "Technology Appraisal",
                "recommendation": "",
                "company_name": "",
                "source_url":   f"https://www.nice.org.uk/guidance/{ta.lower()}",
                "ta_number":    ta,
            })

    return results


# ── Main connector ────────────────────────────────────────────────────────────

def scan_nice_appraisals(molecule: str) -> List[Dict[str, str]]:
    """
    Main entry point.  Returns a list of dicts with keys:
        trial_id, registry_source, trial_title, phase, phase_status,
        trial_location, company_name, source_url, indication
    """
    print(f"  [NICE] Scanning NICE Technology Appraisals for '{molecule}' …",
          file=sys.stderr)

    queries  = _build_queries(molecule)
    raw_hits: List[Dict[str, str]] = []
    seen_ids: set = set()

    # ── Single combined Gemini call with all queries ──────────────────────
    combined_query = " | ".join(queries)
    print(f"  [NICE] Combined query ({len(queries)} sub-queries): {combined_query[:120]}…",
          file=sys.stderr)
    prompt = _EXTRACTION_PROMPT.format(molecule=molecule, query=combined_query)
    raw    = _gemini_call(prompt)

    if raw:
        parsed = _safe_json(raw)
        if isinstance(parsed, dict):
            appraisals = parsed.get("appraisals") or []
            if not isinstance(appraisals, list):
                appraisals = []
        elif isinstance(parsed, list):
            appraisals = parsed
        else:
            appraisals = []

        for item in appraisals:
            if not isinstance(item, dict):
                continue

            ta_number = str(item.get("ta_number") or "").strip()
            trial_id  = str(item.get("trial_id") or ta_number).strip().upper()

            if not trial_id:
                # Try to pull any registry ID from raw text of this item
                item_text = json.dumps(item)
                for extra_id in _regex_scan_ids(item_text):
                    if extra_id not in seen_ids:
                        seen_ids.add(extra_id)
                        raw_hits.append({
                            "trial_id":     extra_id,
                            "trial_title":  str(item.get("trial_title") or ""),
                            "indication":   str(item.get("indication") or ""),
                            "phase":        str(item.get("phase") or ""),
                            "recommendation": str(item.get("recommendation") or ""),
                            "company_name": str(item.get("company_name") or ""),
                            "source_url":   str(item.get("source_url") or "https://www.nice.org.uk"),
                            "ta_number":    ta_number,
                        })
                continue

            if trial_id in seen_ids:
                continue
            seen_ids.add(trial_id)

            raw_hits.append({
                "trial_id":     trial_id,
                "trial_title":  str(item.get("trial_title") or "")[:300],
                "indication":   str(item.get("indication") or ""),
                "phase":        str(item.get("phase") or "Technology Appraisal"),
                "recommendation": str(item.get("recommendation") or ""),
                "company_name": str(item.get("company_name") or ""),
                "source_url":   str(item.get("source_url") or f"https://www.nice.org.uk/guidance/{ta_number.lower()}"),
                "ta_number":    ta_number,
            })

        # Also regex-scan the raw Gemini response for any IDs Gemini
        # mentioned in free text but didn't put in the JSON
        for extra_id in _regex_scan_ids(raw):
            if extra_id not in seen_ids:
                seen_ids.add(extra_id)
                raw_hits.append({
                    "trial_id":     extra_id,
                    "trial_title":  "",
                    "indication":   "",
                    "phase":        "",
                    "recommendation": "",
                    "company_name": "",
                    "source_url":   "https://www.nice.org.uk",
                    "ta_number":    "",
                })

    # If structured extraction got nothing, try free-text fallback
    if not raw_hits:
        print("  [NICE] No structured hits — trying free-text fallback …",
              file=sys.stderr)
        raw_hits = _sweep_free_text(molecule)

    # Build final output dicts
    final: List[Dict[str, str]] = []
    for item in raw_hits:
        tid = item.get("trial_id", "").strip()
        if not tid:
            continue
        phase_status = item.get("recommendation", "")  # maps to phase_status column
        final.append({
            "trial_id":        tid,
            "registry_source": _registry_source(tid),
            "trial_title":     item.get("trial_title", ""),
            "phase":           item.get("phase", "Technology Appraisal"),
            "phase_status":    phase_status,
            "trial_location":  "UK",
            "company_name":    item.get("company_name", ""),
            "source_url":      item.get("source_url", "https://www.nice.org.uk"),
            "indication":      item.get("indication", ""),
        })

    print(f"  [NICE] {len(final)} unique ID(s) found from NICE.", file=sys.stderr)
    return final


# ── CLI (quick test) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    mol = _sys.argv[1] if len(_sys.argv) > 1 else "Semaglutide"
    results = scan_nice_appraisals(mol)
    print(json.dumps(results, indent=2))
