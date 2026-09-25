#!/usr/bin/env python3
"""
innovator_web.py – Extract clinical trial IDs from the innovator company's
own website and trial/pipeline pages.

Strategy (two passes via Gemini + Google Search):

  Pass 1 – Discovery
    Ask Gemini: "Who makes <molecule> and what are the URLs of their
    clinical trials / pipeline pages?"  Returns (company_name, [urls]).

  Pass 2 – Extraction
    For each URL, fetch the page HTML (requests) and ask Gemini to extract
    every trial ID it can find.  Also directly regex-scan the HTML as a
    fast supplement.

Both passes use Gemini with the googleSearch tool so Gemini can look up
live data when the cached URLs are stale or unavailable.

Public API
----------
    results = scan_innovator_website(molecule: str) -> list[dict]

Each dict has keys: trial_id, registry_source, trial_title, source_url,
phase, company_name (all strings, may be empty).

No hard dependency on gcp_utils / google-cloud — wraps the import in
try/except so the module can be imported standalone.
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests

# ── optional dotenv ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Gemini client ─────────────────────────────────────────────────────────────
from .utils import MODEL as _MODEL, get_gemini_client, Innovator_web_sources_calls as _MAX_INNOVATOR_CALLS

try:
    from google import genai
    from google.genai import types as _gtypes
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ── regex patterns for trial IDs ─────────────────────────────────────────────
_TRIAL_ID_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bNCT\d{8}\b',                          re.IGNORECASE),
    re.compile(r'\bEUCTR\d{4}-\d{6}-\d{2}\b',           re.IGNORECASE),
    re.compile(r'\bCTIS\d{4}-\d{6}-\d{2}\b',            re.IGNORECASE),
    re.compile(r'\bCTRI/\d{4}/\d{2}/\d+\b',             re.IGNORECASE),
    re.compile(r'\bISRCTN\d{8}\b',                       re.IGNORECASE),
    re.compile(r'\bACTRN\d{14}\b',                       re.IGNORECASE),
    re.compile(r'\bChiCTR-?[A-Z]{0,5}-?\d{8,10}\b',     re.IGNORECASE),
    re.compile(r'\bJRCT\d{6,12}\b',                      re.IGNORECASE),
    re.compile(r'\bKCT\d{7}\b',                          re.IGNORECASE),
    re.compile(r'\bPACTR\d{15}\b',                       re.IGNORECASE),
    re.compile(r'\bNTR\d{4,6}\b',                        re.IGNORECASE),
    re.compile(r'\bDRKS\d{8}\b',                         re.IGNORECASE),
    re.compile(r'\bReBEC/RBR-[A-Z0-9]{6}\b',            re.IGNORECASE),
    re.compile(r'\bRBR-[A-Z0-9]{5,7}\b',                re.IGNORECASE),
    re.compile(r'\bSLCTR/\d{4}/\d{3}\b',                re.IGNORECASE),
    re.compile(r'\bIRCTN\d{8}\b',                        re.IGNORECASE),
    re.compile(r'\bIRCT\d{10,20}[A-Z0-9]{0,10}\b',      re.IGNORECASE),
    re.compile(r'\bTCTR\d{8,14}\b',                      re.IGNORECASE),
]

_SOURCE_MAP: Dict[str, str] = {
    "NCT":     "ClinicalTrials.gov",
    "EUCTR":   "EudraCT",
    "CTIS":    "EU CTIS",
    "CTRI":    "CTRI (India)",
    "ISRCTN":  "ISRCTN",
    "ACTRN":   "ANZCTR",
    "CHICTR":  "ChiCTR",
    "JRCT":    "JRCT (Japan)",
    "KCT":     "CRIS (Korea)",
    "PACTR":   "PACTR",
    "NTR":     "NTR (Netherlands)",
    "DRKS":    "DRKS (Germany)",
    "REBEC":   "ReBEC",
    "RBR":     "ReBEC",
    "SLCTR":   "SLCTR",
    "IRCT":    "IRCT (Iran)",
    "TCTR":    "TCTR (Thailand)",
}


def _registry_source(trial_id: str) -> str:
    prefix = trial_id.upper().split("/")[0].split("-")[0]
    for key, source in _SOURCE_MAP.items():
        if prefix.startswith(key):
            return source
    return "Innovator Website"


def _regex_scan(text: str) -> List[str]:
    """Return all unique trial IDs found by regex in text."""
    found: List[str] = []
    seen: set = set()
    for pat in _TRIAL_ID_PATTERNS:
        for m in pat.findall(text):
            norm = m.upper()
            if norm not in seen:
                seen.add(norm)
                found.append(norm)
    return found


# ── HTTP helpers ─────────────────────────────────────────────────────────────

_SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_FETCH_TIMEOUT   = 20   # seconds per page
_MAX_HTML_CHARS  = 120_000  # chars sent to Gemini per page (avoids token overflow)


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_SESSION_HEADERS)
    return s


def _fetch_page(url: str, session: requests.Session) -> str:
    """Fetch a URL and return its text content, truncated to _MAX_HTML_CHARS."""
    try:
        resp = session.get(url, timeout=_FETCH_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        text = resp.text or ""
        return text[:_MAX_HTML_CHARS]
    except Exception as exc:
        print(f"  [INNOV] Could not fetch {url}: {exc}", file=sys.stderr)
        return ""


# ── Gemini helpers ────────────────────────────────────────────────────────────

def _gemini_call(prompt: str, use_search: bool = True) -> str:
    """
    Call Gemini with optional Google Search grounding.
    Returns the model's text output or '' on error.
    """
    if not _HAS_GENAI:
        return ""
    try:
        client   = get_gemini_client()
        contents = [_gtypes.Content(role="user",
                                    parts=[_gtypes.Part.from_text(text=prompt)])]
        config_kwargs: Dict[str, Any] = {}
        if use_search:
            config_kwargs["tools"] = [_gtypes.Tool(googleSearch=_gtypes.GoogleSearch())]
        config = _gtypes.GenerateContentConfig(**config_kwargs)

        out = ""
        for chunk in client.models.generate_content_stream(
            model=_MODEL, contents=contents, config=config
        ):
            if chunk.text:
                out += chunk.text
        return out.strip()
    except Exception as exc:
        print(f"  [INNOV] Gemini call failed: {exc}", file=sys.stderr)
        return ""


def _extract_json_block(text: str) -> str:
    """Pull the first JSON object or array out of a Gemini response."""
    text = text.strip()
    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    # Find first { or [
    for i, ch in enumerate(text):
        if ch in "{[":
            return text[i:]
    return text


import json as _json

def _safe_json(text: str) -> Any:
    text = _extract_json_block(text)
    try:
        from json_repair import repair_json
        return repair_json(text, return_objects=True)
    except Exception:
        pass
    try:
        return _json.loads(text)
    except Exception:
        return None


# ── Pass 1: Discover company and trial/pipeline URLs ─────────────────────────

_DISCOVERY_PROMPT = """You are a pharmaceutical research assistant with access to Google Search.

MOLECULE: {molecule}

Your task:
1. Identify the originator / innovator company that developed or owns {molecule}.
2. Find URLs across TWO categories:

   CATEGORY A — Company main website pages:
   Pages on the innovator's primary corporate website that list clinical trials,
   pipeline, research, or study results for {molecule}.  Examples:
     - /pipeline, /research/pipeline, /our-research/pipeline
     - /clinical-trials, /research/clinical-trials
     - /science/clinical-trials, /r-and-d/clinical-studies
     - Investor-relations pipeline or R&D overview pages
     - Press-release or news pages announcing trial results for {molecule}

   CATEGORY B — Innovator-run clinical trial portal / patient finder sites:
   Many large pharma companies operate a SEPARATE dedicated website (different
   domain or subdomain) specifically for browsing or finding their clinical
   trials.  Search for these by name and include their {molecule}-specific
   search result pages.  Well-known examples (not exhaustive):
     - Novartis: clinicaltrials.novartis.com
     - Pfizer: clinicaltrials.pfizer.com  or  pfe.com/trials
     - Lilly: trials.lilly.com
     - Roche/Genentech: clinicaltrials.roche.com
     - AstraZeneca: astrazenecaclinicaltrials.com
     - Sanofi: sanofi.com/clinical-trials-results or sanofi-trialresults.com
     - GSK: gsk-clinicalstudyregister.com
     - Merck (MSD): merck.com/clinical-trials or msd.com/clinical-trials
     - BMS: bms.com/researchers-and-partners/clinical-trials-and-research.html
     - J&J / Janssen: jnj.com/latest-news/clinical-trials or clinicaltrials.janssen.com
     - Boehringer Ingelheim: trials.boehringer-ingelheim.com
     - Takeda: takedaclinicaltrials.com
     - Abbvie: abbvie.com/science/clinical-trials.html
     - Amgen: amgen.com/science/clinical-trials
   Include the URL for a search/filter page scoped to {molecule} where possible.

Return ONLY valid JSON — no markdown, no preamble:

{{
  "company_name": "<Name of the innovator company>",
  "urls": [
    "<url_1>",
    "<url_2>"
  ]
}}

Rules:
- List up to 15 URLs total, mixing both categories.  Prefer specific pages
  (search results for {molecule}, pipeline detail pages) over generic homepages.
- If no dedicated trial portal exists for this company, include their main
  pipeline or research overview page instead.
- Do not invent URLs; only include ones you confirmed via search.
"""


def _discover_innovator(molecule: str) -> Tuple[str, List[str]]:
    """
    Use Gemini+Search to find the innovator company and their trial/pipeline URLs.
    Returns (company_name, list_of_urls).
    """
    prompt = _DISCOVERY_PROMPT.format(molecule=molecule)
    raw    = _gemini_call(prompt, use_search=True)
    if not raw:
        return "", []
    parsed = _safe_json(raw)
    if not isinstance(parsed, dict):
        return "", []
    company = str(parsed.get("company_name") or "").strip()
    urls    = [str(u).strip() for u in (parsed.get("urls") or []) if u]
    return company, urls


# ── Pass 2: Extract trial IDs from a single page ─────────────────────────────

_EXTRACTION_PROMPT = """You are a clinical trial data extraction engine.

MOLECULE: {molecule}
COMPANY:  {company}
PAGE URL: {url}

Below is the page content (HTML/text).  Extract every clinical trial
registration ID that appears anywhere on this page.  Include:
  - NCT numbers  (e.g. NCT01234567)
  - EudraCT / EU CTR numbers  (e.g. 2019-001234-56)
  - CTRI numbers  (e.g. CTRI/2020/01/023456)
  - ISRCTN numbers
  - ACTRN numbers (ANZCTR)
  - ChiCTR numbers
  - JRCT numbers
  - Any other registry IDs
Also extract the trial title / description if present next to the ID.

Return ONLY valid JSON — no markdown, no preamble:

{{
  "trials": [
    {{
      "trial_id":    "<registry ID>",
      "trial_title": "<title or empty string>",
      "phase":       "<phase or empty string>"
    }}
  ]
}}

PAGE CONTENT:
{content}
"""


def _extract_from_page(
    url: str,
    content: str,
    molecule: str,
    company: str,
) -> List[Dict[str, str]]:
    """
    Two-pronged extraction for one page:
    1. Regex scan (fast, catches most standard IDs).
    2. Gemini prompt (catches IDs buried in non-standard text or behind JS).
    Merges and deduplicates results.
    """
    results: List[Dict[str, str]] = []
    seen: set = set()

    # ── Regex pass ──────────────────────────────────────────────────────────
    for tid in _regex_scan(content):
        if tid not in seen:
            seen.add(tid)
            results.append({
                "trial_id":    tid,
                "trial_title": "",
                "phase":       "",
                "source_url":  url,
            })

    # ── Gemini pass (only if content is non-trivial) ─────────────────────
    if len(content) > 200:
        prompt = _EXTRACTION_PROMPT.format(
            molecule=molecule,
            company=company,
            url=url,
            content=content[:_MAX_HTML_CHARS],
        )
        raw    = _gemini_call(prompt, use_search=False)
        parsed = _safe_json(raw)
        if isinstance(parsed, dict):
            for item in (parsed.get("trials") or []):
                tid = str(item.get("trial_id") or "").strip().upper()
                if tid and tid not in seen:
                    seen.add(tid)
                    results.append({
                        "trial_id":    tid,
                        "trial_title": str(item.get("trial_title") or "")[:300],
                        "phase":       str(item.get("phase") or ""),
                        "source_url":  url,
                    })

    return results


# ── Pass 3 (fallback): Ask Gemini directly without fetching pages ─────────────

_FALLBACK_PROMPT = """You are a pharmaceutical research assistant with access to Google Search.

MOLECULE: {molecule}
COMPANY:  {company}

Search the company's website, press releases, investor relations pages,
and any public clinical trial portal for all clinical trial registration
IDs associated with {molecule}.

Return ONLY valid JSON — no markdown, no preamble:

{{
  "trials": [
    {{
      "trial_id":    "<registry ID>",
      "trial_title": "<title or empty string>",
      "phase":       "<phase or empty string>",
      "source_url":  "<url where this ID was found>"
    }}
  ]
}}
"""


def _fallback_gemini_extract(molecule: str, company: str) -> List[Dict[str, str]]:
    """Ask Gemini+Search directly for trial IDs when page fetching fails."""
    prompt = _FALLBACK_PROMPT.format(molecule=molecule, company=company or molecule)
    raw    = _gemini_call(prompt, use_search=True)
    parsed = _safe_json(raw)
    if not isinstance(parsed, dict):
        return []
    results: List[Dict[str, str]] = []
    seen: set = set()
    for item in (parsed.get("trials") or []):
        tid = str(item.get("trial_id") or "").strip().upper()
        if tid and tid not in seen:
            seen.add(tid)
            results.append({
                "trial_id":    tid,
                "trial_title": str(item.get("trial_title") or "")[:300],
                "phase":       str(item.get("phase") or ""),
                "source_url":  str(item.get("source_url") or ""),
            })
    return results


# ── Public API ────────────────────────────────────────────────────────────────

_MAX_WORKERS = 6   # concurrent page fetches


def scan_innovator_website(molecule: str) -> List[Dict[str, str]]:
    """
    Main entry point. Uses at most _MAX_INNOVATOR_CALLS Gemini calls:
      Call 1: Discover the innovator company name
      Call 2: Discover its clinical trial / pipeline website URLs
      Call 3: Search those websites for trial IDs (Gemini+Search)

    Returns a list of dicts with keys:
        trial_id, registry_source, trial_title, phase,
        company_name, source_url
    """
    print(f"  [INNOV] Discovering innovator for '{molecule}' "
          f"(max {_MAX_INNOVATOR_CALLS} Gemini calls) …", file=sys.stderr)

    calls_remaining = _MAX_INNOVATOR_CALLS

    # ── Call 1: Discover company + URLs ───────────────────────────────────
    company, urls = _discover_innovator(molecule)
    calls_remaining -= 1
    if company:
        print(f"  [INNOV] Innovator: {company}  |  {len(urls)} URL(s) found.", file=sys.stderr)
    else:
        print("  [INNOV] Could not identify innovator; using fallback.", file=sys.stderr)

    page_results: List[Dict[str, str]] = []

    # ── Call 2: Fetch pages and regex-scan (no Gemini call, just HTTP) ────
    # Then use remaining calls for Gemini extraction
    if urls and calls_remaining > 0:
        session = _make_session()

        # Fetch all pages via HTTP (no Gemini cost)
        all_page_content: List[Tuple[str, str]] = []  # (url, content)
        for url in urls:
            print(f"  [INNOV] Fetching: {url}", file=sys.stderr)
            content = _fetch_page(url, session)
            if content:
                # Regex-scan each page (free, no Gemini call)
                for tid in _regex_scan(content):
                    page_results.append({
                        "trial_id":    tid,
                        "trial_title": "",
                        "phase":       "",
                        "source_url":  url,
                    })
                all_page_content.append((url, content))

        # ── Remaining calls: Gemini extraction on combined page content ───
        if all_page_content and calls_remaining > 0:
            # Combine all page content into batches for remaining calls
            pages_per_call = max(1, len(all_page_content) // calls_remaining)
            page_batches = []
            for i in range(0, len(all_page_content), pages_per_call):
                page_batches.append(all_page_content[i:i + pages_per_call])
            # Cap to remaining calls
            page_batches = page_batches[:calls_remaining]

            for batch in page_batches:
                # Combine content from all pages in this batch
                combined_content = ""
                batch_urls = []
                for url, content in batch:
                    batch_urls.append(url)
                    # Allocate content space evenly
                    max_chars = _MAX_HTML_CHARS // len(batch)
                    combined_content += f"\n\n=== PAGE: {url} ===\n{content[:max_chars]}\n"

                prompt = _EXTRACTION_PROMPT.format(
                    molecule=molecule,
                    company=company,
                    url=" | ".join(batch_urls),
                    content=combined_content[:_MAX_HTML_CHARS],
                )
                raw = _gemini_call(prompt, use_search=False)
                calls_remaining -= 1

                parsed = _safe_json(raw)
                if isinstance(parsed, dict):
                    for item in (parsed.get("trials") or []):
                        tid = str(item.get("trial_id") or "").strip().upper()
                        if tid:
                            page_results.append({
                                "trial_id":    tid,
                                "trial_title": str(item.get("trial_title") or "")[:300],
                                "phase":       str(item.get("phase") or ""),
                                "source_url":  batch_urls[0] if len(batch_urls) == 1 else ", ".join(batch_urls),
                            })

    # If we got nothing from direct page fetches, fall back to Gemini+Search
    if not page_results and calls_remaining > 0:
        print("  [INNOV] No IDs from page fetch — trying Gemini+Search fallback …",
              file=sys.stderr)
        page_results = _fallback_gemini_extract(molecule, company)

    # Deduplicate by trial_id and annotate
    seen: set = set()
    final: List[Dict[str, str]] = []
    for item in page_results:
        tid = item.get("trial_id", "").strip().upper()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        final.append({
            "trial_id":       tid,
            "registry_source": _registry_source(tid),
            "trial_title":    item.get("trial_title", ""),
            "phase":          item.get("phase", ""),
            "company_name":   company,
            "source_url":     item.get("source_url", ""),
        })

    print(f"  [INNOV] {len(final)} unique trial ID(s) found on innovator website "
          f"({_MAX_INNOVATOR_CALLS - calls_remaining}/{_MAX_INNOVATOR_CALLS} calls used).",
          file=sys.stderr)
    return final


# ── CLI (quick test) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    mol = _sys.argv[1] if len(_sys.argv) > 1 else "Semaglutide"
    trials = scan_innovator_website(mol)
    print(_json.dumps(trials, indent=2))