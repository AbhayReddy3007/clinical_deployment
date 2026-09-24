"""
alias_resolver.py – Resolve a drug name to all its known aliases using
BigQuery (vw_drug_details_full) + Gemini cleaning.

Mirrors the reference code pattern exactly, with the following upgrades:
  - Accepts a drug name and matches it against cleaned_generic_name
    (case-insensitive, also tries alias columns as fallback).
  - Uses gcp_utils for all GCP config (BQ client, project ID, API keys).
  - Returns a deduplicated, cleaned list of search terms:
      [primary_name, alias_1, alias_2, ...]
  - Caches the BQ result for the session so repeated lookups are free.

Public API
----------
    terms = resolve_aliases(drug_name: str) -> list[str]

    The first element is always the original drug_name as passed in.
    Subsequent elements are cleaned aliases from BigQuery.
    If no aliases are found, returns [drug_name] unchanged.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import List, Optional

# ── All GCP config comes from medical_potential.config ───────────────────────
from medical_potential.config import PROJECT_ID
from medical_potential.gcp_utils import get_bq_client
from .utils import (
    get_gemini_client,
    MODEL,
)

_MODEL = MODEL

# ── Gemini ────────────────────────────────────────────────────────────────────
try:
    from google.genai import types as _gtypes
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ── BQ query ─────────────────────────────────────────────────────────────────
_ALIAS_QUERY = """
WITH filtered_data AS (
    SELECT
        cleaned_generic_name,
        TRIM(alias) AS alias
    FROM `{project}.data_mart.vw_drug_details_full`,
    UNNEST(SPLIT(COALESCE(Alias_Name,''), ',')) alias
    WHERE alias IS NOT NULL
      AND TRIM(alias) != ''
)
SELECT
    cleaned_generic_name,
    STRING_AGG(
        DISTINCT alias,
        ', '
        ORDER BY alias
    ) AS Alias_Name
FROM filtered_data
GROUP BY cleaned_generic_name
ORDER BY cleaned_generic_name
"""


@lru_cache(maxsize=1)
def _fetch_all_aliases_from_bq() -> List[dict]:
    """
    Fetch the full alias table from BigQuery once and cache it in memory.
    Returns a list of dicts: [{cleaned_generic_name, Alias_Name}, ...].
    """
    try:
        client  = get_bq_client()
        project = PROJECT_ID or client.project
        query   = _ALIAS_QUERY.format(project=project)
        rows    = list(client.query(query).result())
        print(f"  [ALIAS] Loaded {len(rows)} drug(s) from BigQuery alias table.",
              file=sys.stderr)
        return [{"cleaned_generic_name": r.cleaned_generic_name,
                 "Alias_Name":           r.Alias_Name} for r in rows]
    except Exception as exc:
        print(f"  [ALIAS] BQ query failed: {exc}", file=sys.stderr)
        return []


def _normalise(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    name = name.lower().strip()
    name = re.sub(r"[\s\-_+/]+", " ", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()


def _find_row(drug_name: str, all_rows: List[dict]) -> Optional[dict]:
    """
    Match drug_name against cleaned_generic_name (exact, then normalised,
    then check if drug_name appears anywhere in the alias list).
    Returns the matching row or None.
    """
    norm_drug = _normalise(drug_name)

    # Pass 1: exact match on cleaned_generic_name (case-insensitive)
    for row in all_rows:
        if row["cleaned_generic_name"].strip().lower() == drug_name.strip().lower():
            return row

    # Pass 2: normalised match
    for row in all_rows:
        if _normalise(row["cleaned_generic_name"]) == norm_drug:
            return row

    # Pass 3: drug_name is itself an alias — search the alias column
    for row in all_rows:
        aliases = [a.strip().lower() for a in (row["Alias_Name"] or "").split(",")]
        if drug_name.strip().lower() in aliases:
            return row
        if norm_drug in [_normalise(a) for a in aliases]:
            return row

    return None


def _clean_aliases_with_gemini(generic_name: str, alias_csv: str) -> List[str]:
    """
    Use Gemini to deduplicate, normalise, and filter the raw alias list.
    Returns a cleaned list of alias strings.
    Falls back to a simple split+strip if Gemini is unavailable.
    """
    if not _HAS_GENAI:
        return [a.strip() for a in alias_csv.split(",") if a.strip()]

    prompt = f"""You are a pharmaceutical data normalisation assistant.

Drug generic name: {generic_name}
Raw alias list: {alias_csv}

Task: Clean and normalise the alias names.
- Remove exact or near-duplicate aliases
- Aliases that differ ONLY by a hyphen are duplicates — keep the version
  WITHOUT the hyphen (e.g. "LY3437943" and "LY-3437943" → keep "LY3437943";
  "NN-9536" and "NN9536" → keep "NN9536")
- Strip extra whitespace
- Standardise capitalisation (title case for brand names, lowercase INN)
- Remove entries that are clearly not drug names (e.g. mechanism descriptions,
  therapeutic class names, registry IDs, gene names)
- Remove aliases that are identical to the generic name itself
- Keep well-known brand names, INN synonyms, development codes (e.g. NN9388),
  and combination names (e.g. Cagrilintide+Semaglutide, CagriSema)

Return ONLY a comma-separated list of cleaned aliases, nothing else.
No preamble, no explanation, no bullet points."""

    try:
        client   = get_gemini_client()
        contents = [_gtypes.Content(role="user",
                                    parts=[_gtypes.Part.from_text(text=prompt)])]
        config   = _gtypes.GenerateContentConfig()
        out      = ""
        for chunk in client.models.generate_content_stream(
            model=_MODEL, contents=contents, config=config
        ):
            if chunk.text:
                out += chunk.text
        cleaned = [a.strip() for a in out.strip().split(",") if a.strip()]
        return cleaned
    except Exception as exc:
        print(f"  [ALIAS] Gemini cleaning failed: {exc}", file=sys.stderr)
        return [a.strip() for a in alias_csv.split(",") if a.strip()]


def resolve_aliases(drug_name: str) -> List[str]:
    """
    Main entry point.

    Given a drug name, returns a deduplicated list of search terms:
        [drug_name, alias_1, alias_2, ...]

    The primary drug_name is always first.
    If no BQ match is found, returns [drug_name].
    """
    print(f"  [ALIAS] Resolving aliases for '{drug_name}' …", file=sys.stderr)

    all_rows = _fetch_all_aliases_from_bq()
    if not all_rows:
        print("  [ALIAS] No BQ data — searching with primary name only.",
              file=sys.stderr)
        return [drug_name]

    row = _find_row(drug_name, all_rows)
    if row is None:
        print(f"  [ALIAS] No match found in BQ for '{drug_name}' — "
              "searching with primary name only.", file=sys.stderr)
        return [drug_name]

    matched_name = row["cleaned_generic_name"]
    raw_aliases  = row["Alias_Name"] or ""
    print(f"  [ALIAS] Matched '{drug_name}' → '{matched_name}' "
          f"({len(raw_aliases.split(','))} raw alias(es)).", file=sys.stderr)

    cleaned = _clean_aliases_with_gemini(matched_name, raw_aliases)

    # Build deduped term list: primary name first, then aliases
    seen:  set      = set()
    terms: List[str] = []

    for term in [drug_name] + cleaned:
        norm = _normalise(term)
        if norm and norm not in seen:
            seen.add(norm)
            terms.append(term)

    # Also include matched_name if it differs from drug_name
    norm_matched = _normalise(matched_name)
    if norm_matched not in seen:
        seen.add(norm_matched)
        terms.append(matched_name)

    print(f"  [ALIAS] {len(terms)} search term(s): {terms}", file=sys.stderr)
    return terms


# ── CLI (quick test) ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys as _sys
    drug = " ".join(_sys.argv[1:]) if len(_sys.argv) > 1 else "Semaglutide"
    terms = resolve_aliases(drug)
    print("\nSearch terms:")
    for t in terms:
        print(f"  - {t}")
