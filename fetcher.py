#!/usr/bin/env python3
"""
fetcher.py – Enrichment + clinical efficacy scoring for trial_fetcher.py JSON output.

Step 1 – Gemini enrichment (parallel):
  Reads the Excel file produced by trial_fetcher.py and uses Gemini (with Google Search)
  to fill in per-trial outcome columns:
    dosage
    hba1c_change_pct  hba1c_duration  hba1c_rationale  hba1c_confidence
    weight_change_pct weight_duration weight_rationale  weight_confidence
    alt_reduction_pct alt_duration    alt_rationale     alt_confidence
    mash_change_pct   mash_duration   mash_rationale    mash_confidence

Step 2 – Clinical Efficacy Scoring:
  Scores the molecule across four endpoints using a phase-anchored algorithm.
    Phase 3 -> no penalty  |  Phase 2 -> x0.85  |  Phase 1 -> x0.65
    >=22% -> 5  |  16-21.9% -> 4  |  10-15.9% -> 3  |  5-9.9% -> 2  |  <5% -> 1
    Weights: Weight Loss 40% | HbA1c 40% | MASH 10% | ALT 10%

Usage:
    python fetcher.py Cagrisema
    python fetcher.py Cagrisema --json cagrisema_trials.json --workers 8
    python fetcher.py Cagrisema --no-score
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# -- third-party ---------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("ERROR: google-genai not installed.  Run: pip install google-genai")

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

# -- config from gcp_utils -----------------------------------------------------
from .utils import (
    MODEL, 
    RATIONALE_MODEL, 
    get_gemini_client,
    endpoint_extraction_batch,
    partial_enrichment as _PARTIAL_ENRICHMENT,
    metadata_batch as _METADATA_BATCH_SIZE,
)

# ==============================================================================
# SECTION 1 – CONSTANTS & COLUMN DEFINITIONS
# ==============================================================================

MAX_RETRIES     = 5
INITIAL_BACKOFF = 2.0
BATCH_SIZE      = endpoint_extraction_batch  # from utils config
DEFAULT_WORKERS = 6

OUTCOME_COLS = [
    "dosage",
    "hba1c_change_pct",  "hba1c_duration",  "hba1c_rationale",  "hba1c_confidence",
    "weight_change_pct", "weight_duration",  "weight_rationale", "weight_confidence",
    "alt_reduction_pct", "alt_duration",     "alt_rationale",    "alt_confidence",
    "mash_change_pct",   "mash_duration",    "mash_rationale",   "mash_confidence",
    "eval_hba1c_confidence",  "eval_weight_confidence",
    "eval_mash_confidence",   "eval_alt_confidence",
    "eval_hba1c_pct_change",  "eval_weight_pct_change",
    "eval_mash_pct_change",   "eval_alt_pct_change",
]

# Metadata + location fields extracted alongside endpoints in the combined call.
# These are merged into rows only when the existing value is empty/missing.
META_COLS = [
    "trial_title", "phase", "trial_study", "trial_size",
    "trial_location", "secondary_locations",
    "trial_start_date", "trial_completion_date",
    "phase_status", "company_name", "source_url",
]

ALL_COLUMNS = [
    "molecule_name", "registry_source", "trial_id", "acronym",
    "dosage", "phase", "trial_title", "trial_study", "trial_size",
    "trial_location", "secondary_locations", "trial_start_date", "trial_completion_date", "phase_status",
    "hba1c_change_pct",  "hba1c_duration",  "hba1c_rationale",  "hba1c_confidence",
    "weight_change_pct", "weight_duration",  "weight_rationale", "weight_confidence",
    "alt_reduction_pct", "alt_duration",     "alt_rationale",    "alt_confidence",
    "mash_change_pct",   "mash_duration",    "mash_rationale",   "mash_confidence",
    "company_name", "source_url",
    "llm_confidence", "llm_confidence_rationale",
    "eval_hba1c_confidence",  "eval_weight_confidence",
    "eval_mash_confidence",   "eval_alt_confidence",
    "eval_hba1c_pct_change",  "eval_weight_pct_change",
    "eval_mash_pct_change",   "eval_alt_pct_change",
    "efficacy_score", "efficacy_data_coverage",
    "efficacy_score_breakdown", "efficacy_narrative_rationale",
    "alias_names",
]


# ==============================================================================
# SECTION 2 – GEMINI CLIENT
# ==============================================================================


def get_client() -> genai.Client:
    """Return a Gemini client via gcp_utils.get_gemini_client().

    A fresh client is created per call to avoid "client has been closed"
    errors when multiple asyncio.run() calls share a cached instance.
    """
    return get_gemini_client()


def _check_api_key_early() -> None:
    try:
        get_client()
    except (ValueError, ImportError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)


# ==============================================================================
# SECTION 3 – JSON HELPERS
# ==============================================================================

def _safe_parse(text: str) -> Any:
    if not text:
        return None
    text = text.strip()
    if not text:
        return None
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
    if _HAS_JSON_REPAIR:
        try:
            return repair_json(text, return_objects=True)
        except Exception:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return None


# ==============================================================================
# SECTION 4 – GEMINI CALLS
# ==============================================================================

def _sync_call(prompt: str, use_search: bool = True) -> str:
    client = get_client()
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config_kwargs: Dict[str, Any] = {}
    if use_search:
        config_kwargs["tools"] = [types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(**config_kwargs)
    out = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL, contents=contents, config=config
    ):
        if chunk.text:
            out += chunk.text
    return out.strip()


async def _gemini_call(prompt: str, use_search: bool = True) -> str:
    backoff = INITIAL_BACKOFF
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(_sync_call, prompt, use_search)
        except (ValueError, ImportError):
            raise
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("429", "rate limit", "quota", "resource exhausted")):
                if attempt == MAX_RETRIES:
                    print(f"  X Max retries exceeded: {exc}", file=sys.stderr)
                    raise
                print(f"  ! Rate-limit – waiting {backoff:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...", file=sys.stderr)
                await asyncio.sleep(backoff)
                backoff *= 2
            else:
                print(f"  X Gemini error: {exc}", file=sys.stderr)
                raise
    return ""


# ==============================================================================
# SECTION 5 – ENRICHMENT
# ==============================================================================

def _build_prompt(molecule: str, batch: List[Dict[str, str]]) -> str:
    trial_lines = "\n".join(
        f"  - {t.get('trial_id','?')} | {t.get('trial_title','')[:120]} "
        f"| Phase {t.get('phase','?')} | {t.get('company_name','')} "
        f"| Registry: {t.get('registry_source','')} "
        f"| Indication: {t.get('trial_study','')} "
        f"| Enrollment: {t.get('trial_size','')} "
        f"| URL: {t.get('source_url','')}"
        for t in batch
    )
    return f"""You are a clinical data extraction engine with access to Google Search and live trial registries.

MOLECULE: {molecule}

TRIALS TO ENRICH ({len(batch)} total):
{trial_lines}

════════════════════════════════════════════════════════
SOURCE PRIORITY — search in this exact order for EACH trial.
Stop at the tier that yields a usable numeric result.
════════════════════════════════════════════════════════

PRIORITY 1 — Official Clinical Trial Registries (most authoritative):
Search these registries first. Use the Registry and URL fields above to
go directly to the trial's home registry when available.
  • ClinicalTrials.gov          (clinicaltrials.gov)
  • EU Clinical Trials Register (clinicaltrialsregister.eu)
  • ChiCTR                      (chictr.org.cn)
  • CTRI India                  (ctri.nic.in)
  • JRCT Japan                  (rctportal.niph.go.jp)
  • ANZCTR                      (anzctr.org.au)
  • CRIS Korea                  (cris.nih.go.kr)
  • ReBEC Brazil                (ensaiosclinicos.gov.br)
  • IRCT Iran                   (irct.ir)
  • DRKS Germany                (drks.de)
  • PACTR Africa                (pactr.samrc.ac.za)
  • TCTR Thailand               (thaiclinicaltrials.org)
  • WHO ICTRP                   (trialsearch.who.int)

  → If a complete numeric result (dosage, outcomes, timepoints) is found
    here, record it with confidence "High" and cite the registry page.
  → If only protocol data (no outcomes yet) is found, note it and
    continue to Priority 2.

PRIORITY 2 — Innovator Sources (use only if P1 is incomplete):
  • NICE Technology Appraisals  (nice.org.uk/guidance/ta*)
  • Innovator clinical trial portals (e.g. novonordisk-trials.com,
    clinicaltrials.novartis.com, trials.lilly.com, lilly.com/clinical-trials,
    clinicaltrials.pfizer.com, clinicaltrials.roche.com,
    astrazenecaclinicaltrials.com, gsk-clinicalstudyregister.com,
    takedaclinicaltrials.com, merck.com/clinical-trials, etc.)
  • Innovator press releases on official company newsroom / IR pages

  → If a numeric result is found here, record it with confidence "Medium"
    (unless it references a peer-reviewed publication — then "High").
  → Cite the exact URL and document title found.

PRIORITY 3 — Secondary Literature (use only if P1 and P2 are incomplete):
  • Pharma trade publications: Endpoints News, STAT News, FierceBiotech,
    FiercePharma, BioPharma Dive, Evaluate, Scrip
  • Peer-reviewed journals & preprints: PubMed, NEJM, Lancet, JAMA, BMJ,
    Nature Medicine, JCEM, Diabetes Care, bioRxiv
  • Medical and scientific conferences: ASCO, ESMO, AHA, ACC, ADA, EASD,
    ACR, AASLD, EHA, ASH, WCLC abstracts and proceedings
  • Google Scholar results for the trial ID or program name

  → If a numeric result is found here, record confidence as:
      "High"   — full peer-reviewed publication with primary data
      "Medium" — conference abstract or trade report of published data
      "Low"    — trade publication summary or secondary report

CRITICAL: Do NOT give up easily. For each trial, try MULTIPLE search
queries across all three priority tiers before returning "N/A". Use
the trial ID, program name, AND molecule name in your searches.

════════════════════════════════════════════════════════
FIELDS TO EXTRACT FOR EACH TRIAL
════════════════════════════════════════════════════════

1. dosage          - Primary or highest dose tested (e.g. "2.4 mg OW", "15 mg QD").
                     If multiple doses, pick the highest. Format: "[amount] [unit] [frequency]"

2. hba1c_change_pct  - HbA1c reduction in percentage points (positive number, e.g. "1.8").
                        "N/A" if not a diabetes trial or data unavailable.
   hba1c_duration     - Timepoint of measurement (e.g. "26 wk"). "N/A" if unavailable.
   hba1c_rationale    - 1-2 sentences: state the EXACT source URL/document and priority
                        tier used (e.g. "From ClinicalTrials.gov results tab for NCT01234567
                        [P1]. HbA1c reduction of 1.8pp at 26 weeks reported.").
   hba1c_confidence   - "High" / "Medium" / "Low" per the tier rules above.

3. weight_change_pct - Body weight loss percentage (positive number). "N/A" if unavailable.
   weight_duration    - Timepoint (e.g. "68 wk"). "N/A" if unavailable.
   weight_rationale   - 1-2 sentences citing exact source, URL, and priority tier.
   weight_confidence  - "High" / "Medium" / "Low".

4. alt_reduction_pct - ALT enzyme reduction percentage (positive number). "N/A" if unavailable.
   alt_duration       - Timepoint. "N/A" if unavailable.
   alt_rationale      - 1-2 sentences citing exact source, URL, and priority tier.
   alt_confidence     - "High" / "Medium" / "Low".

5. mash_change_pct   - MASH/NASH resolution rate or fibrosis improvement % (positive number).
                        "N/A" if not a liver trial.
   mash_duration      - Timepoint. "N/A" if unavailable.
   mash_rationale     - 1-2 sentences citing exact source, URL, and priority tier.
   mash_confidence    - "High" / "Medium" / "Low".

════════════════════════════════════════════════════════
RULES
════════════════════════════════════════════════════════
- Always start with P1 registries. Use the Registry and URL in the trial
  line above to go directly to the correct registry page first.
- Only move to P2 if P1 has no numeric outcome data for that field.
- Only move to P3 if both P1 and P2 have no numeric outcome data.
- Report reductions as POSITIVE numbers.
- Use "N/A" for fields with genuinely no data across all three tiers.
- Each rationale MUST name the exact source, URL, and tier (P1/P2/P3).
- One JSON object per trial, keyed by trial_id.
- Do NOT invent numbers. If no source has the data, use "N/A".

EVAL CONFIDENCE (0 to 1):
For each endpoint, also provide an eval confidence score (0-1) that reflects
how confident you are in the EXTRACTED numeric value:
  - eval_hba1c_confidence:  confidence in the extracted hba1c_change_pct value
  - eval_weight_confidence: confidence in the extracted weight_change_pct value
  - eval_mash_confidence:   confidence in the extracted mash_change_pct value
  - eval_alt_confidence:    confidence in the extracted alt_reduction_pct value

Score meaning:
  1.0 = Value directly from a P1 registry results page or peer-reviewed primary paper
  0.8 = Value from a P2 innovator source or well-cited secondary source
  0.6 = Value from a P3 trade publication or conference abstract
  0.4 = Value inferred or approximated from incomplete data
  0.2 = Low confidence — value from an unreliable or ambiguous source
  0.0 = N/A or no data found (use when the endpoint value is "N/A")

════════════════════════════════════════════════════════
METADATA + LOCATION FIELDS (fill in if missing above)
════════════════════════════════════════════════════════

For each trial, ALSO extract the following registry metadata and location
fields. These are needed when the trial row has gaps from the discovery
phase. If the trial line above already has the value, confirm or improve it.

- trial_title:           Full official trial title from the registry
- phase:                 Trial phase (e.g. "Phase 3", "Phase 2", "Phase 1")
- trial_study:           Study type / design (e.g. "Interventional | Randomized, Double-blind")
- trial_size:            Total enrollment number (e.g. "1200")
- trial_location:        Primary country (e.g. "United States")
- secondary_locations:   All OTHER countries, comma-separated (e.g. "Germany, Japan, India").
                         Empty string "" if single-country. Do NOT repeat primary country.
- trial_start_date:      Study start date (YYYY-MM-DD or YYYY-MM)
- trial_completion_date: Primary completion date (YYYY-MM-DD or YYYY-MM)
- phase_status:          Current status (e.g. "Completed", "Recruiting")
- company_name:          Sponsor / lead organization
- source_url:            Direct URL to the trial on its registry

Return ONLY valid JSON, no markdown, no preamble:

{{
  "results": {{
    "<trial_id_1>": {{
      "trial_title": "...", "phase": "...", "trial_study": "...",
      "trial_size": "...", "trial_location": "...", "secondary_locations": "...",
      "trial_start_date": "...", "trial_completion_date": "...",
      "phase_status": "...", "company_name": "...", "source_url": "...",
      "dosage": "...",
      "hba1c_change_pct": "...", "hba1c_duration": "...", "hba1c_rationale": "...", "hba1c_confidence": "...",
      "weight_change_pct": "...", "weight_duration": "...", "weight_rationale": "...", "weight_confidence": "...",
      "alt_reduction_pct": "...", "alt_duration": "...", "alt_rationale": "...", "alt_confidence": "...",
      "mash_change_pct": "...", "mash_duration": "...", "mash_rationale": "...", "mash_confidence": "...",
      "eval_hba1c_confidence": 0.9, "eval_weight_confidence": 1.0,
      "eval_mash_confidence": 0.0, "eval_alt_confidence": 0.0
    }},
    "<trial_id_2>": {{ ... }}
  }}
}}
"""


async def _enrich_batch(
    molecule: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
    total_batches: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, str]]:
    async with semaphore:
        ids = [t.get("trial_id", "?") for t in batch]
        print(f"  Batch {batch_idx+1}/{total_batches} -> {ids}", file=sys.stderr)
        try:
            raw = await _gemini_call(_build_prompt(molecule, batch), use_search=True)
        except Exception:
            return {}
        data = _safe_parse(raw)
        if not data:
            print(f"  X Batch {batch_idx+1}: could not parse response", file=sys.stderr)
            return {}
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, dict):
            results = data
        else:
            print(f"  X Batch {batch_idx+1}: unexpected JSON structure", file=sys.stderr)
            return {}
        print(f"  OK Batch {batch_idx+1}: enriched {len(results)} trial(s)", file=sys.stderr)
        return results


# ==============================================================================
# SECTION 5b – PARTIAL ENRICHMENT HELPERS (metadata-only for existing trials)
# ==============================================================================

def _get_existing_trial_ids(molecule: str) -> set:
    """Query BigQuery for trial IDs that already exist for this molecule."""
    try:
        from medical_potential.gcp_utils import get_bq_client
        from medical_potential.config import GD_CLINICAL_EFFICACY_TABLE_ID
    except ImportError:
        print("[PARTIAL] Could not import BQ helpers — treating all trials as new.",
              file=sys.stderr)
        return set()

    try:
        client = get_bq_client()
        from google.cloud import bigquery
        query = f"""
            SELECT trial_id
            FROM `{GD_CLINICAL_EFFICACY_TABLE_ID}`
            WHERE molecule_name = @molecule_name
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule)
        ])
        return {row.trial_id for row in client.query(query, job_config=job_config).result()}
    except Exception as exc:
        print(f"[PARTIAL] BQ lookup failed: {exc} — treating all trials as new.",
              file=sys.stderr)
        return set()


def _build_metadata_only_prompt(molecule: str, batch: List[Dict[str, str]]) -> str:
    """Build a Gemini prompt for metadata-only extraction (no endpoints)."""
    trial_lines = "\n".join(
        f"  - {r.get('trial_id', '?')} | {r.get('trial_title', '')[:120]} "
        f"| Phase {r.get('phase', '?')} | {r.get('company_name', '')} "
        f"| Registry: {r.get('registry_source', '')} "
        f"| URL: {r.get('source_url', '')}"
        for r in batch
    )
    return f"""You are a clinical trial metadata extraction engine with access to Google Search.

MOLECULE: {molecule}

TRIAL IDs TO LOOK UP ({len(batch)} total):
{trial_lines}

For EACH trial ID above, search the relevant clinical trial registry
(ClinicalTrials.gov, ChiCTR, CTRI, JRCT, ANZCTR, CRIS, ReBEC, EU CTR,
IRCT, DRKS, NTR, PACTR, SLCTR, TCTR, WHO ICTRP, etc.)
and extract the following metadata:

- trial_id:              The trial ID exactly as given above
- trial_title:           Full official trial title from the registry
- phase:                 Trial phase (e.g. "Phase 3", "Phase 2", "Phase 1")
- trial_study:           Study type / design (e.g. "Interventional | Randomized, Double-blind")
- trial_size:            Total enrollment number (e.g. "1200")
- trial_location:        Primary country / region where the trial is led (e.g. "United States")
- secondary_locations:   All other countries where the trial is conducted, comma-separated
                         (e.g. "Germany, Japan, India"). "" if single-country.
- trial_start_date:      Study start date (YYYY-MM-DD or YYYY-MM)
- trial_completion_date: Primary completion date (YYYY-MM-DD or YYYY-MM)
- phase_status:          Current status (e.g. "Completed", "Recruiting", "Active, not recruiting")
- company_name:          Sponsor / lead organization
- source_url:            Direct URL to the trial on its registry

RULES:
- Return one JSON object per trial, keyed by trial_id
- Use "N/A" for genuinely unavailable fields — do NOT guess
- For NCT IDs, use ClinicalTrials.gov as primary source
- For non-NCT IDs, search the appropriate international registry

Return ONLY valid JSON, no markdown, no preamble:

{{
  "results": {{
    "<trial_id_1>": {{
      "trial_title": "...",
      "phase": "...",
      "trial_study": "...",
      "trial_size": "...",
      "trial_location": "...",
      "secondary_locations": "...",
      "trial_start_date": "...",
      "trial_completion_date": "...",
      "phase_status": "...",
      "company_name": "...",
      "source_url": "..."
    }}
  }}
}}
"""


async def _metadata_only_batch(
    molecule: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
    total_batches: int,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Dict[str, str]]:
    """Fetch metadata only (no endpoints) for a batch of existing trials."""
    async with semaphore:
        ids = [t.get("trial_id", "?") for t in batch]
        print(f"  [META-ONLY] Batch {batch_idx+1}/{total_batches} -> {ids}", file=sys.stderr)
        try:
            raw = await _gemini_call(_build_metadata_only_prompt(molecule, batch), use_search=True)
        except Exception:
            return {}
        data = _safe_parse(raw)
        if not data:
            print(f"  X [META-ONLY] Batch {batch_idx+1}: could not parse response", file=sys.stderr)
            return {}
        if isinstance(data, dict) and "results" in data:
            results = data["results"]
        elif isinstance(data, dict):
            results = data
        else:
            print(f"  X [META-ONLY] Batch {batch_idx+1}: unexpected JSON structure", file=sys.stderr)
            return {}
        print(f"  OK [META-ONLY] Batch {batch_idx+1}: metadata for {len(results)} trial(s)",
              file=sys.stderr)
        return results


async def enrich_all(
    molecule: str,
    rows: List[Dict[str, str]],
    max_workers: int = DEFAULT_WORKERS,
    force_partial: Optional[bool] = None,
) -> List[Dict[str, str]]:
    """
    Enrich trial rows with endpoints and metadata.

    When CLINICAL_TRIALS_PARTIAL_ENRICHMENT is enabled (via config or force_partial):
      - NEW trials (not in BQ) → full enrichment (metadata + locations + endpoints)
      - EXISTING trials (already in BQ) → metadata-only backfill (batch size = METADATA_BATCH)
    When disabled, all high-confidence rows get full enrichment.
    """
    use_partial = force_partial if force_partial is not None else _PARTIAL_ENRICHMENT

    # Only enrich trials with llm_confidence > 0.6
    high_rows = []
    for r in rows:
        try:
            conf = float(r.get("llm_confidence", "0") or "0")
        except (ValueError, TypeError):
            conf = 0.0
        if conf > 0.6:
            high_rows.append(r)
    skip_count = len(rows) - len(high_rows)
    if skip_count:
        print(f"[ENRICH] Skipping {skip_count} trial(s) with LLM confidence <= 0.6.",
              file=sys.stderr)
    if not high_rows:
        print("[ENRICH] No trials with confidence > 0.6 to enrich.", file=sys.stderr)
        for row in rows:
            for col in OUTCOME_COLS:
                if col not in row or not row[col]:
                    row[col] = "N/A"
        return rows

    # ── Split new vs existing when partial enrichment is on ───────────────
    new_rows = high_rows
    existing_rows: List[Dict[str, str]] = []

    if use_partial:
        existing_ids = _get_existing_trial_ids(molecule)
        if existing_ids:
            new_rows = []
            existing_rows = []
            for r in high_rows:
                tid = (r.get("trial_id") or "").strip()
                if tid in existing_ids:
                    existing_rows.append(r)
                else:
                    new_rows.append(r)
            print(f"[PARTIAL] {len(new_rows)} new trial(s) → full enrichment, "
                  f"{len(existing_rows)} existing trial(s) → metadata-only.",
                  file=sys.stderr)
        else:
            print("[PARTIAL] No existing trials found in BQ — all trials treated as new.",
                  file=sys.stderr)

    semaphore = asyncio.Semaphore(max_workers)

    async def _staggered(coro, delay: float):
        await asyncio.sleep(delay)
        return await coro

    # ── Full enrichment for new trials ────────────────────────────────────
    full_merged: Dict[str, Dict[str, str]] = {}
    if new_rows:
        batches = [new_rows[i: i + BATCH_SIZE] for i in range(0, len(new_rows), BATCH_SIZE)]
        total = len(batches)
        print(f"\n[ENRICH] {len(new_rows)} new trial(s) across {total} batch(es) "
              f"(max {max_workers} concurrent)...\n", file=sys.stderr)

        staggered_tasks = [
            _staggered(_enrich_batch(molecule, batch, idx, total, semaphore), idx * 0.4)
            for idx, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*staggered_tasks, return_exceptions=False)

        for br in batch_results:
            if isinstance(br, dict):
                full_merged.update(br)

    # ── Metadata-only for existing trials (partial enrichment) ────────────
    meta_merged: Dict[str, Dict[str, str]] = {}
    if existing_rows:
        meta_batches = [existing_rows[i: i + _METADATA_BATCH_SIZE]
                        for i in range(0, len(existing_rows), _METADATA_BATCH_SIZE)]
        meta_total = len(meta_batches)
        print(f"\n[META-ONLY] {len(existing_rows)} existing trial(s) across {meta_total} "
              f"batch(es) (batch size {_METADATA_BATCH_SIZE})...\n", file=sys.stderr)

        meta_tasks = [
            _staggered(
                _metadata_only_batch(molecule, batch, idx, meta_total, semaphore),
                idx * 0.4,
            )
            for idx, batch in enumerate(meta_batches)
        ]
        meta_results = await asyncio.gather(*meta_tasks, return_exceptions=False)

        for mr in meta_results:
            if isinstance(mr, dict):
                meta_merged.update(mr)

    # ── Merge results into rows ───────────────────────────────────────────
    def _find_enrichment(tid: str, lookup: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        result = lookup.get(tid, {})
        if not result:
            for k, v in lookup.items():
                if k.strip().upper() == tid.strip().upper():
                    return v
        return result

    updated = 0

    # Apply full enrichment (metadata + endpoints) to new rows
    for row in new_rows:
        tid = row.get("trial_id", "")
        enrichment = _find_enrichment(tid, full_merged)
        if enrichment:
            for col in OUTCOME_COLS:
                val = enrichment.get(col, "")
                if val and str(val).strip().lower() not in ("n/a", "null", "none", ""):
                    row[col] = str(val).strip()
                elif col not in row or not row[col]:
                    row[col] = "N/A"
            for col in META_COLS:
                val = enrichment.get(col, "")
                if val and str(val).strip().lower() not in ("n/a", "null", "none", ""):
                    if not row.get(col, "").strip():
                        row[col] = str(val).strip()
            updated += 1
        else:
            for col in OUTCOME_COLS:
                if col not in row or not row[col]:
                    row[col] = "N/A"

    # Apply metadata-only to existing rows (no endpoint overwrite)
    meta_updated = 0
    for row in existing_rows:
        tid = row.get("trial_id", "")
        enrichment = _find_enrichment(tid, meta_merged)
        if enrichment:
            for col in META_COLS:
                val = enrichment.get(col, "")
                if val and str(val).strip().lower() not in ("n/a", "null", "none", ""):
                    if not row.get(col, "").strip():
                        row[col] = str(val).strip()
            meta_updated += 1
        # Mark endpoint columns as N/A if still empty (no endpoint extraction done)
        for col in OUTCOME_COLS:
            if col not in row or not row[col]:
                row[col] = "N/A"

    # Mark rows with confidence <= 0.6 with N/A for all outcome columns
    for row in rows:
        try:
            conf = float(row.get("llm_confidence", "0") or "0")
        except (ValueError, TypeError):
            conf = 0.0
        if conf <= 0.6:
            for col in OUTCOME_COLS:
                if col not in row or not row[col]:
                    row[col] = "N/A"

    if existing_rows:
        print(f"\n[ENRICH] Done: {updated}/{len(new_rows)} new trial(s) fully enriched, "
              f"{meta_updated}/{len(existing_rows)} existing trial(s) metadata-updated "
              f"({skip_count} below threshold skipped).\n", file=sys.stderr)
    else:
        print(f"\n[ENRICH] Done: {updated}/{len(high_rows)} trial(s) with confidence > 0.6 "
              f"updated ({skip_count} below threshold skipped).\n", file=sys.stderr)

    # Safety: coerce all values to str (Gemini may return ints/floats for
    # fields like trial_size, confidence scores, etc.)
    for row in rows:
        for k in list(row.keys()):
            if row[k] is not None and not isinstance(row[k], str):
                row[k] = str(row[k])

    return rows


# ==============================================================================
# SECTION 6 – SCORING
# ==============================================================================

SCORE_TABLE = [(22.0, 5), (16.0, 4), (10.0, 3), (5.0, 2), (0.0, 1)]
ENDPOINT_WEIGHTS = {"weight_loss": 0.40, "hba1c": 0.40, "mash": 0.10, "alt": 0.10}
FIELD_MAP = {
    "weight_loss": "weight_change_pct",
    "hba1c":       "hba1c_change_pct",
    "mash":        "mash_change_pct",
    "alt":         "alt_reduction_pct",
}
# Phase 4 (post-marketing) is treated the same as Phase 3 — no penalty.
PHASE_PENALTY = {4: 1.00, 3: 1.00, 2: 0.85, 1: 0.65}


def _parse_phase(raw) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().upper().replace("PHASE", "").strip()
    if s.startswith("4"): return 4
    if s.startswith("3"): return 3
    if s.startswith("2"): return 2
    if s.startswith("1"): return 1
    try:
        v = float(s)
        return 4 if v >= 4 else (3 if v >= 3 else (2 if v >= 2 else 1))
    except ValueError:
        return None


def _parse_float(raw) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("n/a", "", "0", "none", "null"):
        return None
    s = s.rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _pct_to_score(pct: float) -> int:
    for threshold, score in SCORE_TABLE:
        if pct >= threshold:
            return score
    return 1


def _collect_endpoint_rows(
    trials: List[Dict[str, str]],
    value_field: str,
    require_size: bool,
) -> List[Dict[str, Any]]:
    """Gather trials with a usable phase + value for this endpoint.

    When require_size is True, a trial also needs a parseable trial_size > 0
    to qualify. This is relaxed (require_size=False) as a fallback when no
    trial clears that bar, so an endpoint doesn't end up with no data purely
    because trial_size is missing.
    """
    out = []
    for t in trials:
        phase = _parse_phase(t.get("phase"))
        value = _parse_float(t.get(value_field))
        if phase is None or value is None:
            continue
        n = _parse_float(t.get("trial_size")) or 0
        if require_size and n <= 0:
            continue
        trial_id = t.get("trial_id") or t.get("Trial ID") or f"__unknown_{id(t)}"
        out.append({"phase": phase, "value": value, "n": n, "trial_id": trial_id, "full_trial": t})
    return out


def _score_endpoint(trials: List[Dict[str, str]], value_field: str) -> Dict[str, Any]:
    """Score a single endpoint (e.g. weight loss) across a molecule's trials.

    Selection rule: prefer Phase 4/3 data over Phase 2 over Phase 1. Within
    the highest available phase, dedupe multiple rows for the same trial_id
    (keep each trial's best arm), then take the single highest value among
    the deduped trials. There is no confidence-based filtering — any row
    with a parseable phase and value is eligible.

    If no trial has a usable trial_size, we fall back to picking from
    whichever trial has data at the highest available phase, ignoring size.
    """
    valid = _collect_endpoint_rows(trials, value_field, require_size=True)
    used_fallback = False
    if not valid:
        valid = _collect_endpoint_rows(trials, value_field, require_size=False)
        used_fallback = True

    if not valid:
        return {"best_value": None, "raw_value": None, "phase_used": None,
                "penalty": 1.0, "score": None, "trial_details": {},
                "reason": "No valid data for this endpoint"}

    for target_phase in (4, 3, 2, 1):
        phase_trials = [r for r in valid if r["phase"] == target_phase]
        if not phase_trials:
            continue
        trial_groups: Dict[str, list] = {}
        for t in phase_trials:
            trial_groups.setdefault(t["trial_id"], []).append(t)
        deduplicated = [max(arms, key=lambda x: x["value"]) for arms in trial_groups.values()]
        best = max(deduplicated, key=lambda r: r["value"])
        raw  = best["value"]
        pen  = PHASE_PENALTY[target_phase]
        adj  = raw * pen
        ft   = best.get("full_trial", {})
        reason = f"Phase {target_phase} data used" + (f" (x{pen} penalty applied)" if pen < 1 else "")
        if used_fallback:
            reason += " [fallback: no trial had a usable trial_size]"
        return {
            "best_value":  round(adj, 4),
            "raw_value":   round(raw, 4),
            "phase_used":  target_phase,
            "penalty":     pen,
            "score":       _pct_to_score(adj),
            "trial_details": {
                "trial_id":       best["trial_id"],
                "dosage":         ft.get("dosage", "N/A"),
                "weight_duration": ft.get("weight_duration", "N/A"),
                "hba1c_duration":  ft.get("hba1c_duration", "N/A"),
                "mash_duration":   ft.get("mash_duration", "N/A"),
                "alt_duration":    ft.get("alt_duration", "N/A"),
            },
            "reason": reason,
        }

    return {"best_value": None, "raw_value": None, "phase_used": None,
            "penalty": 1.0, "score": None, "trial_details": {}, "reason": "Unexpected state"}


def compute_clinical_efficacy_score(molecule: str, rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    endpoint_results = {
        ep: _score_endpoint(rows, field)
        for ep, field in FIELD_MAP.items()
    }

    score_sum, scored_eps, missing_eps = 0.0, [], []
    for ep, result in endpoint_results.items():
        w = ENDPOINT_WEIGHTS[ep]
        if result["score"] is not None:
            score_sum += result["score"] * w
            scored_eps.append(ep)
        else:
            missing_eps.append(ep)

    lines = []
    for ep, result in endpoint_results.items():
        w_pct = int(ENDPOINT_WEIGHTS[ep] * 100)
        if result["score"] is not None:
            lines.append(f"  {ep:12} | adj={result['best_value']:.2f}%  score={result['score']}  weight={w_pct}%  ({result['reason']})")
        else:
            lines.append(f"  {ep:12} | N/A  weight={w_pct}%  ({result['reason']})")

    coverage = (
        f"{len(scored_eps)}/4 endpoints scored"
        + (f" (missing: {', '.join(missing_eps)})" if missing_eps else "")
    )
    return {
        "molecule":        molecule,
        "total_trials":    total,
        "endpoints":       endpoint_results,
        "weighted_score":  round(score_sum, 3),
        "score_breakdown": "\n".join(lines),
        "data_coverage":   coverage,
    }


# ==============================================================================
# SECTION 7 – JSON I/O
# ==============================================================================

def _read_json(path: str) -> List[Dict[str, str]]:
    """Read a JSON file produced by trial_fetcher.py into a list of row dicts."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    rows = [{k: (str(v) if v is not None else "") for k, v in row.items()} for row in data]
    print(f"[INPUT] Loaded {len(rows)} row(s) from {path}", file=sys.stderr)
    return rows


# ==============================================================================
# SECTION 8 – PUBLIC API
# ==============================================================================

async def run_fetcher(
    molecule: str,
    rows: List[Dict[str, str]],
    max_workers: int = DEFAULT_WORKERS,
    no_score: bool = False,
) -> tuple:
    """
    Run enrichment + scoring on trial rows produced by trial_fetcher.py.
    Returns: (enriched_rows, score_result, score_rationale)
    """
    _check_api_key_early()

    if not rows:
        print("[FETCHER] No rows found in input.", file=sys.stderr)
        return [], None, None

    t0 = time.time()
    enriched_rows = await enrich_all(molecule, rows, max_workers=max_workers)
    print(f"[ENRICH] Time: {time.time() - t0:.1f}s", file=sys.stderr)

    score_result: Optional[Dict[str, Any]] = None
    if not no_score:
        print(f"\n[SCORE] Computing Clinical Efficacy Score...", file=sys.stderr)
        score_result = compute_clinical_efficacy_score(molecule, enriched_rows)
        print(f"  Weighted Score : {score_result['weighted_score']} / 5.0", file=sys.stderr)
        print(f"  Coverage       : {score_result['data_coverage']}", file=sys.stderr)
        print(f"  Breakdown:\n{score_result['score_breakdown']}", file=sys.stderr)

    if score_result:
        for row_data in enriched_rows:
            row_data["efficacy_score"]      = str(score_result.get("weighted_score", ""))
            row_data["efficacy_data_coverage"]       = score_result.get("data_coverage", "")
            row_data["efficacy_score_breakdown"]     = score_result.get("score_breakdown", "")
            row_data["efficacy_narrative_rationale"] = ""

    return enriched_rows, score_result


# ==============================================================================
# SECTION 10 – CLI
# ==============================================================================

def _resolve_input_json(molecule: str, explicit: Optional[str]) -> str:
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"ERROR: File not found: {explicit}")
        return explicit
    candidate = f"{molecule.lower().replace(' ', '_')}_trials.json"
    if os.path.exists(candidate):
        return candidate
    candidates = [f for f in os.listdir(".") if f.endswith("_trials.json")]
    if len(candidates) == 1:
        print(f"  i Auto-discovered: {candidates[0]}", file=sys.stderr)
        return candidates[0]
    sys.exit(f"ERROR: Could not find input JSON. Expected: {candidate}\nOr use: --json <path>")


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich + score clinical trials from JSON (trial_fetcher.py output).")
    ap.add_argument("molecule")
    ap.add_argument("--json",        default=None, help="Input JSON file (default: <molecule>_trials.json)")
    ap.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--run-fetcher", action="store_true", help="Run trial_fetcher.py first to generate the input JSON")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--top-n",       type=int, default=None)
    ap.add_argument("--no-score",    action="store_true")
    args = ap.parse_args()

    molecule = args.molecule.strip()

    print(f"\n{'='*60}\n  FETCHER  -  {molecule}\n{'='*60}\n", file=sys.stderr)

    if args.run_fetcher:
        import subprocess
        cmd = [sys.executable, "trial_fetcher.py", molecule, "--no-enrich"]
        if args.max_records: cmd += ["--max-records", str(args.max_records)]
        if args.top_n:       cmd += ["--top-n", str(args.top_n)]
        print(f"> Running: {' '.join(cmd)}\n", file=sys.stderr)
        if subprocess.run(cmd).returncode != 0:
            sys.exit("ERROR: trial_fetcher.py failed.")

    json_path = _resolve_input_json(molecule, args.json)
    rows = _read_json(json_path)
    t0 = time.time()
    enriched_rows, score_result, score_rationale = asyncio.run(
        run_fetcher(molecule, rows, max_workers=args.workers, no_score=args.no_score)
    )
    if not enriched_rows:
        return 1

    print(
        f"\nDone!\n  Rows       : {len(enriched_rows)}\n  Total time : {time.time()-t0:.1f}s\n",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
