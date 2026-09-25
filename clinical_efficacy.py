#!/usr/bin/env python3
"""
clinical_efficacy.py – Clinical efficacy pipeline for a single molecule.

Steps:
  1. trial_fetcher  – Fetch raw trials from all registries
  2. fetcher        – Enrich trials + compute efficacy score
  3. push_to_bq     – Push enriched rows to BigQuery (incremental)
  4. generate_efficacy_report – Generate PDF report + upload to GCS

Usage:
    python -m medical_potential.clinical_efficacy.clinical_efficacy Semaglutide
    python -m medical_potential.clinical_efficacy.clinical_efficacy CagriSema --skip-fetch
    python -m medical_potential.clinical_efficacy.clinical_efficacy CagriSema --no-report

Programmatic use:
    from medical_potential.clinical_efficacy.clinical_efficacy import clinical_efficacy_assessment

    clinical_efficacy_assessment("CagriSema")
"""

from __future__ import annotations
import json
import os
import sys
import time
from typing import List, Optional

from medical_potential.config import CLINICAL_EFFICACY_DIMENSION_NAME
from medical_potential.gcp_utils import (
    append_dimension_score_to_bigquery,
    upload_dimension_payload_cache_to_gcs,
    upload_dimension_report_pdf_to_gcs,
)


# ==============================================================================
# HELPERS
# ==============================================================================

async def _run_trial_fetcher(
    molecule: str,
    max_records: Optional[int],
    top_n: Optional[int],
    out_json: str,
) -> List[dict]:
    """Fetch raw trials via trial_fetcher.fetch_trials. Exits on failure."""
    from .trial_fetcher import fetch_trials

    print(f"\n[CE] Step 1 – Fetching trials ...", file=sys.stderr)
    rows = await fetch_trials(
        molecule    = molecule,
        max_records = max_records,
        top_n       = top_n,
        no_enrich   = True,
        out_json    = None,
    )
    if not rows:
        sys.exit(f"ERROR: No trials found for {molecule}. Aborting.")
    print(f"[CE] Trials written to {out_json}", file=sys.stderr)
    return rows


# ==============================================================================
# PIPELINE
# ==============================================================================

async def clinical_efficacy_assessment(
    molecule_name: str,
    max_records: Optional[int] = None,
    top_n: Optional[int] = None,
    workers: int = 6,
    no_score: bool = False,
    skip_fetch: bool = False,
    json_path: Optional[str] = None,
    no_report: bool = False,
    step1_only: bool = False,
) -> int:
    """
    Execute the full clinical efficacy pipeline for a single molecule.

    If step1_only=True, runs only Step 1 (fetch) and Step 3 (push to BQ).

    Returns 0 on success, non-zero on failure.
    """
    slug     = molecule_name.lower().replace(" ", "_")
    raw_json = json_path or f"{slug}_trials.json"
    t_start  = time.time()

    print(f"\n{'='*64}", file=sys.stderr)
    print(f"  CLINICAL EFFICACY PIPELINE  –  {molecule_name}"
          f"{'  [STEP 1 ONLY]' if step1_only else ''}", file=sys.stderr)
    print(f"{'='*64}", file=sys.stderr)

    # ── Step 1: Fetch raw trials ───────────────────────────────────────────
    _step1_rows: Optional[List[dict]] = None
    if skip_fetch:
        if not os.path.exists(raw_json):
            print(f"[CE] ERROR: --skip-fetch set but {raw_json} not found.", file=sys.stderr)
            return 1
        print(f"\n[CE] Step 1 – Skipped (using {raw_json})", file=sys.stderr)
    else:
        _step1_rows = await _run_trial_fetcher(molecule_name, max_records, top_n, raw_json)

    if step1_only:
        print(f"\n[CE] Step 2 – Skipped (--step1 mode)", file=sys.stderr)

        if _step1_rows is None:
            print("[CE] No rows to append. Aborting.", file=sys.stderr)
            return 1

        raw_rows = [{k: (str(v) if v is not None else "") for k, v in r.items()}
                    for r in _step1_rows]

        print(f"\n[CE] Step 3 – Pushing {len(raw_rows)} trial(s) to BigQuery ...",
              file=sys.stderr)
        try:
            from .push_to_bq import save_clinical_efficacy_to_bq
        except ImportError:
            print("[CE] ERROR: push_to_bq.py not found.", file=sys.stderr)
            return 1

        save_clinical_efficacy_to_bq(
            molecule_name = molecule_name,
            trials        = raw_rows,
            score_result  = None,
            rationale     = None,
        )

        print(f"\n[CE] Step 4 – Skipped (--step1 mode)", file=sys.stderr)

        elapsed = time.time() - t_start
        print(
            f"\n{'='*64}\n"
            f"  DONE (step1 only)  –  {molecule_name}  ({elapsed:.1f}s)\n"
            f"{'='*64}\n",
            file=sys.stderr,
        )
        return 0

    # ── Step 2: Enrich + score ────────────────────────────────────────────
    print(f"\n[CE] Step 2 – Enrichment + scoring (fetcher.py) ...", file=sys.stderr)
    try:
        from . import fetcher
    except ImportError:
        print("[CE] ERROR: fetcher.py not found.", file=sys.stderr)
        return 1

    if _step1_rows is None:
        print("[CE] No trials found in step 1. Aborting.", file=sys.stderr)
        return 1

    enriched_rows, score_result, = await fetcher.run_fetcher(
        molecule    = molecule_name,
        rows        = _step1_rows,
        max_workers = workers,
        no_score    = no_score,
    )

    if not enriched_rows:
        print("[CE] No enriched rows returned. Aborting.", file=sys.stderr)
        return 1

    print(f"\n[CE] Enrichment complete: {len(enriched_rows)} trial(s)", file=sys.stderr)
    if score_result:
        print(
            f"  Efficacy score : {score_result['weighted_score']} / 5.0\n"
            f"  Coverage       : {score_result['data_coverage']}",
            file=sys.stderr,
        )

    score_rationale = None
    if score_result and not no_score:
        print(f"\n[CE] Step 2b – Generating rationale ...", file=sys.stderr)
        try:
            from .generate_report_and_raitonale import generate_score_rationale
        except ImportError:
            print("[CE] WARN: generate_rationale.py not found – skipping.", file=sys.stderr)
        else:
            try:
                score_rationale = await generate_score_rationale(molecule_name, score_result)
            except Exception as exc:
                print(f"[CE] WARN: Rationale generation failed: {exc}", file=sys.stderr)

    # ── Step 3: Push to BigQuery ──────────────────────────────────────────
    print(f"\n[CE] Step 3 – Pushing to BigQuery ...", file=sys.stderr)
    try:
        from .push_to_bq import save_clinical_efficacy_to_bq
    except ImportError:
        print("[CE] ERROR: push_to_bq.py not found.", file=sys.stderr)
        return 1

    save_clinical_efficacy_to_bq(
        molecule_name = molecule_name,
        trials        = enriched_rows,
        score_result  = score_result,
        rationale     = score_rationale,
    )

    if score_result:
        try:
            append_dimension_score_to_bigquery(
                molecule_name  = molecule_name,
                dimension_name = "Clinical Efficacy",
                score          = score_result.get("weighted_score"),
                pillar_name    = "Medical Potential",
                rationale      = score_rationale,
            )
        except Exception as exc:
            print(f"[CE] Warning: could not append dim score: {exc}", file=sys.stderr)

    # ── Step 4: Generate PDF report ───────────────────────────────────────
    if no_report:
        print(f"\n[CE] Step 4 – Skipped (--no-report)", file=sys.stderr)
    else:
        print(f"\n[CE] Step 4 – Generating efficacy report ...", file=sys.stderr)
        try:
            from .generate_report_and_raitonale import generate_efficacy_report
        except ImportError:
            print("[CE] WARN: generate_efficacy_report.py not found – skipping.", file=sys.stderr)
        else:
            try:
                    pdf_bytes = generate_efficacy_report(molecule_name=molecule_name)
                    if pdf_bytes:
                        report_gcs_uri, report_archive_gcs_uri = upload_dimension_report_pdf_to_gcs(
                            pdf_bytes=pdf_bytes,
                            molecule_name=molecule_name,
                            dimension_name=CLINICAL_EFFICACY_DIMENSION_NAME,
                        )
                        print(
                            f"[CE] Report uploaded to GCS: {report_gcs_uri}",
                            file=sys.stderr,
                        )
                    else:
                        report_gcs_uri = None
                        report_archive_gcs_uri = None
                        print(
                            "[CE] Report generation returned no output "
                            "(check GEMINI_API_KEY and BQ data).",
                            file=sys.stderr,
                        )
            except Exception as exc:
                print(f"[CE] WARN: Report generation failed: {exc}", file=sys.stderr)

        output_payload = {
            "molecule_name": molecule_name,
            "score_result": score_result,
            "score_rationale": score_rationale,
            "report_gcs_uri": locals().get("report_gcs_uri"),
            "report_archive_gcs_uri": locals().get("report_archive_gcs_uri"),
            "enriched_trials": len(enriched_rows),
            "enriched_trials_rows": enriched_rows,
        }

        try:
            cache_gcs_uri = upload_dimension_payload_cache_to_gcs(
                payload=output_payload,
                molecule_name=molecule_name,
                dimension_name=CLINICAL_EFFICACY_DIMENSION_NAME,
            )
            print(f"[CE] Output payload cached to GCS: {cache_gcs_uri}", file=sys.stderr)
        except Exception as exc:
            print(f"[CE] Warning: could not cache output payload: {exc}", file=sys.stderr)

    elapsed = time.time() - t_start
    print(
        f"\n{'='*64}\n"
        f"  DONE  –  {molecule_name}  ({elapsed:.1f}s)\n"
        f"{'='*64}\n",
        file=sys.stderr,
    )