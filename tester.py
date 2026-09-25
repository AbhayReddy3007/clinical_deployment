#!/usr/bin/env python3
"""
tester.py - Re-score EXISTING trial rows with the current scoring logic,
without re-running trial_fetcher / Gemini enrichment / BigQuery push.

Use this after changing the scoring logic in fetcher.py, to see what the
new logic would produce for a molecule's trials you've already fetched -
either from a local JSON file (trial_fetcher.py / fetcher.py output) or
straight from the BigQuery clinical-efficacy table.

This script is READ-ONLY: it never writes anything back to BigQuery or to
any file unless you pass --out, in which case it only writes the score
result (not the trial rows) to the path you give.

Usage (run as part of the package, so the relative imports in fetcher.py
resolve correctly - same way clinical_efficacy.py is run):

    python -m medical_potential.clinical_efficacy.tester "Semaglutide"
    python -m medical_potential.clinical_efficacy.tester "Semaglutide" --json semaglutide_trials.json
    python -m medical_potential.clinical_efficacy.tester "Semaglutide" --out score_result.json

If you see an ImportError about relative imports, you ran this as a plain
script (`python tester.py`) instead of as a module (`python -m ...`) -
see the message printed below for the exact command to use.
"""

from __future__ import annotations

import argparse
import json
import sys


def _load_rows_from_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        sys.exit(f"ERROR: Expected a JSON array of trial rows in {path}, got {type(data).__name__}")
    # Match fetcher._read_json's normalisation: everything as strings.
    return [{k: (str(v) if v is not None else "") for k, v in row.items()} for row in data]


def _load_rows_from_bq(molecule: str) -> list:
    """Pull this molecule's existing rows straight from the clinical
    efficacy BigQuery table - the same table push_to_bq.py writes to."""
    from medical_potential.config import BQ_DATASET_ID, CLINICAL_EFFICACY_TABLE, PROJECT_ID
    from medical_potential.gcp_utils import get_bq_client
    from google.cloud import bigquery

    table_id = f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"
    client = get_bq_client()
    query = f"SELECT * FROM `{table_id}` WHERE molecule_name = @molecule_name"
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule)
    ])
    rows = [dict(row) for row in client.query(query, job_config=job_config).result()]
    if not rows:
        sys.exit(f"ERROR: No existing rows found in {table_id} for molecule_name = {molecule!r}.")
    # Normalise to strings, same convention the scorer expects.
    return [{k: (str(v) if v is not None else "") for k, v in row.items()} for row in rows]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-score existing trial rows with the current scoring logic (no fetch/enrich/push)."
    )
    ap.add_argument("molecule", help="Molecule / drug name (used for BQ lookup and display).")
    ap.add_argument(
        "--json", default=None,
        help="Path to a local enriched trials JSON (trial_fetcher.py / fetcher.py output). "
             "If omitted, rows are pulled from the BigQuery clinical-efficacy table instead.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Optional path to also write the score result (not the trial rows) as JSON.",
    )
    args = ap.parse_args()

    try:
        from .fetcher import compute_clinical_efficacy_score
    except ImportError as exc:
        if "relative import" in str(exc) or "no known parent package" in str(exc):
            sys.exit(
                "ERROR: Could not import fetcher.py as a relative import "
                f"({exc}).\n"
                "This script must be run as a module, the same way clinical_efficacy.py is, e.g.:\n\n"
                "    python -m medical_potential.clinical_efficacy.tester \"Semaglutide\"\n\n"
                "Running it directly as `python tester.py` will not work because fetcher.py "
                "uses package-relative imports."
            )
        # Some other import failed inside fetcher.py (e.g. a missing
        # dependency or config value) - surface that error as-is rather
        # than the (wrong) "run with -m" hint.
        raise

    molecule = args.molecule.strip()

    if args.json:
        print(f"[TESTER] Loading rows from {args.json} ...", file=sys.stderr)
        rows = _load_rows_from_json(args.json)
    else:
        print(f"[TESTER] No --json given - pulling existing rows from BigQuery for '{molecule}' ...",
              file=sys.stderr)
        rows = _load_rows_from_bq(molecule)

    print(f"[TESTER] {len(rows)} row(s) loaded for '{molecule}'.", file=sys.stderr)

    result = compute_clinical_efficacy_score(molecule, rows)

    print(f"\n{'='*64}")
    print(f"  RE-SCORE (current logic)  -  {molecule}")
    print(f"{'='*64}")
    print(f"  Weighted Score : {result['weighted_score']} / 5.0")
    print(f"  Coverage       : {result['data_coverage']}")
    print(f"  Breakdown:\n{result['score_breakdown']}")
    print(f"{'='*64}\n")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"[TESTER] Score result written to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
