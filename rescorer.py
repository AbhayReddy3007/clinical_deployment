#!/usr/bin/env python3
"""
rescorer.py - Recompute the clinical efficacy score for a molecule's
EXISTING BigQuery trial rows using the current scoring logic, and write the
result back to BigQuery.

Unlike tester.py (read-only, just prints the score), this script UPDATES
these four columns in the clinical_efficacy table, for every row belonging
to the molecule:

    efficacy_score
    efficacy_data_coverage
    efficacy_score_breakdown
    efficacy_narrative_rationale

It does NOT re-fetch trials, re-run Gemini endpoint enrichment, or insert
any new rows - it only re-scores trial data that's already in BigQuery
(using compute_clinical_efficacy_score from fetcher.py) and writes those
four score-related columns back onto the existing rows.

Note: if the "no trial had a usable trial_size" fallback was used to pick
an endpoint's value, that is intentionally NOT mentioned in the
efficacy_score_breakdown written back to BigQuery - the breakdown text is
stripped of that annotation either way, so it reads the same as a normal
(non-fallback) selection.

Usage (must run as a module, the same way fetcher.py / tester.py are):

    python -m medical_potential.clinical_efficacy.rescorer "Semaglutide"
    python -m medical_potential.clinical_efficacy.rescorer --all
    python -m medical_potential.clinical_efficacy.rescorer "Semaglutide" --no-rationale
    python -m medical_potential.clinical_efficacy.rescorer "Semaglutide" --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Strips the "[fallback: ...]" annotation that fetcher.py's scorer adds to
# an endpoint's reason string when no trial had a usable trial_size. The
# breakdown written back to BigQuery should read the same whether or not
# that fallback was used.
_FALLBACK_NOTE_RE = re.compile(r"\s*\[fallback: no trial had a usable trial_size\]")


def _strip_fallback_note(breakdown: str) -> str:
    return _FALLBACK_NOTE_RE.sub("", breakdown or "")


def _table_id() -> str:
    from medical_potential.config import BQ_DATASET_ID, CLINICAL_EFFICACY_TABLE, PROJECT_ID
    return f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"


def _load_rows_from_bq(client, table_id: str, molecule: str) -> List[Dict[str, Any]]:
    from google.cloud import bigquery
    query = f"SELECT * FROM `{table_id}` WHERE molecule_name = @molecule_name"
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule)
    ])
    rows = [dict(row) for row in client.query(query, job_config=job_config).result()]
    return [{k: (str(v) if v is not None else "") for k, v in row.items()} for row in rows]


def _list_all_molecules(client, table_id: str) -> List[str]:
    query = f"SELECT DISTINCT molecule_name FROM `{table_id}` WHERE molecule_name IS NOT NULL"
    return [row.molecule_name for row in client.query(query).result() if row.molecule_name]


def _update_bq(
    client,
    table_id: str,
    molecule: str,
    score_result: Dict[str, Any],
    rationale: Optional[str],
    dry_run: bool,
) -> int:
    """Write the score columns back. rationale=None means "leave whatever
    efficacy_narrative_rationale currently holds untouched" (e.g. because
    rationale generation failed or was skipped) rather than blanking it."""
    from google.cloud import bigquery

    breakdown = _strip_fallback_note(score_result.get("score_breakdown", ""))
    efficacy_score = score_result.get("weighted_score")
    coverage = score_result.get("data_coverage", "")
    now = datetime.now(timezone.utc).isoformat()

    set_clauses = [
        "efficacy_score = @efficacy_score",
        "efficacy_data_coverage = @coverage",
        "efficacy_score_breakdown = @breakdown",
        "updated_at = @updated_at",
    ]
    params = [
        bigquery.ScalarQueryParameter("efficacy_score", "FLOAT64", efficacy_score),
        bigquery.ScalarQueryParameter("coverage", "STRING", coverage),
        bigquery.ScalarQueryParameter("breakdown", "STRING", breakdown),
        bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", now),
        bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule),
    ]
    if rationale is not None:
        set_clauses.append("efficacy_narrative_rationale = @rationale")
        params.append(bigquery.ScalarQueryParameter("rationale", "STRING", rationale))

    if dry_run:
        print(f"[DRY-RUN] Would update molecule_name = {molecule!r}:")
        print(f"  efficacy_score               = {efficacy_score}")
        print(f"  efficacy_data_coverage       = {coverage}")
        print(f"  efficacy_score_breakdown     =\n{breakdown}")
        if rationale is not None:
            print(f"  efficacy_narrative_rationale = {rationale!r}")
        else:
            print(f"  efficacy_narrative_rationale = <unchanged>")
        return 0

    sql = f"UPDATE `{table_id}` SET {', '.join(set_clauses)} WHERE molecule_name = @molecule_name"
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    job.result()
    return job.num_dml_affected_rows or 0


async def _rescore_one(
    molecule: str,
    client,
    table_id: str,
    generate_rationale: bool,
    dry_run: bool,
) -> None:
    from .fetcher import compute_clinical_efficacy_score

    print(f"[RESCORE] Loading existing rows for {molecule!r} ...", file=sys.stderr)
    rows = _load_rows_from_bq(client, table_id, molecule)
    if not rows:
        print(f"[SKIP] No existing rows found for {molecule!r}.", file=sys.stderr)
        return

    result = compute_clinical_efficacy_score(molecule, rows)

    rationale: Optional[str] = None
    if generate_rationale:
        try:
            # NOTE: matches the (intentionally renamed) folder name used by
            # clinical_efficacy.py.
            from .generate_report_and_raitonale import generate_score_rationale
        except ImportError as exc:
            print(f"[WARN] Could not import generate_score_rationale ({exc}); "
                  f"leaving efficacy_narrative_rationale unchanged for {molecule!r}.",
                  file=sys.stderr)
        else:
            try:
                rationale = await generate_score_rationale(molecule, result)
            except Exception as exc:
                print(f"[WARN] Rationale generation failed for {molecule!r}: {exc}; "
                      f"leaving efficacy_narrative_rationale unchanged.", file=sys.stderr)

    affected = _update_bq(client, table_id, molecule, result, rationale, dry_run)

    print(
        f"[DONE] {molecule!r}: weighted_score={result['weighted_score']}  "
        f"coverage={result['data_coverage']!r}  rows_updated={affected}"
    )


async def _run(args: argparse.Namespace) -> int:
    from medical_potential.gcp_utils import get_bq_client

    client = get_bq_client()
    table_id = _table_id()

    if args.all:
        molecules = _list_all_molecules(client, table_id)
        if not molecules:
            print("[RESCORE] No molecules found in the table.", file=sys.stderr)
            return 1
        print(f"[RESCORE] Re-scoring {len(molecules)} molecule(s): {molecules}", file=sys.stderr)
    else:
        molecules = [args.molecule.strip()]

    for molecule in molecules:
        await _rescore_one(
            molecule, client, table_id,
            generate_rationale=not args.no_rationale,
            dry_run=args.dry_run,
        )

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-score existing BigQuery trial rows with the current scoring logic "
                    "and write efficacy_score / efficacy_data_coverage / "
                    "efficacy_score_breakdown / efficacy_narrative_rationale back."
    )
    ap.add_argument("molecule", nargs="?", default=None,
                     help="Molecule / drug name to re-score. Required unless --all is given.")
    ap.add_argument("--all", action="store_true",
                     help="Re-score every molecule currently in the table.")
    ap.add_argument("--no-rationale", action="store_true",
                     help="Skip regenerating efficacy_narrative_rationale (no Gemini call); "
                          "leaves the existing value untouched.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Compute and print the new values without writing to BigQuery.")
    args = ap.parse_args()

    if not args.all and not args.molecule:
        ap.error("Provide a molecule name, or use --all to re-score every molecule.")

    try:
        return asyncio.run(_run(args))
    except ImportError as exc:
        if "relative import" in str(exc) or "no known parent package" in str(exc):
            sys.exit(
                "ERROR: Could not import fetcher.py as a relative import "
                f"({exc}).\n"
                "This script must be run as a module, e.g.:\n\n"
                "    python -m medical_potential.clinical_efficacy.rescorer \"Semaglutide\"\n\n"
                "Running it directly as `python rescorer.py` will not work because fetcher.py "
                "uses package-relative imports."
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
