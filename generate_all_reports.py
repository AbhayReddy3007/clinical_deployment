#!/usr/bin/env python3
"""
generate_all_reports.py - Generate the per-molecule clinical efficacy PDF
report for every drug currently in BigQuery (or a specific one), without
running the rest of the pipeline (no fetch, no enrichment, no re-scoring).

This is a thin driver around the existing report builder in
generate_report_and_raitonale/generate_efficacy_report.py:
  - load_from_bigquery()       -> discovers which molecules have data
  - generate_efficacy_report() -> builds one molecule's PDF (its own Gemini
                                   narrative call + its own BQ read)

Each PDF is written to --out-dir (default: ./reports), one file per
molecule. A failure or "no data" on one molecule is logged and skipped -
it does not stop the rest of the batch.

Usage (must run as a module, the same way fetcher.py / tester.py / rescorer.py are):

    python -m medical_potential.clinical_efficacy.generate_all_reports
    python -m medical_potential.clinical_efficacy.generate_all_reports --out-dir ./pdfs
    python -m medical_potential.clinical_efficacy.generate_all_reports "Semaglutide"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return slug or "unknown_molecule"


def _discover_molecules() -> List[str]:
    """All molecules that currently have data in the clinical_efficacy table
    (subject to the same llm_confidence filter generate_efficacy_report.py
    already applies when loading)."""
    from .generate_report_and_raitonale import load_from_bigquery

    molecule_data = load_from_bigquery()
    return sorted(molecule_data.keys())


def _generate_one(molecule: str, out_dir: str) -> Optional[str]:
    """Generate and save one molecule's PDF. Returns the file path written,
    or None if the report was skipped (no data / no API key) or failed."""
    from .generate_report_and_raitonale import generate_efficacy_report

    print(f"\n[REPORT] {molecule} ...", file=sys.stderr)
    try:
        pdf_bytes = generate_efficacy_report(molecule_name=molecule)
    except Exception as exc:
        print(f"[REPORT] FAILED for {molecule!r}: {exc}", file=sys.stderr)
        return None

    if not pdf_bytes:
        print(f"[REPORT] Skipped {molecule!r} (no data or report generation returned nothing).",
              file=sys.stderr)
        return None

    path = os.path.join(out_dir, f"{_slugify(molecule)}_efficacy_report.pdf")
    with open(path, "wb") as fh:
        fh.write(pdf_bytes)
    print(f"[REPORT] Saved {molecule!r} -> {path} ({len(pdf_bytes)} bytes)", file=sys.stderr)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate the clinical efficacy PDF report for every drug in BigQuery "
                    "(or a single named one), without running fetch/enrichment/scoring."
    )
    ap.add_argument("molecule", nargs="?", default=None,
                     help="Generate the report for just this molecule instead of all of them.")
    ap.add_argument("--out-dir", default="./reports",
                     help="Directory to write the PDF files into (default: ./reports).")
    args = ap.parse_args()

    try:
        os.makedirs(args.out_dir, exist_ok=True)

        if args.molecule:
            molecules = [args.molecule.strip()]
        else:
            print("[REPORT] Discovering molecules in BigQuery ...", file=sys.stderr)
            molecules = _discover_molecules()
            if not molecules:
                print("[REPORT] No molecules found.", file=sys.stderr)
                return 1
            print(f"[REPORT] Found {len(molecules)} molecule(s): {molecules}", file=sys.stderr)

        results = {}
        for molecule in molecules:
            results[molecule] = _generate_one(molecule, args.out_dir)

        succeeded = [m for m, p in results.items() if p]
        skipped = [m for m, p in results.items() if not p]

        print(f"\n{'='*64}")
        print(f"  DONE: {len(succeeded)}/{len(results)} report(s) generated in {args.out_dir}")
        if skipped:
            print(f"  Skipped/failed ({len(skipped)}): {skipped}")
        print(f"{'='*64}\n")

        return 0 if succeeded or not results else 1

    except ImportError as exc:
        if "relative import" in str(exc) or "no known parent package" in str(exc):
            sys.exit(
                "ERROR: Could not import generate_report_and_raitonale as a relative import "
                f"({exc}).\n"
                "This script must be run as a module, e.g.:\n\n"
                "    python -m medical_potential.clinical_efficacy.generate_all_reports\n\n"
                "Running it directly as `python generate_all_reports.py` will not work because "
                "these modules use package-relative imports."
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
