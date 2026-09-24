"""
llm_eval.py – LLM-based confidence evaluation for trial-drug matching.

Uses Gemini 3.1 Pro Preview with Google Search grounding to verify whether
each clinical trial genuinely studies the specified drug.

Two modes of operation:

  1. In-memory (called by trial_fetcher.py):
       evaluate_trials(molecule, rows) -> updates rows in-place

  2. Standalone against BigQuery:
       python llm_eval.py "Cagrilintide+Semaglutide"
       python llm_eval.py "Semaglutide" --dry-run
       python llm_eval.py --all

     Reads existing rows from the clinical_efficacy BQ table, evaluates
     any row missing llm_confidence, and writes the results back.

Public API
----------
    evaluate_trials(molecule: str, rows: list[dict], aliases: str = "") -> None
        Updates rows in-place with llm_confidence (0-1) and llm_confidence_rationale.
        Rows from ClinicalTrials.gov or EU CTIS or Global Data are auto-set to 1.0.

    evaluate_from_bq(molecule: str, dry_run: bool = False) -> int
        Reads rows from BQ, evaluates, writes back. Returns count updated.

    evaluate_all_from_bq(dry_run: bool = False) -> dict[str, int]
        Evaluates all molecules in the BQ table.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

# ── GCP config from medical_potential.config ─────────────────────────────────
from medical_potential.config import BQ_DATASET_ID, CLINICAL_EFFICACY_TABLE, PROJECT_ID
from .utils import LLM_EVAL_MODEL, get_gemini_client, full_bq_table, eval_trial_batch

from medical_potential.gcp_utils import get_bq_client

try:
    from google import genai
    from google.genai import types as _gtypes
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

# ── Constants ─────────────────────────────────────────────────────────────────

# LLM_EVAL_MODEL is imported from utils (GEMINI_PRO_MODEL env var,
# default: gemini-3.1-pro-preview)
BATCH_SIZE        = eval_trial_batch  # from utils config (default 6)
MAX_WORKERS       = 1
AUTO_HIGH_SOURCES = {"clinicaltrials.gov", "eu ctis","gd clinical trials"}

# ── Prompt ────────────────────────────────────────────────────────────────────

_EVAL_PROMPT = """You are a pharmaceutical data validation engine.

DRUG: {molecule}
(Also known as: {aliases})

For EACH trial below, determine whether it genuinely studies the drug above
(or a combination containing it).  Consider the trial title, indication,
sponsor, phase, and registry source.

TRIALS:
{trial_lines}

For each trial, assign a confidence score between 0 and 1:
  - 1.0  — clearly studies this drug (drug name or known alias appears
            in the title, or the sponsor matches the known innovator)
  - 0.7-0.9 — probably studies this drug but not 100% certain (generic
               title, no drug name visible, but indication and sponsor fit)
  - 0.4-0.6 — uncertain (could be this drug or a related one, weak signal)
  - 0.1-0.3 — likely unrelated (different drug, different mechanism,
               wrong indication, or the trial ID appears to be a false
               positive from registry search)
  - 0.0  — definitely not related to this drug

Also provide a brief rationale (1-2 sentences) explaining WHY you assigned
that score.

Return ONLY valid JSON — no markdown, no preamble:

{{
  "results": {{
    "<trial_id>": {{
      "confidence": 0.9,
      "rationale": "Trial title explicitly mentions CagriSema and sponsor is Novo Nordisk."
    }},
    "<trial_id>": {{
      "confidence": 0.3,
      "rationale": "Trial studies a GLP-1 agonist but title refers to a different compound."
    }}
  }}
}}
"""


# ── JSON parsing helper ───────────────────────────────────────────────────────

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


# ── Batch evaluation ──────────────────────────────────────────────────────────

def _check_batch(
    client,
    molecule: str,
    aliases: str,
    batch: List[Dict[str, str]],
    batch_idx: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Send one batch of trials to Gemini for confidence evaluation.
    Returns {trial_id: {"confidence": float, "rationale": str}}.
    """
    trial_lines = "\n".join(
        f"  - {r.get('trial_id', '?')} | {r.get('trial_title', '')[:100]} "
        f"| Phase: {r.get('phase', '?')} | Sponsor: {r.get('company_name', '?')} "
        f"| Source: {r.get('registry_source', '?')}"
        for r in batch
    )
    prompt = _EVAL_PROMPT.format(
        molecule=molecule,
        aliases=aliases,
        trial_lines=trial_lines,
    )
    try:
        contents = [_gtypes.Content(role="user",
                                    parts=[_gtypes.Part.from_text(text=prompt)])]
        config = _gtypes.GenerateContentConfig(
            tools=[_gtypes.Tool(googleSearch=_gtypes.GoogleSearch())]
        )
        out = ""
        for chunk in client.models.generate_content_stream(
            model=LLM_EVAL_MODEL, contents=contents, config=config
        ):
            if chunk.text:
                out += chunk.text
        raw = out.strip()

        parsed = _safe_json(raw)
        if isinstance(parsed, dict):
            results = parsed.get("results", parsed)
            if isinstance(results, dict):
                out_dict: Dict[str, Dict[str, Any]] = {}
                for k, v in results.items():
                    tid = k.strip().upper()
                    if isinstance(v, dict):
                        out_dict[tid] = {
                            "confidence": v.get("confidence", ""),
                            "rationale":  str(v.get("rationale", "")),
                        }
                    elif isinstance(v, (int, float)):
                        out_dict[tid] = {
                            "confidence": v,
                            "rationale":  "",
                        }
                return out_dict
        return {}
    except Exception as exc:
        print(f"  [LLM_EVAL] Batch {batch_idx + 1} failed: {exc}", file=sys.stderr)
        return {}


def _apply_results(
    batch: List[Dict[str, str]],
    results: Dict[str, Dict[str, Any]],
) -> int:
    """Apply batch results to rows in-place. Returns count updated."""
    updated = 0
    for r in batch:
        tid = r.get("trial_id", "").strip().upper()
        entry = results.get(tid, {})
        conf = entry.get("confidence", "")
        rationale = entry.get("rationale", "")
        if conf != "" and conf is not None:
            try:
                conf_val = float(conf)
                conf_val = max(0.0, min(1.0, conf_val))
                r["llm_confidence"] = str(round(conf_val, 2))
                r["llm_confidence_rationale"] = rationale
                updated += 1
            except (ValueError, TypeError):
                r["llm_confidence"] = ""
                r["llm_confidence_rationale"] = ""
        else:
            r["llm_confidence"] = ""
            r["llm_confidence_rationale"] = ""
    return updated


# ── Public API: in-memory evaluation ──────────────────────────────────────────

def evaluate_trials(
    molecule: str,
    rows: List[Dict[str, str]],
    aliases: str = "",
) -> None:
    """
    Evaluate llm_confidence for a list of trial rows in-place.

    Rows from ClinicalTrials.gov or EU CTIS or Global Data are auto-set to 1.0.
    Remaining rows are sent to Gemini in batches.

    Parameters
    ----------
    molecule : Primary drug name.
    rows     : List of trial dicts (modified in-place).
    aliases  : Comma-separated alias string for context.
    """
    if not rows:
        return

    # Auto-set authoritative sources
    auto_count = 0
    needs_check: List[Dict[str, str]] = []
    for r in rows:
        src = (r.get("registry_source") or "").strip().lower()
        if src in AUTO_HIGH_SOURCES:
            r["llm_confidence"] = "1"
            r["llm_confidence_rationale"] = (
                f"Auto-assigned: trial sourced directly from {r.get('registry_source', '')} "
                f"(authoritative primary registry)."
            )
            auto_count += 1
        else:
            needs_check.append(r)

    print(f"[LLM_EVAL] {auto_count} trial(s) auto-set to 1.0 "
          f"(ClinicalTrials.gov / EU CTIS/ Global Data trials).", file=sys.stderr)

    if not needs_check:
        print("[LLM_EVAL] No trials need LLM verification.", file=sys.stderr)
        return

    if not _HAS_GENAI:
        print("[LLM_EVAL] Gemini not available — skipping LLM check.", file=sys.stderr)
        for r in needs_check:
            r["llm_confidence"] = ""
            r["llm_confidence_rationale"] = ""
        return

    print(f"[LLM_EVAL] Verifying {len(needs_check)} trial(s) with {LLM_EVAL_MODEL} …",
          file=sys.stderr)

    alias_str = aliases or molecule
    try:
        client = get_gemini_client()
    except (ValueError, ImportError) as exc:
        print(f"[LLM_EVAL] Could not create Gemini client: {exc}", file=sys.stderr)
        return

    batches = [needs_check[i:i + BATCH_SIZE]
               for i in range(0, len(needs_check), BATCH_SIZE)]
    total_checked = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_check_batch, client, molecule, alias_str, batch, idx): (idx, batch)
            for idx, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            idx, batch = futures[fut]
            try:
                results = fut.result()
                total_checked += _apply_results(batch, results)
            except Exception as exc:
                print(f"  [LLM_EVAL] Batch {idx + 1} worker error: {exc}", file=sys.stderr)
                for r in batch:
                    r["llm_confidence"] = ""
                    r["llm_confidence_rationale"] = ""

    print(f"[LLM_EVAL] Checked {total_checked}/{len(needs_check)} trial(s).",
          file=sys.stderr)


# ── BQ schema helper ──────────────────────────────────────────────────────────

def _ensure_bq_columns(bq_client, table_id: str) -> None:
    """
    Ensure llm_confidence and llm_confidence_rationale columns exist in the
    BQ table. Adds them as STRING columns if missing.
    """
    from google.cloud import bigquery

    try:
        table = bq_client.get_table(table_id)
        existing = {f.name.lower() for f in table.schema}

        new_fields = []
        if "llm_confidence" not in existing:
            new_fields.append(bigquery.SchemaField("llm_confidence", "STRING"))
        if "llm_confidence_rationale" not in existing:
            new_fields.append(bigquery.SchemaField("llm_confidence_rationale", "STRING"))

        if new_fields:
            print(f"[LLM_EVAL] Adding column(s) to BQ table: "
                  f"{[f.name for f in new_fields]}", file=sys.stderr)
            table.schema = list(table.schema) + new_fields
            bq_client.update_table(table, ["schema"])
            print("[LLM_EVAL] Columns added successfully.", file=sys.stderr)
    except Exception as exc:
        print(f"[LLM_EVAL] Warning: could not verify/add BQ columns: {exc}",
              file=sys.stderr)


# ── Public API: BigQuery evaluation ───────────────────────────────────────────

def evaluate_from_bq(
    molecule: str,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """
    Read existing rows from the clinical_efficacy BQ table for a molecule,
    evaluate any missing llm_confidence, and write results back.

    Parameters
    ----------
    molecule : Drug name to filter on (matches molecule_name column).
    dry_run  : If True, print results but don't write to BQ.
    force    : If True, re-evaluate even rows that already have llm_confidence.

    Returns count of rows updated.
    """
    print(f"\n[LLM_EVAL] Loading rows for '{molecule}' from BigQuery …", file=sys.stderr)

    bq_client = get_bq_client()
    table_id  = full_bq_table(CLINICAL_EFFICACY_TABLE)

    # Read existing rows
    query = f"""
        SELECT *
        FROM `{table_id}`
        WHERE LOWER(molecule_name) = LOWER(@molecule)
    """
    job_config = __import__("google.cloud.bigquery", fromlist=["QueryJobConfig"]).QueryJobConfig(
        query_parameters=[
            __import__("google.cloud.bigquery", fromlist=["ScalarQueryParameter"]).ScalarQueryParameter(
                "molecule", "STRING", molecule
            )
        ]
    )

    try:
        result = bq_client.query(query, job_config=job_config)
        rows_raw = [dict(row.items()) for row in result]
    except Exception as exc:
        print(f"[LLM_EVAL] BQ query failed: {exc}", file=sys.stderr)
        return 0

    if not rows_raw:
        print(f"[LLM_EVAL] No rows found for '{molecule}' in BQ.", file=sys.stderr)
        return 0

    print(f"[LLM_EVAL] {len(rows_raw)} row(s) loaded.", file=sys.stderr)

    # Convert all values to strings (matching trial_fetcher format)
    rows = [{k: (str(v) if v is not None else "") for k, v in r.items()} for r in rows_raw]

    # Filter to rows needing evaluation
    if force:
        to_eval = rows
    else:
        to_eval = [r for r in rows
                   if not r.get("llm_confidence", "").strip()]

    if not to_eval:
        print("[LLM_EVAL] All rows already have llm_confidence — nothing to do "
              "(use --force to re-evaluate).", file=sys.stderr)
        return 0

    print(f"[LLM_EVAL] {len(to_eval)} row(s) need evaluation.", file=sys.stderr)

    # Run evaluation
    evaluate_trials(molecule, to_eval)

    # Count rows that got a score
    updated_rows = [r for r in to_eval
                    if r.get("llm_confidence", "").strip()]
    updated_count = len(updated_rows)

    if dry_run:
        print(f"\n[LLM_EVAL] DRY RUN — {updated_count} row(s) would be updated:",
              file=sys.stderr)
        for r in updated_rows[:20]:
            print(f"  {r.get('trial_id', '?'):25s} → {r.get('llm_confidence', '?'):5s}  "
                  f"{r.get('llm_confidence_rationale', '')[:80]}", file=sys.stderr)
        if updated_count > 20:
            print(f"  ... and {updated_count - 20} more.", file=sys.stderr)
        return updated_count

    # Write back to BQ
    if not updated_rows:
        print("[LLM_EVAL] No rows to update.", file=sys.stderr)
        return 0

    # Ensure the columns exist in the BQ table (add if missing)
    _ensure_bq_columns(bq_client, table_id)

    print(f"[LLM_EVAL] Updating {updated_count} row(s) in BigQuery …", file=sys.stderr)

    # Use DML UPDATE for each row (merge by trial_id + molecule_name)
    update_count = 0
    for r in updated_rows:
        tid  = r.get("trial_id", "")
        conf = r.get("llm_confidence", "")
        rat  = r.get("llm_confidence_rationale", "").replace("'", "\\'")
        if not tid or not conf:
            continue

        update_sql = f"""
            UPDATE `{table_id}`
            SET llm_confidence = '{conf}',
                llm_confidence_rationale = '{rat}'
            WHERE trial_id = '{tid}'
              AND LOWER(molecule_name) = LOWER('{molecule}')
        """
        try:
            bq_client.query(update_sql).result()
            update_count += 1
        except Exception as exc:
            print(f"  [LLM_EVAL] Failed to update {tid}: {exc}", file=sys.stderr)

    print(f"[LLM_EVAL] {update_count}/{updated_count} row(s) updated in BigQuery.",
          file=sys.stderr)
    return update_count


def evaluate_all_from_bq(
    dry_run: bool = False,
    force: bool = False,
) -> Dict[str, int]:
    """
    Evaluate all molecules in the clinical_efficacy BQ table.
    Returns {molecule: rows_updated}.
    """
    print("[LLM_EVAL] Loading all distinct molecules from BigQuery …", file=sys.stderr)

    bq_client = get_bq_client()
    table_id  = full_bq_table(CLINICAL_EFFICACY_TABLE)

    query = f"SELECT DISTINCT molecule_name FROM `{table_id}` ORDER BY molecule_name"
    try:
        molecules = [row.molecule_name for row in bq_client.query(query).result()]
    except Exception as exc:
        print(f"[LLM_EVAL] BQ query failed: {exc}", file=sys.stderr)
        return {}

    print(f"[LLM_EVAL] Found {len(molecules)} molecule(s): {molecules}", file=sys.stderr)

    results: Dict[str, int] = {}
    for idx, mol in enumerate(molecules, 1):
        print(f"\n[LLM_EVAL] === [{idx}/{len(molecules)}] {mol} ===", file=sys.stderr)
        count = evaluate_from_bq(mol, dry_run=dry_run, force=force)
        results[mol] = count

    # Summary
    total = sum(results.values())
    print(f"\n[LLM_EVAL] === SUMMARY ===", file=sys.stderr)
    for mol, count in results.items():
        print(f"  {mol:40s} {count} row(s) updated", file=sys.stderr)
    print(f"  {'TOTAL':40s} {total} row(s)", file=sys.stderr)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            "LLM confidence evaluation for clinical trial - drug matching. "
            "Reads from BigQuery, evaluates with Gemini, writes back."
        )
    )
    ap.add_argument(
        "molecule", nargs="?", default=None,
        help="Drug name to evaluate. Omit with --all to evaluate everything."
    )
    ap.add_argument("--all", action="store_true",
                    help="Evaluate all molecules in the BQ table.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print results without writing to BQ.")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate rows that already have llm_confidence.")
    args = ap.parse_args()

    if args.all:
        results = evaluate_all_from_bq(dry_run=args.dry_run, force=args.force)
        return 0 if results else 1
    elif args.molecule:
        count = evaluate_from_bq(args.molecule, dry_run=args.dry_run, force=args.force)
        return 0 if count >= 0 else 1
    else:
        ap.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
