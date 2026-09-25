#!/usr/bin/env python3
"""
push_to_bq.py – Push enriched clinical trial data to BigQuery.

Incremental logic:
  - New trial_id  → INSERT full row
  - Existing trial_id with changed phase / phase_status / blank title / study → UPDATE
  - Unchanged → skip

Typically called by clinical_efficacy.py. Also works standalone:
    python push_to_bq.py <molecule_name> --json <enriched.json>
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from google.cloud import bigquery
except ImportError:
    sys.exit("ERROR: google-cloud-bigquery not installed.  Run: pip install google-cloud-bigquery")

from medical_potential.config import BQ_DATASET_ID, CLINICAL_EFFICACY_TABLE, PROJECT_ID

from medical_potential.gcp_utils import get_bq_client

# ==============================================================================
# TABLE SCHEMA
# ==============================================================================

# All confidence columns are FLOAT64 (0-1 numeric).
_SCHEMA = [
    bigquery.SchemaField("molecule_name",               "STRING"),
    bigquery.SchemaField("registry_source",             "STRING"),
    bigquery.SchemaField("trial_id",                    "STRING"),
    bigquery.SchemaField("acronym",                     "STRING"),
    bigquery.SchemaField("dosage",                      "STRING"),
    bigquery.SchemaField("phase",                       "STRING"),
    bigquery.SchemaField("trial_title",                 "STRING"),
    bigquery.SchemaField("trial_study",                 "STRING"),
    bigquery.SchemaField("trial_size",                  "STRING"),
    bigquery.SchemaField("trial_location",              "STRING"),
    bigquery.SchemaField("secondary_locations",         "STRING"),
    bigquery.SchemaField("trial_start_date",            "STRING"),
    bigquery.SchemaField("trial_completion_date",       "STRING"),
    bigquery.SchemaField("phase_status",                "STRING"),
    bigquery.SchemaField("hba1c_change_pct",            "STRING"),
    bigquery.SchemaField("hba1c_duration",              "STRING"),
    bigquery.SchemaField("hba1c_rationale",             "STRING"),
    bigquery.SchemaField("hba1c_confidence",            "FLOAT64"),
    bigquery.SchemaField("weight_change_pct",           "STRING"),
    bigquery.SchemaField("weight_duration",             "STRING"),
    bigquery.SchemaField("weight_rationale",            "STRING"),
    bigquery.SchemaField("weight_confidence",           "FLOAT64"),
    bigquery.SchemaField("alt_reduction_pct",           "STRING"),
    bigquery.SchemaField("alt_duration",                "STRING"),
    bigquery.SchemaField("alt_rationale",               "STRING"),
    bigquery.SchemaField("alt_confidence",              "FLOAT64"),
    bigquery.SchemaField("mash_change_pct",             "STRING"),
    bigquery.SchemaField("mash_duration",               "STRING"),
    bigquery.SchemaField("mash_rationale",              "STRING"),
    bigquery.SchemaField("mash_confidence",             "FLOAT64"),
    bigquery.SchemaField("company_name",                "STRING"),
    bigquery.SchemaField("source_url",                  "STRING"),
    bigquery.SchemaField("llm_confidence",              "FLOAT64"),
    bigquery.SchemaField("llm_confidence_rationale",    "STRING"),
    bigquery.SchemaField("trial_source",                "STRING"),
    bigquery.SchemaField("efficacy_score",      "FLOAT64"),
    bigquery.SchemaField("efficacy_data_coverage",       "STRING"),
    bigquery.SchemaField("efficacy_score_breakdown",     "STRING"),
    bigquery.SchemaField("efficacy_narrative_rationale", "STRING"),
    bigquery.SchemaField("created_at",                  "TIMESTAMP"),
    bigquery.SchemaField("updated_at",                  "TIMESTAMP"),
]

# Columns that may need to be added to an existing table
_NEW_COLS = [
    ("secondary_locations",         "STRING"),
    ("llm_confidence",              "FLOAT64"),
    ("llm_confidence_rationale",    "STRING"),
    ("trial_source",                "STRING"),
    ("efficacy_score",      "FLOAT64"),
    ("efficacy_data_coverage",       "STRING"),
    ("efficacy_score_breakdown",     "STRING"),
    ("efficacy_narrative_rationale", "STRING"),
    ("updated_at",                   "TIMESTAMP"),
]

# All confidence columns — must be numeric 0-1
_CONFIDENCE_COLS = [
    "hba1c_confidence", "weight_confidence", "alt_confidence", "mash_confidence",
    "llm_confidence",
]

# Legacy text-to-number mapping
_CONF_TEXT_MAP = {
    "high": 1.0, "medium": 0.6, "low": 0.4,
    "n/a": 0.0, "na": 0.0, "none": 0.0, "null": 0.0, "": 0.0,
}


def _safe_confidence(val: Any) -> Optional[float]:
    """Convert a confidence value to a float in [0.0, 1.0].

    Handles floats, ints, numeric strings, and legacy text labels (High/Medium/Low).
    Returns None only if the value is truly absent.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    mapped = _CONF_TEXT_MAP.get(s.lower())
    if mapped is not None:
        return mapped
    try:
        f = float(s)
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return 0.0


def _table_id() -> str:
    return f"{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"


def _ensure_table(client: bigquery.Client) -> None:
    tid = _table_id()
    try:
        client.get_table(tid)
    except Exception:
        dataset_ref = bigquery.DatasetReference(PROJECT_ID, BQ_DATASET_ID)
        try:
            client.get_dataset(dataset_ref)
        except Exception:
            client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)
            print(f"[BQ] Created dataset {BQ_DATASET_ID}", file=sys.stderr)
        client.create_table(bigquery.Table(tid, schema=_SCHEMA), exists_ok=True)
        print(f"[BQ] Created table {tid}", file=sys.stderr)


def _ensure_new_columns(client: bigquery.Client) -> None:
    """Add any missing columns to an existing table (migration helper)."""
    tid = _table_id()
    try:
        table = client.get_table(tid)
        existing = {f.name: f for f in table.schema}

        # Rename efficacy_weighted_score → efficacy_score if old column still exists
        if "efficacy_weighted_score" in existing and "efficacy_score" not in existing:
            client.query(
                f"ALTER TABLE `{tid}` RENAME COLUMN efficacy_weighted_score TO efficacy_score"
            ).result()
            print("[BQ] Renamed column efficacy_weighted_score → efficacy_score", file=sys.stderr)
            # Refresh existing map after rename
            existing = {f.name: f for f in client.get_table(tid).schema}

        for col_name, col_type in _NEW_COLS:
            if col_name not in existing:
                client.query(
                    f"ALTER TABLE `{tid}` ADD COLUMN IF NOT EXISTS `{col_name}` {col_type}"
                ).result()
                print(f"[BQ] Added column {col_name} ({col_type})", file=sys.stderr)
            elif col_name in _CONFIDENCE_COLS:
                # Warn if confidence column is STRING — BigQuery requires recreating
                # the column to change type; flag for manual action
                cur_type = existing[col_name].field_type
                if cur_type == "STRING" and col_type == "FLOAT64":
                    print(f"[BQ] Warning: column {col_name} is STRING but should be FLOAT64. "
                          f"Manual migration may be needed.", file=sys.stderr)
    except Exception as exc:
        print(f"[BQ] Warning: could not verify/add columns: {exc}", file=sys.stderr)


def _get_existing_trials(client: bigquery.Client, molecule_name: str) -> Dict[str, Dict[str, Any]]:
    query = f"""
        SELECT trial_id, phase, phase_status, trial_title, trial_study
        FROM `{_table_id()}`
        WHERE molecule_name = @molecule_name
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule_name)
    ])
    try:
        return {row.trial_id: dict(row) for row in client.query(query, job_config=job_config).result()}
    except Exception as exc:
        print(f"[BQ] Could not query existing trials: {exc}", file=sys.stderr)
        return {}


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


# ==============================================================================
# MAIN PUSH FUNCTION
# ==============================================================================

def save_clinical_efficacy_to_bq(
    molecule_name: str,
    trials: List[Dict[str, Any]],
    score_result: Optional[Dict[str, Any]] = None,
    rationale: Optional[str] = None,
) -> None:
    """
    Incrementally push enriched clinical trial rows to BigQuery.

    Args:
        molecule_name  – Drug name.
        trials         – List of row dicts from fetcher.run_fetcher().
        score_result   – Dict from compute_clinical_efficacy_score() (or None).
        rationale      – Narrative string from generate_score_rationale() (or None).
    """
    client = get_bq_client()
    now    = datetime.now(timezone.utc).isoformat()

    _ensure_table(client)
    _ensure_new_columns(client)

    if not trials:
        print(f"[BQ] No trials to save for {molecule_name}", file=sys.stderr)
        return

    efficacy_score     = _safe_float(score_result.get("weighted_score")) if score_result else None
    efficacy_coverage  = score_result.get("data_coverage", "")           if score_result else ""
    efficacy_breakdown = score_result.get("score_breakdown", "")         if score_result else ""
    efficacy_rationale = rationale or ""

    existing_trials = _get_existing_trials(client, molecule_name)
    if existing_trials:
        print(f"[BQ] Found {len(existing_trials)} existing trial(s) for {molecule_name}", file=sys.stderr)

    new_rows: List[Dict[str, Any]] = []
    updates:  List[Dict[str, Any]] = []
    skipped = 0

    for trial in trials:
        trial_id = str(trial.get("trial_id") or "").strip()
        if not trial_id:
            continue

        new_phase        = str(trial.get("phase")        or "")
        new_phase_status = str(trial.get("phase_status") or "")
        new_title        = str(trial.get("trial_title")  or "")
        new_study        = str(trial.get("trial_study")  or "")

        if trial_id in existing_trials:
            ex = existing_trials[trial_id]
            changed = (
                new_phase        != (ex.get("phase")        or "") or
                new_phase_status != (ex.get("phase_status") or "") or
                (not ex.get("trial_title") and new_title) or
                (not ex.get("trial_study") and new_study)
            )
            if changed:
                updates.append({
                    "trial_id": trial_id, "phase": new_phase,
                    "phase_status": new_phase_status, "trial_title": new_title,
                    "trial_study": new_study, "updated_at": now,
                    "molecule_name": molecule_name,
                })
            else:
                skipped += 1
            continue

        new_rows.append({
            "molecule_name":               str(trial.get("molecule_name")        or molecule_name),
            "registry_source":             str(trial.get("registry_source")       or ""),
            "trial_id":                    trial_id,
            "acronym":                     str(trial.get("acronym")               or ""),
            "dosage":                      str(trial.get("dosage")                or ""),
            "phase":                       new_phase,
            "trial_title":                 new_title,
            "trial_study":                 new_study,
            "trial_size":                  str(trial.get("trial_size")            or ""),
            "trial_location":              str(trial.get("trial_location")        or ""),
            "secondary_locations":         str(trial.get("secondary_locations")   or ""),
            "trial_start_date":            str(trial.get("trial_start_date")      or ""),
            "trial_completion_date":       str(trial.get("trial_completion_date") or ""),
            "phase_status":                new_phase_status,
            "hba1c_change_pct":            str(trial.get("hba1c_change_pct")      or ""),
            "hba1c_duration":              str(trial.get("hba1c_duration")        or ""),
            "hba1c_rationale":             str(trial.get("hba1c_rationale")       or ""),
            "hba1c_confidence":            _safe_confidence(trial.get("hba1c_confidence")),
            "weight_change_pct":           str(trial.get("weight_change_pct")     or ""),
            "weight_duration":             str(trial.get("weight_duration")       or ""),
            "weight_rationale":            str(trial.get("weight_rationale")      or ""),
            "weight_confidence":           _safe_confidence(trial.get("weight_confidence")),
            "alt_reduction_pct":           str(trial.get("alt_reduction_pct")     or ""),
            "alt_duration":                str(trial.get("alt_duration")          or ""),
            "alt_rationale":               str(trial.get("alt_rationale")         or ""),
            "alt_confidence":              _safe_confidence(trial.get("alt_confidence")),
            "mash_change_pct":             str(trial.get("mash_change_pct")       or ""),
            "mash_duration":               str(trial.get("mash_duration")         or ""),
            "mash_rationale":              str(trial.get("mash_rationale")        or ""),
            "mash_confidence":             _safe_confidence(trial.get("mash_confidence")),
            "company_name":                str(trial.get("company_name")          or ""),
            "source_url":                  str(trial.get("source_url")            or ""),
            "llm_confidence":              _safe_confidence(trial.get("llm_confidence")),
            "llm_confidence_rationale":    str(trial.get("llm_confidence_rationale") or ""),
            "trial_source":                str(trial.get("trial_source")           or ""),
            "efficacy_score":      efficacy_score,
            "efficacy_data_coverage":       efficacy_coverage,
            "efficacy_score_breakdown":     efficacy_breakdown,
            "efficacy_narrative_rationale": efficacy_rationale,
            "created_at":                  now,
            "updated_at":                  None,
        })

    # Updates
    updated_count = 0
    if updates:
        print(f"[BQ] Updating {len(updates)} trial(s)...", file=sys.stderr)
        tid = _table_id()
        for upd in updates:
            sql = f"""
                UPDATE `{tid}`
                SET phase        = @phase,
                    phase_status = @phase_status,
                    trial_title  = CASE WHEN trial_title IS NULL OR trial_title = ''
                                        THEN @trial_title ELSE trial_title END,
                    trial_study  = CASE WHEN trial_study IS NULL OR trial_study = ''
                                        THEN @trial_study ELSE trial_study END,
                    updated_at   = @updated_at
                WHERE molecule_name = @molecule_name AND trial_id = @trial_id
            """
            jc = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("phase",         "STRING",    upd["phase"]),
                bigquery.ScalarQueryParameter("phase_status",  "STRING",    upd["phase_status"]),
                bigquery.ScalarQueryParameter("trial_title",   "STRING",    upd["trial_title"]),
                bigquery.ScalarQueryParameter("trial_study",   "STRING",    upd["trial_study"]),
                bigquery.ScalarQueryParameter("updated_at",    "TIMESTAMP", upd["updated_at"]),
                bigquery.ScalarQueryParameter("molecule_name", "STRING",    upd["molecule_name"]),
                bigquery.ScalarQueryParameter("trial_id",      "STRING",    upd["trial_id"]),
            ])
            try:
                client.query(sql, job_config=jc).result()
                updated_count += 1
            except Exception as exc:
                print(f"[BQ] Update error for {upd['trial_id']}: {exc}", file=sys.stderr)

    if not new_rows and not updates:
        msg = f"({skipped} trial(s) unchanged)" if skipped else ""
        print(f"[BQ] No changes for {molecule_name} {msg}", file=sys.stderr)
        return

    if new_rows:
        errors = client.insert_rows_json(_table_id(), new_rows)
        if errors:
            print(f"[BQ] Insert errors for {molecule_name}: {errors[:3]}", file=sys.stderr)
        else:
            print(
                f"[BQ] Saved {len(new_rows)} NEW trial(s) for {molecule_name} "
                f"(skipped {skipped} unchanged, updated {updated_count})",
                file=sys.stderr,
            )
    else:
        print(f"[BQ] Updated {updated_count} trial(s) for {molecule_name} (no new trials)", file=sys.stderr)


# ==============================================================================
# STANDALONE CLI
# ==============================================================================

def main() -> int:
    import argparse, json as _json
    ap = argparse.ArgumentParser(description="Push enriched trials JSON to BigQuery.")
    ap.add_argument("molecule", help="Molecule / drug name")
    ap.add_argument("--json",  required=True, help="Path to enriched trials JSON file")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as fh:
        rows = _json.load(fh)
    rows = [{k: (str(v) if v is not None else "") for k, v in r.items()} for r in rows]

    print(f"[BQ-CLI] Loaded {len(rows)} row(s) from {args.json}", file=sys.stderr)

    score_result = None
    if rows and rows[0].get("efficacy_score"):
        score_result = {
            "weighted_score":  rows[0].get("efficacy_score", ""),
            "data_coverage":   rows[0].get("efficacy_data_coverage",  ""),
            "score_breakdown": rows[0].get("efficacy_score_breakdown", ""),
        }
    rationale = rows[0].get("efficacy_narrative_rationale", "") if rows else ""

    save_clinical_efficacy_to_bq(args.molecule, rows, score_result=score_result, rationale=rationale)
    return 0


if __name__ == "__main__":
    sys.exit(main())