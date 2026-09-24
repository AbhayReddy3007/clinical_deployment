import requests
import pandas as pd
import json
import sys
import time
import re
from dataclasses import dataclass
from typing import Any, Dict
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None  # BigQuery not available; BQ features will be skipped

try:
    import config
except ImportError:
    class config:
        DRUG_NAME = ""
        COMPANY_NAME = ""
        DRUG_LIST = []
try:
    from gcs_upload import save_workbook_to_gcs
except ImportError:
    def save_workbook_to_gcs(*args, **kwargs): pass  # stub: gcs_upload not available

def _safe_folder_name(name: str) -> str:
    """Create consistent GCS-safe product folder/file component."""
    safe = str(name or "").strip()
    safe = safe.replace("+", " plus ")
    safe = re.sub(r'[<>:"/\\|?*]+', "_", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_").rstrip(".")
    return safe or "unknown_product"



BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


# Product registry source: direct BigQuery read.
# Fields intentionally match the existing product registry schema used by the researcher scripts.
_BQ_PROJECT  = "prj-portfolio-ai-dev"
_BQ_TABLE    = "prj-portfolio-ai-dev.portfolio_data.product_registry"
_BQ_FIELDS   = (
    "product_name, company_name, inn, active_ingredient, originator_company, "
    "brand_names, alternative_names, search_synonyms, development_codes, salt_forms, "
    "nda_bla_numbers, known_patent_numbers, known_approval_numbers, approval_markets, "
    "formulations, dosage_forms, strengths, cas_number, generic_manufacturers, "
    "competitors_in_class"
)


@dataclass
class RunContext:
    product_name: str
    company_name: str
    registry: Dict[str, Any] | None = None


def _bq_value(row, key, default=None):
    try:
        value = row.get(key)
    except Exception:
        value = getattr(row, key, default)
    return default if value is None else value


def _bq_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


def _dedupe_keep_order(items):
    out, seen = [], set()
    for item in items or []:
        s = str(item or "").strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _bq_row_to_registry(row, fallback_product: str, fallback_company: str) -> Dict[str, Any]:
    registry = {
        "product_name": _bq_value(row, "product_name", fallback_product) or fallback_product,
        "company_name": _bq_value(row, "company_name", fallback_company) or fallback_company,
        "inn": _bq_value(row, "inn", "") or "",
        "active_ingredient": _bq_value(row, "active_ingredient", "") or "",
        "originator_company": _bq_value(row, "originator_company", "") or "",
        "brand_names": _bq_list(_bq_value(row, "brand_names", [])),
        "alternative_names": _bq_list(_bq_value(row, "alternative_names", [])),
        "search_synonyms": _bq_list(_bq_value(row, "search_synonyms", [])),
        "development_codes": _bq_list(_bq_value(row, "development_codes", [])),
        "salt_forms": _bq_list(_bq_value(row, "salt_forms", [])),
        "nda_bla_numbers": _bq_list(_bq_value(row, "nda_bla_numbers", [])),
        "known_patent_numbers": _bq_list(_bq_value(row, "known_patent_numbers", [])),
        "known_approval_numbers": _bq_list(_bq_value(row, "known_approval_numbers", [])),
        "approval_markets": _bq_list(_bq_value(row, "approval_markets", [])),
        "formulations": _bq_list(_bq_value(row, "formulations", [])),
        "dosage_forms": _bq_list(_bq_value(row, "dosage_forms", [])),
        "strengths": _bq_list(_bq_value(row, "strengths", [])),
        "cas_number": _bq_value(row, "cas_number", "") or "",
        "generic_manufacturers": _bq_list(_bq_value(row, "generic_manufacturers", [])),
        "competitors_in_class": _bq_list(_bq_value(row, "competitors_in_class", [])),
    }
    registry["search_terms_used"] = _dedupe_keep_order([
        fallback_product,
        registry.get("product_name", ""),
        registry.get("inn", ""),
        registry.get("active_ingredient", ""),
        *registry.get("brand_names", []),
        *registry.get("alternative_names", []),
        *registry.get("search_synonyms", []),
        *registry.get("development_codes", []),
        *registry.get("salt_forms", []),
    ]) or [fallback_product]
    if not registry.get("brand_names"):
        registry["brand_names"] = [fallback_product]
    return registry


def get_or_build_registry(ctx: RunContext) -> Dict[str, Any]:
    """Fetch product registry directly from BigQuery. No local registry/cache file is used."""
    product_name = ctx.product_name
    company_name = ctx.company_name

    try:
        client = bigquery.Client(project=_BQ_PROJECT)
        sql = f"""
        SELECT {_BQ_FIELDS}
        FROM `{_BQ_TABLE}`
        WHERE LOWER(product_name) = LOWER(@product_name)
           OR LOWER(inn) = LOWER(@product_name)
           OR LOWER(active_ingredient) = LOWER(@product_name)
            OR LOWER(product_name) LIKE CONCAT('%', LOWER(@product_name), '%')
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("product_name", "STRING", product_name)]
        )
        rows = list(client.query(sql, job_config=job_config).result())
        if rows:
            print(f"  [BQ] Registry loaded from {_BQ_TABLE} for {product_name}")
            ctx.registry = _bq_row_to_registry(rows[0], product_name, company_name)
            return ctx.registry
    except Exception as e:
        print(f"  [BQ] Registry lookup failed for {product_name}: {type(e).__name__}: {e}")

    print(f"  [BQ] Registry not found; using minimal fallback for {product_name}")
    ctx.registry = {
        "product_name": product_name,
        "company_name": company_name,
        "inn": product_name,
        "active_ingredient": product_name,
        "brand_names": [product_name],
        "search_terms_used": [product_name],
    }
    return ctx.registry

# -----------------------------------------------
# CLINICAL EVENT CLASSIFICATION — mirrors deep script
# -----------------------------------------------
_STATUS_TO_CLASSIFICATION = {
    "NOT_YET_RECRUITING":      "Trial Initiation / IND-CTA Filed",
    "RECRUITING":              "Enrollment Milestone (Ongoing / Complete)",
    "ENROLLING_BY_INVITATION": "Enrollment Milestone (Ongoing / Complete)",
    "ACTIVE_NOT_RECRUITING":   "Enrollment Milestone (Ongoing / Complete)",
    "TERMINATED":              "Trial Halted / Terminated / Discontinued",
    "WITHDRAWN":               "Trial Halted / Terminated / Discontinued",
    "SUSPENDED":               "Safety Signal / Clinical Hold",
    "UNKNOWN_STATUS":          "Other Clinical Development Event",
}

def derive_classification(status: str, has_results: bool) -> str:
    status_up = (status or "").upper()
    if status_up == "COMPLETED":
        return "Published Full Results" if has_results else "Other Clinical Development Event"
    return _STATUS_TO_CLASSIFICATION.get(status_up, "Other Clinical Development Event")


def extract_list_text(val) -> str:
    """Convert a list or JSON-string-list to comma-separated readable text."""
    if isinstance(val, list):
        return ", ".join(str(x) for x in val if x)
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed if x)
        except Exception:
            pass
        return val
    return str(val) if val is not None else ""


def format_phases(phases_raw) -> str:
    """Convert ['PHASE3'] → 'Phase 3', ['PHASE1', 'PHASE2'] → 'Phase 1, Phase 2'."""
    if not phases_raw:
        return ""
    if isinstance(phases_raw, str):
        try:
            phases_raw = json.loads(phases_raw)
        except Exception:
            return phases_raw
    if not isinstance(phases_raw, list):
        return str(phases_raw)
    cleaned = []
    for p in phases_raw:
        p = str(p).strip()
        p = p.replace("EARLY_PHASE", "Early Phase ").replace("PHASE", "Phase ").strip()
        if p.upper() == "NA":
            p = "N/A"
        cleaned.append(p)
    return ", ".join(cleaned)


def extract_countries(locations) -> str:
    """Extract unique country names from locations list."""
    if not locations:
        return "GLOBAL"
    if isinstance(locations, str):
        try:
            locations = json.loads(locations)
        except Exception:
            return "GLOBAL"
    if not isinstance(locations, list):
        return "GLOBAL"
    seen, countries = set(), []
    for loc in locations:
        if isinstance(loc, dict):
            country = loc.get("country", "")
            if country and country not in seen:
                seen.add(country)
                countries.append(country)
    if not countries:
        return "GLOBAL"
    if len(countries) > 5:
        return f"{', '.join(countries[:5])}... ({len(countries)} countries)"
    return ", ".join(countries)


def fetch_all_studies(drug_name: str) -> list[dict]:
    """Fetch all studies for a given drug name, paginating through all results."""
    all_studies = []
    page_token = None
    page_num = 0

    # First request to get total count
    params = {
        "query.intr": drug_name,
        "pageSize": 1000,
        "countTotal": "true",
        "format": "json",
    }

    print(f"\nSearching ClinicalTrials.gov for: '{drug_name}'")
    print("-" * 50)

    while True:
        page_num += 1
        if page_token:
            params["pageToken"] = page_token
            params.pop("countTotal", None)  # Only needed on first page
        else:
            params["countTotal"] = "true"

        print(f"  Fetching page {page_num}...", end=" ", flush=True)

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"\nHTTP Error: {e}")
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            print("\nConnection error. Check your internet connection.")
            sys.exit(1)
        except requests.exceptions.Timeout:
            print("\nRequest timed out. Retrying in 5s...")
            time.sleep(5)
            continue
        except ValueError:
            print("\nFailed to parse API response as JSON.")
            sys.exit(1)

        studies = data.get("studies", [])
        all_studies.extend(studies)

        if page_num == 1:
            total = data.get("totalCount", "unknown")
            print(f"Total studies found: {total}")

        print(f"  Retrieved {len(studies)} studies (cumulative: {len(all_studies)})")

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        # Be polite to the API
        time.sleep(0.3)

    return all_studies


def studies_to_dataframe(studies: list[dict], drug_name: str) -> pd.DataFrame:
    """
    Extract and align ClinicalTrials.gov study data to match the deep researcher
    output schema (same column names and structure).
    Aligned columns first, then bonus CT-specific context columns.
    """
    rows = []
    for i, study in enumerate(studies, 1):
        ps            = study.get("protocolSection", {}) or {}
        id_mod        = ps.get("identificationModule", {}) or {}
        status_mod    = ps.get("statusModule", {}) or {}
        desc_mod      = ps.get("descriptionModule", {}) or {}
        design_mod    = ps.get("designModule", {}) or {}
        conditions_mod= ps.get("conditionsModule", {}) or {}
        sponsor_mod   = ps.get("sponsorCollaboratorsModule", {}) or {}
        outcomes_mod  = ps.get("outcomesModule", {}) or {}
        contacts_mod  = ps.get("contactsLocationsModule", {}) or {}

        nct_id         = id_mod.get("nctId", "") or ""
        brief_title    = id_mod.get("briefTitle", "") or ""
        official_title = id_mod.get("officialTitle", "") or ""
        overall_status = status_mod.get("overallStatus", "") or ""
        has_results    = bool(study.get("hasResults", False))

        # trial_phase — convert ['PHASE3'] → 'Phase 3'
        trial_phase = format_phases(design_mod.get("phases", []))

        # indication — conditions list → comma string
        indication = extract_list_text(conditions_mod.get("conditions", []))

        # sponsor
        lead_sponsor = (sponsor_mod.get("leadSponsor", {}) or {}).get("name", "") or ""

        # dates
        start_date          = ((status_mod.get("startDateStruct") or {}).get("date") or "")
        primary_completion  = ((status_mod.get("primaryCompletionDateStruct") or {}).get("date") or "")
        completion_date     = ((status_mod.get("completionDateStruct") or {}).get("date") or "")
        last_update         = ((status_mod.get("lastUpdatePostDateStruct") or {}).get("date") or "")

        # enrollment
        enrollment_count = (design_mod.get("enrollmentInfo") or {}).get("count", "") or ""

        # primary endpoints (up to 3 measures)
        primary_outcomes = outcomes_mod.get("primaryOutcomes") or []
        primary_endpoint_text = "; ".join(
            o.get("measure", "") for o in (primary_outcomes or [])[:3]
            if isinstance(o, dict) and o.get("measure")
        )

        # geo — unique countries from locations
        geo = extract_countries(contacts_mod.get("locations") or [])

        # ── Derived deep-script aligned fields ──────────────────────────
        url_resolved = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""

        evidence = f"ClinicalTrials.gov registry | {nct_id}"
        if lead_sponsor:
            evidence += f" | Sponsor: {lead_sponsor}"
        if overall_status:
            evidence += f" | Status: {overall_status}"

        reasoning_parts = [f"Status={overall_status}"]
        if trial_phase:
            reasoning_parts.append(f"Phase={trial_phase}")
        if start_date:
            reasoning_parts.append(f"Start={start_date}")
        if primary_completion:
            reasoning_parts.append(f"PrimaryCompletion={primary_completion}")
        if enrollment_count:
            reasoning_parts.append(f"Enrollment={enrollment_count}")
        if lead_sponsor:
            reasoning_parts.append(f"Sponsor={lead_sponsor}")
        reasoning_parts.append(f"HasResults={has_results}")
        reasoning = " | ".join(reasoning_parts)

        # ── Build aligned row ─────────────────────────────────────────────
        row = {
            # ── Aligned to deep researcher Final_Output columns ──
            "#":                              i,
            "product_name":                   drug_name,
            "trial_phase":                    trial_phase,
            "indication":                     indication,
            "nct_id":                         nct_id,
            "timestamp":                      last_update,
            "timestamp_rationale":            "Last update posted date from ClinicalTrials.gov registry",
            "timestamp_enriched":             "REGISTRY",
            "connector":                      "data_source-1 (ClinicalTrials.gov)",
            "geo":                            geo,
            "raw_text":                       brief_title,
            "brief_summary":                  desc_mod.get("briefSummary") or "",
            "evidence":                       evidence,
            "url_vertexai":                   "",
            "url_resolved":                   url_resolved,
            "url_status":                     "direct",
            "source_validated":               "YES",
            "validation_note":                "Official ClinicalTrials.gov registry record",
            "reasoning":                      reasoning,
            # ── Bonus context columns ──
            "overall_status":                 overall_status,
            "start_date":                     start_date,
            "primary_completion_date":        primary_completion,
            "completion_date":                completion_date,
            "enrollment_count":               enrollment_count,
            "lead_sponsor":                   lead_sponsor,
            "primary_outcomes":               primary_endpoint_text,
            "has_results":                    has_results,
            "official_title":                 official_title,
            "study_type":                     design_mod.get("studyType", "") or "",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_to_excel(df: pd.DataFrame, drug_name: str) -> str:
    """
    Save aligned DataFrame to Excel with styling matching the deep researcher output.
    Sheet 1: Clinical_Trials  — aligned columns, color-coded classification
    Sheet 2: Summary          — breakdown by status, phase, classification
    """

    # ── Styles (mirrors deep researcher) ──────────────────────────────────
    HEADER_FILL  = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT  = Font(color="FFFFFF", bold=True, size=11)
    THIN_BORDER  = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"),  bottom=Side(style="thin"),
    )
    GREEN_FILL   = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    RED_FILL     = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    ORANGE_FILL  = PatternFill(start_color="FCD5B4", end_color="FCD5B4", fill_type="solid")
    BLUE_FILL    = PatternFill(start_color="D9EAF7", end_color="D9EAF7", fill_type="solid")
    GRAY_FILL    = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    CT_FILL      = PatternFill(start_color="EDEDED", end_color="EDEDED", fill_type="solid")  # bonus cols
    DATA_FONT    = Font(size=10)

    safe_name = _safe_folder_name(drug_name)
    filename = f"{safe_name}_clinical_trials.xlsx"

    wb = Workbook()

    # ── Sheet 1: Clinical_Trials ───────────────────────────────────────────
    ws = wb.active
    ws.title = "Clinical_Trials"

    cols = list(df.columns)
    # Column index lookup (1-based)
    col_idx = {name: i + 1 for i, name in enumerate(cols)}

    # Header row
    for c, col_name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c, value=col_name)
        cell.fill      = HEADER_FILL
        cell.font      = HEADER_FONT
        cell.border    = THIN_BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Data rows
    WRAP_COLS = {"raw_text", "indication", "evidence", "reasoning",
                 "timestamp_rationale", "validation_note", "brief_summary",
                 "official_title", "primary_outcomes"}

    BONUS_COLS = {"overall_status", "start_date", "primary_completion_date",
                  "completion_date", "enrollment_count", "lead_sponsor",
                  "primary_outcomes", "has_results", "official_title", "study_type"}

    for r_idx, row in df.iterrows():
        excel_row = r_idx + 2  # header is row 1, df is 0-indexed
        is_even   = (r_idx % 2 == 0)

        for c, col_name in enumerate(cols, 1):
            val  = row[col_name]
            cell = ws.cell(row=excel_row, column=c, value=str(val) if val is not None else "")
            cell.font   = DATA_FONT
            cell.border = THIN_BORDER
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=(col_name in WRAP_COLS),
            )

            # ── Colour logic ──────────────────────────────────────────────
            if col_name == "url_resolved":
                cell.fill = GREEN_FILL if val else RED_FILL

            elif col_name == "source_validated":
                cell.fill = GREEN_FILL   # always YES for registry

            elif col_name == "overall_status":
                s = str(val).upper()
                if s == "COMPLETED":
                    cell.fill = GREEN_FILL
                elif s in ("TERMINATED", "WITHDRAWN", "SUSPENDED"):
                    cell.fill = RED_FILL
                elif s == "RECRUITING":
                    cell.fill = BLUE_FILL
                elif val:
                    cell.fill = CT_FILL

            elif col_name in BONUS_COLS and is_even:
                # Subtle grey tint for bonus context columns on even rows
                if not cell.fill or cell.fill.fill_type == "none":
                    cell.fill = CT_FILL

            elif col_name in ("#", "product_name", "connector",
                              "timestamp_enriched", "url_status") and is_even:
                cell.fill = GRAY_FILL

    # Column widths
    COL_WIDTHS = {
        "#":                             5,
        "product_name":                  18,
        "trial_phase":                   14,
        "indication":                    30,
        "nct_id":                        16,
        "timestamp":                     14,
        "timestamp_rationale":           40,
        "timestamp_enriched":            16,
        "connector":                     30,
        "geo":                           20,
        "raw_text":                      40,
        "brief_summary":                 70,
        "evidence":                      40,
        "url_vertexai":                  10,
        "url_resolved":                  45,
        "url_status":                    12,
        "source_validated":              16,
        "validation_note":               38,
        "reasoning":                     55,
        "overall_status":                20,
        "start_date":                    16,
        "primary_completion_date":       24,
        "completion_date":               20,
        "enrollment_count":              18,
        "lead_sponsor":                  30,
        "primary_outcomes":              50,
        "has_results":                   14,
        "official_title":                50,
        "study_type":                    16,
    }
    for c, col_name in enumerate(cols, 1):
        width = COL_WIDTHS.get(col_name, 18)
        ws.column_dimensions[get_column_letter(c)].width = width

    ws.freeze_panes = "A2"
    if len(df) > 0:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(df) + 1}"

    # ── Sheet 2: Summary ───────────────────────────────────────────────────
    ws2 = wb.create_sheet("Summary")
    LABEL_FONT = Font(bold=True, size=10)
    VALUE_FONT = Font(size=10)
    SEC_FONT   = Font(bold=True, size=10, color="1F4E79")

    # Summary breakdowns
    status_counts = df["overall_status"].value_counts().to_dict() if "overall_status" in df.columns else {}
    phase_counts  = df["trial_phase"].value_counts().to_dict() if "trial_phase" in df.columns else {}

    summary_rows = [
        ("Drug / Intervention",     drug_name),
        ("Total Studies Retrieved",  len(df)),
        ("Data Pulled On",           datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Source",                   "ClinicalTrials.gov API v2"),
        ("", ""),
        ("─── Status Breakdown ───", ""),
    ]
    for s, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
        summary_rows.append((f"  {s}", cnt))
    summary_rows.append(("", ""))
    summary_rows.append(("─── Phase Breakdown ───", ""))
    for p, cnt in sorted(phase_counts.items(), key=lambda x: -x[1]):
        summary_rows.append((f"  {p}", cnt))

    ws2.column_dimensions["A"].width = 45
    ws2.column_dimensions["B"].width = 40

    for r_idx, (label, value) in enumerate(summary_rows, start=1):
        a = ws2.cell(row=r_idx, column=1, value=label)
        b = ws2.cell(row=r_idx, column=2, value=value)
        if r_idx <= 6:
            a.font = LABEL_FONT
            b.font = VALUE_FONT
            b.fill = PatternFill(start_color="EBF3FB", end_color="EBF3FB", fill_type="solid")
        elif str(label).startswith("───"):
            a.font = SEC_FONT
        else:
            a.font = VALUE_FONT
            b.font = VALUE_FONT

    # Write under product-specific folder in GCS, not bucket root.
    # save_workbook_to_gcs accepts object/blob path as filename.
    gcs_object_name = f"{safe_name}/{filename}"
    save_workbook_to_gcs(wb, gcs_object_name)
    return gcs_object_name


def run_for_drug(drug_name: str, company_name: str = "") -> str:
    """
    Batch-runner entrypoint for ClinicalTrials.gov.
    Product/company are passed by batch_runner_sources.py from config.PRODUCTS.
    """
    if not drug_name:
        raise ValueError("drug_name is required")

    print(f"\n{'=' * 60}")
    print("  ClinicalTrials.gov Connector")
    print(f"  Drug: {drug_name}")
    if company_name:
        print(f"  Company: {company_name}")
    print(f"{'=' * 60}")

    ctx = RunContext(product_name=drug_name, company_name=company_name)

    registry = get_or_build_registry(ctx)
    ctx.registry = registry
    brand_names = registry.get("brand_names", [])
    search_terms = [ctx.product_name] + [
        b for b in brand_names
        if b and b.lower() != ctx.product_name.lower()
    ]
    print(f"Search terms: {search_terms}")

    all_studies = []
    for term in search_terms:
        studies = fetch_all_studies(term)
        all_studies.extend(studies)

    if not all_studies:
        print(f"\nNo studies found for '{ctx.product_name}' or its brand names.")
        return ""

    print(f"\nProcessing {len(all_studies)} studies (combined, pre-dedup) into aligned format...")
    df = studies_to_dataframe(all_studies, ctx.product_name)
    before = len(df)
    df = df.drop_duplicates(subset=["nct_id"], keep="first").reset_index(drop=True)
    df["#"] = df.index + 1
    print(f"  → {before} rows before dedup, {len(df)} after (by nct_id) × {len(df.columns)} columns")

    print("Saving to Excel...")
    filepath = save_to_excel(df, ctx.product_name)
    print(f"\nDone! File saved to:\n   {filepath}")
    return filepath


def main():
    # Single-product backward-compatible mode only.
    # Batch mode does not use config.DRUG_NAME.
    drug_name = getattr(config, "DRUG_NAME", "")
    company_name = getattr(config, "COMPANY_NAME", "")
    if not drug_name:
        print("No drug name entered. Exiting.")
        sys.exit(0)
    run_for_drug(drug_name, company_name)


if __name__ == "__main__":
    main()
