"""
WHO ICTRP — International Clinical Trials Registry Platform
Standalone connector with proper ASP.NET pagination & table scraping.

VERIFIED HTML STRUCTURE (from live inspection June 2026):
  • Form fields:  TextBox1 (search input), Button1 (submit)
  • Hidden fields: __VIEWSTATE, __VIEWSTATEGENERATOR, __VIEWSTATEENCRYPTED,
                   __EVENTVALIDATION, ToolkitScriptManager_HiddenField,
                   TextBoxWatermarkExtender1_ClientState
  • Results table ID: GridView1
  • Table layout:
        Row 0 = top pager (page links with spans)
        Row 1 = inner pager row
        Row 2 = header row (Recruitment status | Prospective | Main ID | _ | Public Title | Date | Results)
        Row 3..N = data rows (7 cells each):
            col0 = Recruitment Status
            col1 = Prospective Registration (usually empty)
            col2 = Main ID (trial identifier)
            col3 = (empty spacer)
            col4 = Public Title (contains <a href="Trial2.aspx?TrialID=...">)
            col5 = Date of Registration (YYYY-MM-DD)
            col6 = Results available
        Last row = bottom pager
  • Pagination: __doPostBack('GridView1','Page$N')
  • 10 results per page, ">>" links to Page$Last

Usage:
    python who_ictrp_api.py
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict
import requests
import pandas as pd
from datetime import datetime
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None  # BigQuery not available; BQ features will be skipped
from bs4 import BeautifulSoup
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

try:
    import config
except ImportError:
    class config:
        DRUG_NAME = ""
        COMPANY_NAME = ""
        DRUG_LIST = []
        NCBI_API_KEY = None
try:
    from gcs_upload import save_workbook_to_gcs
except ImportError:
    def save_workbook_to_gcs(*args, **kwargs): pass  # stub: gcs_upload not available

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DRUG_NAME = getattr(config, "DRUG_NAME", "")

def _safe_folder_name(name: str) -> str:
    """Create consistent GCS-safe product folder/file component."""
    safe = str(name or "").strip()
    safe = safe.replace("+", " plus ")
    safe = re.sub(r'[<>:"/\\|?*]+', "_", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_").rstrip(".")
    return safe or "unknown_product"

# ─────────────────────────────────────────────
# ENDPOINTS & SETTINGS
# ─────────────────────────────────────────────
ICTRP_SEARCH_URL = "https://trialsearch.who.int/Default.aspx"
ICTRP_TIMEOUT = 45       # generous timeout for ASP.NET responses
SLEEP_BETWEEN = 1.5      # seconds between page fetches (be polite)
MAX_PAGES = 500         # safety cap per synonym
SKIP_NCT = False         # False = include US (NCT) trials too; True = skip them
DEDUP_PAGES = False      # False = keep ALL rows (incl. duplicates); True = dedup by trial_id per page

TS = datetime.now().strftime("%Y%m%d_%H%M%S")


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

# ─────────────────────────────────────────────
# STATUS → CLASSIFICATION MAPPING
# ─────────────────────────────────────────────
_ICTRP_STATUS_MAP = {
    "not yet recruiting":       "Trial Initiation / IND-CTA Filed",
    "not recruiting":           "Trial Initiation / IND-CTA Filed",
    "recruiting":               "Enrollment Milestone (Ongoing / Complete)",
    "enrolling by invitation":  "Enrollment Milestone (Ongoing / Complete)",
    "active, not recruiting":   "Enrollment Milestone (Ongoing / Complete)",
    "active":                   "Enrollment Milestone (Ongoing / Complete)",
    "completed":                "Published Full Results",
    "terminated":               "Trial Halted / Terminated / Discontinued",
    "withdrawn":                "Trial Halted / Terminated / Discontinued",
    "suspended":                "Safety Signal / Clinical Hold",
    "authorised":               "Trial Initiation / IND-CTA Filed",
    "approved":                 "Trial Initiation / IND-CTA Filed",
    "ongoing":                  "Enrollment Milestone (Ongoing / Complete)",
    "no longer recruiting":     "Enrollment Milestone (Ongoing / Complete)",
    "main results":             "Published Full Results",
    "results available":        "Published Full Results",
}

# ─────────────────────────────────────────────
# REGISTRY DETECTION — from Trial ID prefix
# ─────────────────────────────────────────────
_REGISTRY_PREFIXES = {
    "NCT":          "ClinicalTrials.gov",
    "EUCTR":        "EU Clinical Trials Register",
    "CTRI":         "Clinical Trials Registry - India",
    "JPRN":         "Japan Primary Registries Network",
    "ISRCTN":       "ISRCTN Registry",
    "ACTRN":        "ANZCTR (Australia/NZ)",
    "ChiCTR":       "Chinese Clinical Trial Registry",
    "KCT":          "Korean Clinical Trial Registry",
    "DRKS":         "German Clinical Trials Register",
    "NTR":          "Netherlands Trial Register",
    "PACTR":        "Pan African Clinical Trial Registry",
    "SLCTR":        "Sri Lanka Clinical Trials Registry",
    "TCTR":         "Thai Clinical Trials Registry",
    "RPCEC":        "Cuban Public Registry",
    "IRCT":         "Iranian Registry of Clinical Trials",
    "LBCTR":        "Lebanese Clinical Trials Registry",
    "RBR":          "Brazilian Clinical Trials Registry",
    "PER":          "Peruvian Clinical Trials Registry",
}


def _detect_registry(trial_id: str) -> str:
    """Detect source registry from trial ID prefix."""
    trial_upper = trial_id.upper().strip()
    for prefix, registry in _REGISTRY_PREFIXES.items():
        if trial_upper.startswith(prefix.upper()):
            return registry
    return "Unknown Registry"


def _extract_phase_from_text(text: str) -> str:
    """Extract trial phase from free text."""
    text_lower = (text or "").lower()
    phase_patterns = [
        (r"phase\s*(iv|4)", "Phase 4"),
        (r"phase\s*(iii|3)", "Phase 3"),
        (r"phase\s*(ii[i]?b?/?\s*i{0,3}|2[ab]?/?3?)", "Phase 2"),
        (r"phase\s*(i{1,3}|1[ab]?)", "Phase 1"),
    ]
    for pattern, label in phase_patterns:
        if re.search(pattern, text_lower):
            return label
    return ""


def _extract_nct_from_text(text: str) -> str:
    """Extract NCT IDs from text via regex."""
    matches = re.findall(r"NCT\d{8}", text or "", re.IGNORECASE)
    return ", ".join(sorted(set(m.upper() for m in matches)))


def _normalize_date(date_str: str) -> str:
    """Try multiple date formats → YYYY-MM-DD."""
    if not date_str:
        return ""
    date_str = date_str.strip()
    formats = [
        "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d",
        "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str  # return raw if no format matches


# ═══════════════════════════════════════════════════════════════════════════════
# CORE SCRAPING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_all_form_fields(soup: BeautifulSoup) -> dict:
    """
    Extract ALL hidden fields + form state from the page.
    WHO ICTRP uses ASP.NET with these critical fields:
      - ToolkitScriptManager_HiddenField
      - __VIEWSTATE, __VIEWSTATEGENERATOR, __VIEWSTATEENCRYPTED
      - __EVENTVALIDATION
      - TextBoxWatermarkExtender1_ClientState
    """
    fields = {}
    for inp in soup.find_all("input", {"type": "hidden"}):
        name = inp.get("name", "")
        if name:
            fields[name] = inp.get("value", "")
    return fields


def _parse_ictrp_table(soup: BeautifulSoup, debug: bool = False) -> list[dict]:
    """
    Parse the ICTRP GridView1 results table.

    VERIFIED table structure (June 2026):
      Table ID = "GridView1"
      Row 0 = top pager row (13 cells: full pager + page numbers)
      Row 1 = inner pager row (12 cells: page numbers only)
      Row 2 = header row (7 cells): Status | Prospective | Main ID | _ | Title | Date | Results
      Row 3..N-1 = data rows (7 cells each):
          col0 = Recruitment Status (e.g. "Not Recruiting", "Recruiting")
          col1 = Prospective Registration (usually empty)
          col2 = Main ID (e.g. IRCT20150303021315N42, ChiCTR2600124677)
          col3 = empty spacer column
          col4 = Public Title (contains <a href="Trial2.aspx?TrialID=...">)
          col5 = Date of Registration (YYYY-MM-DD format)
          col6 = Results available (empty or text)
      Row N = bottom pager row
    """
    table = soup.find("table", {"id": "GridView1"})
    if not table:
        return []

    trials = []
    rows = table.find_all("tr")

    for row_num, row in enumerate(rows):
        # CRITICAL FIX: Skip rows that belong to nested tables (sub-trial panels)
        # GridView1 has collapsible panels with nested <table>s for sub-trials.
        # find_all("tr") picks up those nested rows too.
        parent_table = row.find_parent("table")
        if parent_table is not None and parent_table != table:
            if debug:
                print(f"\n      [DBG] Row {row_num}: NESTED (skipped, parent_table_id={parent_table.get('id', 'none')})")
            continue

        # CRITICAL: Use recursive=False to get ONLY direct <td> children of this <tr>.
        # Without this, find_all("td") also picks up <td> elements from nested
        # sub-tables (expand panels), causing:
        #   - Pager rows (nested table) to appear as data rows with numbers as trial_name
        #   - Expandable rows to have 22+ cells (sub-trial <td>s mixed in),
        #     making date_raw pick up "Recruiting" from a sub-trial cell
        cells = row.find_all("td", recursive=False)

        # With recursive=False, data rows have exactly 7 direct <td> children.
        # Pager rows have 1 <td colspan="7"> containing a nested pager table.
        # Header rows have 0 <td> (they use <th>).
        if len(cells) < 7:
            if debug and len(cells) > 0:
                text = row.get_text(strip=True)[:60]
                print(f"\n      [DBG] Row {row_num}: cells={len(cells)} SKIPPED (too few) | text: {text}")
            continue

        # Safety: skip rows with way too many direct cells (shouldn't happen now)
        if len(cells) > 10:
            if debug:
                text = row.get_text(strip=True)[:60]
                print(f"\n      [DBG] Row {row_num}: cells={len(cells)} SKIPPED (too many) | text: {text}")
            continue

        status_raw = cells[0].get_text(strip=True)
        trial_id_raw = cells[2].get_text(strip=True)

        # Skip pager rows: they have cells with page numbers like "1 2 3 4..."
        # Detected by: status looks like a page number, or row has __doPostBack links for "Page$"
        if status_raw.isdigit() or (not status_raw and not trial_id_raw):
            # Check if it's actually a pager row
            row_html = str(row)
            if "Page$" in row_html or re.match(r"^[\d\s.>]+$", row.get_text(strip=True)):
                if debug:
                    print(f"\n      [DBG] Row {row_num}: PAGER row skipped (cells={len(cells)})")
                continue

        # For rows with extra cells (8+), the title/date/results shift right.
        # With recursive=False, data rows should always be 7 cells.
        # But keep dynamic detection as safety fallback.
        title_cell_idx = None
        for idx in range(3, min(len(cells), 8)):  # limit search to first 8 cells
            # find() IS recursive within the cell (to find links inside nested divs)
            a_tag = cells[idx].find("a", href=re.compile(r"Trial2\.aspx", re.I))
            if a_tag:
                title_cell_idx = idx
                break

        if title_cell_idx is not None:
            # Get title text from the link itself (not the whole cell, which may include sub-panel text)
            title_link = cells[title_cell_idx].find("a", href=re.compile(r"Trial2\.aspx", re.I))
            title_raw = title_link.get_text(strip=True) if title_link else cells[title_cell_idx].get_text(strip=True)
            date_raw = cells[title_cell_idx + 1].get_text(strip=True) if title_cell_idx + 1 < len(cells) else ""
            results_raw = cells[title_cell_idx + 2].get_text(strip=True) if title_cell_idx + 2 < len(cells) else ""
        else:
            # Fallback: assume standard 7-cell layout
            title_raw = cells[4].get_text(strip=True)
            date_raw = cells[5].get_text(strip=True)
            results_raw = cells[6].get_text(strip=True)

        # VALIDATION: date_raw should look like a date (YYYY-MM-DD or DD/MM/YYYY), not a status
        # If date_raw looks like a recruitment status, try cells[5] directly
        if date_raw and not re.match(r"^\d", date_raw):
            # date_raw doesn't start with a digit — likely picked up wrong cell
            # Try cells[5] (standard date position)
            fallback_date = cells[5].get_text(strip=True) if len(cells) > 5 else ""
            if fallback_date and re.match(r"^\d", fallback_date):
                date_raw = fallback_date
            else:
                # Last resort: search all cells for a date-like pattern
                for c in cells[4:]:
                    ct = c.get_text(strip=True)
                    if re.match(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}", ct):
                        date_raw = ct
                        break

        # Skip header row (contains "Recruitment status")
        if "recruitment" in status_raw.lower() and "status" in status_raw.lower():
            if debug:
                print(f"\n      [DBG] Row {row_num}: HEADER row skipped")
            continue

        # Clean trial ID
        trial_id = re.sub(r"\s+", " ", trial_id_raw).strip()
        if not trial_id:
            if debug:
                print(f"\n      [DBG] Row {row_num}: EMPTY trial_id, cells={len(cells)}, status='{status_raw}', text='{cells[2].get_text()[:40]}'")
            continue

        # Skip rows where trial_id is purely numeric (pager leak-through)
        if trial_id.isdigit():
            if debug:
                print(f"\n      [DBG] Row {row_num}: NUMERIC trial_id='{trial_id}' (pager row), skipped")
            continue

        # Get detail link from title cell
        detail_url = ""
        link_cell = cells[title_cell_idx] if title_cell_idx is not None else cells[4]
        link = link_cell.find("a")
        if link and link.get("href"):
            href = link["href"]
            if href.startswith("http"):
                detail_url = href
            elif href.startswith("/"):
                detail_url = f"https://trialsearch.who.int{href}"
            else:
                detail_url = f"https://trialsearch.who.int/{href}"

        trials.append({
            "trial_id":        trial_id,
            "title":           title_raw,
            "status":          status_raw,
            "date_registered": date_raw,
            "has_results":     results_raw,
            "detail_url":      detail_url,
        })

    return trials


def _get_total_pages(soup: BeautifulSoup) -> int:
    """
    Detect total pages from the GridView1 pager.
    Pager links use: javascript:__doPostBack('GridView1','Page$N')
    ">>" links to Page$Last (we infer max from numbered links + "..." link)
    """
    table = soup.find("table", {"id": "GridView1"})
    if not table:
        return 1

    max_page = 1

    # Find all pager links in the table
    for link in table.find_all("a"):
        href = link.get("href", "")
        # Match Page$N where N is a number
        m = re.search(r"Page\$(\d+)", href)
        if m:
            max_page = max(max_page, int(m.group(1)))
        # ">>" goes to Page$Last — we can't get the number directly
        # but "..." goes to Page$11 etc. which gives us next batch start

    # Also check <span> for current page (not a link, just bold text)
    # The pager row has spans for the current page number
    all_rows = table.find_all("tr")
    for row in all_rows:
        cells = row.find_all("td")
        # Pager rows have many cells (12-13) with just page numbers
        if len(cells) > 7:
            for span in row.find_all("span"):
                text = span.get_text(strip=True)
                if text.isdigit():
                    max_page = max(max_page, int(text))

    # If we only see pages 1-10 and "..." links to 11, there are likely more
    # We'll handle progressive discovery during pagination
    return max_page


def _navigate_to_page(session: requests.Session, page_num: int,
                      current_soup: BeautifulSoup) -> requests.Response | None:
    """
    Navigate to a specific page using ASP.NET __doPostBack.
    Event target = 'GridView1', argument = 'Page$N'
    Must include ALL hidden form fields to maintain ASP.NET state.
    """
    fields = _get_all_form_fields(current_soup)

    # Override event fields for pagination
    fields["__EVENTTARGET"] = "GridView1"
    fields["__EVENTARGUMENT"] = f"Page${page_num}"

    # Must include the search text to maintain state
    text_box = current_soup.find("input", {"name": "TextBox1"})
    if text_box:
        fields["TextBox1"] = text_box.get("value", "")

    # Remove Button1 (we're paginating, not searching again)
    fields.pop("Button1", None)

    try:
        resp = session.post(ICTRP_SEARCH_URL, data=fields, timeout=ICTRP_TIMEOUT)
        if resp.status_code == 200:
            return resp
    except requests.RequestException as e:
        print(f" [p{page_num}:ERR {e}]", end="", flush=True)

    return None


def fetch_ictrp_trials(synonym: str, session: requests.Session) -> list[dict]:
    """
    Fetch ALL pages of WHO ICTRP results for a single search term.

    Flow:
      1. GET Default.aspx → extract hidden form fields
      2. POST with TextBox1=synonym, Button1=Search → page 1 results
      3. Parse page 1 from GridView1 table, detect page count
      4. Loop pages 2..N via __doPostBack('GridView1','Page$N')
      5. Handle progressive pagination ("..." → next batch of pages)
    """
    all_trials = []

    # Step 1: GET the search page for initial form state
    try:
        init_resp = session.get(ICTRP_SEARCH_URL, timeout=ICTRP_TIMEOUT)
        if init_resp.status_code != 200:
            print(f"GET failed ({init_resp.status_code})")
            return []
    except requests.RequestException as e:
        print(f"GET error: {e}")
        return []

    init_soup = BeautifulSoup(init_resp.text, "html.parser")
    fields = _get_all_form_fields(init_soup)

    # Step 2: POST the search form with correct field names
    fields["TextBox1"] = synonym
    fields["Button1"] = "Search"
    fields["__EVENTTARGET"] = ""
    fields["__EVENTARGUMENT"] = ""
    # Clear watermark state
    fields["TextBoxWatermarkExtender1_ClientState"] = ""

    try:
        search_resp = session.post(ICTRP_SEARCH_URL, data=fields, timeout=ICTRP_TIMEOUT)
        if search_resp.status_code != 200:
            print(f"POST failed ({search_resp.status_code})")
            return []
    except requests.RequestException as e:
        print(f"POST error: {e}")
        return []

    # Step 3: Parse page 1
    page_soup = BeautifulSoup(search_resp.text, "html.parser")

    # Check for "no results" message
    page_text = page_soup.get_text()[:1000]
    if "no records found" in page_text.lower() or "0 records" in page_text.lower():
        print("0 results")
        return []

    # Extract record count to calculate total pages
    total_records = 0
    count_match = re.search(r"(\d+)\s+records?\s+for\s+(\d+)\s+trials?", page_text)
    if count_match:
        total_records = int(count_match.group(1))
        print(f"({count_match.group(1)} records, {count_match.group(2)} trials) ", end="", flush=True)

    # Calculate expected pages: records shown 10 per page
    import math
    expected_pages = math.ceil(total_records / 10) if total_records > 0 else MAX_PAGES

    # Parse first page — track seen_ids for cycle detection (always needed)
    seen_ids = set()
    page_trials = _parse_ictrp_table(page_soup)
    for t in page_trials:
        seen_ids.add(t["trial_id"])
    all_trials.extend(page_trials)
    print(f"p1:{len(page_trials)}", end="", flush=True)

    # Step 4: Sequential pagination with cycle detection.
    # ASP.NET GridView shows page links in batches of 10 (1-10, 11-20, 21-30...).
    # __doPostBack('GridView1','Page$N') can only reach pages in the VISIBLE batch.
    # When we request a page beyond the current batch, the server loops back.
    # FIX: After each page fetch, check if ALL returned trial_ids were already seen.
    # If yes → we've hit the pagination ceiling → stop.
    consecutive_no_new = 0

    for next_page in range(2, min(expected_pages + 1, MAX_PAGES + 1)):
        time.sleep(SLEEP_BETWEEN)

        page_resp = _navigate_to_page(session, next_page, page_soup)
        if not page_resp:
            # Retry once
            time.sleep(2)
            page_resp = _navigate_to_page(session, next_page, page_soup)
            if not page_resp:
                print(f"|p{next_page}:ERR", end="", flush=True)
                consecutive_no_new += 1
                if consecutive_no_new >= 3:
                    break
                continue

        page_soup = BeautifulSoup(page_resp.text, "html.parser")
        page_trials = _parse_ictrp_table(page_soup)

        if not page_trials:
            break

        # Check how many are genuinely new (not seen before)
        new_trials = [t for t in page_trials if t["trial_id"] not in seen_ids]
        for t in new_trials:
            seen_ids.add(t["trial_id"])

        if not new_trials:
            # Server looped back — all trials on this page were already seen
            consecutive_no_new += 1
            print(f"|p{next_page}:dup", end="", flush=True)
            if consecutive_no_new >= 2:
                # Confirmed: pagination has cycled. Stop.
                print("|STOP(cycle)", end="", flush=True)
                break
        else:
            consecutive_no_new = 0
            # DEDUP_PAGES switch: if False, keep ALL rows (including duplicates)
            if DEDUP_PAGES:
                all_trials.extend(new_trials)
                print(f"|p{next_page}:+{len(new_trials)}", end="", flush=True)
            else:
                all_trials.extend(page_trials)
                print(f"|p{next_page}:+{len(page_trials)}", end="", flush=True)

    label = "unique" if DEDUP_PAGES else "total(with dups)"
    print(f" ={len(all_trials)} {label}")
    return all_trials


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


def search_all_synonyms(search_terms: list[str]) -> list[dict]:
    """
    Search WHO ICTRP for all search_terms.
    Deduplicates by trial ID across all searches.
    Optionally skips NCT IDs.
    """
    print("\n" + "═" * 70)
    print("  WHO ICTRP — International Clinical Trials Registry Platform")
    print("═" * 70)
    print(f"  Drug: {DRUG_NAME}")
    print(f"  Search terms: {', '.join(search_terms)}")
    print(f"  Skip NCT IDs: {SKIP_NCT}")
    print(f"  Max pages/term: {MAX_PAGES}")
    print("─" * 70)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })

    all_trials = {}  # trial_id → trial_dict (dedup)

    for synonym in search_terms:
        print(f"\n  🔍 Searching: '{synonym}' ...", end=" ", flush=True)
        trials = fetch_ictrp_trials(synonym, session)

        new_count = 0
        for trial in trials:
            tid = trial["trial_id"]

            # Skip NCT IDs if configured (covered by CT.gov connector)
            if SKIP_NCT and tid.upper().startswith("NCT"):
                continue

            if tid not in all_trials:
                trial["search_term"] = synonym
                trial["source_registry"] = _detect_registry(tid)
                all_trials[tid] = trial
                new_count += 1

        print(f"     → +{new_count} new (dedup total: {len(all_trials)})")
        time.sleep(SLEEP_BETWEEN)

    print(f"\n{'─'*70}")
    print(f"  ✅ Grand Total (deduplicated, non-NCT): {len(all_trials)} trials")
    print(f"{'═'*70}\n")

    return list(all_trials.values())


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMATION — Aligned 20-column output
# ═══════════════════════════════════════════════════════════════════════════════

def trials_to_dataframe(trials: list[dict], drug_name: str) -> pd.DataFrame:
    """Convert list of ICTRP trial dicts → aligned DataFrame."""
    rows = []
    for i, trial in enumerate(trials, 1):
        trial_id    = trial.get("trial_id", "")
        title       = trial.get("title", "")
        status      = trial.get("status", "")
        date_reg    = trial.get("date_registered", "")
        source_reg  = trial.get("source_registry", "")
        detail_url  = trial.get("detail_url", "")
        has_results = trial.get("has_results", "")

        # Derived fields
        phase     = _extract_phase_from_text(title)
        timestamp = _normalize_date(date_reg)

        # Build aligned row
        row = {
            "#":                   i,
            "product_name":        drug_name,
            "trial_phase":         phase,
            "timestamp":           timestamp,
            "timestamp_rationale": f"Date registered in {source_reg} (via WHO ICTRP)",
            "timestamp_enriched":  "ICTRP",
            "connector":           "data_source-3 (WHO ICTRP)",
            "geo":                 _infer_geo(source_reg),
            "raw_text":            title,
            "evidence":            f"Trial ID: {trial_id} | Registry: {source_reg}",
            "url_vertexai":        "",
            "url_resolved":        detail_url,
            "url_status":          "direct" if detail_url else "missing",
            "source_validated":    "YES",
            "validation_note":     "WHO ICTRP international registry platform record",
            "reasoning":           f"Status={status} | Phase={phase or 'N/A'}",
            # ── Bonus columns ──
            "trial_id":            trial_id,
            "source_registry":     source_reg,
            "status":  status,
            "has_results":         has_results,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        cols = [
            "#", "product_name", "trial_phase", "timestamp",
            "timestamp_rationale", "timestamp_enriched", "connector", "geo",
            "raw_text", "evidence", "url_vertexai", "url_resolved",
            "url_status", "source_validated", "validation_note", "reasoning",
            "trial_id", "source_registry", "status", "has_results",
        ]
        df = pd.DataFrame(columns=cols)

    return df


def _infer_geo(registry: str) -> str:
    """Infer geography from registry name."""
    geo_map = {
        "Clinical Trials Registry - India": "India",
        "Japan Primary Registries Network": "Japan",
        "ANZCTR (Australia/NZ)": "Australia/New Zealand",
        "Chinese Clinical Trial Registry": "China",
        "Korean Clinical Trial Registry": "South Korea",
        "German Clinical Trials Register": "Germany",
        "Netherlands Trial Register": "Netherlands",
        "Pan African Clinical Trial Registry": "Africa",
        "Sri Lanka Clinical Trials Registry": "Sri Lanka",
        "Thai Clinical Trials Registry": "Thailand",
        "Cuban Public Registry": "Cuba",
        "Iranian Registry of Clinical Trials": "Iran",
        "Lebanese Clinical Trials Registry": "Lebanon",
        "Brazilian Clinical Trials Registry": "Brazil",
        "Peruvian Clinical Trials Registry": "Peru",
        "EU Clinical Trials Register": "EU",
        "ISRCTN Registry": "International",
        "ClinicalTrials.gov": "US",
    }
    return geo_map.get(registry, "")


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEL OUTPUT — Styled
# ═══════════════════════════════════════════════════════════════════════════════

def save_to_excel(df: pd.DataFrame, drug_name: str) -> str:
    """Save DataFrame to styled Excel file."""
    drug_folder = _safe_folder_name(drug_name)
    drug_short = drug_folder.replace(" ", "_")[:30]
    filename = f"who_ictrp_results_{drug_short}.xlsx"

    # Build workbook fully in memory. Do not create/read a local XLSX file.
    wb = Workbook()
    ws = wb.active
    ws.title = "WHO_ICTRP"

    cols = list(df.columns)

    # Header row
    for c, col_name in enumerate(cols, 1):
        ws.cell(row=1, column=c, value=col_name)

    # Data rows — keep blanks as blanks, avoid literal "nan" in Excel.
    for r_idx, row in enumerate(df.itertuples(index=False, name=None), start=2):
        for c_idx, value in enumerate(row, start=1):
            if pd.isna(value):
                value = ""
            ws.cell(row=r_idx, column=c_idx, value=value)

    # ── Header styling ──
    header_font = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
    header_fill = PatternFill(start_color="1B4F72", end_color="1B4F72", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin")
    )

    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # ── Data styling ──
    data_font = Font(name="Calibri", size=9)
    data_align = Alignment(vertical="top", wrap_text=True)

    for row_idx in range(2, ws.max_row + 1):
        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.font = data_font
            cell.alignment = data_align
            cell.border = thin_border

    # ── Column widths ──
    col_widths = {
        "#": 5, "product_name": 20, "trial_phase": 12,
        "timestamp": 14, "timestamp_rationale": 36, "timestamp_enriched": 14,
        "connector": 28, "geo": 15, "raw_text": 60,
        "evidence": 40, "url_vertexai": 10, "url_resolved": 50,
        "url_status": 12, "source_validated": 16, "validation_note": 38,
        "reasoning": 40, "trial_id": 28, "source_registry": 28,
        "status": 22, "has_results": 14,
    }
    for col_idx in range(1, ws.max_column + 1):
        col_name = ws.cell(row=1, column=col_idx).value
        width = col_widths.get(col_name, 15)
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    # ── Freeze panes & filters ──
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = ws.dimensions

    # ── Summary sheet ──
    ws2 = wb.create_sheet(title="Summary")
    bold       = Font(bold=True, size=10)
    label_font = Font(name="Calibri", bold=True, size=10)
    value_font = Font(name="Calibri", size=10)

    ws2.cell(row=1, column=1, value="SUMMARY").font = Font(bold=True, size=12)
    ws2.cell(row=2, column=1, value="Drug:").font = label_font
    ws2.cell(row=2, column=2, value=drug_name).font = value_font
    ws2.cell(row=3, column=1, value="Total Trials:").font = label_font
    ws2.cell(row=3, column=2, value=len(df)).font = value_font
    ws2.cell(row=4, column=1, value="Run Time:").font = label_font
    ws2.cell(row=4, column=2, value=TS).font = value_font
    ws2.cell(row=5, column=1, value="NCT Filtered:").font = label_font
    ws2.cell(row=5, column=2, value="Yes" if SKIP_NCT else "No").font = value_font

    r = 7
    if not df.empty and "source_registry" in df.columns:
        ws2.cell(row=r, column=1, value="REGISTRY BREAKDOWN").font = bold
        r += 1
        for reg, count in df["source_registry"].value_counts().items():
            ws2.cell(row=r, column=1, value=reg).font = value_font
            ws2.cell(row=r, column=2, value=count).font = value_font
            r += 1

    ws2.column_dimensions["A"].width = 35
    ws2.column_dimensions["B"].width = 20

    # Write under product-specific folder in GCS, not bucket root.
    gcs_object_name = f"{drug_folder}/{filename}"
    save_workbook_to_gcs(wb, gcs_object_name)
    return gcs_object_name


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_for_drug(drug_name: str, company_name: str = "") -> str:
    """
    Batch-runner entrypoint for WHO ICTRP.
    Runs one product at a time, like ClinicalTrials.gov / PubMed.
    """
    if not drug_name:
        raise ValueError("drug_name is required")

    ctx = RunContext(product_name=drug_name, company_name=company_name)

    # Load registry from BigQuery → build search terms: product + brand names
    registry = get_or_build_registry(ctx)
    ctx.registry = registry
    brand_names = registry.get("brand_names", [])
    search_terms = [ctx.product_name] + [
        b for b in brand_names
        if b and b.lower() != ctx.product_name.lower()
    ]

    print(f"\n{'▓'*70}")
    print("  WHO ICTRP CONNECTOR")
    print(f"  Run: {TS}")
    print(f"  Drug: {drug_name}")
    if company_name:
        print(f"  Company: {company_name}")
    print(f"  Search terms: {search_terms}")
    print(f"{'▓'*70}")

    start_time = time.time()
    trials = search_all_synonyms(search_terms)

    if not trials:
        print("\n  ⚠ No trials found. Empty output file will be created.")

    df = trials_to_dataframe(trials, ctx.product_name)
    print(f"\n  📊 DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns")

    if not df.empty:
        print("\n  Registry breakdown:")
        for reg, count in df["source_registry"].value_counts().items():
            print(f"    • {reg}: {count}")

        print("\n  Geography breakdown:")
        for geo, count in df["geo"].value_counts().items():
            if geo:
                print(f"    • {geo}: {count}")

    filename = save_to_excel(df, ctx.product_name)
    elapsed = time.time() - start_time

    print(f"\n  💾 Saved: {filename}")
    print(f"  ⏱  Runtime: {elapsed:.1f}s")
    print(f"{'▓'*70}\n")
    return filename


def main():
    """Single-product backward-compatible mode only."""
    drug_name = DRUG_NAME
    company_name = getattr(config, "COMPANY_NAME", "")
    if not drug_name:
        print("No DRUG_NAME set in config.py. Exiting.")
        return
    run_for_drug(drug_name, company_name)


if __name__ == "__main__":
    main()
