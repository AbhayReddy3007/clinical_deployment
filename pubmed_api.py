"""
PubMed Literature Connector — Clinical Trial Publications
- Fetches published clinical trial results from PubMed (NCBI E-utilities API)
- Aligned output schema matches deep researcher and clinical_trials_org_api
- Extracts NCT IDs from DataBankList (structured) + abstract regex (fallback)
- Publication types → clinical_event_classification
- MeSH terms → indication proxy
- DOI preferred over PubMed URL → url_resolved
- Color-coded Excel output with Summary sheet (same style as deep researcher)
- No auth required; add NCBI_API_KEY to config.py for 10x rate limit


Usage:
    python pubmed_api.py
"""
"""
note: # One call → returns ALL known names for a drug
https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/tirzepatide/synonyms/JSON
# Returns: Mounjaro, Zepbound, LY3298176, tirzepatide, 
#          2450006-71-8 (CAS), etc. — hundreds of entries
"""
import requests
import pandas as pd
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict
import re
import xml.etree.ElementTree as ET
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
        NCBI_API_KEY = None
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


# ═══════════════════ CONFIG ═══════════════════
# Batch runner passes product/company into run_for_drug().
# These globals are only for backward-compatible single-product CLI runs.
DRUG_NAME    = getattr(config, "DRUG_NAME", "")
COMPANY_NAME = getattr(config, "COMPANY_NAME", "")
NCBI_API_KEY = getattr(config, "NCBI_API_KEY", None)   # optional — add to config.py for 10x rate limit

ESEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
FETCH_BATCH  = 100      # articles per efetch call
MAX_RESULTS  = 10000    # safety cap for esearch retmax
SLEEP_SEC    = 0.4      # NCBI polite delay: ≤3 req/sec free, ≤10/sec with API key

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

# -----------------------------------------------
# PUBLICATION TYPE → clinical_event_classification
# First match in list wins (ordered by priority)
# -----------------------------------------------
_PUBTYPE_MAP = [
    ("Clinical Trial, Phase IV",    "Phase IV / Post-Marketing Study"),
    ("Clinical Trial, Phase III",   "Published Full Results"),
    ("Clinical Trial, Phase II",    "Published Full Results"),
    ("Clinical Trial, Phase I",     "Trial Initiation / IND-CTA Filed"),
    ("Randomized Controlled Trial", "Published Full Results"),
    ("Multicenter Study",           "Published Full Results"),
    ("Meta-Analysis",               "Biomarker / Subgroup Analysis"),
    ("Systematic Review",           "Biomarker / Subgroup Analysis"),
    ("Observational Study",         "Other Clinical Development Event"),
    ("Clinical Trial",              "Published Full Results"),
]

_PHASE_MAP = [
    ("Phase IV",  "Phase 4"),
    ("Phase III", "Phase 3"),
    ("Phase II",  "Phase 2"),
    ("Phase I",   "Phase 1"),
]


# -----------------------------------------------
# HELPERS
# -----------------------------------------------
def derive_classification(pub_types: list[str]) -> str:
    """Map list of publication types → clinical_event_classification (first match wins)."""
    for pt_key, cls in _PUBTYPE_MAP:
        if pt_key in pub_types:
            return cls
    return "Other Clinical Development Event"


def derive_trial_phase(pub_types: list[str]) -> str:
    """Derive trial phase label from publication types."""
    for pt_key, label in _PHASE_MAP:
        for pt in pub_types:
            if pt_key in pt:
                return label
    return ""


# ─────────────────────────────────────────────
# NON-NCT TRIAL REGISTRY EXTRACTION
# ClinicalTrials.gov (NCT) IDs are intentionally excluded — that source is
# already covered by the dedicated ctgov_port.py connector.
# ─────────────────────────────────────────────
_TRIAL_REGISTRY_PATTERNS = [
    (r"EUCTR\d{4}-\d{6}-\d{2}(?:-[A-Z]{2})?", "EU Clinical Trials Register"),
    (r"CTRI/\d{4}/\d{2}/\d{6}",               "Clinical Trials Registry - India"),
    (r"ISRCTN\d{8}",                          "ISRCTN Registry"),
    (r"ACTRN\d{14}",                          "ANZCTR (Australia/NZ)"),
    (r"ChiCTR-?[A-Z]{0,5}-?\d{8,10}",         "Chinese Clinical Trial Registry"),
    (r"JPRN-[A-Za-z0-9\-]+",                  "Japan Primary Registries Network"),
    (r"UMIN\d{9}",                            "Japan Primary Registries Network (UMIN)"),
    (r"KCT\d{7}",                             "Korean Clinical Trial Registry"),
    (r"DRKS\d{8}",                            "German Clinical Trials Register"),
    (r"NTR\d{3,4}",                           "Netherlands Trial Register"),
    (r"PACTR\d{15}",                          "Pan African Clinical Trial Registry"),
    (r"SLCTR/\d{4}/\d{3}",                    "Sri Lanka Clinical Trials Registry"),
    (r"TCTR\d{11}",                           "Thai Clinical Trials Registry"),
    (r"RPCEC\d{8}",                           "Cuban Public Registry"),
    (r"IRCT\d{14,20}N\d{1,3}",                "Iranian Registry of Clinical Trials"),
    (r"LBCTR\d{14}",                          "Lebanese Clinical Trials Registry"),
    (r"RBR-[a-z0-9]{6,8}",                    "Brazilian Clinical Trials Registry"),
    (r"PER-\d{3}-\d{2}",                      "Peruvian Clinical Trials Registry"),
]

# DataBankName values PubMed uses for non-ClinicalTrials.gov registries.
# (PubMed's own DataBank element already tells us the registry, so we don't
# need to guess it from the accession number format for these.)
_NON_NCT_DATABANK_NAMES = {
    "ISRCTN", "ChiCTR", "EudraCT", "UMIN-CTR", "JapicCTI", "jRCT",
    "Netherlands Trial Register", "German Clinical Trials Register",
    "ANZCTR", "CTRI", "IRCT", "PACTR", "TCTR", "DRKS",
}


def extract_trial_ids(article_elem: ET.Element, abstract_text: str) -> tuple[str, str]:
    """
    Extract non-NCT trial registry IDs from:
      1. DataBankList XML (structured — most reliable), skipping ClinicalTrials.gov
      2. Abstract free text regex (fallback)
    Returns (comma-separated trial IDs, comma-separated source registries).
    ClinicalTrials.gov (NCT) IDs are skipped on purpose — see ctgov_port.py.
    """
    trial_ids: set[str] = set()
    registries: set[str] = set()

    # Structured DataBankList — any registry that isn't ClinicalTrials.gov
    for db in article_elem.findall(".//DataBank"):
        db_name = (db.findtext("DataBankName") or "").strip()
        if not db_name or "ClinicalTrials" in db_name:
            continue
        for acc in db.findall(".//AccessionNumber"):
            if acc.text and acc.text.strip():
                trial_ids.add(acc.text.strip().upper())
                registries.add(db_name)

    # Regex fallback on abstract for known non-NCT registry formats
    for pattern, registry_name in _TRIAL_REGISTRY_PATTERNS:
        for match in re.findall(pattern, abstract_text or "", re.IGNORECASE):
            trial_ids.add(match.upper())
            registries.add(registry_name)

    return ", ".join(sorted(trial_ids)), ", ".join(sorted(registries))


def extract_mesh_terms(medline_citation: ET.Element, limit: int = 10) -> str:
    """Extract MeSH descriptor names as comma-separated string."""
    terms = []
    for mh in medline_citation.findall(".//MeshHeading"):
        desc = mh.findtext("DescriptorName")
        if desc:
            terms.append(desc.strip())
    return ", ".join(terms[:limit])


def parse_pubdate(journal_elem: ET.Element, article_elem: ET.Element | None = None) -> str:
    """Parse publication date — prefers ArticleDate (electronic) over Journal/PubDate (print)."""
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Aug": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }
    # Prefer ArticleDate (electronic pub date — more precise and earlier)
    if article_elem is not None:
        art_date = article_elem.find("ArticleDate")
        if art_date is not None:
            year  = art_date.findtext("Year")  or ""
            month = art_date.findtext("Month") or ""
            day   = art_date.findtext("Day")   or ""
            month = month_map.get(month, month)
            if year and month and day:
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            if year and month:
                return f"{year}-{month.zfill(2)}"
            if year:
                return year
    # Fall back to Journal/PubDate (print date)
    if journal_elem is None:
        return ""
    pub_date = journal_elem.find(".//PubDate")
    if pub_date is None:
        return ""
    year  = pub_date.findtext("Year")  or ""
    month = pub_date.findtext("Month") or ""
    day   = pub_date.findtext("Day")   or ""
    month = month_map.get(month, month)
    if year and month and day:
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    if year and month:
        return f"{year}-{month.zfill(2)}"
    return year


def safe_itertext(elem: ET.Element | None) -> str:
    """Safely get all text from an element including sub-elements."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()



# -----------------------------------------------
# STEP 1 — SEARCH: esearch → PMIDs
# -----------------------------------------------
def search_pubmed(drug_name: str) -> list[str]:
    """
    Run esearch filtered to clinical publication types.
    Returns list of PMIDs (most recent first).
    """
    # Broad query: title/abstract OR supplementary concept (registry term)
    query = (
        f'("{drug_name}"[Title/Abstract] OR "{drug_name}"[Supplementary Concept]) '
        f'AND ("clinical trial"[Publication Type] '
        f'OR "randomized controlled trial"[Publication Type] '
        f'OR "meta-analysis"[Publication Type] '
        f'OR "systematic review"[Publication Type] '
        f'OR "observational study"[Publication Type])'
    )
    print(query)

    params: dict = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  MAX_RESULTS,
        "retmode": "json",
        "sort":    "pub_date",   # most recent first
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    print(f"\nSearching PubMed for: '{drug_name}'")
    print(f"Query: {query[:140]}...")
    print("-" * 55)

    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] esearch failed: {e}")
        sys.exit(1)

    result = data.get("esearchresult", {})
    total  = int(result.get("count", 0))
    pmids  = result.get("idlist", [])

    print(f"  Total PubMed articles found : {total}")
    print(f"  Retrieving                  : {len(pmids)} PMIDs (cap={MAX_RESULTS})")
    return pmids


# -----------------------------------------------
# STEP 2 — FETCH: efetch → XML elements
# -----------------------------------------------
def fetch_articles(pmids: list[str]) -> list[ET.Element]:
    """
    Fetch full article XML via efetch in batches of FETCH_BATCH.
    Returns list of <PubmedArticle> XML elements.
    """
    articles: list[ET.Element] = []
    total_batches = (len(pmids) + FETCH_BATCH - 1) // FETCH_BATCH

    for batch_num, start in enumerate(range(0, len(pmids), FETCH_BATCH), 1):
        batch = pmids[start : start + FETCH_BATCH]
        print(
            f"  Batch {batch_num}/{total_batches} — fetching {len(batch)} articles...",
            end=" ", flush=True,
        )
        params: dict = {
            "db":      "pubmed",
            "id":      ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        try:
            resp = requests.get(EFETCH_URL, params=params, timeout=60)
            resp.raise_for_status()
            root         = ET.fromstring(resp.content)
            batch_arts   = root.findall("PubmedArticle")
            articles.extend(batch_arts)
            print(f"OK ({len(batch_arts)} parsed, cumulative={len(articles)})")
        except ET.ParseError as e:
            print(f"XML parse error — {e} — skipping batch")
        except requests.exceptions.RequestException as e:
            print(f"Request error — {e} — skipping batch")

        time.sleep(SLEEP_SEC)

    return articles


# -----------------------------------------------
# STEP 3 — TRANSFORM: XML elements → aligned DataFrame
# -----------------------------------------------
def articles_to_dataframe(articles: list[ET.Element], drug_name: str) -> pd.DataFrame:
    """
    Convert PubmedArticle XML elements to an aligned DataFrame.
    20 columns match the deep researcher Final_Output schema exactly.
    7 bonus pm_ columns provide PubMed-specific context.
    """
    rows = []

    for i, article in enumerate(articles, 1):
        mc  = article.find("MedlineCitation")
        art = mc.find("Article") if mc is not None else None
        if mc is None or art is None:
            continue

        # ── Core identifiers ────────────────────────────────────────────
        pmid_elem = mc.find("PMID")
        pmid = pmid_elem.text.strip() if (pmid_elem is not None and pmid_elem.text) else ""

        # ── Title ───────────────────────────────────────────────────────
        title = safe_itertext(art.find("ArticleTitle"))

        # ── Abstract — handles structured labels (BACKGROUND, METHODS…) ─
        abstract_parts = []
        for ab in art.findall(".//AbstractText"):
            label = ab.get("Label")
            text  = "".join(ab.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)

        # ── Journal & publication date ───────────────────────────────────
        journal_elem = art.find("Journal")
        journal_name = ""
        if journal_elem is not None:
            journal_name = (
                journal_elem.findtext("Title")
                or journal_elem.findtext("ISOAbbreviation")
                or ""
            )
        pub_date = parse_pubdate(journal_elem, art)

        # ── Publication types ────────────────────────────────────────────
        pub_types = [
            pt.text.strip()
            for pt in art.findall(".//PublicationType")
            if pt.text
        ]

        # ── Authors (first 3, then "et al.") ────────────────────────────
        all_authors = art.findall(".//Author")
        author_parts = []
        for author in all_authors[:3]:
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or author.findtext("Initials") or ""
            if last:
                author_parts.append(f"{last} {fore}".strip())
        author_str = "; ".join(author_parts)
        if len(all_authors) > 3:
            author_str += " et al."

        # ── DOI ─────────────────────────────────────────────────────────
        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        # ── Non-NCT trial IDs — structured DataBankList + abstract regex ──
        trial_id, source_registry = extract_trial_ids(article, abstract)

        # ── Derived clinical fields ──────────────────────────────────────
        trial_phase = derive_trial_phase(pub_types)

        # ── URLs: PubMed URL always in url_resolved; DOI in doi field only ──
        url_pubmed   = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        url_resolved = url_pubmed

        # ── Evidence & reasoning strings ─────────────────────────────────
        evidence = f"PubMed PMID: {pmid}"
        if journal_name:
            evidence += f" | Journal: {journal_name}"
        if pub_types:
            evidence += f" | PubType: {'; '.join(pub_types[:3])}"

        reasoning_parts = [f"PMID={pmid}"]
        if journal_name:
            reasoning_parts.append(f"Journal={journal_name}")
        if pub_types:
            reasoning_parts.append(f"PubTypes={'; '.join(pub_types[:3])}")
        if trial_phase:
            reasoning_parts.append(f"Phase={trial_phase}")
        if trial_id:
            reasoning_parts.append(f"TrialID={trial_id} ({source_registry})")
        if author_str:
            reasoning_parts.append(f"Authors={author_str}")
        reasoning = " | ".join(reasoning_parts)

        # ── Build aligned row ─────────────────────────────────────────────
        row = {
            "#":                     i,
            "product_name":          drug_name,
            "trial_phase":           trial_phase,
            "trial_id":              trial_id,
            "source_registry":       source_registry,
            "raw_text":              title,
            "brief_summary":         abstract,
            "timestamp":             pub_date,
            "timestamp_rationale":   "PubMed publication date (electronic or print)",
            "timestamp_enriched":    "PUBMED",
            "connector":             "data_source-2 (PubMed)",
            "evidence":              evidence,
            "url_vertexai":          "",
            "url_resolved":          url_resolved,
            "url_status":            "direct",
            "source_validated":      "YES",
            "validation_note":       "Peer-reviewed publication — PubMed/NCBI indexed",
            "reasoning":             reasoning,
            # ── PubMed-specific context columns ───────────────────────────
            "pmid":                  pmid,
            "journal":               journal_name,
            "pub_types":             "; ".join(pub_types),
            "authors":               author_str,
            "doi":                   doi,
        }
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------
# STEP 4 — EXCEL: styled output aligned to deep researcher
# -----------------------------------------------
def save_to_excel(df: pd.DataFrame, drug_name: str, company_name: str = "") -> str:
    """
    Save aligned DataFrame to Excel.
    Sheet 1: Publications  — 20 aligned + 7 bonus pm_ columns, color-coded
    Sheet 2: Summary       — classification / phase / publication-type breakdown
    Styling matches deep researcher and clinical_trials_org_api exactly.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    # ── Shared styles (identical to deep researcher) ──────────────────────
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
    THIN_BORDER = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"),  bottom=Side(style="thin"),
    )
    GREEN_FILL  = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    RED_FILL    = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    ORANGE_FILL = PatternFill(start_color="FCD5B4", end_color="FCD5B4", fill_type="solid")
    BLUE_FILL   = PatternFill(start_color="D9EAF7", end_color="D9EAF7", fill_type="solid")
    GRAY_FILL   = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    PM_FILL     = PatternFill(start_color="F0EBF8", end_color="F0EBF8", fill_type="solid")  # purple tint for pm_ cols
    DATA_FONT   = Font(size=10)

    safe_name = _safe_folder_name(drug_name)
    filename  = f"{safe_name}_pubmed_literature.xlsx"
    wb        = Workbook()

    # ── Sheet 1: Publications ─────────────────────────────────────────────
    ws   = wb.active
    ws.title = "Publications"
    cols = list(df.columns)

    # Header row
    for c, col_name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c, value=col_name)
        cell.fill      = HEADER_FILL
        cell.font      = HEADER_FONT
        cell.border    = THIN_BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    WRAP_COLS = {
        "raw_text", "brief_summary", "evidence", "reasoning",
        "timestamp_rationale", "validation_note",
        "pub_types", "authors",
    }

    for r_idx, row in df.iterrows():
        excel_row = r_idx + 2    # df is 0-indexed; row 1 = header
        is_even   = (r_idx % 2 == 0)

        for c, col_name in enumerate(cols, 1):
            val  = row[col_name]
            cell = ws.cell(
                row=excel_row, column=c,
                value=str(val) if val is not None else "",
            )
            cell.font      = DATA_FONT
            cell.border    = THIN_BORDER
            cell.alignment = Alignment(
                vertical="top", wrap_text=(col_name in WRAP_COLS)
            )

            # ── Color logic (mirrors deep researcher) ─────────────────────
            if col_name == "url_resolved":
                cell.fill = GREEN_FILL if str(val).strip() else RED_FILL

            elif col_name == "source_validated":
                cell.fill = GREEN_FILL   # always YES — PubMed is authoritative

            elif col_name in {"pmid", "journal", "pub_types", "authors", "doi"} and is_even:
                cell.fill = PM_FILL      # subtle purple tint for bonus context cols

            elif col_name in ("#", "product_name", "connector",
                              "timestamp_enriched", "url_status") and is_even:
                cell.fill = GRAY_FILL

    # Column widths
    COL_WIDTHS = {
        "#":                     5,
        "product_name":         18,
        "trial_phase":          14,
        "trial_id":             22,
        "source_registry":      32,
        "raw_text":             70,
        "brief_summary":        80,
        "timestamp":            14,
        "timestamp_rationale":  42,
        "timestamp_enriched":   16,
        "connector":            18,
        "evidence":             48,
        "url_vertexai":         10,
        "url_resolved":         52,
        "url_status":           12,
        "source_validated":     16,
        "validation_note":      40,
        "reasoning":            60,
        "pmid":                 14,
        "journal":              36,
        "pub_types":            48,
        "authors":              42,
        "doi":                  36,
    }
    for c, col_name in enumerate(cols, 1):
        ws.column_dimensions[get_column_letter(c)].width = COL_WIDTHS.get(col_name, 18)

    ws.freeze_panes = "A2"
    if len(df) > 0:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{len(df) + 1}"

    # ── Sheet 2: Summary ──────────────────────────────────────────────────
    ws2        = wb.create_sheet("Summary")
    LABEL_FONT = Font(bold=True, size=10)
    VALUE_FONT = Font(size=10)
    SEC_FONT   = Font(bold=True, size=10, color="1F4E79")

    phase_counts = (
        df["trial_phase"].value_counts().to_dict()
        if "trial_phase" in df.columns else {}
    )

    # Flatten semicolon-separated pub_types for count breakdown
    type_counts: dict[str, int] = {}
    if "pub_types" in df.columns:
        for pts in df["pub_types"].dropna():
            for pt in str(pts).split(";"):
                pt = pt.strip()
                if pt:
                    type_counts[pt] = type_counts.get(pt, 0) + 1

    summary_rows: list[tuple] = [
        ("Drug / Intervention",           drug_name),
        ("Company",                       company_name),
        ("Total Publications Retrieved",  len(df)),
        ("Output Columns",                len(df.columns) if len(df) > 0 else 0),
        ("Data Pulled On",                datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Source",                        "PubMed NCBI E-utilities API"),
        ("NCBI API Key Used",             "YES" if NCBI_API_KEY else "NO (free tier, 3 req/s)"),
        ("", ""),
        ("─── Trial Phase Breakdown ───", ""),
    ]
    for p, cnt in sorted(phase_counts.items(), key=lambda x: -x[1]):
        summary_rows.append((f"  {p}", cnt))

    summary_rows += [("", ""), ("─── Publication Type Breakdown (top 15) ───", "")]
    for pt, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:15]:
        summary_rows.append((f"  {pt}", cnt))

    ws2.column_dimensions["A"].width = 50
    ws2.column_dimensions["B"].width = 40

    for r_idx, (label, value) in enumerate(summary_rows, start=1):
        a = ws2.cell(row=r_idx, column=1, value=label)
        b = ws2.cell(row=r_idx, column=2, value=value)
        if r_idx <= 8:
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


# -----------------------------------------------
# MAIN
# -----------------------------------------------
def run_for_drug(drug_name: str, company_name: str = "") -> str:
    """
    Batch-runner entrypoint for PubMed.
    Product/company are passed by batch_runner_sources.py from config.PRODUCTS.
    """
    if not drug_name:
        raise ValueError("drug_name is required")

    ctx = RunContext(product_name=drug_name, company_name=company_name)

    registry = get_or_build_registry(ctx)
    ctx.registry = registry
    brand_names = registry.get("brand_names", [])
    search_terms = [ctx.product_name] + [
        b for b in brand_names
        if b and b.lower() != ctx.product_name.lower()
    ]

    print(f"\n{'=' * 60}")
    print("  PubMed Literature Connector — Clinical Trial Publications")
    print(f"  Drug: {drug_name}")
    if company_name:
        print(f"  Company: {company_name}")
    print(f"  Search terms       : {search_terms}")
    print(f"  NCBI API Key: {'YES' if NCBI_API_KEY else 'NO (free tier)'}")
    print(f"  Batch size: {FETCH_BATCH} | Max results: {MAX_RESULTS}")
    print(f"{'=' * 60}")

    all_pmids: list[str] = []
    seen_pmids: set[str] = set()
    for term in search_terms:
        pmids = search_pubmed(term)
        new = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new)
        all_pmids.extend(new)
        print(f"  '{term}': {len(pmids)} hits, {len(new)} new unique PMIDs")

    if not all_pmids:
        print(f"\nNo PubMed articles found for '{drug_name}' or its brand names.")
        return ""

    print(f"\n  Total unique PMIDs : {len(all_pmids)}")

    print(f"\nFetching full article data for {len(all_pmids)} PMIDs...")
    articles = fetch_articles(all_pmids)
    print(f"  → {len(articles)} articles fetched and parsed")

    if not articles:
        print("No articles could be parsed. Exiting.")
        return ""

    print("\nBuilding aligned DataFrame...")
    df = articles_to_dataframe(articles, ctx.product_name)
    before = len(df)
    df = df.drop_duplicates(subset=["pmid"], keep="first").reset_index(drop=True)
    df["#"] = df.index + 1
    print(f"  → {before} rows before dedup, {len(df)} after (by pmid) × {len(df.columns)} columns")

    print("\nSaving to Excel...")
    filepath = save_to_excel(df, ctx.product_name, ctx.company_name)

    phase_counts = df["trial_phase"].value_counts().to_dict() if len(df) > 0 else {}

    print(f"\n{'=' * 60}")
    print("  ✅ DONE")
    print(f"  Search terms       : {search_terms}")
    print(f"  Articles fetched   : {len(articles)}")
    print(f"  Rows in output     : {len(df)}")
    print("  ─── Trial Phase ───")
    for p, cnt in sorted(phase_counts.items(), key=lambda x: -x[1]):
        print(f"    {p}: {cnt}")
    print(f"  📊 {filepath}")
    print(f"{'=' * 60}\n")
    return filepath


def main():
    # Single-product backward-compatible mode only.
    # Batch mode does not use config.DRUG_NAME.
    if not DRUG_NAME:
        print("No DRUG_NAME set in config.py. Exiting.")
        sys.exit(0)
    run_for_drug(DRUG_NAME, COMPANY_NAME)


if __name__ == "__main__":
    main()
