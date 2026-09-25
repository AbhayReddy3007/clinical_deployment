#!/usr/bin/env python3
"""
ctri_trials.py – Clinical Trials Registry - India (CTRI).

    Search : https://ctri.nic.in/Clinicaltrials/advsearch.php
    Trial  : https://ctri.nic.in/Clinicaltrials/pmaindet2.php?EncHid=<hash>

CTRI has no public API. The site is a classic PHP app whose trial pages are
label/value tables, so we harvest every label/value pair on the page rather
than hard-coding a field list. CTRI is slow and rate-sensitive, so requests
are throttled.

registry_common utilities are inlined below — no external dependency needed.
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup
import requests

# ==============================================================================
# INLINED registry_common
# ==============================================================================

SRC_CTGOV = "ClinicalTrials.gov"
SRC_CTRI  = "CTRI (India)"

UNIFIED_COLUMNS = [
    "registry_source", "trial_id", "secondary_ids", "title", "public_title",
    "status", "phase", "study_type", "study_design", "conditions",
    "interventions", "drug_names", "sponsor", "sponsor_type", "collaborators",
    "countries", "sites", "target_enrollment", "actual_enrollment",
    "age_min", "age_max", "gender", "healthy_volunteers",
    "inclusion_criteria", "exclusion_criteria", "primary_objective",
    "primary_outcome", "secondary_outcome", "start_date", "completion_date",
    "registration_date", "last_updated", "results_available", "findings",
    "contact", "ethics_approval", "url",
]


def blank_row(source: str) -> Dict[str, Any]:
    row = {c: "" for c in UNIFIED_COLUMNS}
    row["registry_source"] = source
    return row


def clean(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def first_nonempty(*values: Any) -> str:
    for v in values:
        s = clean(v)
        if s:
            return s
    return ""


def join(items: Iterable[Any], sep: str = "; ") -> str:
    parts = [clean(i) for i in items if clean(i)]
    return sep.join(parts)


def make_session(extra_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    })
    if extra_headers:
        s.headers.update(extra_headers)
    return s


def http_get(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, str]] = None,
    expect_json: bool = False,
    timeout: int = 30,
    retries: int = 3,
) -> Any:
    backoff = 2.0
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json() if expect_json else resp.text
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2
    raise last_exc


def http_post(
    session: requests.Session,
    url: str,
    data: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> str:
    resp = session.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def find_field(pairs: Dict[str, str], *labels: str) -> str:
    for label in labels:
        if label in pairs:
            return pairs[label]
        lo = label.lower()
        for k, v in pairs.items():
            if k.lower().startswith(lo):
                return v
    return ""


def extract_label_value_pairs(soup) -> Dict[str, str]:
    """Extract label→value pairs from a BeautifulSoup-parsed CTRI detail page."""
    pairs: Dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            label = clean(cells[0].get_text(" ", strip=True)).rstrip(":")
            value = clean(cells[1].get_text(" ", strip=True))
            if label and value:
                pairs[label] = value
    return pairs


def run_cli(fetch_fn, source_name: str, description: str) -> int:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("drug", help="Drug / molecule name to search for")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--out", default=None, help="Output JSON file")
    args = ap.parse_args()

    rows = fetch_fn(args.drug, max_records=args.max_records)
    out_path = args.out or f"{args.drug.lower().replace(' ', '_')}_{source_name.lower().replace(' ', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} row(s) → {out_path}", file=sys.stderr)
    return 0


# ==============================================================================
# ctri_trials logic
# ==============================================================================

BASE       = "https://ctri.nic.in"
SEARCH_URL = BASE + "/Clinicaltrials/advancesearchmain.php"
SEARCH_GET = BASE + "/Clinicaltrials/showallp.php?mid1=&EncHid=&userName={q}"
DETAIL_RE  = re.compile(r"(?:showallp|pmaindet2)\.php\?(?:mid1=\d*&)?EncHid=([^&\"'\s]*)", re.I)
DETAIL_ALT = re.compile(r"showallp\.php\?mid1=(\d+)", re.I)
DELAY      = 1.5


def _search_links(drug: str, session, max_records: Optional[int]) -> List[str]:
    """Collect detail-page hrefs for trials matching the drug."""
    links: List[str] = []
    seen  = set()

    candidates = [
        SEARCH_GET.format(q=quote_plus(drug)),
        f"{BASE}/Clinicaltrials/showallp.php?mid1=&EncHid=&userName={quote_plus(drug)}",
    ]

    html = ""
    for url in candidates:
        try:
            html = http_get(session, url)
            if DETAIL_RE.search(html):
                break
        except Exception as exc:
            print(f"  [CTRI] {url} -> {exc}", file=sys.stderr)

    if not html or not DETAIL_RE.search(html):
        try:
            html = http_post(session, SEARCH_URL,
                             data={"drug": drug, "search": "Search",
                                   "public_title": drug})
        except Exception as exc:
            print(f"  [CTRI] search POST failed: {exc}", file=sys.stderr)
            return links

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        m = DETAIL_RE.search(a["href"]) or DETAIL_ALT.search(a["href"])
        if not m:
            continue
        href = urljoin(BASE + "/Clinicaltrials/", a["href"])
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
        if max_records and len(links) >= max_records:
            break
    return links


def _map_detail(html: str, url: str) -> Dict[str, Any]:
    soup  = BeautifulSoup(html, "html.parser")
    pairs = extract_label_value_pairs(soup)
    text  = clean(soup.get_text(" ", strip=True))

    ctri_no = find_field(pairs, "CTRI Number", "CTRI No")
    if not ctri_no:
        m = re.search(r"(CTRI/\d{4}/\d{2}/\d+)", text)
        ctri_no = m.group(1) if m else ""

    row = blank_row(SRC_CTRI)
    row.update({
        "trial_id":      ctri_no,
        "secondary_ids": join([find_field(pairs, "Secondary IDs if Any",
                                          "Secondary ID"),
                               find_field(pairs, "Protocol Number"),
                               find_field(pairs, "UTN")]),
        "title":         first_nonempty(find_field(pairs, "Scientific Title of Study",
                                                   "Scientific Title"),
                                        find_field(pairs, "Public Title of Study",
                                                   "Public Title")),
        "public_title":  find_field(pairs, "Public Title of Study", "Public Title",
                                    "Brief Summary"),
        "status":        find_field(pairs, "Recruitment Status of Trial",
                                    "Recruitment Status", "Trial Status"),
        "phase":         find_field(pairs, "Phase of Trial", "Phase"),
        "study_type":    find_field(pairs, "Type of Study", "Study Type",
                                    "Type of Trial"),
        "study_design":  join([find_field(pairs, "Study Design"),
                               find_field(pairs, "Method of generating random sequence"),
                               find_field(pairs, "Method of Concealment"),
                               find_field(pairs, "Blinding/Masking")]),
        "conditions":    join([find_field(pairs, "Health Condition", "Condition"),
                               find_field(pairs, "Health Type")]),
        "interventions": join([find_field(pairs, "Intervention", "Intervention/Comparator Agent"),
                               find_field(pairs, "Comparator Agent")]),
        "drug_names":    join([find_field(pairs, "Intervention"),
                               find_field(pairs, "Comparator Agent")]),
        "sponsor":       find_field(pairs, "Primary Sponsor", "Name of Primary Sponsor"),
        "sponsor_type":  find_field(pairs, "Type of Sponsor", "Sponsor Type"),
        "collaborators": join([find_field(pairs, "Secondary Sponsor"),
                               find_field(pairs, "Source of Monetary or Material Support"),
                               find_field(pairs, "Details of Secondary Sponsor")]),
        "countries":     first_nonempty(find_field(pairs, "Countries of Recruitment"),
                                        "India"),
        "sites":         join([find_field(pairs, "Sites of Study", "Site of Study"),
                               find_field(pairs, "Name of the Site")]),
        "target_enrollment": join([find_field(pairs, "Target Sample Size"),
                                   find_field(pairs, "Total Sample Size"),
                                   find_field(pairs, "Sample Size from India")]),
        "actual_enrollment": find_field(pairs, "Final Enrollment numbers achieved",
                                        "Actual Sample Size"),
        "age_min":       find_field(pairs, "Age From", "Minimum Age"),
        "age_max":       find_field(pairs, "Age To", "Maximum Age"),
        "gender":        find_field(pairs, "Gender", "Sex"),
        "healthy_volunteers": find_field(pairs, "Healthy Volunteers"),
        "inclusion_criteria": find_field(pairs, "Inclusion Criteria"),
        "exclusion_criteria": find_field(pairs, "Exclusion Criteria"),
        "primary_objective":  find_field(pairs, "Brief Summary", "Objective"),
        "primary_outcome":    find_field(pairs, "Primary Outcome"),
        "secondary_outcome":  find_field(pairs, "Secondary Outcome"),
        "start_date":    join([find_field(pairs, "Date of First Enrollment (India)",
                                          "Date of First Enrollment"),
                               find_field(pairs, "Date of Study Commencement")]),
        "completion_date": join([find_field(pairs, "Date of Study Completion"),
                                 find_field(pairs, "Estimated Duration of Trial")]),
        "registration_date": find_field(pairs, "Date of Registration", "Registered on"),
        "last_updated":  find_field(pairs, "Last Modified On", "Modified On"),
        "results_available": find_field(pairs, "Publication Details",
                                        "Results Available"),
        "findings":      join([find_field(pairs, "Summary of Results", "Brief Results"),
                               find_field(pairs, "Publication Details"),
                               find_field(pairs, "Outcome of the trial")]),
        "contact":       join([find_field(pairs, "Contact Person (Scientific Query)"),
                               find_field(pairs, "Contact Person (Public Query)"),
                               find_field(pairs, "Email")]),
        "ethics_approval": join([find_field(pairs, "Ethics Committee"),
                                 find_field(pairs, "Status of Ethics Committee"),
                                 find_field(pairs, "Approval Status"),
                                 find_field(pairs, "Regulatory Clearance Status")]),
        "url": url,
    })

    for k, v in pairs.items():
        col = "ctri." + re.sub(r"\s+", "_", k)[:80]
        if col not in row:
            row[col] = v
    return row


def fetch(drug: str, max_records: Optional[int] = None,
          details: bool = True) -> List[Dict[str, Any]]:
    session = make_session({"Referer": BASE + "/Clinicaltrials/login.php"})
    links   = _search_links(drug, session, max_records)
    print(f"  CTRI search returned {len(links)} trial link(s).", file=sys.stderr)

    rows: List[Dict[str, Any]] = []
    for i, url in enumerate(links, start=1):
        try:
            rows.append(_map_detail(http_get(session, url), url))
        except Exception as exc:
            print(f"  ! CTRI detail failed for {url}: {exc}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  ...CTRI {i}/{len(links)}", file=sys.stderr)
        time.sleep(DELAY)
    return rows


if __name__ == "__main__":
    sys.exit(run_cli(fetch, SRC_CTRI, "Fetch trials from CTRI (India)."))