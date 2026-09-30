# Clinical Efficacy Scoring Pipeline

The `clinical_efficacy_assessment()` pipeline identifies and scores a drug's
clinical efficacy for its current use, based on evidence pooled from clinical
trial registries, published literature, and public pipeline intelligence. It
discovers every trial associated with the drug across eleven independent
sources, filters that evidence by an LLM-derived confidence score, extracts
efficacy endpoints from the highest-confidence trials, computes a
phase-anchored composite score, and produces a business-facing PDF report
with a per-endpoint evidence breakdown.

> **Confidence gating.** Not every discovered trial is treated as usable
> evidence. Each trial is expected to carry an `llm_confidence` value (0–1);
> only trials **above 0.6** proceed to full endpoint enrichment and scoring —
> trials at or below the threshold are marked `N/A` on every endpoint and
> excluded from the weighted score, though they remain in the trial record.

---

## Pipeline Overview

```text
Step 1   Discovery
         ├── 1.1  Alias Resolution              → deduplicated search-term list (BigQuery)
         ├── 1.2  Registry & Literature APIs     → raw trial records (CTGOV, CTIS, EudraCT, CTRI, PubMed, WHO)
         ├── 1.3  Gemini-Augmented Discovery     → raw trial records (Int'l registries, NICE, Trade/Conf, Innovator web)
         ├── 1.4  Curated Table Lookup           → raw trial records (GD BigQuery table)
         └── 1.5  Dedup & Trim                   → unified trial list, one row per trial_id
                    │
                    ▼
Step 2   Confidence Gating & Enrichment
         ├── 2.1  LLM Confidence Evaluation      → llm_confidence + rationale per trial (BigQuery)
         ├── 2.2  Endpoint Extraction (new)      → weight loss / HbA1c / ALT / MASH per trial (BigQuery)
         └── 2.3  Metadata Backfill (existing)   → refreshed phase / dates / status per trial (BigQuery)
                    │
                    ▼
Step 3   Score Calculation
         Phase-anchored weighted efficacy score → weighted_score (1–5) + data_coverage
                    │
                    ▼
Step 4   Rationale Generation
         Score result → Gemini → 3-sentence narrative rationale
                    │
                    ▼
Step 5   Push to BigQuery
         Incremental INSERT / UPDATE of enriched trials + score → clinical_efficacy table
                    │
                    ▼
Step 6   Report Generation
         BQ data → (optional thin-data enrichment) → business narrative → PDF (GCS) + JSON payload (GCS)
```

> Steps 1–5 run for every molecule submitted to the pipeline. Step 6 (the
> PDF report) can be skipped independently via `--no-report`; Steps 2–4
> (enrichment, scoring, rationale) can be skipped via `--no-score` if only
> raw trial discovery is needed.

---

## Step 1 — Discovery

The pipeline casts a wide net across eleven independent trial sources to
collect every registered or publicly discussed trial for the drug. All
sources run against the primary drug name first, then again — batched —
against every resolved alias.

### 1.1 Alias Resolution

Expands the primary drug name into every name the trial registries might use
for it.

- Looks the drug up in a BigQuery alias table to retrieve its known raw
  alias string (brand names, INN synonyms, development codes).
- Passes the raw alias string to Gemini (no search grounding) to clean it
  into a deduplicated term list, discarding entries that are clearly not
  drug names (mechanism descriptions, gene names, registry IDs).
- If no BigQuery match is found, the pipeline proceeds with the primary name
  only.

**Artifacts produced:** an ordered list of search terms — primary name
first, then cleaned aliases — used to drive every source in 1.2–1.4.

### 1.2 Registry & Literature APIs

Queries public trial registries and literature databases directly — no LLM
involved, since each source exposes a structured, queryable API or a
predictable page layout.

| Source | What it does |
|---|---|
| **CTGOV** | ClinicalTrials.gov REST API v2 (US) — full study record flattened into columns |
| **CTIS** | EU Clinical Trials Information System public search API |
| **EudraCT** | Legacy EU-CTR register (HTML), including trial results |
| **CTRI** | Clinical Trials Registry – India (HTML, throttled — no public API) |
| **PubMed** | NCBI E-utilities — publications, with NCT IDs extracted from structured DataBankList fields and abstract text |
| **WHO** | WHO ICTRP international registry (ASP.NET table scrape) |

Each source is called once for the primary drug name, then once more per
resolved alias, all in parallel.

### 1.3 Gemini-Augmented Discovery

Covers sources with no queryable API — where finding the right trial or
signal requires reading and reasoning over free text, so each call uses
Gemini with Google Search grounding.

| Source | What it does |
|---|---|
| **International registries** | 6 additional country/region registries, searched via Gemini + Google Search, batched 3 registries per call |
| **NICE** | UK NICE Technology Appraisals and Highly Specialised Technology guidance |
| **Trade & Conferences** | Medical conference abstracts (ASCO, ESMO, ADA, etc.) and pharma trade press (Endpoints, STAT, FierceBiotech) |
| **Innovator web** | Sponsor/innovator company websites — a two-pass discovery-then-fallback scan |

For the primary drug name, each of these runs once. For aliases: NICE, Trade
& Conferences, and Innovator web each run **once** with all aliases combined
into a single prompt; international registries run once per batch of
aliases (default 10 aliases/batch), each internally batched at 3 registries
per call.

### 1.4 Curated Table Lookup

- Pulls any pre-vetted trials for the drug from a curated "GD Clinical
  Trials" BigQuery table.
- Rows sourced this way are auto-assigned `llm_confidence = 1` at ingestion,
  since they've already been through prior curation — they skip Step 2.1
  entirely.

### 1.5 Dedup & Trim

- Merges rows from every source and de-duplicates on `trial_id`, keeping the
  highest reported phase when the same trial appears in multiple registries.
- If a `top_n` cap is set, trims to the most complete trials before
  enrichment.

**Artifacts produced:** a single unified list of raw trial records, ready
for confidence evaluation.

---

## Step 2 — Confidence Gating & Enrichment

### 2.1 LLM Confidence Evaluation

- Every trial not sourced from the curated GD table needs an `llm_confidence`
  score before it can be enriched — trials sourced from the API/literature
  and Gemini-augmented discovery steps arrive without one.
- Sends trials to Gemini with Google Search grounding, in batches of 6, to
  verify the trial is genuinely associated with the drug and assign a
  confidence score (0–1) with a short rationale.
- Results are written back to BigQuery so subsequent runs don't
  re-evaluate trials that already have a score.

### 2.2 Endpoint Extraction (new trials)

- Trials with `llm_confidence > 0.6` that are not already in BigQuery are
  sent to Gemini (optionally with Google Search grounding) in batches of 6,
  to extract four efficacy endpoints: **Weight Loss %, HbA1c change %, ALT
  reduction %, MASH resolution %** — each with a duration, a short
  rationale, and a per-endpoint confidence flag.
- Trials at or below the 0.6 threshold are marked `N/A` on every endpoint
  column and skipped.

### 2.3 Metadata Backfill (existing trials)

- Trials already present in BigQuery only get a lighter refresh — title,
  phase, dates, status — in batches of 10, via Gemini with Google Search
  grounding. Endpoint values already on record are left untouched.

**Artifacts produced:** enriched trial rows (endpoints for new trials,
refreshed metadata for existing ones), ready for scoring.

---

## Step 3 — Score Calculation

Computes a **phase-anchored weighted efficacy score** (range: 1–5) from the
enriched trial set.

| # | Component | What it captures |
|---|-----------|-------------------|
| 1 | Per-endpoint best trial | For each of the 4 endpoints, prefer the highest-value Phase 3 trial; fall back to Phase 2, then Phase 1, if none is available |
| 2 | Phase penalty | Phase 3/4 → ×1.00, Phase 2 → ×0.85, Phase 1 → ×0.65, applied to the raw endpoint value |
| 3 | `_pct_to_score` | Buckets the phase-adjusted percentage into a 1–5 score: ≥22%→5, 16–21.9%→4, 10–15.9%→3, 5–9.9%→2, <5%→1 |
| 4 | Endpoint weights | Weight Loss 0.40, HbA1c 0.40, MASH Resolution 0.10, ALT Reduction 0.10 |
| 5 | `weighted_score` | Sum of each endpoint's bucketed score × its weight |
| 6 | `data_coverage` | How many of the 4 endpoints had usable data (e.g. "3/4 endpoints scored (missing: mash)") |

Endpoints are deduplicated by `trial_id` per phase tier (keeping the highest
value) before the best trial is selected, so a trial reported twice doesn't
double-count.

**Artifacts produced:** `weighted_score`, per-endpoint breakdown, and
`data_coverage` — attached to the trial set for the given molecule.

---

## Step 4 — Rationale Generation

- Sends the score result (weighted score, per-endpoint breakdown, the exact
  trial IDs used) to Gemini, no search grounding.
- Returns a 3-sentence, plain-text narrative citing specific numbers
  (percentages, trial IDs, phase, dosage, duration) — written as
  documentation for regulatory/pharma stakeholders.

**Artifacts produced:** `score_rationale`, attached alongside the score.

---

## Step 5 — Push to BigQuery

`save_clinical_efficacy_to_bq()` writes every enriched trial row, plus the
score and rationale, into the `clinical_efficacy` BigQuery table:

- **New `trial_id`** → INSERT full row.
- **Existing `trial_id`** with a changed phase, phase_status, or a
  previously-blank title/study → UPDATE.
- **Unchanged** → skip.

The molecule's overall `weighted_score` is also appended to a separate
cross-dimension scoring table, tagged under the "Medical Potential" pillar /
"Clinical Efficacy" dimension, for dashboard consumption alongside other
scoring pipelines.

---

## Step 6 — Report Generation

### 6.1 Thin-Data Enrichment (conditional)

- If the BigQuery data pulled back for the molecule is too sparse to
  summarize meaningfully, calls Gemini once **with** Google Search grounding
  to pull endpoint data directly from public trial results, FDA labels, and
  registries.

### 6.2 Business Narrative

- Calls Gemini again (no search grounding) to write a 2-page narrative aimed
  at a Medical Affairs business audience, covering per-endpoint performance
  and overall efficacy positioning.

### 6.3 PDF Rendering

- Renders the narrative plus per-endpoint summary tables into a PDF (via
  `reportlab`) and returns the raw bytes — the function itself never writes
  to disk.

### 6.4 Persistence

| Artifact | Location | Contents |
|---|---|---|
| PDF report | GCS | Full business-facing efficacy report |
| JSON payload | GCS | Score, rationale, and all enriched trial rows |
| Dimension score | BigQuery | `weighted_score` + rationale for dashboard consumption |

---

## Step Input / Output Reference

| Step | Input | Output |
|------|-------|--------|
| 1.1 Alias Resolution | Drug name; BigQuery alias table | Ordered list of search terms (primary + cleaned aliases) |
| 1.2 Registry & Literature APIs | Search terms | Raw trial rows from CTGOV, CTIS, EudraCT, CTRI, PubMed, WHO |
| 1.3 Gemini-Augmented Discovery | Search terms | Raw trial rows from int'l registries, NICE, Trade/Conf, Innovator web |
| 1.4 Curated Table Lookup | Drug name | Raw trial rows from the GD BigQuery table, `llm_confidence` pre-set to 1 |
| 1.5 Dedup & Trim | Raw rows from 1.2–1.4 | Unified trial list, one row per `trial_id`, highest reported phase kept |
| 2.1 LLM Confidence Evaluation | Unified trial list, batched 6/call | `llm_confidence` (0–1) + rationale per trial, written to BigQuery |
| 2.2 Endpoint Extraction | Trials with `llm_confidence > 0.6`, not yet in BigQuery, batched 6/call | Weight loss / HbA1c / ALT / MASH values, durations, rationale, confidence |
| 2.3 Metadata Backfill | Trials already in BigQuery, batched 10/call | Refreshed title, phase, dates, status |
| 3. Score Calculation | Enriched trial set | `weighted_score` (1–5), per-endpoint breakdown, `data_coverage` |
| 4. Rationale Generation | Score result | 3-sentence narrative rationale |
| 5. Push to BigQuery | Enriched rows, score, rationale | Incremental INSERT/UPDATE in `clinical_efficacy`; appended dimension score |
| 6.1 Thin-Data Enrichment | BigQuery data for the molecule (if sparse) | Supplemental endpoint data from public sources |
| 6.2 Business Narrative | Aggregated endpoint data, score, rationale | 2-page narrative text |
| 6.3 PDF Rendering | Narrative + endpoint summary tables | PDF bytes |
| 6.4 Persistence | PDF bytes + full payload | PDF (GCS), JSON payload (GCS), dimension score (BigQuery) |

---

## Gemini Usage

Gemini is used at nine distinct points in the pipeline. Note that Step 1.2
(the six direct-API/literature sources) makes **no** Gemini calls at all —
it only escalates to Gemini where no structured API exists or free-text
reasoning is required.

| Step | Task | Gemini capability used |
|------|------|------------------------|
| 1.1 Alias Resolution | Clean a raw alias string into a deduplicated list of genuine drug names | Text cleaning / classification |
| 1.3 International Registries | Search 6 additional registries for trials matching the drug | Grounded generation with Google Search |
| 1.3 NICE | Find UK Technology Appraisals concerning the drug | Grounded generation with Google Search |
| 1.3 Trade & Conferences | Find conference abstracts and trade-press mentions of the drug | Grounded generation with Google Search |
| 1.3 Innovator Web | Discover the innovator company and scan its site for trial data | Grounded generation with Google Search (discovery + fallback passes) |
| 2.1 LLM Confidence Evaluation | Verify each trial is genuinely associated with the drug; assign a confidence score | Grounded verification with Google Search |
| 2.2 Endpoint Extraction | Extract weight loss / HbA1c / ALT / MASH values from a trial record | Structured extraction, optionally grounded with Google Search |
| 2.3 Metadata Backfill | Refresh title, phase, dates, and status for an existing trial | Grounded retrieval with Google Search |
| 4. Rationale Generation | Write a 3-sentence, citation-backed narrative explaining the score | Structured text generation |
| 6.1 / 6.2 Report Generation | Fill gaps in thin BigQuery data; write the 2-page business narrative | Grounded retrieval (6.1) + long-form generation (6.2) |

---

## Pipeline Re-entry

The pipeline supports skipping stages that have already been completed, so a
molecule can be reprocessed without repeating expensive discovery or
enrichment work.

```text
discovery (Step 1)
    ↓                 --skip-fetch   → reuse existing <slug>_trials.json, skip Step 1
enrich_and_score (Steps 2-3)
    ↓                 --no-score     → skip enrichment, scoring, and rationale entirely
rationale (Step 4)
    ↓
push_to_bq (Step 5)
    ↓                 --step1-only   → stop here; push raw (unscored) trials only
report_generation (Step 6)
    ↓                 --no-report    → skip PDF generation
```

- `--skip-fetch` reuses a previously written `<slug>_trials.json` instead of
  re-running Step 1 discovery.
- `--step1-only` runs only discovery (Step 1) and the BigQuery push (Step 5)
  of the raw, unscored trial set — useful for topping up trial coverage
  without triggering enrichment or a report.
- `--no-score` skips Steps 2–4 (confidence-gated enrichment, scoring,
  rationale) while still pushing whatever trial data was fetched.
- `--no-report` skips Step 6 (PDF report + GCS upload) while still
  completing scoring and the BigQuery push.
- **`tester.py`** offers a separate, off-pipeline re-entry point: it
  re-runs only Step 3 (scoring) against trials already fetched — from a
  local JSON file or straight from BigQuery — using the current scoring
  logic, without touching Gemini or writing back to BigQuery. This is the
  fast path for validating a change to the scoring formula.
