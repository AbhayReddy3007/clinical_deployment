#!/usr/bin/env python3
"""
gcp_utils.py – Shared Google Cloud helpers for BigQuery and GCS operations.

All config is sourced from medical_potential.config.
Every other pipeline script (fetcher.py, push_to_bq.py,
generate_efficacy_report.py, clinical_efficacy.py) imports clients and
constants from here — never instantiate BQ/GCS clients elsewhere.

Quick check:
    python gcp_utils.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# Load .env file before any os.getenv calls so GEMINI_API_KEY etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from medical_potential.config import (
    BQ_DATASET_ID,
    GCS_BUCKET,
    PROJECT_ID,
)

logger = logging.getLogger(__name__)

# Re-export so other files can do: from gcp_utils import PROJECT_ID, ...
__all__ = [
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MODEL",
    "RATIONALE_MODEL",
    "EVAL_MODEL",
    "LLM_EVAL_MODEL",
    "get_gemini_client",
    "get_active_api_key",
    "full_bq_table",
    "gcs_path",
    "validate_config",
    # Pipeline config variables
    "alias_batch",
    "trial_registries_per_call",
    "international_alias_batch",
    "Innovator_web_sources_calls",
    "metadata_batch",
    "endpoint_extraction_batch",
    "eval_trial_batch",
    "trade_calls",
    "conference_calls",
    "partial_enrichment",
]

# These are not in medical_potential.config — set them here or via env
GEMINI_API_KEY:  str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
GOOGLE_API_KEY:  str = os.getenv("GEMINI_API_KEY","")
MODEL:           str = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-3-flash-preview")
RATIONALE_MODEL: str = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-3-flash-preview")
EVAL_MODEL:      str = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-3-flash-preview")
LLM_EVAL_MODEL:  str = os.getenv("GEMINI_PRO_MODEL",  "gemini-3.1-pro-preview")
SEARCH_MODEL:  str = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-3-flash-preview")

# ==============================================================================
# PIPELINE CONFIGURATION VARIABLES
# ==============================================================================

# Number of aliases that can be clubbed together while searching
alias_batch: int = int(os.getenv("ALIAS_BATCH", "10"))

# Number of international trial registries searched per single Gemini call
trial_registries_per_call: int = int(os.getenv("TRIAL_REGISTRIES_PER_CALL", "3"))

# Number of aliases batched together per international registry search call
international_alias_batch: int = int(os.getenv("INTERNATIONAL_ALIAS_BATCH", "10"))

# Max number of Gemini calls for innovator website scanning
# Call 1: discover company, Call 2: discover its websites/URLs,
# Call 3: search those websites for trial IDs
Innovator_web_sources_calls: int = int(os.getenv("INNOVATOR_WEB_SOURCES_CALLS", "3"))

# Number of trials per batch for metadata + location backfill
metadata_batch: int = int(os.getenv("METADATA_BATCH", "10"))

# Number of trials per batch for endpoint extraction (enrichment)
endpoint_extraction_batch: int = int(os.getenv("ENDPOINT_EXTRACTION_BATCH", "6"))

# Number of trials per batch for LLM confidence evaluation
eval_trial_batch: int = int(os.getenv("EVAL_TRIAL_BATCH", "6"))

# Max number of Gemini calls for all trade publication sources
trade_calls: int = int(os.getenv("TRADE_CALLS", "4"))

# Max number of Gemini calls for all conference sources
conference_calls: int = int(os.getenv("CONFERENCE_CALLS", "4"))

# When True, only NEW trials (not in BQ) get full metadata+locations+endpoints
# enrichment. Existing trials get metadata-only backfill (batch size = METADATA_BATCH).
# When False, all high-confidence trials get full enrichment.
partial_enrichment: bool = os.getenv("CLINICAL_TRIALS_PARTIAL_ENRICHMENT", "false").lower() in ("true", "1", "yes")

# ==============================================================================
# CLIENT HELPERS
# ==============================================================================

def get_gemini_client():
    """Return a google-genai Client initialised with the configured API key.

    A **new** client is created on every call.  The google-genai Client wraps
    an httpx session that is tied to the event-loop it was created on.  Caching
    a single instance causes "Cannot send a request, as the client has been
    closed" errors when multiple ``asyncio.run()`` calls share the singleton
    across different event loops (each ``asyncio.run()`` closes the loop — and
    the httpx session — on exit).

    Import is deferred so the rest of gcp_utils works even when google-genai
    is not installed (e.g. in BQ-only contexts).
    """
    try:
        from google import genai  # type: ignore
    except ImportError:
        raise ImportError(
            "google-genai is not installed.  Run: pip install google-genai"
        )
    api_key = get_active_api_key()
    if not api_key:
        raise ValueError(
            "No Gemini API key found.  Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return genai.Client(api_key=api_key)


# ==============================================================================
# CONVENIENCE HELPERS
# ==============================================================================

def full_bq_table(table_name: str) -> str:
    """Return a fully-qualified BQ table ID: project.dataset.table."""
    return f"{PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"


def gcs_path(*parts: str) -> str:
    """Join GCS path segments cleanly."""
    return "/".join(p.strip("/") for p in parts if p)


def get_active_api_key() -> str:
    """Return whichever Gemini/Google API key is set."""
    return GEMINI_API_KEY or GOOGLE_API_KEY


def validate_config(raise_on_error: bool = True) -> list[str]:
    """Check that all required variables are set."""
    required = {
        "PROJECT_ID":                      PROJECT_ID,
        "GEMINI_API_KEY / GOOGLE_API_KEY": get_active_api_key(),
        "GCS_BUCKET":                      GCS_BUCKET,
    }
    missing = [name for name, val in required.items() if not val]
    if missing and raise_on_error:
        raise ValueError(
            "Missing required config (set in medical_potential.config or env):\n"
            + "\n".join(f"  - {m}" for m in missing)
        )
    return missing