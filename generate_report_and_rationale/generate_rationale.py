"""Prompt-based rationale generator for clinical efficacy scoring."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from google.genai import types

from ..utils import RATIONALE_MODEL, get_gemini_client

logger = logging.getLogger(__name__)

NOT_GENERATED_MESSAGE = "Rationale has not been generated."


def _prepare_prompt_payload(molecule: str, score_result: Dict[str, Any]) -> dict[str, Any]:
    endpoints = score_result.get("endpoints", {}) if isinstance(score_result, dict) else {}

    return {
        "molecule": molecule,
        "weighted_score": score_result.get("weighted_score") if isinstance(score_result, dict) else None,
        "data_coverage": score_result.get("data_coverage") if isinstance(score_result, dict) else None,
        "endpoint_performance": {
            "Weight Loss (40% weight)": {
                "score": endpoints.get("weight_loss", {}).get("score"),
                "best_value": endpoints.get("weight_loss", {}).get("best_value"),
                "raw_value": endpoints.get("weight_loss", {}).get("raw_value"),
                "phase_used": endpoints.get("weight_loss", {}).get("phase_used"),
                "trial_used": endpoints.get("weight_loss", {}).get("trial_details", {}).get("trial_id"),
                "dosage": endpoints.get("weight_loss", {}).get("trial_details", {}).get("dosage"),
                "duration": endpoints.get("weight_loss", {}).get("trial_details", {}).get("weight_duration"),
            },
            "HbA1c Reduction (40% weight)": {
                "score": endpoints.get("hba1c", {}).get("score"),
                "best_value": endpoints.get("hba1c", {}).get("best_value"),
                "raw_value": endpoints.get("hba1c", {}).get("raw_value"),
                "phase_used": endpoints.get("hba1c", {}).get("phase_used"),
                "trial_used": endpoints.get("hba1c", {}).get("trial_details", {}).get("trial_id"),
                "dosage": endpoints.get("hba1c", {}).get("trial_details", {}).get("dosage"),
                "duration": endpoints.get("hba1c", {}).get("trial_details", {}).get("hba1c_duration"),
            },
            "MASH Resolution (10% weight)": {
                "score": endpoints.get("mash", {}).get("score"),
                "best_value": endpoints.get("mash", {}).get("best_value"),
                "raw_value": endpoints.get("mash", {}).get("raw_value"),
                "phase_used": endpoints.get("mash", {}).get("phase_used"),
                "trial_used": endpoints.get("mash", {}).get("trial_details", {}).get("trial_id"),
                "dosage": endpoints.get("mash", {}).get("trial_details", {}).get("dosage"),
                "duration": endpoints.get("mash", {}).get("trial_details", {}).get("mash_duration"),
            },
            "ALT Reduction (10% weight)": {
                "score": endpoints.get("alt", {}).get("score"),
                "best_value": endpoints.get("alt", {}).get("best_value"),
                "raw_value": endpoints.get("alt", {}).get("raw_value"),
                "phase_used": endpoints.get("alt", {}).get("phase_used"),
                "trial_used": endpoints.get("alt", {}).get("trial_details", {}).get("trial_id"),
                "dosage": endpoints.get("alt", {}).get("trial_details", {}).get("dosage"),
                "duration": endpoints.get("alt", {}).get("trial_details", {}).get("alt_duration"),
            },
        },
        "total_trials": score_result.get("total_trials") if isinstance(score_result, dict) else None,
    }


async def generate_score_rationale(molecule: str, score_result: Dict[str, Any]) -> str:
    """Generate a concise narrative rationale for the clinical efficacy score."""
    logger.info("[SCORE] Generating narrative rationale via Gemini...")
    payload = _prepare_prompt_payload(molecule, score_result)

    prompt = f"""You are a clinical pharmacology expert. Generate a concise, evidence-based rationale explaining the clinical efficacy score for {molecule}.

SCORING RESULTS:
- Clinical Efficacy Score: {payload['weighted_score']} / 5
- Coverage: {payload['data_coverage']}

ENDPOINT PERFORMANCE (EXACT trials used for scoring):
{json.dumps(payload['endpoint_performance'], indent=2)}

SCORING METHODOLOGY:
- Score ranges: 5 = >=22%, 4 = 16-21.9%, 3 = 10-15.9%, 2 = 5-9.9%, 1 = <5%
- Phase penalties: Phase 3 = no penalty, Phase 2 = x0.85, Phase 1 = x0.65
- Weighted average: Weight Loss (40%) + HbA1c (40%) + MASH (10%) + ALT (10%)

YOUR TASK:
Write a concise clinical rationale in EXACTLY 3 sentences:
1. State the overall clinical efficacy score and briefly summarise {molecule}'s performance across the scored endpoints.
2. Highlight the strongest endpoint(s) with specific percentage, dosage, duration, phase, and trial ID.
3. Note any missing endpoints or data gaps and state what the score reflects about the molecule's overall clinical profile.

WRITING GUIDELINES:
- EXACTLY 3 sentences - no more, no less
- Include specific numbers (percentages, trial IDs, phase info, dosage, duration) where available
- Plain text only - no markdown, no headers, no bullets
- Write as documentation for regulatory or pharma stakeholders

IMPORTANT: Use trial_id, dosage, and duration ONLY from the ENDPOINT PERFORMANCE section above.

Generate the rationale now:"""

    try:
        client = get_gemini_client()
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(temperature=0.3, response_mime_type="text/plain")
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=RATIONALE_MODEL,
            contents=contents,
            config=config,
        )
        rationale = (response.text or "").strip().replace("\n\n\n", "\n\n")
        if rationale:
            logger.info("[SCORE] Rationale generated.")
            return rationale
        logger.warning("[SCORE] Prompt output was empty; rationale was not generated.")
    except Exception:
        logger.exception("[SCORE] Rationale generation failed; falling back to summary.")

    return NOT_GENERATED_MESSAGE
