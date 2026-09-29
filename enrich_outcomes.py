"""Optional enrich-outcomes hook.

This package does not bundle an enrich_outcomes implementation in the
workspace, so importing it raises ImportError and the caller's optional import
fallback can disable that path cleanly.
"""

from __future__ import annotations

raise ImportError("medical_potential.clinical_efficacy.enrich_outcomes is not bundled")
