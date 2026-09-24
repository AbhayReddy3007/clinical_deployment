"""Public API for the clinical_efficacy package.

Expose the main pipeline entry point for programmatic use.
"""

from .clinical_efficacy import clinical_efficacy_assessment

__all__ = ["clinical_efficacy_assessment"]
