"""Rule-based structured attribute extraction (Phase 2)."""

from belgrade_recommender.extract.attributes import (
    extract_event_attributes,
    extract_event_attributes_maybe_lazy_gemini,
)

__all__ = ["extract_event_attributes", "extract_event_attributes_maybe_lazy_gemini"]
