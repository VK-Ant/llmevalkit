"""llmevalkit document evaluation module.

Metrics for evaluating document extraction and parsing accuracy.
All metrics work offline (fuzzy matching, format validation) or
with API (LLM-as-judge for semantic matching).

Usage:
    from llmevalkit.doceval import FieldAccuracy, FieldCompleteness
    from llmevalkit.doceval import FieldHallucination, FormatValidation
    from llmevalkit.doceval import ExtractionConsistency
"""

from llmevalkit.doceval.field_accuracy import FieldAccuracy
from llmevalkit.doceval.field_completeness import FieldCompleteness
from llmevalkit.doceval.field_hallucination import FieldHallucination
from llmevalkit.doceval.format_validation import FormatValidation
from llmevalkit.doceval.extraction_consistency import ExtractionConsistency

__all__ = [
    "FieldAccuracy",
    "FieldCompleteness",
    "FieldHallucination",
    "FormatValidation",
    "ExtractionConsistency",
]
