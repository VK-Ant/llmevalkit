"""Completeness metric — evaluates thoroughness of the answer."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import COMPLETENESS_PROMPT


class Completeness(BaseMetric):
    """Evaluates whether the answer thoroughly covers all aspects of the question.
    
    Supports optional reference answer for more precise evaluation.
    
    Score interpretation:
        1.0: Comprehensive coverage
        0.75: Mostly complete
        0.5: Main points covered, gaps remain
        0.25: Significant gaps
        0.0: Barely addresses the question
    """
    
    name = "completeness"
    prompt_template = COMPLETENESS_PROMPT

    @property
    def required_fields(self) -> list[str]:
        return ["question", "answer"]
