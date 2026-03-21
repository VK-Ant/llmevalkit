"""Coherence metric — evaluates logical flow, clarity, and readability."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import COHERENCE_PROMPT


class Coherence(BaseMetric):
    """Evaluates the logical structure, clarity, and readability of the answer.
    
    Score interpretation:
        1.0: Exceptionally clear and well-organized
        0.75: Well-written with minor issues
        0.5: Adequate but could improve
        0.25: Confusing or poorly structured
        0.0: Incoherent
    """
    
    name = "coherence"
    prompt_template = COHERENCE_PROMPT

    @property
    def required_fields(self) -> list[str]:
        return ["question", "answer"]
