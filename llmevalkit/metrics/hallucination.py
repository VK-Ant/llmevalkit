"""Hallucination detection metric — reference-free capability."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import HALLUCINATION_PROMPT


class Hallucination(BaseMetric):
    """Detects hallucinated, fabricated, or unverifiable content in the answer.
    
    **KEY DIFFERENTIATOR**: Works in both reference-based and REFERENCE-FREE mode.
    When no context is provided, uses the LLM's own knowledge to assess factual accuracy
    (inspired by SelfCheckGPT approach).
    
    Score interpretation (inverted — higher is better):
        1.0: No hallucinations detected
        0.75: Minor inaccuracies
        0.5: Some hallucinated content
        0.25: Significant hallucinations
        0.0: Mostly hallucinated
    """
    
    name = "hallucination"
    prompt_template = HALLUCINATION_PROMPT
    invert_score = True  # Lower hallucination = higher score

    @property
    def required_fields(self) -> list[str]:
        return ["question", "answer"]  # Context is optional — reference-free!
