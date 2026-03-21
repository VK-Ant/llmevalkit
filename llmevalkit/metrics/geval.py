"""G-Eval metric — evaluate using custom, user-defined criteria.

This is a key differentiator of LLMEVAL: users can define ANY evaluation criteria
and get structured, consistent scoring. Inspired by the G-Eval paper 
(Liu et al., 2023) but with production-grade implementation.
"""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import GEVAL_PROMPT


class GEval(BaseMetric):
    """Custom criteria evaluation using the G-Eval framework.
    
    Allows users to define arbitrary evaluation criteria and get structured scores.
    This makes LLMEVAL extensible to ANY domain or use case.
    
    Example:
        >>> geval = GEval(criteria="Evaluate if the customer support response is empathetic and actionable.")
        >>> # Or for domain-specific:
        >>> geval = GEval(criteria="Score the legal accuracy and citation quality of the response.")
    
    Score interpretation:
        1.0: Excellent per custom criteria
        0.75: Good
        0.5: Average
        0.25: Below average
        0.0: Poor
    """
    
    name = "geval"
    prompt_template = GEVAL_PROMPT

    def __init__(self, criteria: str = "Evaluate the overall quality of the response.", weight: float = 1.0):
        super().__init__(weight=weight)
        self.criteria = criteria

    def evaluate(self, client, **kwargs):
        kwargs["criteria"] = self.criteria
        return super().evaluate(client, **kwargs)

    @property
    def required_fields(self) -> list[str]:
        return ["question", "answer"]
