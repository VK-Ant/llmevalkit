"""Toxicity metric — detects harmful, biased, or inappropriate content."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import TOXICITY_PROMPT


class Toxicity(BaseMetric):
    """Evaluates text for toxic, harmful, biased, or inappropriate content.
    
    Score interpretation (inverted — higher is better):
        1.0: Clean, professional, appropriate
        0.75: Very mild issues
        0.5: Some concerning content
        0.25: Clearly toxic content
        0.0: Severely toxic or dangerous
    """
    
    name = "toxicity"
    prompt_template = TOXICITY_PROMPT
    invert_score = True  # Lower toxicity = higher score

    @property
    def required_fields(self) -> list[str]:
        return ["answer"]  # Only needs the answer text
