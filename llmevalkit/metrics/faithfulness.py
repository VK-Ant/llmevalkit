"""Faithfulness metric — measures if the answer is grounded in the provided context."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import FAITHFULNESS_PROMPT


class Faithfulness(BaseMetric):
    """Evaluates whether the answer only contains information supported by the context.
    
    This is the most critical metric for RAG systems. A faithful answer should not
    introduce any information beyond what's in the retrieved context.
    
    Score interpretation:
        1.0: All claims fully supported by context
        0.75: Almost all claims supported
        0.5: Mix of supported and unsupported claims
        0.25: Many unsupported claims
        0.0: Answer contradicts or ignores context
    """
    
    name = "faithfulness"
    prompt_template = FAITHFULNESS_PROMPT

    @property
    def required_fields(self) -> list[str]:
        return ["question", "answer", "context"]
