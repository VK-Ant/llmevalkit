"""Context Relevance metric — measures if the retrieved context is useful for the question."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import CONTEXT_RELEVANCE_PROMPT


class ContextRelevance(BaseMetric):
    """Evaluates whether the retrieved context contains relevant information for the question.
    
    Critical for diagnosing retrieval quality in RAG systems.
    
    Score interpretation:
        1.0: Context fully covers the question
        0.75: Context mostly relevant
        0.5: Partially relevant
        0.25: Minimal relevance
        0.0: Completely irrelevant context
    """
    
    name = "context_relevance"
    prompt_template = CONTEXT_RELEVANCE_PROMPT

    @property
    def required_fields(self) -> list[str]:
        return ["question", "context"]
