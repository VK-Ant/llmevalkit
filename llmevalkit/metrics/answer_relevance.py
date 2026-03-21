"""Answer Relevance metric — measures if the answer addresses the question."""

from llmevalkit.metrics.base import BaseMetric
from llmevalkit.prompts import ANSWER_RELEVANCE_PROMPT


class AnswerRelevance(BaseMetric):
    """Evaluates whether the answer directly and completely addresses the question.
    
    Score interpretation:
        1.0: Perfect, direct, complete answer
        0.75: Mostly addresses the question
        0.5: Partially addresses with some off-topic content
        0.25: Tangentially related
        0.0: Completely irrelevant
    """
    
    name = "answer_relevance"
    prompt_template = ANSWER_RELEVANCE_PROMPT

    @property
    def required_fields(self) -> list[str]:
        return ["question", "answer"]
