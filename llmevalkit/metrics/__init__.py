"""LLMEVAL Metrics — LLM-as-Judge AND Pure Math, all in one library.

Two modes:
    - LLM metrics: Use an LLM to judge quality (needs API)
    - Math metrics: Pure statistical calculations (NO API needed, zero cost)
"""

# LLM-as-Judge metrics (require API)
from llmevalkit.metrics.base import BaseMetric
from llmevalkit.metrics.faithfulness import Faithfulness
from llmevalkit.metrics.answer_relevance import AnswerRelevance
from llmevalkit.metrics.context_relevance import ContextRelevance
from llmevalkit.metrics.hallucination import Hallucination
from llmevalkit.metrics.toxicity import Toxicity
from llmevalkit.metrics.coherence import Coherence
from llmevalkit.metrics.completeness import Completeness
from llmevalkit.metrics.geval import GEval

# Pure Math metrics (NO API needed)
from llmevalkit.metrics.math_metrics import (
    MathMetric,
    BLEUScore,
    ROUGEScore,
    TokenOverlap,
    SemanticSimilarity,
    AnswerLength,
    ReadabilityScore,
    KeywordCoverage,
)

__all__ = [
    # Base
    "BaseMetric",
    "MathMetric",
    # LLM-as-Judge
    "Faithfulness",
    "AnswerRelevance",
    "ContextRelevance",
    "Hallucination",
    "Toxicity",
    "Coherence",
    "Completeness",
    "GEval",
    # Pure Math
    "BLEUScore",
    "ROUGEScore",
    "TokenOverlap",
    "SemanticSimilarity",
    "AnswerLength",
    "ReadabilityScore",
    "KeywordCoverage",
]
