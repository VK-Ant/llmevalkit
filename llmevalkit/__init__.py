"""
LLMEVAL - Comprehensive LLM Evaluation Library
================================================

A production-grade, reference-free evaluation library for RAG pipelines,
chatbots, and generative AI systems.

Developed by Venkatkumar Rajan (@VK_Venkatkumar)

Quick Start:
    >>> from llmeval import Evaluator
    >>> evaluator = Evaluator(provider="openai", model="gpt-4o-mini")
    >>> result = evaluator.evaluate(
    ...     question="What is photosynthesis?",
    ...     answer="Photosynthesis is the process by which plants convert sunlight into energy.",
    ...     context="Photosynthesis is a biological process where plants use sunlight, water, and CO2 to produce glucose and oxygen."
    ... )
    >>> print(result)
"""

__version__ = "1.0.3"
__author__ = "Venkatkumar Rajan"

from llmevalkit.evaluator import Evaluator
from llmevalkit.models import EvalResult, EvalConfig, MetricResult
from llmevalkit.metrics import (
    # LLM-as-Judge metrics
    Faithfulness,
    AnswerRelevance,
    ContextRelevance,
    Hallucination,
    Toxicity,
    Coherence,
    Completeness,
    GEval,
    # Pure Math metrics (NO API needed)
    BLEUScore,
    ROUGEScore,
    TokenOverlap,
    SemanticSimilarity,
    AnswerLength,
    ReadabilityScore,
    KeywordCoverage,
)

__all__ = [
    "Evaluator",
    "EvalResult",
    "EvalConfig",
    "MetricResult",
    # LLM-as-Judge Metrics
    "Faithfulness",
    "AnswerRelevance",
    "ContextRelevance",
    "Hallucination",
    "Toxicity",
    "Coherence",
    "Completeness",
    "GEval",
    # Pure Math Metrics
    "BLEUScore",
    "ROUGEScore",
    "TokenOverlap",
    "SemanticSimilarity",
    "AnswerLength",
    "ReadabilityScore",
    "KeywordCoverage",
]
