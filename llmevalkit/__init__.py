"""
llmevalkit - LLM Evaluation, Compliance, Document Parsing, and Security Library
================================================================================

46 metrics for LLM evaluation, compliance, document parsing, governance,
security, and multimodal testing. Everything works with or without API.

Developed by Venkatkumar Rajan (@VK_Venkatkumar)

Quick Start:
    >>> from llmevalkit import Evaluator
    >>> evaluator = Evaluator(provider="none", preset="math")
    >>> result = evaluator.evaluate(
    ...     question="What is Python?",
    ...     answer="Python is a programming language.",
    ...     context="Python is a high-level, interpreted programming language."
    ... )
    >>> print(result.summary())

Compliance:
    >>> from llmevalkit.compliance import PIIDetector, HIPAACheck
    >>> evaluator = Evaluator(provider="none", metrics=[PIIDetector(), HIPAACheck()])
    >>> result = evaluator.evaluate(answer="Patient John Smith, SSN 123-45-6789")
"""

__version__ = "4.0.0"
__author__ = "Venkatkumar Rajan"

from llmevalkit.evaluator import Evaluator
from llmevalkit.models import EvalResult, EvalConfig, MetricResult
from llmevalkit.metrics import (
    # API metrics (LLM-as-judge)
    Faithfulness,
    AnswerRelevance,
    ContextRelevance,
    Hallucination,
    Toxicity,
    Coherence,
    Completeness,
    GEval,
    # Local metrics (no API needed)
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
    # API metrics
    "Faithfulness",
    "AnswerRelevance",
    "ContextRelevance",
    "Hallucination",
    "Toxicity",
    "Coherence",
    "Completeness",
    "GEval",
    # Local metrics
    "BLEUScore",
    "ROUGEScore",
    "TokenOverlap",
    "SemanticSimilarity",
    "AnswerLength",
    "ReadabilityScore",
    "KeywordCoverage",
]
