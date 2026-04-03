"""LLMEVAL Evaluator — the main entry point for all evaluations."""

from __future__ import annotations

import time
import concurrent.futures
from typing import Any, Optional, Union

from rich.console import Console
from rich.table import Table

from llmevalkit.llm_client import LLMClient
from llmevalkit.models import (
    BatchResult,
    EvalConfig,
    EvalResult,
    MetricResult,
    Provider,
    TestCase,
)
from llmevalkit.metrics import (
    AnswerRelevance,
    Coherence,
    Completeness,
    ContextRelevance,
    Faithfulness,
    GEval,
    Hallucination,
    Toxicity,
    # Math metrics
    BLEUScore,
    ROUGEScore,
    TokenOverlap,
    SemanticSimilarity,
    AnswerLength,
    ReadabilityScore,
    KeywordCoverage,
)
from llmevalkit.metrics.base import BaseMetric
from llmevalkit.metrics.math_metrics import MathMetric
from llmevalkit.compliance import (
    PIIDetector,
    HIPAACheck,
    GDPRCheck,
    DPDPCheck,
    EUAIActCheck,
)
from llmevalkit.doceval import (
    FieldAccuracy,
    FieldCompleteness,
    FieldHallucination,
    FormatValidation,
    ExtractionConsistency,
)
from llmevalkit.governance import NISTCheck, CoSAICheck, ISO42001Check, SOC2Check
from llmevalkit.security import PromptInjectionCheck, BiasDetector
from llmevalkit.multimodal import (
    OCRAccuracy,
    AudioTranscriptionAccuracy,
    ImageTextAlignment,
    VisionQAAccuracy,
)


# Preset metric collections
METRIC_PRESETS = {
    # --- Quality evaluation presets (v1) ---
    "rag": [Faithfulness, AnswerRelevance, ContextRelevance, Hallucination],
    "chatbot": [AnswerRelevance, Coherence, Toxicity, Hallucination],
    "summarization": [Faithfulness, Completeness, Coherence],
    "safety": [Toxicity, Hallucination],
    "all": [Faithfulness, AnswerRelevance, ContextRelevance, Hallucination, Toxicity, Coherence, Completeness],
    "minimal": [Faithfulness, AnswerRelevance],
    "local": [BLEUScore, ROUGEScore, TokenOverlap, KeywordCoverage, AnswerLength, ReadabilityScore],
    "math": [BLEUScore, ROUGEScore, TokenOverlap, KeywordCoverage, AnswerLength, ReadabilityScore],
    "math_similarity": [BLEUScore, ROUGEScore, TokenOverlap, SemanticSimilarity],
    "math_minimal": [TokenOverlap, AnswerLength],
    "hybrid_rag": [TokenOverlap, BLEUScore, KeywordCoverage, Faithfulness, Hallucination],
    "hybrid_chatbot": [ReadabilityScore, AnswerLength, Coherence, Toxicity],
    # --- Compliance presets (v2) ---
    "pii": [PIIDetector],
    "hipaa": [PIIDetector, HIPAACheck],
    "gdpr": [PIIDetector, GDPRCheck],
    "india": [PIIDetector, DPDPCheck],
    "dpdp": [PIIDetector, DPDPCheck],
    "eu_ai": [PIIDetector, GDPRCheck, EUAIActCheck],
    "compliance_all": [PIIDetector, HIPAACheck, GDPRCheck, DPDPCheck, EUAIActCheck],
    "rag_hipaa": [Faithfulness, Hallucination, AnswerRelevance, PIIDetector, HIPAACheck],
    "rag_gdpr": [Faithfulness, Hallucination, AnswerRelevance, PIIDetector, GDPRCheck],
    "rag_india": [Faithfulness, Hallucination, AnswerRelevance, PIIDetector, DPDPCheck],
    # --- Document evaluation presets (v3) ---
    "doceval": [FieldAccuracy, FieldCompleteness, FieldHallucination, FormatValidation],
    "doceval_full": [FieldAccuracy, FieldCompleteness, FieldHallucination, FormatValidation, ExtractionConsistency],
    "doceval_hipaa": [FieldAccuracy, FieldCompleteness, FieldHallucination, PIIDetector, HIPAACheck],
    # --- Governance presets (v3) ---
    "governance": [NISTCheck, CoSAICheck, ISO42001Check, SOC2Check],
    "nist": [NISTCheck],
    # --- Security presets (v3) ---
    "security": [PromptInjectionCheck, BiasDetector],
    "security_full": [PromptInjectionCheck, BiasDetector, PIIDetector, Toxicity],
    # --- Multimodal presets (v3) ---
    "ocr": [OCRAccuracy],
    "multimodal": [OCRAccuracy, AudioTranscriptionAccuracy, ImageTextAlignment, VisionQAAccuracy],
    # --- Full audit preset ---
    "full_audit": [
        BLEUScore, ROUGEScore, TokenOverlap, KeywordCoverage,
        PIIDetector, HIPAACheck, GDPRCheck, DPDPCheck, EUAIActCheck,
        PromptInjectionCheck, BiasDetector,
    ],
    "enterprise": [
        Faithfulness, Hallucination, AnswerRelevance,
        PIIDetector, HIPAACheck, GDPRCheck,
        PromptInjectionCheck, BiasDetector,
        NISTCheck,
    ],
}


class Evaluator:
    """Main evaluation engine for LLMEVAL.
    
    Supports TWO evaluation modes:
        - LLM-as-Judge: Uses an LLM API to evaluate (provider="openai", "azure", etc.)
        - Pure Math: Statistical metrics only, NO API needed (provider="none")
        - Hybrid: Mix both in one evaluation
    
    Quick Start (LLM mode):
        >>> evaluator = Evaluator(provider="openai", model="gpt-4o-mini")
        >>> result = evaluator.evaluate(question="...", answer="...", context="...")
    
    Quick Start (Math mode — NO API, zero cost):
        >>> evaluator = Evaluator(provider="none", preset="math")
        >>> result = evaluator.evaluate(question="...", answer="...", context="...")
    
    Hybrid mode:
        >>> evaluator = Evaluator(provider="openai", preset="hybrid_rag")
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        metrics: Optional[list[Union[BaseMetric, MathMetric, type]]] = None,
        preset: Optional[str] = None,
        threshold: float = 0.5,
        verbose: bool = False,
        temperature: float = 0.0,
    ):
        """Initialize the Evaluator.
        
        Args:
            provider: LLM provider ("openai", "azure", "anthropic", "groq", "ollama", "custom", "none")
                      Use "none" for pure math metrics — zero API cost!
            model: Model name/deployment (ignored when provider="none")
            api_key: API key (or use environment variable)
            base_url: Custom API endpoint
            api_version: API version (for Azure)
            metrics: List of metric instances or classes. If None, uses preset.
            preset: Metric preset ("rag", "math", "hybrid_rag", "chatbot", "math_similarity", etc.)
            threshold: Pass/fail threshold (default 0.5)
            verbose: Enable verbose output
            temperature: LLM temperature for evaluation calls
        """
        self.console = Console()
        self.is_math_only = provider == "none"
        
        if self.is_math_only:
            self.config = EvalConfig(
                provider="openai",  # Placeholder, won't be used
                model="none",
                threshold=threshold,
                verbose=verbose,
            )
            self.client = None  # No LLM client needed!
        else:
            self.config = EvalConfig(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
                api_version=api_version,
                threshold=threshold,
                verbose=verbose,
                temperature=temperature,
            )
            self.client = LLMClient(self.config)
        
        # Set up metrics
        if metrics:
            self.metrics = [m() if isinstance(m, type) else m for m in metrics]
        elif preset:
            if preset not in METRIC_PRESETS:
                raise ValueError(f"Unknown preset '{preset}'. Available: {list(METRIC_PRESETS.keys())}")
            self.metrics = [m() for m in METRIC_PRESETS[preset]]
        else:
            # Default: RAG preset
            self.metrics = [m() for m in METRIC_PRESETS["rag"]]

    def evaluate(
        self,
        question: str = "",
        answer: str = "",
        context: str = "",
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Evaluate a single QA pair.
        
        Args:
            question: The input question/prompt
            answer: The LLM-generated answer to evaluate
            context: Retrieved context (for RAG evaluation)
            reference: Ground truth / reference answer (optional)
            **kwargs: Additional fields passed to metrics
        
        Returns:
            EvalResult with all metric scores and overall score
        """
        eval_kwargs = {
            "question": question,
            "answer": answer,
            "context": context,
            "reference": reference or "",
            **kwargs,
        }
        
        if self.config.verbose:
            self.console.print(f"\n[bold blue]LLMEVAL[/] Evaluating with {len(self.metrics)} metrics...")
        
        metric_results: dict[str, MetricResult] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # Separate metrics into local (instant) and API (need LLM call).
        local_metrics = []
        api_metrics = []

        for metric in self.metrics:
            if not metric.validate_inputs(**eval_kwargs):
                if self.config.verbose:
                    self.console.print("  [yellow]Skipping {} (missing required fields)[/]".format(metric.name))
                continue

            # Check if metric can run without API.
            is_local = hasattr(metric, '_compute')  # MathMetric
            is_compliance_local = hasattr(metric, 'use_llm') and not metric.use_llm

            if is_local or is_compliance_local:
                local_metrics.append(metric)
            elif self.client is None:
                # API metric but no provider set.
                result = MetricResult(
                    name=metric.name,
                    score=0.0,
                    reason="Skipped: needs an API provider. Use provider='openai' or 'groq' etc.",
                    details={"error": "no_provider"},
                )
                metric_results[metric.name] = result
                if self.config.verbose:
                    self.console.print(
                        "  [red]* {:<20} SKIPPED (needs API provider)[/]".format(metric.name)
                    )
            else:
                api_metrics.append(metric)

        # Run local metrics (instant, sequential is fine).
        for metric in local_metrics:
            start = time.time()
            result = metric.evaluate(self.client, **eval_kwargs)
            elapsed = time.time() - start
            metric_results[metric.name] = result
            weighted_sum += result.score * metric.weight
            total_weight += metric.weight
            if self.config.verbose:
                color = "green" if result.score >= 0.7 else "yellow" if result.score >= 0.4 else "red"
                self.console.print(
                    "  [{}]* {:<20} {:.3f}[/]  ({:.1f}s)".format(
                        color, metric.name, result.score, elapsed
                    )
                )

        # Run API metrics in parallel.
        if api_metrics:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                future_map = {}
                start_times = {}
                for metric in api_metrics:
                    start_times[metric.name] = time.time()
                    future = pool.submit(metric.evaluate, self.client, **eval_kwargs)
                    future_map[future] = metric

                for future in concurrent.futures.as_completed(future_map):
                    metric = future_map[future]
                    elapsed = time.time() - start_times[metric.name]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = MetricResult(
                            name=metric.name,
                            score=0.0,
                            reason="Failed: {}".format(str(e)),
                            details={"error": str(e)},
                        )
                    metric_results[metric.name] = result
                    weighted_sum += result.score * metric.weight
                    total_weight += metric.weight
                    if self.config.verbose:
                        color = "green" if result.score >= 0.7 else "yellow" if result.score >= 0.4 else "red"
                        self.console.print(
                            "  [{}]* {:<20} {:.3f}[/]  ({:.1f}s)".format(
                                color, metric.name, result.score, elapsed
                            )
                        )
        
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        eval_result = EvalResult(
            question=question,
            answer=answer,
            context=context,
            reference=reference,
            metrics=metric_results,
            overall_score=round(overall, 4),
            metadata={"threshold": self.config.threshold, "model": self.config.model},
        )
        
        if self.config.verbose:
            self.console.print(f"\n  [bold]Overall: {overall:.3f}[/]  {'[green]PASSED' if eval_result.passed else '[red]FAILED'}[/]")
        
        return eval_result

    def evaluate_batch(
        self,
        test_cases: list[Union[TestCase, dict]],
        show_progress: bool = True,
    ) -> BatchResult:
        """Evaluate multiple test cases.
        
        Args:
            test_cases: List of TestCase objects or dicts with question/answer/context
            show_progress: Show progress bar
        
        Returns:
            BatchResult with all results and aggregate statistics
        """
        results = []
        total = len(test_cases)
        
        for i, case in enumerate(test_cases):
            if isinstance(case, dict):
                case = TestCase(**case)
            
            if show_progress:
                self.console.print(f"[blue]Evaluating[/] [{i+1}/{total}] {case.question[:50]}...")
            
            result = self.evaluate(
                question=case.question,
                answer=case.answer,
                context=case.context,
                reference=case.reference,
            )
            results.append(result)
        
        batch = BatchResult(results=results)
        
        if show_progress:
            self.console.print(batch.summary())
        
        return batch

    def add_metric(self, metric: Union[BaseMetric, type]) -> "Evaluator":
        """Add a metric to the evaluator. Returns self for chaining."""
        if isinstance(metric, type):
            metric = metric()
        self.metrics.append(metric)
        return self

    def remove_metric(self, name: str) -> "Evaluator":
        """Remove a metric by name. Returns self for chaining."""
        self.metrics = [m for m in self.metrics if m.name != name]
        return self

    def print_report(self, result: Union[EvalResult, BatchResult]):
        """Pretty-print evaluation results using Rich."""
        if isinstance(result, EvalResult):
            self.console.print(result.summary())
        elif isinstance(result, BatchResult):
            self.console.print(result.summary())
            
            # Detailed table
            table = Table(title="Detailed Results")
            table.add_column("Question", style="cyan", max_width=40)
            table.add_column("Overall", justify="center")
            table.add_column("Status", justify="center")
            
            metric_names = list(result.results[0].metrics.keys()) if result.results else []
            for name in metric_names:
                table.add_column(name, justify="center")
            
            for r in result.results:
                row = [
                    r.question[:40] + "..." if len(r.question) > 40 else r.question,
                    f"{r.overall_score:.3f}",
                    "PASS" if r.passed else "FAIL",
                ]
                for name in metric_names:
                    if name in r.metrics:
                        score = r.metrics[name].score
                        color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
                        row.append(f"[{color}]{score:.3f}[/]")
                    else:
                        row.append("-")
                table.add_row(*row)
            
            self.console.print(table)
