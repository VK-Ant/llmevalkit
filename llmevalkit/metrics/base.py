"""Base metric class for all LLMEVAL metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from jinja2 import Template

from llmevalkit.llm_client import LLMClient
from llmevalkit.models import MetricResult


class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.
    
    All metrics follow the same pattern:
    1. Render a prompt template with the input data
    2. Send to LLM for evaluation
    3. Parse the JSON response
    4. Normalize score to 0-1 range
    5. Return MetricResult
    """
    
    name: str = "base"
    prompt_template: str = ""
    # If True, score is inverted (1 - normalized_score) because the raw score
    # measures a negative quality (e.g., hallucination, toxicity)
    invert_score: bool = False

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, client: LLMClient, **kwargs) -> MetricResult:
        """Run the metric evaluation.
        
        Args:
            client: LLMClient instance for LLM calls
            **kwargs: question, answer, context, reference, etc.
        
        Returns:
            MetricResult with score, reason, and details
        """
        # Render prompt
        template = Template(self.prompt_template)
        prompt = template.render(**kwargs)
        
        # System prompt for consistent evaluation
        system = (
            "You are a precise evaluation judge. Always respond with valid JSON only. "
            "No markdown, no extra text. Be strict and consistent in scoring."
        )
        
        try:
            result = client.generate_json(prompt, system=system)
            return self._parse_result(result)
        except Exception as e:
            # Return a zero-score result on failure rather than crashing
            return MetricResult(
                name=self.name,
                score=0.0,
                reason=f"Evaluation failed: {str(e)}",
                details={"error": str(e)},
            )

    def _parse_result(self, result: dict) -> MetricResult:
        """Parse LLM JSON response into MetricResult."""
        raw_score = result.get("score", 3)
        # Normalize from 1-5 scale to 0-1
        normalized = (raw_score - 1) / 4.0
        normalized = max(0.0, min(1.0, normalized))
        
        if self.invert_score:
            normalized = 1.0 - normalized
        
        # Remove score from details to avoid duplication
        details = {k: v for k, v in result.items() if k not in ("score", "reason")}
        
        return MetricResult(
            name=self.name,
            score=round(normalized, 4),
            reason=result.get("reason", ""),
            details=details,
        )

    @property
    def required_fields(self) -> list[str]:
        """Fields required by this metric. Override in subclasses."""
        return ["question", "answer"]
    
    def validate_inputs(self, **kwargs) -> bool:
        """Check that required fields are present and non-empty."""
        for field in self.required_fields:
            if not kwargs.get(field):
                return False
        return True
