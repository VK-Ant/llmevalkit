"""Base metric class for all evaluation metrics."""

from __future__ import annotations

import sys
from abc import ABC
from typing import Optional

from jinja2 import Template

from llmevalkit.llm_client import LLMClient
from llmevalkit.models import MetricResult


class BaseMetric(ABC):
    """Base class for LLM-as-judge metrics."""

    name = "base"
    prompt_template = ""
    invert_score = False

    def __init__(self, weight=1.0):
        self.weight = weight

    def evaluate(self, client, **kwargs):
        """Run the metric. Returns MetricResult."""
        template = Template(self.prompt_template)
        prompt = template.render(**kwargs)

        system = (
            "You are a precise evaluation judge. Always respond with valid JSON only. "
            "No markdown, no extra text. Be strict and consistent in scoring."
        )

        try:
            result = client.generate_json(prompt, system=system)
            return self._parse_result(result)
        except Exception as e:
            # Print the error so the user can see what went wrong.
            print("ERROR in {}: {}".format(self.name, e), file=sys.stderr)
            return MetricResult(
                name=self.name,
                score=0.0,
                reason="Evaluation failed: {}".format(str(e)),
                details={"error": str(e)},
            )

    def _parse_result(self, result):
        """Parse LLM JSON response into MetricResult."""
        raw_score = result.get("score", 3)

        # Handle string scores like "4" instead of 4.
        if isinstance(raw_score, str):
            try:
                raw_score = float(raw_score)
            except ValueError:
                raw_score = 3

        # Normalize from 1-5 scale to 0-1.
        normalized = (raw_score - 1) / 4.0
        normalized = max(0.0, min(1.0, normalized))

        if self.invert_score:
            normalized = 1.0 - normalized

        details = {}
        for k, v in result.items():
            if k not in ("score", "reason"):
                details[k] = v

        return MetricResult(
            name=self.name,
            score=round(normalized, 4),
            reason=result.get("reason", ""),
            details=details,
        )

    @property
    def required_fields(self):
        return ["question", "answer"]

    def validate_inputs(self, **kwargs):
        for field in self.required_fields:
            if not kwargs.get(field):
                return False
        return True
