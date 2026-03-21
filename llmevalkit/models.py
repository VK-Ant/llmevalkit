"""Data models for LLMEVAL."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Provider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GROQ = "groq"
    CUSTOM = "custom"


class MetricResult(BaseModel):
    """Result from a single metric."""
    name: str = Field(description="Metric name")
    score: float = Field(ge=0.0, le=1.0, description="Score from 0 to 1")
    reason: str = Field(default="", description="Why this score was given")
    details: Dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return "MetricResult(name='{}', score={:.3f})".format(self.name, self.score)


class EvalResult(BaseModel):
    """Complete evaluation result for one sample."""
    question: str = Field(default="")
    answer: str = Field(default="")
    context: str = Field(default="")
    reference: Optional[str] = Field(default=None)
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        threshold = self.metadata.get("threshold", 0.5)
        return self.overall_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        flat = {
            "question": self.question,
            "answer": self.answer,
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
        }
        for name, metric in self.metrics.items():
            flat["metric_{}_score".format(name)] = round(metric.score, 4)
            flat["metric_{}_reason".format(name)] = metric.reason
        return flat

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  LLMEVAL Evaluation Result")
        lines.append("=" * 60)
        lines.append("  Question : {}".format(self.question[:80]))
        lines.append("  Answer   : {}".format(self.answer[:80]))
        lines.append("-" * 60)
        for name, metric in self.metrics.items():
            filled = int(metric.score * 20)
            bar = "#" * filled + "." * (20 - filled)
            lines.append("  {:<20} [{}] {:.3f}".format(name, bar, metric.score))
        lines.append("-" * 60)
        status = "PASSED" if self.passed else "FAILED"
        lines.append("  Overall Score: {:.3f}  {}".format(self.overall_score, status))
        lines.append("=" * 60)
        return "\n".join(lines)


class EvalConfig(BaseModel):
    provider: Provider = Provider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0)
    timeout: int = Field(default=60, ge=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    verbose: bool = False
    model_config = {"use_enum_values": True}


class TestCase(BaseModel):
    question: str
    answer: str
    context: str = ""
    reference: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchResult(BaseModel):
    results: List[EvalResult] = Field(default_factory=list)

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def metric_averages(self) -> Dict[str, float]:
        scores = {}  # type: Dict[str, List[float]]
        for result in self.results:
            for name, metric in result.metrics.items():
                if name not in scores:
                    scores[name] = []
                scores[name].append(metric.score)
        return {name: sum(s) / len(s) for name, s in scores.items()}

    def summary(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  LLMEVAL Batch Summary ({} samples)".format(len(self.results)))
        lines.append("=" * 60)
        lines.append("  Average Score : {:.3f}".format(self.average_score))
        lines.append("  Pass Rate     : {:.1%}".format(self.pass_rate))
        lines.append("-" * 60)
        for name, avg in self.metric_averages().items():
            filled = int(avg * 20)
            bar = "#" * filled + "." * (20 - filled)
            lines.append("    {:<20} [{}] {:.3f}".format(name, bar, avg))
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self):
        try:
            import pandas as pd
            return pd.DataFrame([r.to_dict() for r in self.results])
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")
