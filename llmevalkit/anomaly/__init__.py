"""llmevalkit anomaly detection module.

Detect unusual LLM outputs and sudden score changes.
"""

from __future__ import annotations

import math
import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.observe import EvalLogger


ANOMALY_LLM_PROMPT = """Analyze if this output is anomalous compared to what you would expect
given the context and question.

{% if question %}Question: {{ question }}{% endif %}
{% if context %}Context: {{ context }}{% endif %}

Output:
\"\"\"
{{ answer }}
\"\"\"

Check for: unexpected tone, off-topic content, unusual formatting,
suspicious patterns, extreme sentiment, gibberish, or inappropriate content.

Respond with ONLY valid JSON:
{
    "is_anomalous": false,
    "anomalies": [],
    "severity": "none",
    "score": 5
}

score 1-5: 1=highly anomalous, 5=completely normal."""


class OutputAnomalyDetector:
    """Detect unusual LLM outputs that deviate from expected patterns.

    Checks: length anomaly, vocabulary shift, sentiment extremes,
    topic drift, formatting issues, gibberish detection.
    """

    name = "output_anomaly_detector"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")
        context = kwargs.get("context", "")

        if not answer:
            return MetricResult(name=self.name, score=1.0, reason="Empty output")

        anomalies = []

        # 1. Length anomaly -- extremely short or long
        words = answer.split()
        if len(words) < 3:
            anomalies.append({"type": "too_short", "detail": "{} words".format(len(words))})
        elif len(words) > 2000:
            anomalies.append({"type": "too_long", "detail": "{} words".format(len(words))})

        # 2. Repetition anomaly -- same sentence repeated
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        if len(sentences) >= 3:
            from collections import Counter
            sent_counts = Counter(sentences)
            repeated = {s: c for s, c in sent_counts.items() if c >= 3}
            if repeated:
                anomalies.append({"type": "repetition_loop",
                                  "detail": "sentence repeated {}+ times".format(max(repeated.values()))})

        # 3. Character anomaly -- unusual characters or encoding issues
        non_ascii = sum(1 for c in answer if ord(c) > 127 and ord(c) < 256)
        if non_ascii > len(answer) * 0.1:
            anomalies.append({"type": "encoding_issue",
                              "detail": "{} unusual characters".format(non_ascii)})

        # 4. All caps or no punctuation
        if answer == answer.upper() and len(words) > 5:
            anomalies.append({"type": "all_caps", "detail": "entire output is uppercase"})

        if len(answer) > 50 and not re.search(r'[.!?,;:]', answer):
            anomalies.append({"type": "no_punctuation", "detail": "no punctuation in long text"})

        # 5. Sentiment extremes
        extreme_words = ['URGENT', 'DANGER', 'WARNING', 'EMERGENCY', 'IMMEDIATELY',
                         'BUY NOW', 'ACT NOW', 'LIMITED TIME', 'FREE MONEY']
        extreme_found = [w for w in extreme_words if w.lower() in answer.lower()]
        if extreme_found:
            anomalies.append({"type": "extreme_sentiment",
                              "detail": "found: {}".format(", ".join(extreme_found[:3]))})

        # 6. Topic drift -- if context provided, check overlap
        if context:
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'to', 'for', 'of', 'and', 'or'}
            ctx_words = set(context.lower().split()) - stopwords
            ans_words = set(answer.lower().split()) - stopwords
            if ctx_words and ans_words:
                overlap = len(ctx_words & ans_words) / max(len(ans_words), 1)
                if overlap < 0.05 and len(words) > 10:
                    anomalies.append({"type": "topic_drift",
                                      "detail": "output has <5% overlap with context"})

        # LLM mode
        if self.use_llm and client:
            llm_result = self._check_with_llm(client, answer, context, kwargs.get("question", ""))
            if llm_result and llm_result.get("is_anomalous"):
                for a in llm_result.get("anomalies", []):
                    anomalies.append({"type": "llm_detected", "detail": str(a)})

        if not anomalies:
            return MetricResult(name=self.name, score=1.0,
                                reason="No anomalies detected", details={"anomalies": []})

        score = max(0.0, 1.0 - len(anomalies) * 0.2)
        types = [a["type"] for a in anomalies]

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} anomaly(s): {}".format(len(anomalies), ", ".join(types)),
            details={"anomalies": anomalies, "count": len(anomalies)})

    def _check_with_llm(self, client, answer, context, question):
        from jinja2 import Template
        try:
            template = Template(ANOMALY_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context or "", question=question or "")
            return client.generate_json(prompt, system="You are an output quality expert. Respond with valid JSON only.")
        except Exception:
            return None


class ScoreAnomalyDetector:
    """Detect sudden score changes across evaluation history.

    Uses z-score to flag evaluations that are 2+ standard deviations
    from the historical mean.
    """

    name = "score_anomaly_detector"

    def __init__(self, log_dir=None, z_threshold=2.0):
        self.logger = EvalLogger(log_dir)
        self.z_threshold = z_threshold

    def check(self, last_n=100):
        """Check for score anomalies in recent history."""
        entries = self.logger.read_logs(last_n=last_n)

        if len(entries) < 10:
            return {
                "status": "insufficient_data",
                "message": "Need 10+ evaluations (have {})".format(len(entries)),
            }

        scores = [e["overall_score"] for e in entries]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance) if variance > 0 else 0.001

        anomalies = []
        for i, entry in enumerate(entries):
            z = (entry["overall_score"] - mean) / std
            if abs(z) >= self.z_threshold:
                anomalies.append({
                    "index": i,
                    "score": entry["overall_score"],
                    "z_score": round(z, 2),
                    "timestamp": entry.get("timestamp", ""),
                    "direction": "low" if z < 0 else "high",
                })

        return {
            "status": "anomalies_found" if anomalies else "normal",
            "mean": round(mean, 4),
            "std": round(std, 4),
            "z_threshold": self.z_threshold,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "total_checked": len(entries),
        }
