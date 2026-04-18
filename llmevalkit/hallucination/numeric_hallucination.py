"""Numeric hallucination detection.

Checks if numbers, dates, percentages, and amounts in the output
match the source context. The most common hallucination type in
production LLM systems.
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult


def _extract_numbers(text):
    """Extract all numeric values with their context."""
    numbers = []
    # Currency amounts
    for m in re.finditer(r'[\$\u20ac\u00a3\u20b9]?\s?[\d,]+\.?\d*\s?(?:million|billion|thousand|M|B|K|%)?', text):
        val = m.group().strip()
        clean = re.sub(r'[^\d.]', '', val.split()[0] if val.split() else val)
        try:
            num = float(clean)
            multiplier = 1
            lower = val.lower()
            if 'million' in lower or lower.endswith('m'):
                multiplier = 1_000_000
            elif 'billion' in lower or lower.endswith('b'):
                multiplier = 1_000_000_000
            elif 'thousand' in lower or lower.endswith('k'):
                multiplier = 1_000
            numbers.append({"raw": val, "value": num * multiplier, "position": m.start()})
        except ValueError:
            pass
    # Percentages
    for m in re.finditer(r'(\d+\.?\d*)\s?%', text):
        try:
            numbers.append({"raw": m.group(), "value": float(m.group(1)), "position": m.start(), "type": "percent"})
        except ValueError:
            pass
    # Years
    for m in re.finditer(r'\b(1[89]\d{2}|20[0-3]\d)\b', text):
        numbers.append({"raw": m.group(), "value": int(m.group(1)), "position": m.start(), "type": "year"})
    return numbers


NUMERIC_LLM_PROMPT = """Check all numbers, dates, amounts, and percentages in the output against the context.

Context:
\"\"\"
{{ context }}
\"\"\"

Output:
\"\"\"
{{ answer }}
\"\"\"

Respond with ONLY valid JSON:
{
    "checks": [
        {"value_in_output": "$5M", "value_in_context": "$3M", "match": false, "reason": "amount differs"}
    ],
    "correct_count": 2,
    "wrong_count": 1,
    "total": 3
}"""


class NumericHallucination:
    """Detect numeric hallucinations -- wrong numbers, amounts, dates.

    NumericHallucination()              -- regex extraction + comparison, free
    NumericHallucination(use_llm=True)  -- adds LLM analysis
    """

    name = "numeric_hallucination"

    def __init__(self, use_llm=False, tolerance=0.01, weight=1.0):
        self.use_llm = use_llm
        self.tolerance = tolerance
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
        if not context:
            return MetricResult(name=self.name, score=1.0,
                                reason="No context to check numbers against")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, context)

        answer_nums = _extract_numbers(answer)
        context_nums = _extract_numbers(context)

        if not answer_nums:
            return MetricResult(name=self.name, score=1.0,
                                reason="No numbers found in output",
                                details={"numbers_checked": 0})

        checks = []
        for a_num in answer_nums:
            best_match = None
            best_diff = float('inf')
            for c_num in context_nums:
                if a_num["value"] == 0 and c_num["value"] == 0:
                    best_match = c_num
                    best_diff = 0
                    break
                if c_num["value"] != 0:
                    diff = abs(a_num["value"] - c_num["value"]) / abs(c_num["value"])
                else:
                    diff = abs(a_num["value"])
                if diff < best_diff:
                    best_diff = diff
                    best_match = c_num

            if best_match and best_diff <= self.tolerance:
                checks.append({"output": a_num["raw"], "context": best_match["raw"],
                                "match": True, "diff": round(best_diff, 4)})
            elif best_match and best_diff < 1.0:
                checks.append({"output": a_num["raw"], "context": best_match["raw"],
                                "match": False, "diff": round(best_diff, 4),
                                "reason": "value differs by {:.1%}".format(best_diff)})
            else:
                checks.append({"output": a_num["raw"], "context": "not found",
                                "match": False, "reason": "number not in context"})

        correct = sum(1 for c in checks if c["match"])
        total = len(checks)
        score = correct / total if total > 0 else 1.0

        wrong = [c for c in checks if not c["match"]]
        if not wrong:
            reason = "All {} numbers match context".format(total)
        else:
            reason = "{} of {} numbers wrong: {}".format(
                len(wrong), total, ", ".join(w["output"] for w in wrong[:3]))

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"checks": checks, "correct": correct, "wrong": len(wrong), "total": total})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(NUMERIC_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a fact-checking expert. Respond with valid JSON only.")
            total = result.get("total", 1)
            correct = result.get("correct_count", 0)
            score = correct / total if total > 0 else 1.0
            return MetricResult(
                name=self.name, score=round(score, 4),
                reason="{} of {} numbers correct".format(correct, total), details=result)
        except Exception as e:
            print("NumericHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
