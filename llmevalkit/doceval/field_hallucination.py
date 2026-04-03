"""Field hallucination detection for document extraction.

Checks if extracted values actually exist in the source document.
If a value is not found in the source, it is likely hallucinated.
"""

from __future__ import annotations

import json
import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _fuzzy_ratio, _try_thefuzz, _normalize_amount


def _parse_fields(text):
    if not text:
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {k: str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, TypeError):
        pass
    fields = {}
    for line in text.split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip().lower().replace(' ', '_')
            value = parts[1].strip()
            if key and value:
                fields[key] = value
    return fields


HALLUCINATION_LLM_PROMPT = """Check each extracted value against the source document.
Flag any value that does NOT appear in the source and was likely
fabricated by the model.

Source document:
\"\"\"
{{ context }}
\"\"\"

Extracted fields:
\"\"\"
{{ answer }}
\"\"\"

Respond with ONLY a valid JSON object:
{
    "field_checks": [
        {"field": "vendor", "value": "Acme Corp", "in_source": true, "reason": "found as Acme Corporation"},
        {"field": "amount", "value": "$5000", "in_source": false, "reason": "source says $1,250, not $5000"}
    ],
    "hallucinated_count": 1,
    "total_fields": 4,
    "is_clean": false
}"""


class FieldHallucination:
    """Detect fabricated values in extraction output.

    FieldHallucination()              -- string search in source, free
    FieldHallucination(use_llm=True)  -- LLM checks each value
    """

    name = "field_hallucination"

    def __init__(self, use_llm=False, threshold=0.6, weight=1.0):
        self.use_llm = use_llm
        self.threshold = threshold
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer", "context"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer")) and bool(kwargs.get("context"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")
        context = kwargs.get("context", "")

        if not answer or not context:
            return MetricResult(name=self.name, score=0.0,
                                reason="Need both extraction output and source document")

        fields = _parse_fields(answer)
        if not fields:
            return MetricResult(name=self.name, score=0.0,
                                reason="Could not parse fields")

        field_checks = []
        context_lower = context.lower()

        if not (self.use_llm and client):
            for field_name, value in fields.items():
                check = {"field": field_name, "value": value}

                # Skip very short or generic values
                if len(value) <= 2 or value.lower() in ('yes', 'no', 'true', 'false', 'n/a', 'none'):
                    check["in_source"] = True
                    check["reason"] = "generic value, skipped"
                    check["score"] = 1.0
                    field_checks.append(check)
                    continue

                # Check 1: exact substring match
                if value.lower() in context_lower:
                    check["in_source"] = True
                    check["reason"] = "exact match in source"
                    check["score"] = 1.0
                    field_checks.append(check)
                    continue

                # Check 2: normalized amount match
                norm_val = _normalize_amount(value)
                if norm_val:
                    amounts = re.findall(r'[\$]?[\d,]+\.?\d*', context)
                    found = False
                    for amt in amounts:
                        if _normalize_amount(amt) and abs(_normalize_amount(amt) - norm_val) < 0.01:
                            check["in_source"] = True
                            check["reason"] = "amount match after normalization"
                            check["score"] = 1.0
                            found = True
                            break
                    if found:
                        field_checks.append(check)
                        continue

                # Check 3: fuzzy match against source chunks
                words = context.split()
                best = 0.0
                for start in range(len(words)):
                    for length in range(1, min(len(value.split()) + 3, len(words) - start + 1)):
                        chunk = ' '.join(words[start:start+length])
                        ratio = _try_thefuzz(value, chunk)
                        best = max(best, ratio)
                        if best >= 0.85:
                            break
                    if best >= 0.85:
                        break

                if best >= 0.85:
                    check["in_source"] = True
                    check["reason"] = "fuzzy match ({:.0%})".format(best)
                    check["score"] = best
                else:
                    check["in_source"] = False
                    check["reason"] = "NOT found in source (best match: {:.0%})".format(best)
                    check["score"] = best

                field_checks.append(check)

        # LLM mode
        if self.use_llm and client:
            llm_checks = self._check_with_llm(client, answer, context)
            if llm_checks:
                field_checks = llm_checks

        # Score: percentage of fields that are NOT hallucinated
        total = len(field_checks)
        grounded = sum(1 for c in field_checks if c.get("in_source", False))
        hallucinated = total - grounded

        score = grounded / total if total > 0 else 0.0

        if hallucinated == 0:
            reason = "All {} fields grounded in source document".format(total)
        else:
            bad_fields = [c["field"] for c in field_checks if not c.get("in_source", False)]
            reason = "{} of {} fields may be hallucinated: {}".format(
                hallucinated, total, ", ".join(bad_fields)
            )

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=reason,
            details={
                "field_checks": field_checks,
                "total_fields": total,
                "grounded": grounded,
                "hallucinated": hallucinated,
            },
        )

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(HALLUCINATION_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            system = "You are a document extraction verification expert. Respond with valid JSON only."
            result = client.generate_json(prompt, system=system)
            return result.get("field_checks", [])
        except Exception as e:
            print("FieldHallucination LLM error: {}".format(e), file=sys.stderr)
            return None
