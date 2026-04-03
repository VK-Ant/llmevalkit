"""Field completeness evaluation for document extraction.

Checks if all expected fields are present and non-empty in the
extraction output.
"""

from __future__ import annotations

import json
import sys

from llmevalkit.models import MetricResult


def _parse_fields(text):
    """Parse fields from JSON string or key-value text."""
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
            if key:
                fields[key] = value
    return fields


COMPLETENESS_LLM_PROMPT = """Analyze this document and the extraction output.
Identify which fields exist in the source document but are missing
from the extraction.

Source document:
\"\"\"
{{ context }}
\"\"\"

Extraction output:
\"\"\"
{{ answer }}
\"\"\"

Respond with ONLY a valid JSON object:
{
    "extracted_fields": ["vendor", "amount"],
    "missing_fields": ["date", "invoice_number"],
    "extra_fields": [],
    "completeness": 0.5,
    "summary": "2 of 4 expected fields extracted"
}"""


class FieldCompleteness:
    """Check if all expected fields are present in extraction output.

    FieldCompleteness(expected_fields=[...])   -- checks against expected list
    FieldCompleteness(use_llm=True)            -- LLM identifies missing fields
    """

    name = "field_completeness"

    def __init__(self, expected_fields=None, use_llm=False, weight=1.0):
        self.expected_fields = expected_fields or []
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
            return MetricResult(name=self.name, score=0.0, reason="Empty extraction")

        fields = _parse_fields(answer)

        # If expected fields provided, check against them.
        if self.expected_fields:
            present = []
            missing = []
            empty = []

            for field in self.expected_fields:
                field_lower = field.lower().replace(' ', '_')
                # Check if field exists in extracted fields (case-insensitive)
                matched = False
                for key in fields:
                    if key.lower() == field_lower or field.lower() in key.lower():
                        if fields[key] and fields[key].lower() not in ('', 'none', 'null', 'n/a', 'unknown'):
                            present.append(field)
                        else:
                            empty.append(field)
                        matched = True
                        break
                if not matched:
                    missing.append(field)

            total = len(self.expected_fields)
            found = len(present)
            score = found / total if total > 0 else 0.0

            reason = "{} of {} expected fields present".format(found, total)
            if missing:
                reason += ". Missing: {}".format(", ".join(missing))
            if empty:
                reason += ". Empty: {}".format(", ".join(empty))

            return MetricResult(
                name=self.name,
                score=round(score, 4),
                reason=reason,
                details={
                    "present": present,
                    "missing": missing,
                    "empty": empty,
                    "total_expected": total,
                    "found_count": found,
                },
            )

        # If no expected fields but LLM available, ask LLM to find missing fields.
        if self.use_llm and client and context:
            return self._check_with_llm(client, answer, context)

        # No expected fields, no LLM -- just report what was extracted.
        non_empty = {k: v for k, v in fields.items()
                     if v and v.lower() not in ('', 'none', 'null', 'n/a')}

        return MetricResult(
            name=self.name,
            score=1.0 if non_empty else 0.0,
            reason="{} fields extracted".format(len(non_empty)),
            details={
                "extracted_fields": list(non_empty.keys()),
                "field_count": len(non_empty),
                "note": "Provide expected_fields for completeness scoring",
            },
        )

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(COMPLETENESS_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            system = "You are a document extraction expert. Respond with valid JSON only."
            result = client.generate_json(prompt, system=system)

            extracted = result.get("extracted_fields", [])
            missing = result.get("missing_fields", [])
            total = len(extracted) + len(missing)
            score = len(extracted) / total if total > 0 else 0.0

            return MetricResult(
                name=self.name,
                score=round(score, 4),
                reason=result.get("summary", ""),
                details={
                    "extracted_fields": extracted,
                    "missing_fields": missing,
                    "total_expected": total,
                },
            )
        except Exception as e:
            print("FieldCompleteness LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0,
                                reason="LLM evaluation failed: {}".format(e))
