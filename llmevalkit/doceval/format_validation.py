"""Format validation for extracted fields.

Checks if extracted values have correct formats: valid dates,
valid amounts, valid emails, valid phone numbers, custom regex patterns.
Works entirely offline, no API needed.
"""

from __future__ import annotations

import json
import re

from llmevalkit.models import MetricResult


# Built-in format validators
FORMAT_VALIDATORS = {
    "date": {
        "patterns": [
            r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$',
            r'^\d{4}[/\-]\d{1,2}[/\-]\d{1,2}$',
            r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}$',
            r'^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}$',
        ],
        "description": "Valid date format",
    },
    "currency": {
        "patterns": [
            r'^[\$\u20ac\u00a3\u20b9]?\s?[\d,]+\.?\d*$',
            r'^[\d,]+\.?\d*\s?(?:USD|EUR|GBP|INR|dollars?|rupees?)$',
        ],
        "description": "Valid currency amount",
    },
    "email": {
        "patterns": [r'^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$'],
        "description": "Valid email address",
    },
    "phone": {
        "patterns": [
            r'^[\+]?[\d\s\-\(\)]{7,15}$',
        ],
        "description": "Valid phone number",
    },
    "number": {
        "patterns": [r'^[\-]?[\d,]+\.?\d*$'],
        "description": "Valid number",
    },
    "percentage": {
        "patterns": [r'^[\d.]+\s?%$'],
        "description": "Valid percentage",
    },
}


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


class FormatValidation:
    """Validate formats of extracted field values.

    Works entirely offline. No API needed.

    Example:
        fv = FormatValidation(field_formats={
            "date": "date",
            "amount": "currency",
            "email": "email",
            "invoice_number": r"INV-\\d{4,}",  # custom regex
        })
    """

    name = "format_validation"

    def __init__(self, field_formats=None, weight=1.0):
        self.field_formats = field_formats or {}
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")

        if not answer:
            return MetricResult(name=self.name, score=0.0, reason="Empty extraction")

        fields = _parse_fields(answer)
        if not fields:
            return MetricResult(name=self.name, score=0.0,
                                reason="Could not parse fields")

        if not self.field_formats:
            return MetricResult(name=self.name, score=1.0,
                                reason="No format rules defined",
                                details={"note": "Provide field_formats to validate"})

        validations = []

        for field_name, expected_format in self.field_formats.items():
            field_lower = field_name.lower().replace(' ', '_')

            # Find the field in extracted data
            value = None
            for key in fields:
                if key.lower() == field_lower:
                    value = fields[key]
                    break

            if value is None:
                validations.append({
                    "field": field_name,
                    "format": expected_format,
                    "value": None,
                    "valid": False,
                    "reason": "field not found in extraction",
                })
                continue

            # Validate format
            if expected_format in FORMAT_VALIDATORS:
                # Built-in format
                patterns = FORMAT_VALIDATORS[expected_format]["patterns"]
                matched = any(re.match(p, value.strip(), re.IGNORECASE) for p in patterns)
                validations.append({
                    "field": field_name,
                    "format": expected_format,
                    "value": value,
                    "valid": matched,
                    "reason": "valid" if matched else "does not match {} format".format(expected_format),
                })
            else:
                # Custom regex pattern
                try:
                    matched = bool(re.match(expected_format, value.strip()))
                    validations.append({
                        "field": field_name,
                        "format": "custom",
                        "value": value,
                        "valid": matched,
                        "reason": "valid" if matched else "does not match pattern",
                    })
                except re.error:
                    validations.append({
                        "field": field_name,
                        "format": "custom",
                        "value": value,
                        "valid": False,
                        "reason": "invalid regex pattern",
                    })

        total = len(validations)
        passed = sum(1 for v in validations if v["valid"])
        score = passed / total if total > 0 else 0.0

        if score >= 1.0:
            reason = "All {} fields have valid formats".format(total)
        else:
            failed = [v["field"] for v in validations if not v["valid"]]
            reason = "{} of {} fields have invalid format: {}".format(
                total - passed, total, ", ".join(failed)
            )

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=reason,
            details={
                "validations": validations,
                "total_fields": total,
                "valid_count": passed,
                "invalid_count": total - passed,
            },
        )
