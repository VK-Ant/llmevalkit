"""Field accuracy evaluation for document extraction.

Checks if extracted field values match the source document.
Without API: fuzzy string matching, normalized comparison for
dates, amounts, phone numbers.
With API: LLM judges semantic equivalence.
"""

from __future__ import annotations

import json
import re
import sys

from llmevalkit.models import MetricResult


def _normalize_amount(text):
    """Normalize currency amounts for comparison. $1,250.00 -> 1250.0"""
    clean = re.sub(r'[^\d.]', '', text)
    try:
        return float(clean)
    except ValueError:
        return None


def _normalize_date(text):
    """Normalize date formats for comparison."""
    # Remove common separators and words
    clean = text.strip().lower()
    clean = re.sub(r'(st|nd|rd|th)', '', clean)

    # Try common formats
    import re as _re
    patterns = [
        (r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})', lambda m: '{}-{}-{}'.format(m.group(3), m.group(1).zfill(2), m.group(2).zfill(2))),
        (r'(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})', lambda m: '{}-{}-{}'.format(m.group(1), m.group(2).zfill(2), m.group(3).zfill(2))),
    ]

    months = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'june': '06', 'july': '07', 'august': '08', 'september': '09',
        'october': '10', 'november': '11', 'december': '12',
    }

    # Try month name format: "March 15, 2024" or "15 March 2024"
    for month_name, month_num in months.items():
        if month_name in clean:
            digits = re.findall(r'\d+', clean)
            if len(digits) >= 2:
                day = [d for d in digits if int(d) <= 31]
                year = [d for d in digits if int(d) > 31]
                if day and year:
                    return '{}-{}-{}'.format(year[0], month_num, day[0].zfill(2))

    for pattern, formatter in patterns:
        match = re.search(pattern, clean)
        if match:
            return formatter(match)

    return clean


def _normalize_phone(text):
    """Normalize phone numbers. (555) 123-4567 -> 5551234567"""
    return re.sub(r'[^\d]', '', text)


def _fuzzy_ratio(s1, s2):
    """Simple fuzzy string matching ratio without external dependencies.
    Returns similarity between 0.0 and 1.0."""
    if not s1 or not s2:
        return 0.0
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if s1 == s2:
        return 1.0

    # Levenshtein distance
    len1, len2 = len(s1), len(s2)
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,
                matrix[i][j-1] + 1,
                matrix[i-1][j-1] + cost,
            )
    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


def _try_thefuzz(s1, s2):
    """Try using thefuzz library if available, fallback to built-in."""
    try:
        from thefuzz import fuzz
        return fuzz.token_sort_ratio(s1, s2) / 100.0
    except ImportError:
        return _fuzzy_ratio(s1, s2)


def _parse_fields(text):
    """Parse extracted fields from JSON string or key-value text."""
    if not text:
        return {}
    text = text.strip()

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {k: str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, TypeError):
        pass

    # Try key: value format
    fields = {}
    for line in text.split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            key = parts[0].strip().lower().replace(' ', '_')
            value = parts[1].strip()
            if key and value:
                fields[key] = value
    return fields


FIELD_ACCURACY_LLM_PROMPT = """Compare extracted fields against the source document.

Source document:
\"\"\"
{{ context }}
\"\"\"

Extracted fields:
\"\"\"
{{ answer }}
\"\"\"

For each extracted field, check if the value is correct based on the source.

Respond with ONLY a valid JSON object:
{
    "field_results": [
        {"field": "vendor", "extracted": "Acme Corp", "correct": true,
         "source_value": "Acme Corporation", "reason": "abbreviation match"}
    ],
    "correct_count": 3,
    "total_fields": 4,
    "accuracy": 0.75
}"""


class FieldAccuracy:
    """Check if extracted field values match the source document.

    FieldAccuracy()              -- fuzzy matching, free
    FieldAccuracy(use_llm=True)  -- adds LLM semantic matching
    """

    name = "field_accuracy"

    def __init__(self, use_llm=False, threshold=0.85, weight=1.0):
        self.use_llm = use_llm
        self.threshold = threshold
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")
        context = kwargs.get("context", "")
        reference = kwargs.get("reference", "")

        if not answer:
            return MetricResult(name=self.name, score=0.0, reason="Empty extraction")

        fields = _parse_fields(answer)
        if not fields:
            return MetricResult(name=self.name, score=0.0,
                                reason="Could not parse fields from extraction output")

        # Source to compare against (prefer reference/ground truth, fallback to context)
        source = reference or context

        field_results = []

        if source and not (self.use_llm and client):
            # Offline mode: fuzzy matching against source
            source_lower = source.lower()

            for field_name, extracted_value in fields.items():
                result = {"field": field_name, "extracted": extracted_value}

                # Try exact match first
                if extracted_value.lower() in source_lower:
                    result["match"] = "exact"
                    result["score"] = 1.0
                else:
                    # Try normalized comparison based on field type
                    score = 0.0
                    method = "fuzzy"

                    # Amount normalization
                    if any(c in extracted_value for c in ['$', ',']) or re.match(r'^\d+\.?\d*$', extracted_value):
                        norm_ext = _normalize_amount(extracted_value)
                        amounts_in_source = re.findall(r'[\$]?[\d,]+\.?\d*', source)
                        for amt in amounts_in_source:
                            norm_src = _normalize_amount(amt)
                            if norm_ext and norm_src and abs(norm_ext - norm_src) < 0.01:
                                score = 1.0
                                method = "amount_normalized"
                                break

                    # Date normalization
                    if score < 1.0 and re.search(r'\d{1,4}[/\-.]', extracted_value):
                        norm_ext = _normalize_date(extracted_value)
                        date_patterns = re.findall(r'\d{1,2}[/\-.]?\d{1,2}[/\-.]?\d{2,4}', source)
                        for date_str in date_patterns:
                            norm_src = _normalize_date(date_str)
                            if norm_ext == norm_src:
                                score = 1.0
                                method = "date_normalized"
                                break

                    # Phone normalization
                    if score < 1.0 and re.search(r'[\d\-\(\)\+]{7,}', extracted_value):
                        norm_ext = _normalize_phone(extracted_value)
                        phones_in_source = re.findall(r'[\d\-\(\)\+\s]{7,}', source)
                        for phone in phones_in_source:
                            if _normalize_phone(phone) == norm_ext:
                                score = 1.0
                                method = "phone_normalized"
                                break

                    # Fuzzy string matching as fallback
                    if score < 1.0:
                        # Search in chunks of source text
                        words = source.split()
                        best_score = 0.0
                        for start in range(len(words)):
                            for length in range(1, min(len(extracted_value.split()) + 3, len(words) - start + 1)):
                                chunk = ' '.join(words[start:start+length])
                                ratio = _try_thefuzz(extracted_value, chunk)
                                best_score = max(best_score, ratio)
                        score = best_score
                        method = "fuzzy"

                    result["match"] = method
                    result["score"] = round(score, 3)

                result["passed"] = result["score"] >= self.threshold
                field_results.append(result)

        # LLM mode
        if self.use_llm and client and source:
            llm_results = self._check_with_llm(client, answer, source)
            if llm_results:
                field_results = llm_results

        # Calculate overall score
        if field_results:
            passed = sum(1 for r in field_results if r.get("passed", r.get("correct", False)))
            total = len(field_results)
            score = passed / total if total > 0 else 0.0
        elif not source:
            score = 0.0
            field_results = [{"note": "No source document provided for comparison"}]
        else:
            score = 0.0

        if score >= 0.8:
            reason = "Field accuracy: {}/{} fields match source".format(
                sum(1 for r in field_results if r.get("passed", r.get("correct", False))),
                len(field_results)
            )
        else:
            failed = [r["field"] for r in field_results
                      if not r.get("passed", r.get("correct", False)) and "field" in r]
            reason = "Low accuracy. Failed fields: {}".format(", ".join(failed))

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=reason,
            details={
                "field_results": field_results,
                "total_fields": len(field_results),
                "passed_fields": sum(1 for r in field_results if r.get("passed", r.get("correct", False))),
                "threshold": self.threshold,
            },
        )

    def _check_with_llm(self, client, answer, source):
        from jinja2 import Template
        try:
            template = Template(FIELD_ACCURACY_LLM_PROMPT)
            prompt = template.render(answer=answer, context=source)
            system = "You are a document extraction accuracy expert. Respond with valid JSON only."
            result = client.generate_json(prompt, system=system)
            items = result.get("field_results", [])
            for item in items:
                item["passed"] = item.get("correct", False)
            return items
        except Exception as e:
            print("FieldAccuracy LLM error: {}".format(e), file=sys.stderr)
            return None
