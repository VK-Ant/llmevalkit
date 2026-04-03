"""Extraction consistency evaluation.

Checks if the same document processed multiple times produces
consistent extraction results. No ground truth needed.
Works entirely offline.
"""

from __future__ import annotations

import json

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz, _normalize_amount, _normalize_phone


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


class ExtractionConsistency:
    """Check if multiple extraction runs produce consistent results.

    No ground truth needed. No API needed.
    Pass multiple extraction outputs for the same document.

    Usage:
        ec = ExtractionConsistency()
        result = ec.evaluate(answer=[
            '{"vendor": "Acme Corp", "amount": "$1250"}',
            '{"vendor": "Acme Corp", "amount": "$1,250.00"}',
            '{"vendor": "Acme Corporation", "amount": "$1250"}',
        ])
    """

    name = "extraction_consistency"

    def __init__(self, threshold=0.85, weight=1.0):
        self.threshold = threshold
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        answer = kwargs.get("answer")
        return isinstance(answer, list) and len(answer) >= 2

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", [])

        if not isinstance(answer, list) or len(answer) < 2:
            return MetricResult(
                name=self.name, score=0.0,
                reason="Need a list of 2+ extraction outputs to compare",
            )

        # Parse all extraction outputs
        all_fields = [_parse_fields(a) for a in answer]
        all_fields = [f for f in all_fields if f]

        if len(all_fields) < 2:
            return MetricResult(name=self.name, score=0.0,
                                reason="Could not parse enough extraction outputs")

        # Get all unique field names across all runs
        all_keys = set()
        for fields in all_fields:
            all_keys.update(fields.keys())

        field_consistency = []

        for key in sorted(all_keys):
            values = [fields.get(key, "") for fields in all_fields]
            non_empty = [v for v in values if v]

            if len(non_empty) <= 1:
                field_consistency.append({
                    "field": key,
                    "values": values,
                    "agreement": 0.0 if len(non_empty) == 0 else 1.0,
                    "reason": "only present in {} of {} runs".format(len(non_empty), len(all_fields)),
                })
                continue

            # Compare all pairs
            pair_scores = []
            for i in range(len(non_empty)):
                for j in range(i + 1, len(non_empty)):
                    v1, v2 = non_empty[i], non_empty[j]

                    # Exact match
                    if v1.lower().strip() == v2.lower().strip():
                        pair_scores.append(1.0)
                        continue

                    # Normalized amount match
                    n1, n2 = _normalize_amount(v1), _normalize_amount(v2)
                    if n1 and n2 and abs(n1 - n2) < 0.01:
                        pair_scores.append(1.0)
                        continue

                    # Normalized phone match
                    p1, p2 = _normalize_phone(v1), _normalize_phone(v2)
                    if len(p1) >= 7 and p1 == p2:
                        pair_scores.append(1.0)
                        continue

                    # Fuzzy match
                    pair_scores.append(_try_thefuzz(v1, v2))

            agreement = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

            field_consistency.append({
                "field": key,
                "values": non_empty,
                "agreement": round(agreement, 3),
                "consistent": agreement >= self.threshold,
            })

        # Overall score
        if field_consistency:
            score = sum(f["agreement"] for f in field_consistency) / len(field_consistency)
        else:
            score = 0.0

        consistent = [f for f in field_consistency if f.get("consistent", f["agreement"] >= self.threshold)]
        inconsistent = [f for f in field_consistency if not f.get("consistent", f["agreement"] >= self.threshold)]

        if not inconsistent:
            reason = "All {} fields are consistent across {} runs".format(
                len(field_consistency), len(all_fields)
            )
        else:
            bad = [f["field"] for f in inconsistent]
            reason = "{} of {} fields inconsistent: {}".format(
                len(inconsistent), len(field_consistency), ", ".join(bad)
            )

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=reason,
            details={
                "field_consistency": field_consistency,
                "total_fields": len(field_consistency),
                "consistent_count": len(consistent),
                "inconsistent_count": len(inconsistent),
                "num_runs": len(all_fields),
            },
        )
