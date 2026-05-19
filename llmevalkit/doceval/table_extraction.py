"""Table extraction accuracy metric.

Evaluates if table structure (rows, columns, cells) is correctly extracted.
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz


def _parse_table(text):
    """Parse table from text. Supports pipe-delimited, CSV, tab-delimited."""
    rows = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or set(line) <= {'-', '|', '+', '=', ' '}:
            continue
        if '|' in line:
            cells = [c.strip() for c in line.split('|') if c.strip()]
        elif '\t' in line:
            cells = [c.strip() for c in line.split('\t') if c.strip()]
        elif ',' in line:
            cells = [c.strip().strip('"') for c in line.split(',') if c.strip()]
        else:
            cells = [line.strip()]
        if cells:
            rows.append(cells)
    return rows


class TableExtractionAccuracy:
    """Are table rows and columns extracted correctly?

    Compares extracted table against reference table.
    Checks: row count, column count, cell-level accuracy.
    """
    name = "table_extraction_accuracy"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer", "reference"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer")) and bool(kwargs.get("reference"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")
        reference = kwargs.get("reference", "")

        if not answer or not reference:
            return MetricResult(name=self.name, score=0.0, reason="Need both extraction and reference")

        ext_rows = _parse_table(answer)
        ref_rows = _parse_table(reference)

        if not ref_rows:
            return MetricResult(name=self.name, score=0.5, reason="Could not parse reference table")
        if not ext_rows:
            return MetricResult(name=self.name, score=0.0, reason="Could not parse extracted table")

        # Row count accuracy
        row_score = min(len(ext_rows), len(ref_rows)) / max(len(ext_rows), len(ref_rows))

        # Column count accuracy (based on first row)
        ext_cols = len(ext_rows[0]) if ext_rows else 0
        ref_cols = len(ref_rows[0]) if ref_rows else 0
        col_score = min(ext_cols, ref_cols) / max(ext_cols, ref_cols) if max(ext_cols, ref_cols) > 0 else 0

        # Cell-level accuracy
        cell_correct = 0
        cell_total = 0
        cell_details = []

        for i in range(min(len(ext_rows), len(ref_rows))):
            for j in range(min(len(ext_rows[i]), len(ref_rows[i]))):
                cell_total += 1
                ext_val = ext_rows[i][j].strip()
                ref_val = ref_rows[i][j].strip()

                if ext_val.lower() == ref_val.lower():
                    cell_correct += 1
                    match = "exact"
                else:
                    fuzzy = _try_thefuzz(ext_val, ref_val)
                    if fuzzy >= 0.8:
                        cell_correct += 1
                        match = "fuzzy"
                    else:
                        match = "mismatch"
                        cell_details.append({"row": i, "col": j, "expected": ref_val[:30], "got": ext_val[:30]})

        cell_score = cell_correct / cell_total if cell_total > 0 else 0

        # Combined score
        score = row_score * 0.2 + col_score * 0.2 + cell_score * 0.6

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="Rows: {}/{}, Cols: {}/{}, Cells: {}/{}".format(
                len(ext_rows), len(ref_rows), ext_cols, ref_cols, cell_correct, cell_total),
            details={
                "row_count_extracted": len(ext_rows), "row_count_reference": len(ref_rows),
                "col_count_extracted": ext_cols, "col_count_reference": ref_cols,
                "cell_correct": cell_correct, "cell_total": cell_total,
                "cell_accuracy": round(cell_score, 4),
                "mismatches": cell_details[:10],
            })
