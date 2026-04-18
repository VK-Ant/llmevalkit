"""Additional multimodal evaluation metrics.

DocumentLayoutAccuracy: checks if document structure (headers, tables, sections) is preserved
MultimodalConsistency: checks if text + image/audio descriptions are consistent
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz


LAYOUT_LLM_PROMPT = """Compare the extracted document layout against the expected structure.

Expected structure:
\"\"\"
{{ reference }}
\"\"\"

Extracted structure:
\"\"\"
{{ answer }}
\"\"\"

Check: headers preserved? Tables intact? Sections in correct order? Key elements present?

Respond with ONLY valid JSON:
{
    "elements_checked": [
        {"element": "header: Invoice Details", "found": true},
        {"element": "table: line items", "found": false, "reason": "table merged into paragraph"}
    ],
    "preserved_count": 3,
    "total_elements": 5,
    "score": 0.6
}"""


class DocumentLayoutAccuracy:
    """Check if document structure is preserved during extraction.

    Compares extracted layout elements (headers, tables, lists, sections)
    against expected structure.

    Without API: keyword matching for structural markers
    With API: LLM judges structural preservation
    """

    name = "document_layout_accuracy"

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
        reference = kwargs.get("reference", "") or kwargs.get("context", "")

        if not answer or not reference:
            return MetricResult(name=self.name, score=0.0,
                                reason="Need both extracted and expected layout")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, reference)

        # Offline: detect structural elements
        def find_elements(text):
            elements = []
            # Headers (markdown or caps)
            for m in re.finditer(r'^#{1,6}\s+(.+)', text, re.MULTILINE):
                elements.append(("header", m.group(1).strip()))
            for m in re.finditer(r'^([A-Z][A-Z\s]{3,})$', text, re.MULTILINE):
                elements.append(("header", m.group(1).strip()))
            # Tables (pipe-separated or tab-separated)
            table_lines = [l for l in text.split('\n') if '|' in l or '\t' in l]
            if len(table_lines) >= 2:
                elements.append(("table", "{} rows".format(len(table_lines))))
            # Lists
            list_items = re.findall(r'^\s*[\-\*\d+\.]\s+', text, re.MULTILINE)
            if list_items:
                elements.append(("list", "{} items".format(len(list_items))))
            # Sections (numbered or lettered)
            sections = re.findall(r'^\s*(?:Section|Part|Chapter)\s+\d+', text, re.MULTILINE | re.IGNORECASE)
            for s in sections:
                elements.append(("section", s.strip()))
            return elements

        ref_elements = find_elements(reference)
        ans_elements = find_elements(answer)

        if not ref_elements:
            # Try simple line count and paragraph comparison
            ref_lines = len([l for l in reference.split('\n') if l.strip()])
            ans_lines = len([l for l in answer.split('\n') if l.strip()])
            ratio = min(ref_lines, ans_lines) / max(ref_lines, ans_lines) if max(ref_lines, ans_lines) > 0 else 1.0
            return MetricResult(name=self.name, score=round(ratio, 4),
                                reason="Line structure: {}/{} lines".format(ans_lines, ref_lines),
                                details={"ref_lines": ref_lines, "ans_lines": ans_lines})

        # Match elements
        found = 0
        results = []
        for r_type, r_val in ref_elements:
            matched = False
            for a_type, a_val in ans_elements:
                if r_type == a_type and _try_thefuzz(r_val, a_val) > 0.6:
                    matched = True
                    break
            results.append({"element": "{}: {}".format(r_type, r_val[:40]), "found": matched})
            if matched:
                found += 1

        total = len(ref_elements)
        score = found / total if total > 0 else 1.0

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} of {} layout elements preserved".format(found, total),
            details={"elements": results, "found": found, "total": total})

    def _check_with_llm(self, client, answer, reference):
        from jinja2 import Template
        try:
            template = Template(LAYOUT_LLM_PROMPT)
            prompt = template.render(answer=answer, reference=reference)
            result = client.generate_json(prompt, system="You are a document layout expert. Respond with valid JSON only.")
            score = result.get("score", 0.5)
            if isinstance(score, str):
                score = float(score)
            return MetricResult(name=self.name, score=round(max(0,min(1,score)), 4),
                                reason="{} of {} elements preserved".format(
                                    result.get("preserved_count",0), result.get("total_elements",0)),
                                details=result)
        except Exception as e:
            print("DocumentLayoutAccuracy LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


CONSISTENCY_LLM_PROMPT = """Check if these multimodal descriptions are consistent with each other.

Text description:
\"\"\"
{{ answer }}
\"\"\"

Other modality description (image caption / audio transcript / metadata):
\"\"\"
{{ reference }}
\"\"\"

Do they describe the same content? Any contradictions?

Respond with ONLY valid JSON:
{
    "consistent": true,
    "contradictions": [],
    "overlap_topics": ["dog", "park"],
    "score": 4
}

score 1-5: 1=completely contradictory, 5=fully consistent."""


class MultimodalConsistency:
    """Check if text and other modality descriptions are consistent.

    Compares text output against image captions, audio transcripts,
    or metadata from another modality.
    """

    name = "multimodal_consistency"

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
        reference = kwargs.get("reference", "") or kwargs.get("context", "")

        if not answer or not reference:
            return MetricResult(name=self.name, score=0.0,
                                reason="Need both text and reference modality description")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, reference)

        # Offline: keyword overlap + fuzzy matching
        stopwords = {'the','a','an','is','are','was','were','in','on','at','to',
                     'for','of','and','or','but','with','this','that','it','by','from','as'}

        a_words = set(answer.lower().split()) - stopwords
        r_words = set(reference.lower().split()) - stopwords

        if not r_words or not a_words:
            return MetricResult(name=self.name, score=0.0, reason="Text too short to compare")

        common = a_words & r_words
        overlap = len(common) / max(len(a_words), len(r_words))
        fuzzy = _try_thefuzz(answer, reference)
        score = (overlap * 0.5) + (fuzzy * 0.5)

        return MetricResult(
            name=self.name, score=round(min(1.0, score), 4),
            reason="Consistency: {:.0%} overlap, {:.0%} fuzzy".format(overlap, fuzzy),
            details={"keyword_overlap": round(overlap, 4), "fuzzy_score": round(fuzzy, 4),
                      "common_words": list(common)[:10]})

    def _check_with_llm(self, client, answer, reference):
        from jinja2 import Template
        try:
            template = Template(CONSISTENCY_LLM_PROMPT)
            prompt = template.render(answer=answer, reference=reference)
            result = client.generate_json(prompt, system="You are a multimodal consistency expert. Respond with valid JSON only.")
            raw = result.get("score", 3)
            if isinstance(raw, str):
                raw = float(raw)
            score = max(0.0, min(1.0, (raw - 1) / 4.0))
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="Consistent: {}".format(result.get("consistent", "unknown")),
                                details=result)
        except Exception as e:
            print("MultimodalConsistency LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
