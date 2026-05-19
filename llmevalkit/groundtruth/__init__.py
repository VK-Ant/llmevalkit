"""llmevalkit ground truth testing module.

Reference-based evaluation: compare LLM output against known correct answers.

ExactMatchAccuracy: exact string match
FuzzyMatchAccuracy: fuzzy string match with Levenshtein
GroundTruthF1: token-level F1 score
ContextualPrecision: are relevant docs ranked higher?
ContextualRecall: does context cover the expected output?
JSONCorrectness: is output valid JSON matching expected schema?
"""

from __future__ import annotations

import json
import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz, _fuzzy_ratio


GT_LLM_PROMPT = """Compare the actual output against the expected output.

Expected (ground truth):
\"\"\"
{{ reference }}
\"\"\"

Actual output:
\"\"\"
{{ answer }}
\"\"\"

Respond with ONLY valid JSON:
{
    "match_score": 0.85,
    "correct_parts": ["part1"],
    "incorrect_parts": ["part2"],
    "reason": "mostly correct but missed X"
}"""


class ExactMatchAccuracy:
    """Does the answer exactly match the ground truth?

    Case-insensitive by default. Strips whitespace.
    """
    name = "exact_match_accuracy"

    def __init__(self, case_sensitive=False, weight=1.0):
        self.case_sensitive = case_sensitive
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer", "reference"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer")) and bool(kwargs.get("reference"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "").strip()
        reference = kwargs.get("reference", "").strip()

        if not answer or not reference:
            return MetricResult(name=self.name, score=0.0, reason="Need both answer and ground truth")

        if self.case_sensitive:
            match = answer == reference
        else:
            match = answer.lower() == reference.lower()

        return MetricResult(
            name=self.name, score=1.0 if match else 0.0,
            reason="Exact match" if match else "Does not match",
            details={"exact_match": match, "answer_length": len(answer), "reference_length": len(reference)})


class FuzzyMatchAccuracy:
    """How close is the answer to ground truth using fuzzy matching?"""
    name = "fuzzy_match_accuracy"

    def __init__(self, threshold=0.8, weight=1.0):
        self.threshold = threshold
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer", "reference"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer")) and bool(kwargs.get("reference"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "").strip()
        reference = kwargs.get("reference", "").strip()

        if not answer or not reference:
            return MetricResult(name=self.name, score=0.0, reason="Need both answer and ground truth")

        score = _try_thefuzz(answer, reference)
        passed = score >= self.threshold

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="Fuzzy match: {:.1%}{}".format(score, " (passed)" if passed else " (below threshold)"),
            details={"fuzzy_score": round(score, 4), "threshold": self.threshold, "passed": passed})


class GroundTruthF1:
    """Token-level F1 score against ground truth.

    Precision: what fraction of answer tokens are in ground truth.
    Recall: what fraction of ground truth tokens are in answer.
    F1: harmonic mean.
    """
    name = "ground_truth_f1"

    def __init__(self, weight=1.0):
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
            return MetricResult(name=self.name, score=0.0, reason="Need both answer and ground truth")

        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                     'to', 'for', 'of', 'and', 'or', 'but', 'with', 'this', 'that', 'it'}

        a_tokens = set(answer.lower().split()) - stopwords
        r_tokens = set(reference.lower().split()) - stopwords

        if not a_tokens or not r_tokens:
            return MetricResult(name=self.name, score=0.0, reason="Empty tokens after stopword removal")

        common = a_tokens & r_tokens
        precision = len(common) / len(a_tokens) if a_tokens else 0.0
        recall = len(common) / len(r_tokens) if r_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            name=self.name, score=round(f1, 4),
            reason="F1: {:.3f} (P: {:.3f}, R: {:.3f})".format(f1, precision, recall),
            details={"f1": round(f1, 4), "precision": round(precision, 4),
                      "recall": round(recall, 4), "common_tokens": len(common)})


class ContextualPrecision:
    """Are relevant documents ranked higher in retrieval?

    Checks if context chunks that contain ground truth info
    appear earlier than irrelevant chunks.
    """
    name = "contextual_precision"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer", "context"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("context"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")
        context = kwargs.get("context", "")
        reference = kwargs.get("reference", "") or answer

        if not context:
            return MetricResult(name=self.name, score=0.0, reason="No context provided")

        if self.use_llm and client:
            return self._check_with_llm(client, context, reference)

        # Split context into chunks (by paragraphs or double newlines)
        chunks = [c.strip() for c in re.split(r'\n\n+|\n', context) if c.strip() and len(c.strip()) > 10]

        if not chunks:
            chunks = [context]

        if len(chunks) == 1:
            # Single chunk, check relevance
            overlap = self._overlap(chunks[0], reference)
            return MetricResult(name=self.name, score=round(overlap, 4),
                                reason="Single context chunk, relevance: {:.1%}".format(overlap),
                                details={"chunks": 1, "relevance": round(overlap, 4)})

        # Score each chunk by relevance to reference
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            overlap = self._overlap(chunk, reference)
            chunk_scores.append({"rank": i + 1, "relevance": round(overlap, 3), "text": chunk[:60]})

        # Precision: are relevant chunks ranked higher?
        relevant = [c for c in chunk_scores if c["relevance"] > 0.3]
        if not relevant:
            return MetricResult(name=self.name, score=0.0,
                                reason="No relevant chunks found", details={"chunks": chunk_scores})

        # Average position of relevant chunks (lower = better)
        avg_rank = sum(c["rank"] for c in relevant) / len(relevant)
        ideal_avg = (len(relevant) + 1) / 2
        total = len(chunks)

        score = max(0.0, 1.0 - (avg_rank - ideal_avg) / total)

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} relevant of {} chunks, avg rank: {:.1f}".format(len(relevant), total, avg_rank),
            details={"chunks": chunk_scores, "relevant_count": len(relevant), "avg_rank": round(avg_rank, 1)})

    def _overlap(self, text1, text2):
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'to', 'for', 'of', 'and', 'or'}
        w1 = set(text1.lower().split()) - stopwords
        w2 = set(text2.lower().split()) - stopwords
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / max(len(w1), len(w2))

    def _check_with_llm(self, client, context, reference):
        try:
            prompt = "Rate how well this context supports the expected answer.\n\nContext:\n{}\n\nExpected:\n{}\n\nRespond JSON: {{\"score\": 0.8, \"reason\": \"...\"}}".format(context[:1500], reference[:500])
            result = client.generate_json(prompt, system="You are a retrieval evaluation expert. Respond with valid JSON only.")
            score = result.get("score", 0.5)
            if isinstance(score, str): score = float(score)
            return MetricResult(name=self.name, score=round(max(0, min(1, score)), 4),
                                reason=result.get("reason", ""), details=result)
        except Exception as e:
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


class ContextualRecall:
    """Does the context cover the expected output?

    Checks how much of the ground truth answer can be found
    in the retrieved context.
    """
    name = "contextual_recall"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["context", "reference"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("context")) and bool(kwargs.get("reference"))

    def evaluate(self, client=None, **kwargs):
        context = kwargs.get("context", "")
        reference = kwargs.get("reference", "")

        if not context or not reference:
            return MetricResult(name=self.name, score=0.0, reason="Need both context and ground truth")

        if self.use_llm and client:
            return self._check_with_llm(client, context, reference)

        # Split reference into sentences, check each against context
        sentences = re.split(r'(?<=[.!?])\s+', reference.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

        if not sentences:
            # Fall back to token overlap
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'to', 'for', 'of', 'and', 'or'}
            ref_tokens = set(reference.lower().split()) - stopwords
            ctx_tokens = set(context.lower().split()) - stopwords
            recall = len(ref_tokens & ctx_tokens) / len(ref_tokens) if ref_tokens else 0.0
            return MetricResult(name=self.name, score=round(recall, 4),
                                reason="Token recall: {:.1%}".format(recall))

        ctx_lower = context.lower()
        covered = 0
        results = []
        for sent in sentences:
            # Check if sentence content is in context
            sent_words = set(sent.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'to', 'for', 'of'}
            ctx_words = set(ctx_lower.split())
            overlap = len(sent_words & ctx_words) / max(len(sent_words), 1)
            found = overlap >= 0.4
            if found:
                covered += 1
            results.append({"text": sent[:60], "covered": found, "overlap": round(overlap, 3)})

        recall = covered / len(sentences) if sentences else 0.0

        return MetricResult(
            name=self.name, score=round(recall, 4),
            reason="{} of {} ground truth sentences covered by context".format(covered, len(sentences)),
            details={"sentences": results, "covered": covered, "total": len(sentences)})

    def _check_with_llm(self, client, context, reference):
        try:
            prompt = "How much of the expected answer is covered by the context?\n\nContext:\n{}\n\nExpected:\n{}\n\nRespond JSON: {{\"recall\": 0.8, \"covered_parts\": [], \"missing_parts\": []}}".format(context[:1500], reference[:500])
            result = client.generate_json(prompt, system="You are a recall evaluation expert. Respond with valid JSON only.")
            score = result.get("recall", 0.5)
            if isinstance(score, str): score = float(score)
            return MetricResult(name=self.name, score=round(max(0, min(1, score)), 4),
                                reason="LLM recall: {:.1%}".format(score), details=result)
        except Exception as e:
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


class JSONCorrectness:
    """Is the output valid JSON matching expected schema?

    Checks: valid JSON syntax, required keys present,
    value types match expected types.
    """
    name = "json_correctness"

    def __init__(self, required_keys=None, schema=None, weight=1.0):
        self.required_keys = required_keys or []
        self.schema = schema  # dict of key: expected_type ("str", "int", "float", "bool", "list", "dict")
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "").strip()

        if not answer:
            return MetricResult(name=self.name, score=0.0, reason="Empty output")

        # Check valid JSON
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError as e:
            return MetricResult(name=self.name, score=0.0,
                                reason="Invalid JSON: {}".format(str(e)[:60]),
                                details={"valid_json": False, "error": str(e)})

        issues = []
        checks_total = 1  # JSON validity itself
        checks_passed = 1

        # Check required keys
        if self.required_keys and isinstance(parsed, dict):
            for key in self.required_keys:
                checks_total += 1
                if key in parsed:
                    checks_passed += 1
                else:
                    issues.append("missing key: {}".format(key))

        # Check schema types
        if self.schema and isinstance(parsed, dict):
            type_map = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}
            for key, expected_type in self.schema.items():
                checks_total += 1
                if key in parsed:
                    expected = type_map.get(expected_type)
                    if expected and isinstance(parsed[key], expected):
                        checks_passed += 1
                    elif expected:
                        issues.append("{}: expected {}, got {}".format(key, expected_type, type(parsed[key]).__name__))
                    else:
                        checks_passed += 1
                else:
                    issues.append("missing key for type check: {}".format(key))

        score = checks_passed / checks_total if checks_total > 0 else 0.0

        if not issues:
            reason = "Valid JSON, all {} checks passed".format(checks_total)
        else:
            reason = "{} of {} checks passed. Issues: {}".format(checks_passed, checks_total, "; ".join(issues[:3]))

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"valid_json": True, "checks_passed": checks_passed,
                      "checks_total": checks_total, "issues": issues})


__all__ = [
    "ExactMatchAccuracy", "FuzzyMatchAccuracy", "GroundTruthF1",
    "ContextualPrecision", "ContextualRecall", "JSONCorrectness",
]
