"""Core hallucination detection metrics.

NegationHallucination: detects logical flips (approved vs not approved)
FabricatedInfo: detects statements with no evidence in context
ContradictionDetector: detects direct contradictions between output and context
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz


NEGATION_WORDS = {'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing',
                  'nowhere', 'nobody', "n't", "don't", "doesn't", "didn't",
                  "won't", "wouldn't", "can't", "cannot", "isn't", "aren't",
                  "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                  "shouldn't", "couldn't", "mustn't"}


def _split_sentences(text):
    """Simple sentence splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


def _has_negation(sentence):
    """Check if sentence contains negation."""
    words = set(sentence.lower().split())
    return bool(words & NEGATION_WORDS) or "n't" in sentence.lower()


def _sentence_similarity(s1, s2):
    """Check if two sentences are about the same topic."""
    w1 = set(s1.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    w2 = set(s2.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / min(len(w1), len(w2))


NEGATION_LLM_PROMPT = """Check if the output flips any negation from the context.
Example: context says "not approved" but output says "approved".

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
    "flips": [
        {"context_says": "is not approved", "output_says": "is approved", "severity": "high"}
    ],
    "flip_count": 1,
    "is_clean": false
}"""


class NegationHallucination:
    """Detect negation flips between context and output.

    "Is not approved" in context but "is approved" in output.
    """

    name = "negation_hallucination"

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
        if not context:
            return MetricResult(name=self.name, score=1.0, reason="No context provided")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, context)

        context_sents = _split_sentences(context)
        answer_sents = _split_sentences(answer)
        flips = []

        for a_sent in answer_sents:
            for c_sent in context_sents:
                sim = _sentence_similarity(a_sent, c_sent)
                if sim > 0.4:
                    a_neg = _has_negation(a_sent)
                    c_neg = _has_negation(c_sent)
                    if a_neg != c_neg:
                        flips.append({
                            "context_says": c_sent[:80],
                            "output_says": a_sent[:80],
                            "context_negated": c_neg,
                            "output_negated": a_neg,
                        })

        if not flips:
            return MetricResult(name=self.name, score=1.0,
                                reason="No negation flips detected", details={"flips": []})

        score = max(0.0, 1.0 - len(flips) * 0.25)
        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} negation flip(s) detected".format(len(flips)),
            details={"flips": flips, "flip_count": len(flips)})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(NEGATION_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a logical consistency expert. Respond with valid JSON only.")
            clean = result.get("is_clean", True)
            score = 1.0 if clean else max(0.0, 1.0 - result.get("flip_count", 1) * 0.25)
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="{} negation flips".format(result.get("flip_count", 0)), details=result)
        except Exception as e:
            print("NegationHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


FABRICATED_LLM_PROMPT = """Check each statement in the output. Is it supported by the context?

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
    "statements": [
        {"text": "statement text", "supported": true, "reason": "found in context"}
    ],
    "supported_count": 2,
    "unsupported_count": 1,
    "total": 3
}"""


class FabricatedInfo:
    """Detect fabricated information -- statements with no evidence in context.

    Splits output into sentences and checks each against the context.
    """

    name = "fabricated_info"

    def __init__(self, use_llm=False, threshold=0.25, weight=1.0):
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

        if not answer:
            return MetricResult(name=self.name, score=1.0, reason="Empty output")
        if not context:
            return MetricResult(name=self.name, score=1.0, reason="No context provided")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, context)

        answer_sents = _split_sentences(answer)
        if not answer_sents:
            return MetricResult(name=self.name, score=1.0, reason="No statements to check")

        context_lower = context.lower()
        context_words = set(context_lower.split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'to', 'for', 'of'}

        statements = []
        for sent in answer_sents:
            sent_words = set(sent.lower().split()) - {'the', 'a', 'an', 'is', 'are', 'was', 'in', 'on', 'to', 'for', 'of'}
            if not sent_words:
                continue
            overlap = len(sent_words & context_words) / len(sent_words)
            fuzzy = _try_thefuzz(sent, context) if len(sent) > 10 else overlap
            coverage = max(overlap, fuzzy)
            supported = coverage >= self.threshold
            statements.append({
                "text": sent[:100],
                "coverage": round(coverage, 3),
                "supported": supported,
            })

        if not statements:
            return MetricResult(name=self.name, score=1.0, reason="No statements to check")

        supported = sum(1 for s in statements if s["supported"])
        total = len(statements)
        score = supported / total if total > 0 else 1.0

        unsupported = [s for s in statements if not s["supported"]]
        if not unsupported:
            reason = "All {} statements supported by context".format(total)
        else:
            reason = "{} of {} statements may be fabricated".format(len(unsupported), total)

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"statements": statements, "supported": supported,
                      "unsupported": len(unsupported), "total": total})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(FABRICATED_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a fact verification expert. Respond with valid JSON only.")
            total = result.get("total", 1)
            supported = result.get("supported_count", 0)
            score = supported / total if total > 0 else 1.0
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="{} of {} statements supported".format(supported, total), details=result)
        except Exception as e:
            print("FabricatedInfo LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


CONTRADICTION_LLM_PROMPT = """Find direct contradictions between the output and context.

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
    "contradictions": [
        {"context_says": "...", "output_says": "...", "reason": "direct opposite"}
    ],
    "contradiction_count": 1,
    "is_consistent": false
}"""


class ContradictionDetector:
    """Detect direct contradictions between output and context."""

    name = "contradiction_detector"

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
        if not context:
            return MetricResult(name=self.name, score=1.0, reason="No context provided")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, context)

        # Offline: combine negation flips + antonym detection + numeric mismatch
        contradictions = []
        a_sents = _split_sentences(answer)
        c_sents = _split_sentences(context)

        antonyms = {
            'approved': 'rejected', 'accepted': 'denied', 'increase': 'decrease',
            'profit': 'loss', 'success': 'failure', 'safe': 'dangerous',
            'true': 'false', 'correct': 'incorrect', 'valid': 'invalid',
            'active': 'inactive', 'positive': 'negative', 'present': 'absent',
            'include': 'exclude', 'allow': 'prohibit', 'compliant': 'non-compliant',
        }
        antonyms.update({v: k for k, v in antonyms.items()})

        for a_sent in a_sents:
            for c_sent in c_sents:
                sim = _sentence_similarity(a_sent, c_sent)
                if sim < 0.3:
                    continue

                a_lower = a_sent.lower()
                c_lower = c_sent.lower()

                # Check antonyms
                for word, opposite in antonyms.items():
                    if word in a_lower and opposite in c_lower:
                        contradictions.append({
                            "context_says": c_sent[:80],
                            "output_says": a_sent[:80],
                            "type": "antonym",
                            "words": "{} vs {}".format(word, opposite),
                        })
                        break

                # Check negation flips
                a_neg = _has_negation(a_sent)
                c_neg = _has_negation(c_sent)
                if a_neg != c_neg and sim > 0.5:
                    already_found = any(c["output_says"] == a_sent[:80] for c in contradictions)
                    if not already_found:
                        contradictions.append({
                            "context_says": c_sent[:80],
                            "output_says": a_sent[:80],
                            "type": "negation_flip",
                        })

        if not contradictions:
            return MetricResult(name=self.name, score=1.0,
                                reason="No contradictions detected", details={"contradictions": []})

        score = max(0.0, 1.0 - len(contradictions) * 0.3)
        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} contradiction(s) detected".format(len(contradictions)),
            details={"contradictions": contradictions, "count": len(contradictions)})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(CONTRADICTION_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a contradiction detection expert. Respond with valid JSON only.")
            consistent = result.get("is_consistent", True)
            count = result.get("contradiction_count", 0)
            score = 1.0 if consistent else max(0.0, 1.0 - count * 0.3)
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="{} contradictions found".format(count), details=result)
        except Exception as e:
            print("ContradictionDetector LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
