"""Additional hallucination detection metrics.

SourceCoverage: what percentage of the output is supported by context
TemporalHallucination: wrong dates, timelines, sequences
CausalHallucination: wrong cause-effect relationships
RankingHallucination: wrong ordering, ranking, comparisons
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz


def _split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]


SOURCE_COVERAGE_LLM_PROMPT = """For each sentence in the output, determine if it is supported by the context.

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
    "sentences": [
        {"text": "sentence text", "supported": true, "coverage": 0.9, "reason": "found in context"}
    ],
    "overall_coverage": 0.75,
    "unsupported_count": 1,
    "total": 4
}"""


class SourceCoverage:
    """Measure what percentage of the output is grounded in context.

    Unlike FabricatedInfo which flags unsupported statements,
    SourceCoverage gives a continuous coverage score showing
    how much of the output traces back to the source.

    SourceCoverage()              -- keyword overlap per sentence, free
    SourceCoverage(use_llm=True)  -- LLM judges each sentence
    """

    name = "source_coverage"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
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
                                reason="Need both output and context")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, context)

        sentences = _split_sentences(answer)
        if not sentences:
            return MetricResult(name=self.name, score=1.0, reason="No sentences to check")

        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                     'to', 'for', 'of', 'and', 'or', 'but', 'with', 'this', 'that', 'it', 'by'}
        context_words = set(context.lower().split()) - stopwords

        sentence_scores = []
        for sent in sentences:
            sent_words = set(sent.lower().split()) - stopwords
            if not sent_words:
                sentence_scores.append({"text": sent[:80], "coverage": 1.0})
                continue
            overlap = len(sent_words & context_words) / len(sent_words)
            sentence_scores.append({"text": sent[:80], "coverage": round(overlap, 3)})

        avg_coverage = sum(s["coverage"] for s in sentence_scores) / len(sentence_scores)
        low = [s for s in sentence_scores if s["coverage"] < 0.3]

        reason = "Source coverage: {:.0%} ({} of {} sentences well-grounded)".format(
            avg_coverage, len(sentence_scores) - len(low), len(sentence_scores))

        return MetricResult(
            name=self.name, score=round(avg_coverage, 4), reason=reason,
            details={"sentences": sentence_scores, "avg_coverage": round(avg_coverage, 4),
                      "low_coverage_count": len(low), "total_sentences": len(sentence_scores)})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(SOURCE_COVERAGE_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a source verification expert. Respond with valid JSON only.")
            score = result.get("overall_coverage", 0.5)
            if isinstance(score, str):
                score = float(score)
            return MetricResult(name=self.name, score=round(max(0, min(1, score)), 4),
                                reason="{} of {} sentences supported".format(
                                    result.get("total", 0) - result.get("unsupported_count", 0),
                                    result.get("total", 0)),
                                details=result)
        except Exception as e:
            print("SourceCoverage LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


TEMPORAL_LLM_PROMPT = """Check if dates, timelines, and temporal sequences in the output match the context.

Context:
\"\"\"
{{ context }}
\"\"\"

Output:
\"\"\"
{{ answer }}
\"\"\"

Check for: wrong dates, wrong year, wrong order of events, wrong durations.

Respond with ONLY valid JSON:
{
    "temporal_checks": [
        {"output_says": "founded in 2010", "context_says": "founded in 2015", "match": false}
    ],
    "correct_count": 2,
    "wrong_count": 1,
    "total": 3
}"""


class TemporalHallucination:
    """Detect wrong dates, timelines, and temporal sequences.

    "Company founded in 2010" when context says "founded in 2015".
    "Treatment takes 6 months" when context says "3-month period".
    """

    name = "temporal_hallucination"

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

        # Extract temporal expressions
        date_patterns = [
            r'\b\d{4}\b',  # years
            r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',  # dates
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b',  # durations
            r'\b(?:before|after|during|since|until|from|to)\s+\d{4}\b',  # temporal refs
        ]

        answer_dates = set()
        context_dates = set()

        for pattern in date_patterns:
            answer_dates.update(m.group() for m in re.finditer(pattern, answer, re.IGNORECASE))
            context_dates.update(m.group() for m in re.finditer(pattern, context, re.IGNORECASE))

        if not answer_dates:
            return MetricResult(name=self.name, score=1.0,
                                reason="No temporal expressions found",
                                details={"temporal_checked": 0})

        checks = []
        for a_date in answer_dates:
            found = False
            for c_date in context_dates:
                if a_date.lower() == c_date.lower():
                    checks.append({"output": a_date, "context": c_date, "match": True})
                    found = True
                    break
                # Check year match
                a_years = re.findall(r'\d{4}', a_date)
                c_years = re.findall(r'\d{4}', c_date)
                if a_years and c_years and a_years[0] == c_years[0]:
                    checks.append({"output": a_date, "context": c_date, "match": True})
                    found = True
                    break
            if not found:
                # Check if the date appears anywhere in context
                if a_date.lower() in context.lower():
                    checks.append({"output": a_date, "match": True, "reason": "found in context"})
                else:
                    checks.append({"output": a_date, "match": False, "reason": "not in context"})

        correct = sum(1 for c in checks if c["match"])
        total = len(checks)
        score = correct / total if total > 0 else 1.0

        wrong = [c for c in checks if not c["match"]]
        if not wrong:
            reason = "All {} temporal references match context".format(total)
        else:
            reason = "{} of {} temporal references wrong".format(len(wrong), total)

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"checks": checks, "correct": correct, "wrong": len(wrong), "total": total})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(TEMPORAL_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a temporal fact-checking expert. Respond with valid JSON only.")
            total = result.get("total", 1)
            correct = result.get("correct_count", 0)
            score = correct / total if total > 0 else 1.0
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="{} of {} temporal refs correct".format(correct, total), details=result)
        except Exception as e:
            print("TemporalHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


CAUSAL_LLM_PROMPT = """Check if cause-effect relationships in the output match the context.

Context:
\"\"\"
{{ context }}
\"\"\"

Output:
\"\"\"
{{ answer }}
\"\"\"

Look for: "X caused Y", "because of X", "X leads to Y", "due to X", "as a result of X".
Are these causal claims supported by the context?

Respond with ONLY valid JSON:
{
    "causal_checks": [
        {"claim": "rain caused flooding", "supported": true, "reason": "context confirms"}
    ],
    "correct": 2,
    "wrong": 1,
    "total": 3
}"""

CAUSAL_MARKERS = [
    r'\b(?:because|caused?|due\s+to|leads?\s+to|results?\s+in|as\s+a\s+result)',
    r'\b(?:therefore|consequently|hence|thus|so\s+that|in\s+order\s+to)',
    r'\b(?:if\s+.+\s+then|since|owing\s+to|thanks\s+to|on\s+account\s+of)',
    r'\b(?:responsible\s+for|contributes?\s+to|triggers?|prevents?|enables?)',
]


class CausalHallucination:
    """Detect wrong cause-effect relationships.

    "Drug X causes weight loss" when context says "Drug X causes weight gain".
    """

    name = "causal_hallucination"

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

        # Find sentences with causal language
        sentences = _split_sentences(answer)
        causal_sentences = []
        for sent in sentences:
            for pattern in CAUSAL_MARKERS:
                if re.search(pattern, sent, re.IGNORECASE):
                    causal_sentences.append(sent)
                    break

        if not causal_sentences:
            return MetricResult(name=self.name, score=1.0,
                                reason="No causal claims found",
                                details={"causal_claims": 0})

        # Check if causal sentences are grounded in context
        context_lower = context.lower()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'to', 'for', 'of', 'and', 'or', 'in', 'on'}
        checks = []
        for sent in causal_sentences:
            sent_words = set(sent.lower().split()) - stopwords
            ctx_words = set(context_lower.split()) - stopwords
            overlap = len(sent_words & ctx_words) / max(len(sent_words), 1)
            fuzzy = _try_thefuzz(sent, context) if len(sent) > 15 else overlap
            coverage = max(overlap, fuzzy)
            checks.append({
                "claim": sent[:80],
                "supported": coverage >= 0.3,
                "coverage": round(coverage, 3),
            })

        supported = sum(1 for c in checks if c["supported"])
        total = len(checks)
        score = supported / total if total > 0 else 1.0

        unsupported = [c for c in checks if not c["supported"]]
        if not unsupported:
            reason = "All {} causal claims supported".format(total)
        else:
            reason = "{} of {} causal claims unsupported".format(len(unsupported), total)

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"checks": checks, "supported": supported,
                      "unsupported": len(unsupported), "total": total})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(CAUSAL_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a causal reasoning expert. Respond with valid JSON only.")
            total = result.get("total", 1)
            correct = result.get("correct", 0)
            score = correct / total if total > 0 else 1.0
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="{} of {} causal claims correct".format(correct, total), details=result)
        except Exception as e:
            print("CausalHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


RANKING_LLM_PROMPT = """Check if rankings, orderings, or comparisons in the output match the context.

Context:
\"\"\"
{{ context }}
\"\"\"

Output:
\"\"\"
{{ answer }}
\"\"\"

Check: "X is larger than Y", "first...then...finally", "best/worst/most/least",
rankings, ordered lists, superlatives.

Respond with ONLY valid JSON:
{
    "ranking_checks": [
        {"claim": "A is better than B", "supported": true, "reason": "context confirms"}
    ],
    "correct": 1,
    "wrong": 1,
    "total": 2
}"""

RANKING_MARKERS = [
    r'\b(?:first|second|third|fourth|fifth|last|final)\b',
    r'\b(?:best|worst|most|least|largest|smallest|highest|lowest|fastest|slowest)\b',
    r'\b(?:better|worse|more|less|greater|fewer|higher|lower)\s+than\b',
    r'\b(?:ranked|ranking|top|bottom|leading|trailing)\b',
    r'\b(?:number\s+(?:one|two|three|1|2|3))\b',
    r'\b(?:primary|secondary|main|major|minor)\b',
]


class RankingHallucination:
    """Detect wrong orderings, rankings, and comparisons.

    "Company A is the largest" when context says "Company B is the largest".
    """

    name = "ranking_hallucination"

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

        sentences = _split_sentences(answer)
        ranking_sentences = []
        for sent in sentences:
            for pattern in RANKING_MARKERS:
                if re.search(pattern, sent, re.IGNORECASE):
                    ranking_sentences.append(sent)
                    break

        if not ranking_sentences:
            return MetricResult(name=self.name, score=1.0,
                                reason="No ranking/comparison claims found",
                                details={"ranking_claims": 0})

        context_lower = context.lower()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'to', 'for', 'of', 'and', 'or', 'in', 'on'}
        checks = []
        for sent in ranking_sentences:
            sent_words = set(sent.lower().split()) - stopwords
            ctx_words = set(context_lower.split()) - stopwords
            overlap = len(sent_words & ctx_words) / max(len(sent_words), 1)
            fuzzy = _try_thefuzz(sent, context) if len(sent) > 15 else overlap
            coverage = max(overlap, fuzzy)
            checks.append({
                "claim": sent[:80],
                "supported": coverage >= 0.3,
                "coverage": round(coverage, 3),
            })

        supported = sum(1 for c in checks if c["supported"])
        total = len(checks)
        score = supported / total if total > 0 else 1.0

        unsupported = [c for c in checks if not c["supported"]]
        if not unsupported:
            reason = "All {} ranking claims supported".format(total)
        else:
            reason = "{} of {} ranking claims unsupported".format(len(unsupported), total)

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"checks": checks, "supported": supported, "total": total})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(RANKING_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a fact-checking expert for rankings and comparisons. Respond with valid JSON only.")
            total = result.get("total", 1)
            correct = result.get("correct", 0)
            score = correct / total if total > 0 else 1.0
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="{} of {} ranking claims correct".format(correct, total), details=result)
        except Exception as e:
            print("RankingHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
