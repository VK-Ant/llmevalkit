"""Advanced hallucination detection metrics.

SelfConsistency: checks if multiple runs give same answer (no context needed)
ConfidenceCalibration: checks if confidence words match actual accuracy
InstructionHallucination: checks if output addresses the question/instruction
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz


HIGH_CONFIDENCE = {'definitely', 'certainly', 'absolutely', 'always', 'never',
                   'guaranteed', 'undoubtedly', 'without doubt', 'clearly',
                   'obviously', '100%', 'exactly', 'precisely', 'surely'}

LOW_CONFIDENCE = {'probably', 'maybe', 'perhaps', 'possibly', 'might',
                  'could be', 'i think', 'i believe', 'likely', 'unlikely',
                  'not sure', 'uncertain', 'approximately', 'roughly',
                  'around', 'it seems', 'appears to'}


class SelfConsistency:
    """Check if the same query produces consistent answers.

    No context needed. Pass multiple outputs for the same question.
    If outputs disagree, the model is hallucinating on some of them.

    Usage:
        sc = SelfConsistency()
        result = sc.evaluate(answer=[
            "Python was created in 1991.",
            "Python was created in 1989.",
            "Python was created in 1991.",
        ])
    """

    name = "self_consistency"

    def __init__(self, threshold=0.8, weight=1.0):
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
            return MetricResult(name=self.name, score=0.0,
                                reason="Need a list of 2+ outputs to compare")

        # Compare all pairs
        pair_scores = []
        for i in range(len(answer)):
            for j in range(i + 1, len(answer)):
                sim = _try_thefuzz(str(answer[i]), str(answer[j]))
                pair_scores.append({
                    "pair": "{}-{}".format(i+1, j+1),
                    "similarity": round(sim, 3),
                    "consistent": sim >= self.threshold,
                })

        avg_sim = sum(p["similarity"] for p in pair_scores) / len(pair_scores)
        inconsistent = [p for p in pair_scores if not p["consistent"]]

        if not inconsistent:
            reason = "All {} outputs are consistent (avg: {:.1%})".format(len(answer), avg_sim)
        else:
            reason = "{} of {} pairs inconsistent (avg: {:.1%})".format(
                len(inconsistent), len(pair_scores), avg_sim)

        return MetricResult(
            name=self.name, score=round(avg_sim, 4), reason=reason,
            details={"pairs": pair_scores, "avg_similarity": round(avg_sim, 4),
                      "num_outputs": len(answer), "consistent_pairs": len(pair_scores) - len(inconsistent)})


CONFIDENCE_LLM_PROMPT = """Analyze if the confidence level in this output matches its actual accuracy.

Context (source of truth):
\"\"\"
{{ context }}
\"\"\"

Output to check:
\"\"\"
{{ answer }}
\"\"\"

Check: does the output use high-confidence words ("definitely", "always") for incorrect info?
Or low-confidence words ("maybe", "probably") for correct info?

Respond with ONLY valid JSON:
{
    "confidence_issues": [
        {"text": "definitely costs $5M", "confidence": "high", "accurate": false, "reason": "context says $3M"}
    ],
    "calibration_score": 3,
    "summary": "overconfident on one claim"
}

calibration_score 1-5: 1=badly miscalibrated, 5=well calibrated."""


class ConfidenceCalibration:
    """Check if confidence words match actual accuracy.

    Detects overconfidence ("definitely" + wrong answer) and
    underconfidence ("maybe" + correct answer).
    """

    name = "confidence_calibration"

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

        if self.use_llm and client and context:
            return self._check_with_llm(client, answer, context)

        # Offline: detect confidence signals
        answer_lower = answer.lower()

        high_found = [w for w in HIGH_CONFIDENCE if w in answer_lower]
        low_found = [w for w in LOW_CONFIDENCE if w in answer_lower]

        if not high_found and not low_found:
            return MetricResult(name=self.name, score=1.0,
                                reason="No strong confidence signals detected",
                                details={"high_confidence": [], "low_confidence": []})

        # Without context, we can only flag the signals
        if not context:
            score = 1.0
            reason = "Confidence signals found but no context to verify accuracy"
            details = {"high_confidence": high_found, "low_confidence": low_found,
                        "note": "Provide context for full calibration check"}
            return MetricResult(name=self.name, score=score, reason=reason, details=details)

        # With context: check if high-confidence statements are actually correct
        context_lower = context.lower()
        issues = []

        for word in high_found:
            idx = answer_lower.find(word)
            if idx >= 0:
                # Extract surrounding sentence
                start = max(0, answer_lower.rfind('.', 0, idx) + 1)
                end = answer_lower.find('.', idx)
                if end == -1:
                    end = len(answer_lower)
                sentence = answer[start:end].strip()
                # Check if sentence content is in context
                sent_words = set(sentence.lower().split()) - {'the','a','an','is','are','was'}
                ctx_words = set(context_lower.split())
                overlap = len(sent_words & ctx_words) / max(len(sent_words), 1)
                if overlap < 0.3:
                    issues.append({
                        "text": sentence[:80],
                        "confidence": "high",
                        "word": word,
                        "context_support": round(overlap, 2),
                        "issue": "high confidence but low context support",
                    })

        if not issues:
            score = 1.0
            reason = "Confidence signals are calibrated with context"
        else:
            score = max(0.0, 1.0 - len(issues) * 0.25)
            reason = "{} overconfident statement(s) detected".format(len(issues))

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"issues": issues, "high_confidence": high_found,
                      "low_confidence": low_found})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(CONFIDENCE_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a calibration analysis expert. Respond with valid JSON only.")
            raw = result.get("calibration_score", 3)
            if isinstance(raw, str):
                raw = float(raw)
            score = max(0.0, min(1.0, (raw - 1) / 4.0))
            return MetricResult(name=self.name, score=round(score, 4),
                                reason=result.get("summary", ""), details=result)
        except Exception as e:
            print("ConfidenceCalibration LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


INSTRUCTION_LLM_PROMPT = """Check if the output actually answers the question/instruction.

Question/Instruction:
\"\"\"
{{ question }}
\"\"\"

Output:
\"\"\"
{{ answer }}
\"\"\"

Does the output address what was asked? Or does it talk about something else?

Respond with ONLY valid JSON:
{
    "addresses_question": true,
    "topic_match": 0.9,
    "issues": ["partially answers but misses the main point"],
    "score": 4
}

score 1-5: 1=completely off topic, 5=directly answers the question."""


class InstructionHallucination:
    """Check if output addresses the question/instruction.

    Detects when the model answers a different question than asked
    or goes off-topic.
    """

    name = "instruction_hallucination"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer", "question"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer")) and bool(kwargs.get("question"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")
        question = kwargs.get("question", "")

        if not answer or not question:
            return MetricResult(name=self.name, score=1.0,
                                reason="Need both question and answer")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, question)

        # Offline: keyword overlap between question and answer
        stopwords = {'what','how','why','when','where','who','which','is','are','was',
                     'were','do','does','did','the','a','an','in','on','at','to','for',
                     'of','and','or','but','can','could','would','should','will','have',
                     'has','had','be','been','being','this','that','it','with','from','by',
                     'about','me','my','i','you','your','we','our','they','their'}

        q_words = set(question.lower().split()) - stopwords
        a_words = set(answer.lower().split()) - stopwords

        if not q_words:
            return MetricResult(name=self.name, score=1.0, reason="Question too short to analyze")

        # How many question keywords appear in answer?
        covered = len(q_words & a_words)
        coverage = covered / len(q_words) if q_words else 0.0

        # Fuzzy check
        fuzzy = _try_thefuzz(question, answer[:200])

        score = (coverage * 0.7) + (fuzzy * 0.3)
        score = min(1.0, score)

        if score >= 0.5:
            reason = "Output addresses the question ({:.0%} topic match)".format(score)
        elif score >= 0.2:
            reason = "Output partially addresses the question ({:.0%} match)".format(score)
        else:
            reason = "Output may not address the question ({:.0%} match)".format(score)

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"question_keywords": list(q_words)[:10],
                      "covered_keywords": list(q_words & a_words)[:10],
                      "coverage": round(coverage, 3), "fuzzy": round(fuzzy, 3)})

    def _check_with_llm(self, client, answer, question):
        from jinja2 import Template
        try:
            template = Template(INSTRUCTION_LLM_PROMPT)
            prompt = template.render(answer=answer, question=question)
            result = client.generate_json(prompt, system="You are an instruction-following evaluation expert. Respond with valid JSON only.")
            raw = result.get("score", 3)
            if isinstance(raw, str):
                raw = float(raw)
            score = max(0.0, min(1.0, (raw - 1) / 4.0))
            return MetricResult(name=self.name, score=round(score, 4),
                                reason="Addresses question: {}".format(result.get("addresses_question", "unknown")),
                                details=result)
        except Exception as e:
            print("InstructionHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
