"""Basic multimodal evaluation metrics.

OCRAccuracy: Word/character error rate for OCR outputs
AudioTranscriptionAccuracy: WER/CER for speech-to-text
ImageTextAlignment: checks if text matches image description
VisionQAAccuracy: checks if visual QA answer is correct

All work offline using string comparison. LLM mode adds
semantic understanding.
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult
from llmevalkit.doceval.field_accuracy import _try_thefuzz


def _word_error_rate(reference, hypothesis):
    """Calculate Word Error Rate (WER).
    WER = (S + D + I) / N where S=substitutions, D=deletions, I=insertions, N=ref words.
    Returns WER as a float (0.0 = perfect, 1.0 = all wrong).
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n = len(ref_words)
    m = len(hyp_words)
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,     # deletion
                    d[i][j-1] + 1,     # insertion
                    d[i-1][j-1] + 1,   # substitution
                )

    wer = d[n][m] / n
    return min(wer, 1.0)


def _character_error_rate(reference, hypothesis):
    """Calculate Character Error Rate (CER).
    Same as WER but at character level.
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    n = len(ref_chars)
    m = len(hyp_chars)
    d = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i-1] == hyp_chars[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+1)

    cer = d[n][m] / n
    return min(cer, 1.0)


MULTIMODAL_LLM_PROMPT = """Compare the {{ task_type }} output against the reference.

{{ task_type }} output (what the system produced):
\"\"\"
{{ answer }}
\"\"\"

Reference (what it should be):
\"\"\"
{{ reference }}
\"\"\"

{% if context %}
Additional context:
\"\"\"
{{ context }}
\"\"\"
{% endif %}

Respond with ONLY a valid JSON object:
{
    "score": 4,
    "is_correct": true,
    "errors": ["list of specific errors found"],
    "reason": "detailed assessment"
}

Score 1-5 where 1 = completely wrong, 5 = perfect match."""


class OCRAccuracy:
    """Evaluate OCR output accuracy using WER and CER.

    OCRAccuracy()              -- WER/CER calculation, free
    OCRAccuracy(use_llm=True)  -- adds LLM semantic comparison
    """
    name = "ocr_accuracy"

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
                                reason="Need both OCR output and reference text")

        wer = _word_error_rate(reference, answer)
        cer = _character_error_rate(reference, answer)
        accuracy = 1.0 - wer  # convert error rate to accuracy

        details = {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "word_accuracy": round(1.0 - wer, 4),
            "char_accuracy": round(1.0 - cer, 4),
        }

        if self.use_llm and client:
            llm_result = self._check_with_llm(client, answer, reference)
            if llm_result:
                details["llm_assessment"] = llm_result

        score = max(0.0, accuracy)
        reason = "WER: {:.1%}, CER: {:.1%}, Accuracy: {:.1%}".format(wer, cer, accuracy)

        return MetricResult(name=self.name, score=round(score, 4),
                            reason=reason, details=details)

    def _check_with_llm(self, client, answer, reference):
        from jinja2 import Template
        try:
            template = Template(MULTIMODAL_LLM_PROMPT)
            prompt = template.render(answer=answer, reference=reference, task_type="OCR", context="")
            system = "You are an OCR quality evaluation expert. Respond with valid JSON only."
            return client.generate_json(prompt, system=system)
        except Exception as e:
            print("OCRAccuracy LLM error: {}".format(e), file=sys.stderr)
            return None


class AudioTranscriptionAccuracy:
    """Evaluate speech-to-text transcription accuracy.

    Uses WER (Word Error Rate) and CER (Character Error Rate).
    Standard metrics in speech recognition evaluation.
    """
    name = "audio_transcription_accuracy"

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
                                reason="Need both transcription and reference text")

        wer = _word_error_rate(reference, answer)
        cer = _character_error_rate(reference, answer)
        accuracy = 1.0 - wer

        details = {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "word_accuracy": round(1.0 - wer, 4),
            "char_accuracy": round(1.0 - cer, 4),
            "ref_word_count": len(reference.split()),
            "hyp_word_count": len(answer.split()),
        }

        if self.use_llm and client:
            llm_result = self._check_with_llm(client, answer, reference)
            if llm_result:
                details["llm_assessment"] = llm_result

        score = max(0.0, accuracy)
        reason = "WER: {:.1%}, CER: {:.1%}, Accuracy: {:.1%}".format(wer, cer, accuracy)

        return MetricResult(name=self.name, score=round(score, 4),
                            reason=reason, details=details)

    def _check_with_llm(self, client, answer, reference):
        from jinja2 import Template
        try:
            template = Template(MULTIMODAL_LLM_PROMPT)
            prompt = template.render(answer=answer, reference=reference,
                                     task_type="Audio Transcription", context="")
            system = "You are a speech recognition evaluation expert. Respond with valid JSON only."
            return client.generate_json(prompt, system=system)
        except Exception as e:
            print("AudioTranscriptionAccuracy LLM error: {}".format(e), file=sys.stderr)
            return None


class ImageTextAlignment:
    """Check if generated text aligns with image description/caption.

    Without API: keyword overlap between caption and generated text.
    With API: LLM judges semantic alignment.
    """
    name = "image_text_alignment"

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
        context = kwargs.get("context", "")  # image description or caption

        if not answer or not context:
            return MetricResult(name=self.name, score=0.0,
                                reason="Need both generated text and image description")

        if not (self.use_llm and client):
            # Offline: keyword overlap + fuzzy matching
            answer_words = set(answer.lower().split())
            context_words = set(context.lower().split())
            stopwords = {'the','a','an','is','are','was','were','in','on','at','to','for',
                         'of','and','or','but','with','this','that','it','by','from','as'}
            answer_words -= stopwords
            context_words -= stopwords

            if not context_words:
                overlap = 0.0
            else:
                common = answer_words & context_words
                overlap = len(common) / len(context_words)

            fuzzy = _try_thefuzz(answer, context)
            score = (overlap * 0.6) + (fuzzy * 0.4)

            return MetricResult(
                name=self.name, score=round(score, 4),
                reason="Keyword overlap: {:.1%}, Fuzzy: {:.1%}".format(overlap, fuzzy),
                details={"keyword_overlap": round(overlap, 4), "fuzzy_score": round(fuzzy, 4)},
            )

        # LLM mode
        from jinja2 import Template
        try:
            template = Template(MULTIMODAL_LLM_PROMPT)
            prompt = template.render(answer=answer, reference="", context=context,
                                     task_type="Image-Text Alignment")
            system = "You are a multimodal evaluation expert. Respond with valid JSON only."
            result = client.generate_json(prompt, system=system)
            raw = result.get("score", 3)
            if isinstance(raw, str):
                raw = float(raw)
            normalized = max(0.0, min(1.0, (raw - 1) / 4.0))
            return MetricResult(
                name=self.name, score=round(normalized, 4),
                reason=result.get("reason", ""),
                details={"llm_result": result},
            )
        except Exception as e:
            print("ImageTextAlignment LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))


class VisionQAAccuracy:
    """Check if visual question answering response is correct.

    Without API: fuzzy match against expected answer.
    With API: LLM judges correctness.
    """
    name = "vision_qa_accuracy"

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
            return MetricResult(name=self.name, score=0.0,
                                reason="Need both VQA answer and expected answer")

        if not (self.use_llm and client):
            # Offline: exact match + fuzzy match
            exact = 1.0 if answer.lower().strip() == reference.lower().strip() else 0.0
            fuzzy = _try_thefuzz(answer, reference)
            score = max(exact, fuzzy)

            return MetricResult(
                name=self.name, score=round(score, 4),
                reason="Exact: {}, Fuzzy: {:.1%}".format("yes" if exact else "no", fuzzy),
                details={"exact_match": exact > 0, "fuzzy_score": round(fuzzy, 4)},
            )

        # LLM mode
        from jinja2 import Template
        try:
            template = Template(MULTIMODAL_LLM_PROMPT)
            prompt = template.render(answer=answer, reference=reference,
                                     task_type="Vision QA", context=kwargs.get("context", ""))
            system = "You are a visual question answering evaluation expert. Respond with valid JSON only."
            result = client.generate_json(prompt, system=system)
            raw = result.get("score", 3)
            if isinstance(raw, str):
                raw = float(raw)
            normalized = max(0.0, min(1.0, (raw - 1) / 4.0))
            return MetricResult(
                name=self.name, score=round(normalized, 4),
                reason=result.get("reason", ""),
                details={"llm_result": result},
            )
        except Exception as e:
            print("VisionQAAccuracy LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
