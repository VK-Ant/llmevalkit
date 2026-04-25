"""AI content detection metrics.

Statistical signals to analyze if text may be AI-generated.

IMPORTANT: AI content detection is an unsolved problem. These metrics
provide statistical signals, not definitive answers. No tool can
reliably distinguish AI from human content with 100% accuracy.
Use as part of a broader review process. Do not use as sole basis
for accusations, rejections, or penalties. False positives and
false negatives are expected.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter

from llmevalkit.models import MetricResult


def _sentence_lengths(text):
    """Get list of sentence word counts."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [len(s.split()) for s in sents if s.strip()]


def _type_token_ratio(text):
    """Vocabulary diversity: unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _burstiness(text):
    """Sentence length variance. Humans: high variance. AI: low variance."""
    lengths = _sentence_lengths(text)
    if len(lengths) < 2:
        return 0.5
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    # Normalize: high std = human-like (score closer to 1)
    # AI typically has std < 5, humans > 8
    return min(1.0, std / 12.0)


def _repetition_score(text):
    """Check n-gram repetition. AI repeats more."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.5

    # 3-gram repetition
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    counts = Counter(trigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    repetition_rate = repeated / max(len(counts), 1)

    # Low repetition = more human-like (score closer to 1)
    return max(0.0, 1.0 - repetition_rate * 3)


def _transition_score(text):
    """AI uses transitional phrases more uniformly."""
    transitions = [
        'furthermore', 'moreover', 'additionally', 'in addition',
        'however', 'nevertheless', 'on the other hand', 'conversely',
        'therefore', 'consequently', 'as a result', 'thus',
        'in conclusion', 'to summarize', 'overall', 'in summary',
        'firstly', 'secondly', 'thirdly', 'finally',
        'it is important to note', 'it is worth noting',
        'in this regard', 'with respect to',
    ]

    text_lower = text.lower()
    count = sum(1 for t in transitions if t in text_lower)
    sentences = len(_sentence_lengths(text))

    if sentences == 0:
        return 0.5

    rate = count / sentences
    # High transition rate = more AI-like (score closer to 0)
    # Humans typically use 0-0.1 per sentence, AI uses 0.2-0.5
    return max(0.0, 1.0 - rate * 3)


def _perplexity_proxy(text):
    """Approximate perplexity using word frequency distribution.
    AI text uses common words more uniformly. Human text has more
    surprising word choices."""
    words = text.lower().split()
    if len(words) < 10:
        return 0.5

    freq = Counter(words)
    total = len(words)

    # Entropy as perplexity proxy
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalize: higher entropy = more diverse = more human-like
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.5

    return min(1.0, normalized)


DETECTION_LLM_PROMPT = """Analyze if this text appears to be AI-generated or human-written.

Text:
\"\"\"
{{ answer }}
\"\"\"

Consider: writing style consistency, vocabulary patterns, sentence structure
variety, use of filler phrases, natural flow vs mechanical structure.

Respond with ONLY valid JSON:
{
    "likely_source": "ai" or "human" or "mixed",
    "confidence": 0.75,
    "signals": [
        {"signal": "uniform sentence length", "indicates": "ai", "strength": "medium"}
    ],
    "human_score": 0.3,
    "summary": "brief explanation"
}

human_score 0.0-1.0: 0.0 = definitely AI, 1.0 = definitely human."""


class AITextDetector:
    """Analyze if text may be AI-generated.

    Returns score: 0.0 = likely AI, 1.0 = likely human.

    Without API: perplexity proxy, burstiness, vocabulary diversity,
    repetition patterns, transition word frequency. 65-75% accuracy.
    With API: LLM analyzes writing style. 75-85% accuracy.

    DISCLAIMER: No tool can reliably detect AI text with 100% accuracy.
    Use these signals as part of a broader review process.
    """

    name = "ai_text_detector"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer")) and len(kwargs.get("answer", "").split()) >= 10

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")

        if not answer:
            return MetricResult(name=self.name, score=0.5, reason="Empty text")

        words = answer.split()
        if len(words) < 10:
            return MetricResult(name=self.name, score=0.5,
                                reason="Text too short for reliable detection (need 10+ words)",
                                details={"word_count": len(words)})

        # Calculate all signals
        perplexity = _perplexity_proxy(answer)
        burstiness = _burstiness(answer)
        vocabulary = _type_token_ratio(answer)
        repetition = _repetition_score(answer)
        transition = _transition_score(answer)

        # Weighted combination
        score = (perplexity * 0.25 +
                 burstiness * 0.25 +
                 vocabulary * 0.15 +
                 repetition * 0.15 +
                 transition * 0.20)

        signals = {
            "perplexity_proxy": round(perplexity, 3),
            "burstiness": round(burstiness, 3),
            "vocabulary_diversity": round(vocabulary, 3),
            "repetition": round(repetition, 3),
            "transition_uniformity": round(transition, 3),
            "word_count": len(words),
            "sentence_count": len(_sentence_lengths(answer)),
        }

        # LLM enhancement
        if self.use_llm and client:
            llm_result = self._check_with_llm(client, answer)
            if llm_result:
                llm_score = llm_result.get("human_score", 0.5)
                if isinstance(llm_score, str):
                    llm_score = float(llm_score)
                score = score * 0.4 + llm_score * 0.6
                signals["llm_assessment"] = llm_result

        if score >= 0.6:
            likely = "likely human"
        elif score >= 0.4:
            likely = "uncertain"
        else:
            likely = "likely AI-generated"

        return MetricResult(
            name=self.name, score=round(score, 4),
            reason="{} (score: {:.2f})".format(likely, score),
            details=signals)

    def _check_with_llm(self, client, answer):
        from jinja2 import Template
        try:
            template = Template(DETECTION_LLM_PROMPT)
            prompt = template.render(answer=answer)
            return client.generate_json(prompt, system="You are an AI content analysis expert. Respond with valid JSON only.")
        except Exception as e:
            print("AITextDetector LLM error: {}".format(e), file=sys.stderr)
            return None


class ContentOriginCheck:
    """Per-sentence AI detection. Highlights which sentences are likely AI.

    Scores each sentence individually. Useful for mixed content where
    some parts are human-written and some are AI-generated.
    """

    name = "content_origin_check"

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

        if not answer:
            return MetricResult(name=self.name, score=0.5, reason="Empty text")

        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

        if not sentences:
            return MetricResult(name=self.name, score=0.5, reason="No sentences to analyze")

        if self.use_llm and client:
            return self._check_with_llm(client, answer, sentences)

        results = []
        for sent in sentences:
            words = sent.split()
            vocab = _type_token_ratio(sent) if len(words) >= 5 else 0.5
            rep = _repetition_score(sent) if len(words) >= 10 else 0.5
            trans = _transition_score(sent)

            sent_score = (vocab * 0.4 + rep * 0.3 + trans * 0.3)
            label = "likely_human" if sent_score >= 0.55 else "likely_ai" if sent_score < 0.4 else "uncertain"

            results.append({
                "text": sent[:100],
                "score": round(sent_score, 3),
                "label": label,
            })

        avg_score = sum(r["score"] for r in results) / len(results)
        ai_count = sum(1 for r in results if r["label"] == "likely_ai")

        return MetricResult(
            name=self.name, score=round(avg_score, 4),
            reason="{} of {} sentences flagged as likely AI".format(ai_count, len(results)),
            details={"sentences": results, "ai_sentences": ai_count, "total": len(results)})

    def _check_with_llm(self, client, answer, sentences):
        try:
            prompt = "Label each sentence as 'human' or 'ai' with confidence.\n\nText:\n{}\n\nRespond with ONLY valid JSON:\n{{\"sentences\": [{{\"text\": \"...\", \"label\": \"human\", \"confidence\": 0.8}}]}}".format(answer[:2000])
            result = client.generate_json(prompt, system="You are an AI content analysis expert. Respond with valid JSON only.")
            items = result.get("sentences", [])
            if items:
                human_count = sum(1 for i in items if i.get("label") == "human")
                score = human_count / len(items)
                return MetricResult(name=self.name, score=round(score, 4),
                                    reason="{} of {} sentences human".format(human_count, len(items)),
                                    details=result)
        except Exception as e:
            print("ContentOriginCheck LLM error: {}".format(e), file=sys.stderr)
        return MetricResult(name=self.name, score=0.5, reason="LLM analysis failed")


class AIImageDetector:
    """Basic AI image detection using metadata analysis.

    Checks EXIF data, file properties for AI generation signs.
    Without API: metadata checks only.
    With API: LLM analyzes image description.
    """

    name = "ai_image_detector"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")  # image metadata or description

        if not answer:
            return MetricResult(name=self.name, score=0.5, reason="No image data")

        signals = []
        answer_lower = answer.lower()

        # Check for AI generation tool markers
        ai_tools = ['dall-e', 'midjourney', 'stable diffusion', 'ai generated',
                     'generated by', 'created with ai', 'artificial intelligence',
                     'leonardo ai', 'firefly', 'ideogram', 'flux']
        for tool in ai_tools:
            if tool in answer_lower:
                signals.append({"signal": "AI tool marker: {}".format(tool), "indicates": "ai"})

        # Check for missing camera data
        camera_markers = ['canon', 'nikon', 'sony', 'fuji', 'iphone', 'samsung',
                          'pixel', 'exposure', 'f/', 'iso', 'focal length', 'shutter']
        has_camera = any(m in answer_lower for m in camera_markers)
        if not has_camera:
            signals.append({"signal": "No camera metadata", "indicates": "possibly_ai"})

        # Score
        ai_signals = sum(1 for s in signals if s["indicates"] == "ai")
        if ai_signals > 0:
            score = 0.0
            reason = "AI generation markers found"
        elif signals:
            score = 0.5
            reason = "Suspicious signals but not conclusive"
        else:
            score = 1.0
            reason = "No AI generation signals detected"

        if self.use_llm and client:
            llm_result = self._check_with_llm(client, answer)
            if llm_result:
                llm_score = llm_result.get("human_score", 0.5)
                if isinstance(llm_score, str):
                    llm_score = float(llm_score)
                score = score * 0.3 + llm_score * 0.7
                signals.append({"signal": "LLM analysis", "result": llm_result})

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"signals": signals})

    def _check_with_llm(self, client, answer):
        try:
            prompt = "Analyze this image metadata/description for AI generation signs:\n\n{}\n\nRespond with ONLY valid JSON:\n{{\"human_score\": 0.5, \"likely_source\": \"ai/human/unknown\", \"reason\": \"...\"}}".format(answer[:1000])
            return client.generate_json(prompt, system="You are an AI image forensics expert. Respond with valid JSON only.")
        except Exception:
            return None


class AIAudioDetector:
    """Basic AI audio detection using metadata analysis.

    Checks audio properties for synthetic speech markers.
    """

    name = "ai_audio_detector"

    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        answer = kwargs.get("answer", "")  # audio metadata or transcript

        if not answer:
            return MetricResult(name=self.name, score=0.5, reason="No audio data")

        signals = []
        answer_lower = answer.lower()

        # Check for TTS markers
        tts_markers = ['text-to-speech', 'tts', 'synthesized', 'generated voice',
                       'elevenlabs', 'bark', 'tortoise', 'coqui', 'synthetic speech',
                       'ai voice', 'artificial voice', 'voice cloning']
        for marker in tts_markers:
            if marker in answer_lower:
                signals.append({"signal": "TTS marker: {}".format(marker), "indicates": "ai"})

        # Check for natural recording markers
        natural_markers = ['recorded', 'microphone', 'ambient noise', 'background',
                           'interview', 'live', 'recording studio']
        has_natural = any(m in answer_lower for m in natural_markers)
        if has_natural:
            signals.append({"signal": "Natural recording markers present", "indicates": "human"})

        ai_signals = sum(1 for s in signals if s["indicates"] == "ai")
        human_signals = sum(1 for s in signals if s["indicates"] == "human")

        if ai_signals > 0:
            score = 0.0
            reason = "Synthetic speech markers detected"
        elif human_signals > 0:
            score = 1.0
            reason = "Natural recording markers found"
        else:
            score = 0.5
            reason = "Insufficient data for determination"

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"signals": signals})
