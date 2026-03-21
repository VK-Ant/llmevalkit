"""
Pure mathematical / statistical metrics — NO API, NO cost, runs 100% locally.

These metrics use math formulas, string operations, and statistical calculations.
No LLM judge needed. Instant results.

Available metrics:
    - BLEUScore: N-gram precision (machine translation standard)
    - ROUGEScore: Recall-oriented overlap (summarization standard)  
    - TokenOverlap: Simple token-level precision/recall/F1
    - SemanticSimilarity: Cosine similarity using sentence embeddings
    - AnswerLength: Checks if answer meets length expectations
    - ReadabilityScore: Flesch-Kincaid readability grade level
    - KeywordCoverage: Checks if answer covers key terms from context
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional

from llmevalkit.models import MetricResult


class MathMetric:
    """Base class for pure mathematical metrics. No API needed."""
    
    name: str = "math_metric"
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def evaluate(self, client=None, **kwargs) -> MetricResult:
        """Evaluate using pure math. Client parameter accepted but ignored."""
        return self._compute(**kwargs)
    
    def _compute(self, **kwargs) -> MetricResult:
        raise NotImplementedError
    
    def validate_inputs(self, **kwargs) -> bool:
        for field in self.required_fields:
            if not kwargs.get(field):
                return False
        return True
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'\b\w+\b', text.lower())

    @staticmethod
    def _get_ngrams(tokens: list[str], n: int) -> list[tuple]:
        """Extract n-grams from token list."""
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class BLEUScore(MathMetric):
    """BLEU (Bilingual Evaluation Understudy) score.
    
    Measures n-gram precision between answer and reference.
    Standard metric in machine translation and text generation.
    
    Works with reference answer OR context as reference.
    
    Score interpretation:
        1.0: Perfect n-gram match
        0.5: Moderate overlap
        0.0: No matching n-grams
    """
    
    name = "bleu"
    
    def __init__(self, max_n: int = 4, weight: float = 1.0):
        super().__init__(weight=weight)
        self.max_n = max_n
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]  # reference or context used if available
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        # Use reference if available, otherwise use context
        reference = kwargs.get("reference", "") or kwargs.get("context", "")
        
        if not reference:
            return MetricResult(
                name=self.name, score=0.0,
                reason="No reference or context provided for BLEU calculation",
                details={"error": "missing_reference"},
            )
        
        answer_tokens = self._tokenize(answer)
        ref_tokens = self._tokenize(reference)
        
        if not answer_tokens or not ref_tokens:
            return MetricResult(name=self.name, score=0.0, reason="Empty input")
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            ans_ngrams = self._get_ngrams(answer_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if not ans_ngrams:
                precisions.append(0.0)
                continue
            
            ref_counts = Counter(ref_ngrams)
            ans_counts = Counter(ans_ngrams)
            
            clipped = sum(min(ans_counts[ng], ref_counts[ng]) for ng in ans_counts)
            total = sum(ans_counts.values())
            precisions.append(clipped / total if total > 0 else 0.0)
        
        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(answer_tokens))) if answer_tokens else 0.0
        
        # Geometric mean of precisions
        log_avg = 0.0
        valid_n = 0
        for p in precisions:
            if p > 0:
                log_avg += math.log(p)
                valid_n += 1
        
        if valid_n == 0:
            score = 0.0
        else:
            score = bp * math.exp(log_avg / valid_n)
        
        score = max(0.0, min(1.0, score))
        
        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=f"BLEU-{self.max_n}: {score:.3f} (BP={bp:.3f})",
            details={
                "precisions": [round(p, 4) for p in precisions],
                "brevity_penalty": round(bp, 4),
                "answer_length": len(answer_tokens),
                "reference_length": len(ref_tokens),
            },
        )


class ROUGEScore(MathMetric):
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score.
    
    Measures recall-oriented overlap. Standard for summarization evaluation.
    Computes ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (longest common subsequence).
    
    Score interpretation:
        1.0: Perfect recall of reference content
        0.5: Moderate coverage
        0.0: No overlap
    """
    
    name = "rouge"
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        reference = kwargs.get("reference", "") or kwargs.get("context", "")
        
        if not reference:
            return MetricResult(
                name=self.name, score=0.0,
                reason="No reference or context provided for ROUGE calculation",
                details={"error": "missing_reference"},
            )
        
        ans_tokens = self._tokenize(answer)
        ref_tokens = self._tokenize(reference)
        
        if not ans_tokens or not ref_tokens:
            return MetricResult(name=self.name, score=0.0, reason="Empty input")
        
        # ROUGE-1 (unigram F1)
        rouge1 = self._rouge_n(ans_tokens, ref_tokens, 1)
        
        # ROUGE-2 (bigram F1)
        rouge2 = self._rouge_n(ans_tokens, ref_tokens, 2)
        
        # ROUGE-L (LCS-based F1)
        rougel = self._rouge_l(ans_tokens, ref_tokens)
        
        # Combined score (weighted average)
        score = 0.4 * rouge1["f1"] + 0.3 * rouge2["f1"] + 0.3 * rougel["f1"]
        score = max(0.0, min(1.0, score))
        
        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=f"ROUGE-1={rouge1['f1']:.3f}, ROUGE-2={rouge2['f1']:.3f}, ROUGE-L={rougel['f1']:.3f}",
            details={
                "rouge1": rouge1,
                "rouge2": rouge2,
                "rougeL": rougel,
            },
        )
    
    def _rouge_n(self, ans_tokens: list, ref_tokens: list, n: int) -> dict:
        ans_ngrams = Counter(self._get_ngrams(ans_tokens, n))
        ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))
        
        overlap = sum(min(ans_ngrams[ng], ref_ngrams[ng]) for ng in ans_ngrams)
        precision = overlap / sum(ans_ngrams.values()) if ans_ngrams else 0.0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    
    def _rouge_l(self, ans_tokens: list, ref_tokens: list) -> dict:
        lcs_len = self._lcs_length(ans_tokens, ref_tokens)
        precision = lcs_len / len(ans_tokens) if ans_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
    
    @staticmethod
    def _lcs_length(x: list, y: list) -> int:
        m, n = len(x), len(y)
        # Space-optimized LCS
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]


class TokenOverlap(MathMetric):
    """Token-level overlap between answer and context/reference.
    
    Simple but effective — measures what percentage of important words
    from the context appear in the answer, and vice versa.
    
    Score interpretation:
        1.0: Perfect token overlap
        0.5: Half the tokens match
        0.0: No common tokens
    """
    
    name = "token_overlap"
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        reference = kwargs.get("reference", "") or kwargs.get("context", "")
        
        if not reference:
            return MetricResult(
                name=self.name, score=0.0,
                reason="No reference or context provided",
            )
        
        # Remove stopwords for more meaningful overlap
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
            "both", "either", "neither", "each", "every", "all", "any", "few",
            "more", "most", "other", "some", "such", "no", "only", "own", "same",
            "than", "too", "very", "just", "about", "this", "that", "these", "those",
            "it", "its", "they", "them", "their", "we", "our", "you", "your", "i",
            "me", "my", "he", "him", "his", "she", "her",
        }
        
        ans_tokens = set(self._tokenize(answer)) - stopwords
        ref_tokens = set(self._tokenize(reference)) - stopwords
        
        if not ans_tokens or not ref_tokens:
            return MetricResult(name=self.name, score=0.0, reason="Empty tokens after stopword removal")
        
        overlap = ans_tokens & ref_tokens
        precision = len(overlap) / len(ans_tokens)
        recall = len(overlap) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            score=round(f1, 4),
            reason=f"F1={f1:.3f} (P={precision:.3f}, R={recall:.3f}), {len(overlap)} common tokens",
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "common_tokens": sorted(list(overlap))[:20],  # Top 20
                "answer_only": sorted(list(ans_tokens - ref_tokens))[:10],
                "reference_only": sorted(list(ref_tokens - ans_tokens))[:10],
            },
        )


class SemanticSimilarity(MathMetric):
    """Cosine similarity using sentence embeddings.
    
    Uses sentence-transformers for embedding, then cosine similarity.
    Falls back to token overlap if sentence-transformers is not installed.
    
    NO API call — runs 100% locally using the model on your machine.
    
    Score interpretation:
        1.0: Semantically identical
        0.7+: Very similar meaning
        0.5: Moderate similarity
        0.0: Completely unrelated
    """
    
    name = "semantic_similarity"
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", weight: float = 1.0):
        super().__init__(weight=weight)
        self.model_name = model_name
        self._model = None
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                return None
        return self._model
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        reference = kwargs.get("reference", "") or kwargs.get("context", "")
        
        if not reference:
            return MetricResult(
                name=self.name, score=0.0,
                reason="No reference or context provided",
            )
        
        model = self._get_model()
        
        if model is not None:
            # Use sentence-transformers
            import numpy as np
            embeddings = model.encode([answer, reference])
            cosine = float(np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            ))
            score = max(0.0, min(1.0, cosine))
            method = f"sentence-transformers ({self.model_name})"
        else:
            # Fallback: bag-of-words cosine similarity
            score = self._bow_cosine(answer, reference)
            method = "bag-of-words fallback (install sentence-transformers for better results)"
        
        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=f"Cosine similarity: {score:.3f} via {method}",
            details={"method": method, "cosine_similarity": round(score, 4)},
        )
    
    def _bow_cosine(self, text1: str, text2: str) -> float:
        """Bag-of-words cosine similarity as fallback."""
        tokens1 = Counter(self._tokenize(text1))
        tokens2 = Counter(self._tokenize(text2))
        
        all_tokens = set(tokens1.keys()) | set(tokens2.keys())
        if not all_tokens:
            return 0.0
        
        dot = sum(tokens1.get(t, 0) * tokens2.get(t, 0) for t in all_tokens)
        norm1 = math.sqrt(sum(v ** 2 for v in tokens1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in tokens2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class AnswerLength(MathMetric):
    """Checks if answer meets expected length (not too short, not too long).
    
    Useful for detecting empty/stub answers or excessively verbose outputs.
    
    Score interpretation:
        1.0: Answer is within ideal length range
        0.5: Slightly too short or too long
        0.0: Extremely short (empty) or extremely long
    """
    
    name = "answer_length"
    
    def __init__(self, min_words: int = 5, max_words: int = 500, weight: float = 1.0):
        super().__init__(weight=weight)
        self.min_words = min_words
        self.max_words = max_words
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        words = self._tokenize(answer)
        word_count = len(words)
        
        if word_count < self.min_words:
            # Too short — linear penalty
            score = word_count / self.min_words if self.min_words > 0 else 0.0
            reason = f"Too short: {word_count} words (minimum: {self.min_words})"
        elif word_count > self.max_words:
            # Too long — gradual penalty
            over = word_count - self.max_words
            score = max(0.0, 1.0 - (over / self.max_words))
            reason = f"Too long: {word_count} words (maximum: {self.max_words})"
        else:
            score = 1.0
            reason = f"Good length: {word_count} words (range: {self.min_words}-{self.max_words})"
        
        return MetricResult(
            name=self.name,
            score=round(max(0.0, min(1.0, score)), 4),
            reason=reason,
            details={"word_count": word_count, "min": self.min_words, "max": self.max_words},
        )


class ReadabilityScore(MathMetric):
    """Flesch-Kincaid readability assessment.
    
    Measures how easy the text is to read using established readability formulas.
    No API needed — pure math based on syllable counts and sentence lengths.
    
    Score interpretation (normalized 0-1):
        1.0: Very easy to read (grade 5)
        0.7: Standard (grade 8-10)
        0.5: Somewhat difficult (grade 12)
        0.2: Very difficult (college level)
    """
    
    name = "readability"
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = self._tokenize(answer)
        
        if not sentences or not words:
            return MetricResult(name=self.name, score=0.0, reason="Empty input")
        
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(w) for w in words)
        
        # Flesch Reading Ease (0-100 scale)
        fre = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        fre = max(0.0, min(100.0, fre))
        
        # Flesch-Kincaid Grade Level
        fkgl = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        fkgl = max(0.0, fkgl)
        
        # Normalize Flesch Reading Ease to 0-1 (higher = easier = better)
        score = fre / 100.0
        
        # Determine readability level
        if fre >= 80:
            level = "very easy (6th grade)"
        elif fre >= 60:
            level = "standard (8th-10th grade)"
        elif fre >= 40:
            level = "somewhat difficult (college)"
        elif fre >= 20:
            level = "difficult (college graduate)"
        else:
            level = "very difficult (professional)"
        
        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=f"Readability: {level} (Flesch: {fre:.1f}, Grade: {fkgl:.1f})",
            details={
                "flesch_reading_ease": round(fre, 2),
                "flesch_kincaid_grade": round(fkgl, 2),
                "level": level,
                "num_sentences": num_sentences,
                "num_words": num_words,
                "avg_words_per_sentence": round(num_words / num_sentences, 1),
            },
        )
    
    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 2:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)


class KeywordCoverage(MathMetric):
    """Checks if the answer covers important keywords from the context/question.
    
    Extracts key terms from the context and checks how many appear in the answer.
    Good proxy for completeness without needing an LLM judge.
    
    Score interpretation:
        1.0: All key terms from context covered
        0.5: Half the key terms covered
        0.0: No key terms present
    """
    
    name = "keyword_coverage"
    
    @property
    def required_fields(self) -> list[str]:
        return ["answer"]
    
    def _compute(self, **kwargs) -> MetricResult:
        answer = kwargs.get("answer", "")
        context = kwargs.get("context", "") or kwargs.get("reference", "")
        question = kwargs.get("question", "")
        
        source = context or question
        if not source:
            return MetricResult(
                name=self.name, score=0.0,
                reason="No context or question to extract keywords from",
            )
        
        # Extract important words (nouns, verbs, adjectives — approximated by non-stopwords)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
            "it", "its", "they", "them", "their", "we", "our", "you", "your",
            "this", "that", "these", "those", "what", "which", "who", "how",
            "about", "also", "just", "than", "very", "more", "most", "some",
            "such", "each", "every", "both", "all", "any", "other",
        }
        
        source_tokens = Counter(self._tokenize(source))
        # Keep tokens that appear at least once and aren't stopwords
        keywords = {w for w, c in source_tokens.items() if w not in stopwords and len(w) > 2}
        
        if not keywords:
            return MetricResult(name=self.name, score=1.0, reason="No keywords to check")
        
        answer_tokens = set(self._tokenize(answer))
        covered = keywords & answer_tokens
        
        score = len(covered) / len(keywords) if keywords else 0.0
        
        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=f"Covered {len(covered)}/{len(keywords)} key terms ({score:.1%})",
            details={
                "covered": sorted(list(covered))[:20],
                "missing": sorted(list(keywords - covered))[:20],
                "total_keywords": len(keywords),
            },
        )
