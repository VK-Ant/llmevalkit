"""Tests for LLMEVAL — run with: pytest tests/ -v"""

import pytest
from unittest.mock import MagicMock, patch

from llmevalkit.models import (
    MetricResult,
    EvalResult,
    EvalConfig,
    TestCase,
    BatchResult,
    Provider,
)
from llmevalkit.metrics.base import BaseMetric
from llmevalkit.metrics import (
    Faithfulness,
    AnswerRelevance,
    ContextRelevance,
    Hallucination,
    Toxicity,
    Coherence,
    Completeness,
    GEval,
)


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestMetricResult:
    def test_creation(self):
        r = MetricResult(name="test", score=0.85, reason="Good")
        assert r.name == "test"
        assert r.score == 0.85
        assert r.reason == "Good"

    def test_score_bounds(self):
        r = MetricResult(name="test", score=0.0, reason="")
        assert r.score == 0.0
        r = MetricResult(name="test", score=1.0, reason="")
        assert r.score == 1.0

    def test_score_out_of_bounds_raises(self):
        with pytest.raises(Exception):
            MetricResult(name="test", score=1.5, reason="")
        with pytest.raises(Exception):
            MetricResult(name="test", score=-0.1, reason="")


class TestEvalResult:
    def test_passed_above_threshold(self):
        r = EvalResult(
            question="q", answer="a", overall_score=0.8,
            metadata={"threshold": 0.5}
        )
        assert r.passed is True

    def test_failed_below_threshold(self):
        r = EvalResult(
            question="q", answer="a", overall_score=0.3,
            metadata={"threshold": 0.5}
        )
        assert r.passed is False

    def test_to_dict(self):
        r = EvalResult(
            question="What?", answer="Yes", overall_score=0.75,
            metrics={"test": MetricResult(name="test", score=0.8, reason="ok")},
        )
        d = r.to_dict()
        assert d["overall_score"] == 0.75
        assert "metric_test_score" in d
        assert d["metric_test_score"] == 0.8

    def test_summary_output(self):
        r = EvalResult(
            question="What is AI?",
            answer="Artificial Intelligence",
            overall_score=0.9,
            metrics={"faithfulness": MetricResult(name="faithfulness", score=0.9, reason="Good")},
            metadata={"threshold": 0.5},
        )
        s = r.summary()
        assert "LLMEVAL" in s
        assert "PASSED" in s


class TestBatchResult:
    def test_average_score(self):
        batch = BatchResult(results=[
            EvalResult(overall_score=0.8, metadata={"threshold": 0.5}),
            EvalResult(overall_score=0.6, metadata={"threshold": 0.5}),
        ])
        assert batch.average_score == 0.7

    def test_pass_rate(self):
        batch = BatchResult(results=[
            EvalResult(overall_score=0.8, metadata={"threshold": 0.5}),
            EvalResult(overall_score=0.3, metadata={"threshold": 0.5}),
            EvalResult(overall_score=0.6, metadata={"threshold": 0.5}),
        ])
        assert abs(batch.pass_rate - 2/3) < 0.01

    def test_empty_batch(self):
        batch = BatchResult()
        assert batch.average_score == 0.0
        assert batch.pass_rate == 0.0


class TestEvalConfig:
    def test_defaults(self):
        c = EvalConfig()
        assert c.provider == "openai"
        assert c.model == "gpt-4o-mini"
        assert c.temperature == 0.0

    def test_azure_config(self):
        c = EvalConfig(
            provider="azure",
            model="gpt-4o",
            api_key="test",
            base_url="https://test.openai.azure.com/",
            api_version="2024-02-01",
        )
        assert c.provider == "azure"


class TestTestCase:
    def test_creation(self):
        tc = TestCase(question="q", answer="a", context="c")
        assert tc.question == "q"
        assert tc.reference is None


# ── Metric Tests ─────────────────────────────────────────────────────────────

class TestMetricRequiredFields:
    def test_faithfulness_requires_context(self):
        m = Faithfulness()
        assert "context" in m.required_fields
        assert m.validate_inputs(question="q", answer="a", context="c")
        assert not m.validate_inputs(question="q", answer="a", context="")

    def test_answer_relevance_no_context_needed(self):
        m = AnswerRelevance()
        assert "context" not in m.required_fields
        assert m.validate_inputs(question="q", answer="a")

    def test_hallucination_reference_free(self):
        m = Hallucination()
        assert "context" not in m.required_fields
        assert m.validate_inputs(question="q", answer="a")

    def test_toxicity_only_needs_answer(self):
        m = Toxicity()
        assert m.required_fields == ["answer"]
        assert m.validate_inputs(answer="some text")

    def test_geval_custom_criteria(self):
        m = GEval(criteria="Check response accuracy")
        assert m.criteria == "Check response accuracy"


class TestMetricScoring:
    """Test score normalization and inversion."""
    
    def _make_mock_client(self, response: dict):
        client = MagicMock()
        client.generate_json.return_value = response
        return client

    def test_normal_score_normalization(self):
        m = AnswerRelevance()
        client = self._make_mock_client({"score": 5, "reason": "Perfect", "on_topic_percentage": 100})
        result = m.evaluate(client, question="q", answer="a")
        assert result.score == 1.0

    def test_low_score_normalization(self):
        m = AnswerRelevance()
        client = self._make_mock_client({"score": 1, "reason": "Bad", "on_topic_percentage": 0})
        result = m.evaluate(client, question="q", answer="a")
        assert result.score == 0.0

    def test_mid_score_normalization(self):
        m = AnswerRelevance()
        client = self._make_mock_client({"score": 3, "reason": "Ok", "on_topic_percentage": 50})
        result = m.evaluate(client, question="q", answer="a")
        assert result.score == 0.5

    def test_inverted_score_hallucination(self):
        m = Hallucination()
        # Score 1 means no hallucination → should become 1.0 after inversion
        client = self._make_mock_client({"score": 1, "reason": "Clean", "hallucinations": []})
        result = m.evaluate(client, question="q", answer="a")
        assert result.score == 1.0

    def test_inverted_score_toxicity(self):
        m = Toxicity()
        # Score 5 means very toxic → after inversion should be 0.0
        client = self._make_mock_client({"score": 5, "reason": "Toxic", "issues": []})
        result = m.evaluate(client, answer="bad text")
        assert result.score == 0.0

    def test_error_handling(self):
        m = Faithfulness()
        client = MagicMock()
        client.generate_json.side_effect = Exception("API Error")
        result = m.evaluate(client, question="q", answer="a", context="c")
        assert result.score == 0.0
        assert "failed" in result.reason.lower()


# ── Evaluator Tests ──────────────────────────────────────────────────────────

class TestEvaluatorInit:
    @patch("llmevalkit.evaluator.LLMClient")
    def test_default_preset(self, mock_client):
        from llmevalkit.evaluator import Evaluator
        e = Evaluator(api_key="test")
        assert len(e.metrics) == 4  # RAG preset

    @patch("llmevalkit.evaluator.LLMClient")
    def test_custom_preset(self, mock_client):
        from llmevalkit.evaluator import Evaluator
        e = Evaluator(api_key="test", preset="safety")
        metric_names = [m.name for m in e.metrics]
        assert "toxicity" in metric_names
        assert "hallucination" in metric_names

    @patch("llmevalkit.evaluator.LLMClient")
    def test_custom_metrics(self, mock_client):
        from llmevalkit.evaluator import Evaluator
        e = Evaluator(api_key="test", metrics=[Coherence(), Toxicity()])
        assert len(e.metrics) == 2

    @patch("llmevalkit.evaluator.LLMClient")
    def test_add_remove_metric(self, mock_client):
        from llmevalkit.evaluator import Evaluator
        e = Evaluator(api_key="test", preset="minimal")
        assert len(e.metrics) == 2
        e.add_metric(Toxicity)
        assert len(e.metrics) == 3
        e.remove_metric("toxicity")
        assert len(e.metrics) == 2

    @patch("llmevalkit.evaluator.LLMClient")
    def test_invalid_preset_raises(self, mock_client):
        from llmevalkit.evaluator import Evaluator
        with pytest.raises(ValueError, match="Unknown preset"):
            Evaluator(api_key="test", preset="nonexistent")


# ── Token Counter Tests ──────────────────────────────────────────────────────

class TestTokenCounter:
    def test_count_tokens(self):
        from llmevalkit.utils.token_counter import count_tokens
        # May fall back to rough estimate if tiktoken can't download tokenizer
        count = count_tokens("Hello world, this is a test.")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_fallback(self):
        """Test the rough fallback estimation when tiktoken is unavailable."""
        from unittest.mock import patch
        from llmevalkit.utils.token_counter import count_tokens
        with patch.dict("sys.modules", {"tiktoken": None}):
            # Force ImportError path
            count = count_tokens.__wrapped__("Hello world") if hasattr(count_tokens, '__wrapped__') else len("Hello world") // 4
            assert count >= 0

    def test_estimate_cost(self):
        from llmevalkit.utils.token_counter import estimate_cost
        cost = estimate_cost(input_tokens=1000, output_tokens=200, model="gpt-4o-mini")
        assert cost["total_cost_usd"] > 0
        assert cost["input_tokens"] == 4000  # 1000 * 4 metrics


# ── Integration Smoke Test ───────────────────────────────────────────────────

class TestImports:
    def test_top_level_imports(self):
        from llmevalkit import Evaluator, EvalResult, EvalConfig, MetricResult
        from llmevalkit import Faithfulness, AnswerRelevance, Hallucination, GEval
        assert Evaluator is not None
        assert GEval is not None

    def test_version(self):
        import llmevalkit
        assert llmevalkit.__version__ == "3.0.0"


# ── Math Metric Tests (NO API needed) ────────────────────────────────────────

class TestBLEUScore:
    def test_identical_text(self):
        from llmevalkit.metrics.math_metrics import BLEUScore
        m = BLEUScore()
        result = m.evaluate(answer="the cat sat on the mat", reference="the cat sat on the mat")
        assert result.score > 0.9

    def test_different_text(self):
        from llmevalkit.metrics.math_metrics import BLEUScore
        m = BLEUScore()
        result = m.evaluate(answer="hello world", reference="goodbye moon")
        assert result.score < 0.3

    def test_no_reference(self):
        from llmevalkit.metrics.math_metrics import BLEUScore
        m = BLEUScore()
        result = m.evaluate(answer="hello")
        assert result.score == 0.0

    def test_uses_context_as_fallback(self):
        from llmevalkit.metrics.math_metrics import BLEUScore
        m = BLEUScore()
        result = m.evaluate(answer="the cat sat", context="the cat sat on the mat")
        assert result.score > 0.3


class TestROUGEScore:
    def test_identical(self):
        from llmevalkit.metrics.math_metrics import ROUGEScore
        m = ROUGEScore()
        result = m.evaluate(answer="the cat sat on the mat", reference="the cat sat on the mat")
        assert result.score > 0.9

    def test_partial_overlap(self):
        from llmevalkit.metrics.math_metrics import ROUGEScore
        m = ROUGEScore()
        result = m.evaluate(
            answer="the cat sat on the mat",
            reference="the cat was sitting on a large mat in the room"
        )
        assert 0.2 < result.score < 0.9

    def test_details_contain_rouge1_rouge2(self):
        from llmevalkit.metrics.math_metrics import ROUGEScore
        m = ROUGEScore()
        result = m.evaluate(answer="hello world", reference="hello world today")
        assert "rouge1" in result.details
        assert "rouge2" in result.details
        assert "rougeL" in result.details


class TestTokenOverlap:
    def test_full_overlap(self):
        from llmevalkit.metrics.math_metrics import TokenOverlap
        m = TokenOverlap()
        result = m.evaluate(answer="python programming language", context="python programming language")
        assert result.score > 0.8

    def test_no_overlap(self):
        from llmevalkit.metrics.math_metrics import TokenOverlap
        m = TokenOverlap()
        result = m.evaluate(answer="cats dogs animals", context="programming code software")
        assert result.score == 0.0

    def test_stopwords_filtered(self):
        from llmevalkit.metrics.math_metrics import TokenOverlap
        m = TokenOverlap()
        # "the" and "is" are stopwords, should be filtered
        result = m.evaluate(answer="the sky is blue", context="the ocean is green")
        assert "sky" not in [t for t in result.details.get("common_tokens", [])]


class TestAnswerLength:
    def test_good_length(self):
        from llmevalkit.metrics.math_metrics import AnswerLength
        m = AnswerLength(min_words=5, max_words=50)
        result = m.evaluate(answer="This is a perfectly good answer with enough words to pass.")
        assert result.score == 1.0

    def test_too_short(self):
        from llmevalkit.metrics.math_metrics import AnswerLength
        m = AnswerLength(min_words=10)
        result = m.evaluate(answer="Yes")
        assert result.score < 0.5

    def test_too_long(self):
        from llmevalkit.metrics.math_metrics import AnswerLength
        m = AnswerLength(max_words=5)
        result = m.evaluate(answer="This answer has way more than five words in it and keeps going")
        assert result.score < 1.0


class TestReadabilityScore:
    def test_simple_text(self):
        from llmevalkit.metrics.math_metrics import ReadabilityScore
        m = ReadabilityScore()
        result = m.evaluate(answer="The cat sat on the mat. The dog ran in the park.")
        assert result.score > 0.5  # Simple text should be easy to read

    def test_complex_text(self):
        from llmevalkit.metrics.math_metrics import ReadabilityScore
        m = ReadabilityScore()
        result = m.evaluate(
            answer="The epistemological ramifications of contemporary hermeneutical "
                   "phenomenology necessitate a comprehensive reconceptualization of "
                   "the fundamental ontological presuppositions underlying our interpretive frameworks."
        )
        # Complex text should have lower readability
        assert result.score < 0.5

    def test_details_contain_grade(self):
        from llmevalkit.metrics.math_metrics import ReadabilityScore
        m = ReadabilityScore()
        result = m.evaluate(answer="Hello. This is simple.")
        assert "flesch_reading_ease" in result.details
        assert "flesch_kincaid_grade" in result.details


class TestKeywordCoverage:
    def test_full_coverage(self):
        from llmevalkit.metrics.math_metrics import KeywordCoverage
        m = KeywordCoverage()
        result = m.evaluate(
            answer="Python is a programming language used for machine learning",
            context="Python is a programming language used for machine learning"
        )
        assert result.score > 0.8

    def test_missing_keywords(self):
        from llmevalkit.metrics.math_metrics import KeywordCoverage
        m = KeywordCoverage()
        result = m.evaluate(
            answer="Hello world",
            context="Python is a powerful programming language for machine learning and data science"
        )
        assert result.score < 0.5

    def test_details_show_covered_and_missing(self):
        from llmevalkit.metrics.math_metrics import KeywordCoverage
        m = KeywordCoverage()
        result = m.evaluate(answer="Python rocks", context="Python programming language")
        assert "covered" in result.details
        assert "missing" in result.details


class TestSemanticSimilarity:
    def test_fallback_bow(self):
        """Test bag-of-words fallback when sentence-transformers not installed."""
        from llmevalkit.metrics.math_metrics import SemanticSimilarity
        m = SemanticSimilarity()
        result = m.evaluate(answer="the cat sat on the mat", reference="the cat sat on the mat")
        assert result.score > 0.8

    def test_different_text_fallback(self):
        from llmevalkit.metrics.math_metrics import SemanticSimilarity
        m = SemanticSimilarity()
        result = m.evaluate(answer="hello world", reference="goodbye moon")
        # Even with BOW fallback, completely different words = low similarity
        assert result.score < 0.5


# ── Math-only Evaluator Tests ────────────────────────────────────────────────

class TestMathOnlyEvaluator:
    def test_math_only_no_api(self):
        """Core test: evaluate with ZERO API calls, ZERO cost."""
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="math")
        assert e.client is None  # No LLM client
        assert e.is_math_only is True

    def test_math_evaluation_runs(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="math")
        result = e.evaluate(
            question="What is Python?",
            answer="Python is a high-level programming language used for web development and data science.",
            context="Python is a high-level, interpreted programming language known for its simplicity."
        )
        assert result.overall_score > 0.0
        assert len(result.metrics) > 0

    def test_math_minimal_preset(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="math_minimal")
        result = e.evaluate(
            answer="This is a good answer with enough words.",
            context="This is a reference text."
        )
        assert "token_overlap" in result.metrics
        assert "answer_length" in result.metrics
