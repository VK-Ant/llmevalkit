"""Tests for llmevalkit v5: extended hallucination, detection, observe, anomaly."""

import unittest
import os
import tempfile
import shutil


# ============================================================
# NEW HALLUCINATION METRICS
# ============================================================

class TestSourceCoverage(unittest.TestCase):
    def test_high_coverage(self):
        from llmevalkit.hallucination import SourceCoverage
        sc = SourceCoverage()
        r = sc.evaluate(
            answer="Solar energy is renewable and reduces electricity costs.",
            context="Solar energy is a renewable source that lowers electricity costs."
        )
        self.assertGreater(r.score, 0.3)

    def test_low_coverage(self):
        from llmevalkit.hallucination import SourceCoverage
        sc = SourceCoverage()
        r = sc.evaluate(
            answer="Quantum computing will replace all classical computers by 2025.",
            context="Solar energy is a renewable source."
        )
        self.assertLess(r.score, 0.5)


class TestTemporalHallucination(unittest.TestCase):
    def test_correct_dates(self):
        from llmevalkit.hallucination import TemporalHallucination
        th = TemporalHallucination()
        r = th.evaluate(
            answer="The company was founded in 2015.",
            context="Established in 2015, the company grew rapidly."
        )
        self.assertGreater(r.score, 0.5)

    def test_wrong_date(self):
        from llmevalkit.hallucination import TemporalHallucination
        th = TemporalHallucination()
        r = th.evaluate(
            answer="The company was founded in 2010.",
            context="Established in 2015, the company grew rapidly."
        )
        self.assertLess(r.score, 1.0)


class TestCausalHallucination(unittest.TestCase):
    def test_supported_cause(self):
        from llmevalkit.hallucination import CausalHallucination
        ch = CausalHallucination()
        r = ch.evaluate(
            answer="Revenue increased because of strong product sales.",
            context="Strong product sales drove revenue increase."
        )
        self.assertGreater(r.score, 0.3)

    def test_no_causal_claims(self):
        from llmevalkit.hallucination import CausalHallucination
        ch = CausalHallucination()
        r = ch.evaluate(answer="The sky is blue.", context="The sky appears blue.")
        self.assertEqual(r.score, 1.0)


class TestRankingHallucination(unittest.TestCase):
    def test_no_ranking(self):
        from llmevalkit.hallucination import RankingHallucination
        rh = RankingHallucination()
        r = rh.evaluate(answer="Python is a language.", context="Python is a programming language.")
        self.assertEqual(r.score, 1.0)

    def test_ranking_present(self):
        from llmevalkit.hallucination import RankingHallucination
        rh = RankingHallucination()
        r = rh.evaluate(
            answer="Company A is the largest in the industry.",
            context="Company A is the largest player in the industry."
        )
        self.assertGreater(r.score, 0.0)


# ============================================================
# AI CONTENT DETECTION
# ============================================================

class TestAITextDetector(unittest.TestCase):
    def test_short_text(self):
        from llmevalkit.detection import AITextDetector
        d = AITextDetector()
        r = d.evaluate(answer="Hello world.")
        self.assertEqual(r.score, 0.5)

    def test_normal_text(self):
        from llmevalkit.detection import AITextDetector
        d = AITextDetector()
        r = d.evaluate(
            answer="Python is a versatile programming language used widely in data science "
                   "and web development. Many developers prefer it for its clean syntax."
        )
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 1.0)
        self.assertIn("perplexity_proxy", r.details)
        self.assertIn("burstiness", r.details)

    def test_details_present(self):
        from llmevalkit.detection import AITextDetector
        d = AITextDetector()
        text = "Furthermore, it is important to note that additionally, the system provides. Moreover, the implementation ensures."
        r = d.evaluate(answer=text * 3)
        self.assertIn("vocabulary_diversity", r.details)
        self.assertIn("transition_uniformity", r.details)


class TestContentOriginCheck(unittest.TestCase):
    def test_per_sentence(self):
        from llmevalkit.detection import ContentOriginCheck
        c = ContentOriginCheck()
        r = c.evaluate(answer="First sentence here. Second sentence there. Third one as well.")
        self.assertIn("sentences", r.details)
        self.assertGreater(r.details["total"], 0)


class TestAIImageDetector(unittest.TestCase):
    def test_ai_markers(self):
        from llmevalkit.detection import AIImageDetector
        d = AIImageDetector()
        r = d.evaluate(answer="Generated by DALL-E, 1024x1024, no EXIF data")
        self.assertEqual(r.score, 0.0)

    def test_real_camera(self):
        from llmevalkit.detection import AIImageDetector
        d = AIImageDetector()
        r = d.evaluate(answer="Canon EOS R5, ISO 400, f/2.8, 1/250s, 85mm focal length")
        self.assertEqual(r.score, 1.0)


class TestAIAudioDetector(unittest.TestCase):
    def test_tts_markers(self):
        from llmevalkit.detection import AIAudioDetector
        d = AIAudioDetector()
        r = d.evaluate(answer="Generated using ElevenLabs text-to-speech")
        self.assertEqual(r.score, 0.0)

    def test_natural(self):
        from llmevalkit.detection import AIAudioDetector
        d = AIAudioDetector()
        r = d.evaluate(answer="Recorded with microphone in studio with ambient noise")
        self.assertEqual(r.score, 1.0)


# ============================================================
# OBSERVABILITY
# ============================================================

class TestEvalLogger(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_log_and_read(self):
        from llmevalkit.observe import EvalLogger
        logger = EvalLogger(log_dir=self.tmpdir)
        logger.log({"overall_score": 0.85, "passed": True, "metrics": {"bleu": {"score": 0.7}}})
        logger.log({"overall_score": 0.65, "passed": True, "metrics": {"bleu": {"score": 0.5}}})
        entries = logger.read_logs()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["overall_score"], 0.85)

    def test_clear(self):
        from llmevalkit.observe import EvalLogger
        logger = EvalLogger(log_dir=self.tmpdir)
        logger.log({"overall_score": 0.9, "passed": True, "metrics": {}})
        logger.clear()
        self.assertEqual(len(logger.read_logs()), 0)


class TestScoreDrift(unittest.TestCase):
    def test_insufficient_data(self):
        from llmevalkit.observe import ScoreDrift
        tmpdir = tempfile.mkdtemp()
        sd = ScoreDrift(log_dir=tmpdir, window=5)
        result = sd.check()
        self.assertEqual(result["status"], "insufficient_data")
        shutil.rmtree(tmpdir)


class TestEvalReport(unittest.TestCase):
    def test_no_data(self):
        from llmevalkit.observe import EvalReport
        tmpdir = tempfile.mkdtemp()
        report = EvalReport(log_dir=tmpdir)
        result = report.summary()
        self.assertEqual(result["status"], "no_data")
        shutil.rmtree(tmpdir)


class TestThresholdAlert(unittest.TestCase):
    def test_no_thresholds(self):
        from llmevalkit.observe import ThresholdAlert
        ta = ThresholdAlert()
        result = ta.check()
        self.assertEqual(result["status"], "no_thresholds")


class TestEvalComparison(unittest.TestCase):
    def test_compare(self):
        from llmevalkit.observe import EvalComparison
        from llmevalkit import Evaluator, BLEUScore

        e = Evaluator(provider="none", metrics=[BLEUScore()])
        r1 = e.evaluate(question="q", answer="Python is a language.", context="Python is a programming language.")
        r2 = e.evaluate(question="q", answer="Python is a high-level programming language.", context="Python is a programming language.")

        comp = EvalComparison.compare(r1, r2, "Prompt A", "Prompt B")
        self.assertIn("winner", comp)
        self.assertIn("metrics", comp)


# ============================================================
# ANOMALY DETECTION
# ============================================================

class TestOutputAnomalyDetector(unittest.TestCase):
    def test_normal_output(self):
        from llmevalkit.anomaly import OutputAnomalyDetector
        ad = OutputAnomalyDetector()
        r = ad.evaluate(answer="Solar energy is renewable and reduces electricity costs.")
        self.assertEqual(r.score, 1.0)

    def test_too_short(self):
        from llmevalkit.anomaly import OutputAnomalyDetector
        ad = OutputAnomalyDetector()
        r = ad.evaluate(answer="Yes.")
        self.assertLess(r.score, 1.0)

    def test_all_caps(self):
        from llmevalkit.anomaly import OutputAnomalyDetector
        ad = OutputAnomalyDetector()
        r = ad.evaluate(answer="BUY NOW URGENT ACT IMMEDIATELY FREE MONEY LIMITED TIME")
        self.assertLess(r.score, 1.0)

    def test_topic_drift(self):
        from llmevalkit.anomaly import OutputAnomalyDetector
        ad = OutputAnomalyDetector()
        r = ad.evaluate(
            answer="Quantum physics experiments with particle accelerators demonstrate fascinating phenomena in subatomic particles and wave functions.",
            context="Solar energy is a renewable source that lowers electricity costs and reduces carbon footprint."
        )
        self.assertLessEqual(r.score, 1.0)


class TestScoreAnomalyDetector(unittest.TestCase):
    def test_insufficient_data(self):
        from llmevalkit.anomaly import ScoreAnomalyDetector
        tmpdir = tempfile.mkdtemp()
        sad = ScoreAnomalyDetector(log_dir=tmpdir)
        result = sad.check()
        self.assertEqual(result["status"], "insufficient_data")
        shutil.rmtree(tmpdir)


# ============================================================
# PRESET TESTS
# ============================================================

class TestV5Presets(unittest.TestCase):
    def test_hallucination_expanded(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination")
        self.assertEqual(len(e.metrics), 12)

    def test_detection_preset(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="detection")
        self.assertEqual(len(e.metrics), 4)

    def test_detection_text(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="detection_text")
        self.assertEqual(len(e.metrics), 2)

    def test_anomaly_preset(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="anomaly")
        self.assertEqual(len(e.metrics), 1)

    def test_production_preset(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="production")
        self.assertEqual(len(e.metrics), 9)

    def test_hallucination_financial(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination_financial")
        self.assertEqual(len(e.metrics), 4)

    def test_auto_logging(self):
        from llmevalkit import Evaluator, BLEUScore
        tmpdir = tempfile.mkdtemp()
        e = Evaluator(provider="none", metrics=[BLEUScore()], log_path=tmpdir)
        e.evaluate(question="q", answer="Python is a language.", context="Python is a language.")
        from llmevalkit.observe import EvalLogger
        logger = EvalLogger(log_dir=tmpdir)
        entries = logger.read_logs()
        self.assertGreater(len(entries), 0)
        shutil.rmtree(tmpdir)

    def test_auto_logging_disabled(self):
        from llmevalkit import Evaluator, BLEUScore
        e = Evaluator(provider="none", metrics=[BLEUScore()], auto_log=False)
        self.assertIsNone(e._logger)

    def test_all_previous_presets(self):
        from llmevalkit import Evaluator
        for preset in ["math", "hipaa", "gdpr", "doceval", "security", "governance", "multimodal"]:
            e = Evaluator(provider="none", preset=preset)
            self.assertGreater(len(e.metrics), 0)


if __name__ == "__main__":
    unittest.main()
