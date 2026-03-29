"""Tests for llmevalkit compliance module."""

import unittest
from llmevalkit.compliance import (
    PIIDetector, HIPAACheck, GDPRCheck, DPDPCheck, EUAIActCheck, CustomRule,
)
from llmevalkit.compliance.pii import detect_pii_patterns, _luhn_check


class TestPIIPatterns(unittest.TestCase):

    def test_detect_email(self):
        found = detect_pii_patterns("Contact us at john@example.com")
        types = [p["type"] for p in found]
        self.assertIn("email", types)

    def test_detect_ssn(self):
        found = detect_pii_patterns("SSN is 123-45-6789")
        types = [p["type"] for p in found]
        self.assertIn("ssn", types)

    def test_detect_phone_us(self):
        found = detect_pii_patterns("Call (555) 123-4567")
        types = [p["type"] for p in found]
        self.assertIn("phone_us", types)

    def test_detect_phone_india(self):
        found = detect_pii_patterns("Call +91 98765 43210")
        types = [p["type"] for p in found]
        self.assertIn("phone_india", types)

    def test_detect_aadhaar(self):
        found = detect_pii_patterns("Aadhaar: 1234 5678 9012")
        types = [p["type"] for p in found]
        self.assertIn("aadhaar", types)

    def test_detect_pan(self):
        found = detect_pii_patterns("PAN: ABCDE1234F")
        types = [p["type"] for p in found]
        self.assertIn("pan_india", types)

    def test_detect_ip(self):
        found = detect_pii_patterns("Server at 192.168.1.100")
        types = [p["type"] for p in found]
        self.assertIn("ip_address", types)

    def test_clean_text(self):
        found = detect_pii_patterns("The weather is sunny today")
        self.assertEqual(len(found), 0)

    def test_luhn_valid(self):
        self.assertTrue(_luhn_check("4111111111111111"))

    def test_luhn_invalid(self):
        self.assertFalse(_luhn_check("1234567890123456"))


class TestPIIDetector(unittest.TestCase):

    def test_clean_text_passes(self):
        pii = PIIDetector()
        result = pii.evaluate(answer="The weather is sunny today")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.details["pii_count"], 0)

    def test_email_detected(self):
        pii = PIIDetector()
        result = pii.evaluate(answer="Email john@gmail.com for details")
        self.assertEqual(result.score, 0.0)
        self.assertGreater(result.details["pii_count"], 0)

    def test_ssn_detected(self):
        pii = PIIDetector()
        result = pii.evaluate(answer="SSN: 123-45-6789")
        self.assertEqual(result.score, 0.0)

    def test_multiple_pii(self):
        pii = PIIDetector()
        result = pii.evaluate(
            answer="Contact raj@gmail.com or call +91 98765 43210. PAN: ABCDE1234F"
        )
        self.assertEqual(result.score, 0.0)
        self.assertGreaterEqual(result.details["pii_count"], 3)

    def test_empty_text(self):
        pii = PIIDetector()
        result = pii.evaluate(answer="")
        self.assertEqual(result.score, 1.0)


class TestHIPAACheck(unittest.TestCase):

    def test_clean_text(self):
        h = HIPAACheck()
        result = h.evaluate(answer="The patient recovery rate improved by 15%")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.details["total_violations"], 0)

    def test_ssn_violation(self):
        h = HIPAACheck()
        result = h.evaluate(answer="Patient SSN: 123-45-6789")
        self.assertEqual(result.score, 0.0)
        self.assertIn(7, result.details["identifiers_found"])

    def test_mrn_violation(self):
        h = HIPAACheck()
        result = h.evaluate(answer="MRN: 12345678 shows positive result")
        self.assertEqual(result.score, 0.0)
        self.assertIn(8, result.details["identifiers_found"])

    def test_email_violation(self):
        h = HIPAACheck()
        result = h.evaluate(answer="Email patient at john@hospital.com")
        self.assertEqual(result.score, 0.0)
        self.assertIn(6, result.details["identifiers_found"])

    def test_multiple_identifiers(self):
        h = HIPAACheck()
        result = h.evaluate(
            answer="Patient John, SSN 123-45-6789, email john@hospital.com, DOB 03/15/1980"
        )
        self.assertEqual(result.score, 0.0)
        self.assertGreaterEqual(result.details["identifiers_found_count"], 2)


class TestGDPRCheck(unittest.TestCase):

    def test_clean_text(self):
        g = GDPRCheck()
        result = g.evaluate(answer="Our service uses industry standard encryption")
        self.assertGreater(result.score, 0.5)

    def test_pii_exposure(self):
        g = GDPRCheck()
        result = g.evaluate(answer="User email is john@example.com and SSN is 123-45-6789")
        self.assertLess(result.score, 1.0)
        issues = result.details["issues"]
        articles = [i["article"] for i in issues]
        self.assertIn("5(1)(c)", articles)

    def test_erasure_right_not_acknowledged(self):
        g = GDPRCheck()
        result = g.evaluate(
            question="How do I delete my data?",
            answer="Thank you for your question. We store all data securely."
        )
        issues = result.details["issues"]
        articles = [i["article"] for i in issues]
        self.assertIn("17", articles)

    def test_erasure_right_acknowledged(self):
        g = GDPRCheck()
        result = g.evaluate(
            question="How do I delete my data?",
            answer="You can request erasure of your data by contacting our DPO."
        )
        issues = result.details.get("issues", [])
        articles = [i["article"] for i in issues]
        self.assertNotIn("17", articles)


class TestDPDPCheck(unittest.TestCase):

    def test_clean_text(self):
        d = DPDPCheck()
        result = d.evaluate(answer="We follow all data protection guidelines")
        self.assertGreater(result.score, 0.5)

    def test_aadhaar_exposure(self):
        d = DPDPCheck()
        result = d.evaluate(answer="User Aadhaar: 1234 5678 9012")
        self.assertLess(result.score, 1.0)

    def test_pan_exposure(self):
        d = DPDPCheck()
        result = d.evaluate(answer="PAN number is ABCDE1234F")
        self.assertLess(result.score, 1.0)

    def test_children_data_without_consent(self):
        d = DPDPCheck()
        result = d.evaluate(
            answer="We collect student data and share it with partners for targeted advertising."
        )
        self.assertLess(result.score, 1.0)
        issues = result.details["issues"]
        sections = [i["section"] for i in issues]
        self.assertIn("9", sections)


class TestEUAIActCheck(unittest.TestCase):

    def test_clean_text(self):
        e = EUAIActCheck()
        result = e.evaluate(answer="Here is a summary of the document you uploaded.")
        self.assertGreater(result.score, 0.5)

    def test_social_scoring(self):
        e = EUAIActCheck()
        result = e.evaluate(
            answer="We calculate a social score for each citizen based on their behavior."
        )
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.details["risk_level"], "unacceptable")

    def test_high_risk_without_oversight(self):
        e = EUAIActCheck()
        result = e.evaluate(
            answer="Based on the medical diagnosis, the patient should take medication X."
        )
        self.assertLess(result.score, 1.0)
        self.assertTrue(result.details["is_high_risk"])

    def test_high_risk_with_oversight(self):
        e = EUAIActCheck()
        result = e.evaluate(
            answer="Based on the analysis, this may indicate condition X. Please consult a doctor for professional advice."
        )
        issues = [i for i in result.details["issues"] if i["article"] == "14"]
        self.assertEqual(len(issues), 0)


class TestCustomRule(unittest.TestCase):

    def test_keyword_match_fails(self):
        c = CustomRule(
            rule="No API keys in output",
            keywords=["api_key", "secret", "password"],
            use_llm=False,
        )
        result = c.evaluate(answer="Set your api_key=sk-123456")
        self.assertEqual(result.score, 0.0)

    def test_keyword_no_match_passes(self):
        c = CustomRule(
            rule="No API keys in output",
            keywords=["api_key", "secret", "password"],
            use_llm=False,
        )
        result = c.evaluate(answer="The service is running normally")
        self.assertEqual(result.score, 1.0)

    def test_no_keywords_no_llm_returns_zero(self):
        c = CustomRule(rule="Some rule", use_llm=False)
        result = c.evaluate(answer="Some text")
        self.assertEqual(result.score, 0.0)


class TestEvaluatorWithCompliance(unittest.TestCase):

    def test_compliance_preset_hipaa(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hipaa")
        result = e.evaluate(answer="Patient SSN: 123-45-6789")
        self.assertLess(result.overall_score, 0.5)

    def test_compliance_preset_clean(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hipaa")
        result = e.evaluate(answer="The treatment improved outcomes by 20%")
        self.assertEqual(result.overall_score, 1.0)

    def test_compliance_preset_gdpr(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="gdpr")
        result = e.evaluate(answer="User email is john@example.com")
        self.assertLess(result.overall_score, 1.0)

    def test_compliance_all_preset(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="compliance_all")
        result = e.evaluate(answer="Safe output with no personal data")
        self.assertGreater(result.overall_score, 0.5)

    def test_v1_presets_still_work(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="math")
        result = e.evaluate(
            question="What is Python?",
            answer="Python is a programming language.",
            context="Python is a high-level, interpreted programming language.",
        )
        self.assertGreater(result.overall_score, 0.0)
        self.assertIn("bleu", result.metrics)

    def test_mixed_quality_and_compliance(self):
        from llmevalkit import Evaluator
        from llmevalkit.compliance import PIIDetector
        from llmevalkit.metrics.math_metrics import BLEUScore

        e = Evaluator(
            provider="none",
            metrics=[BLEUScore(), PIIDetector()],
        )
        result = e.evaluate(
            answer="Python is a programming language.",
            context="Python is a high-level, interpreted programming language.",
        )
        self.assertIn("bleu", result.metrics)
        self.assertIn("pii_detector", result.metrics)


if __name__ == "__main__":
    unittest.main()
