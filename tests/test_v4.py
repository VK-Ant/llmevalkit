"""Tests for llmevalkit v4: hallucination detection + new multimodal metrics."""

import unittest


# ============================================================
# HALLUCINATION TESTS
# ============================================================

class TestEntityHallucination(unittest.TestCase):

    def test_grounded_entities(self):
        from llmevalkit.hallucination import EntityHallucination
        eh = EntityHallucination()
        result = eh.evaluate(
            answer="Dr. Smith works at Acme Hospital in New York.",
            context="Dr. Smith is a physician at Acme Hospital, located in New York City."
        )
        self.assertGreater(result.score, 0.5)

    def test_hallucinated_entity(self):
        from llmevalkit.hallucination import EntityHallucination
        eh = EntityHallucination()
        result = eh.evaluate(
            answer="Dr. Kumar recommended the treatment at Mayo Clinic.",
            context="Dr. Smith works at Acme Hospital."
        )
        self.assertLess(result.score, 1.0)

    def test_no_context(self):
        from llmevalkit.hallucination import EntityHallucination
        eh = EntityHallucination()
        result = eh.evaluate(answer="Dr. Smith is here.")
        self.assertEqual(result.score, 1.0)


class TestNumericHallucination(unittest.TestCase):

    def test_correct_numbers(self):
        from llmevalkit.hallucination import NumericHallucination
        nh = NumericHallucination()
        result = nh.evaluate(
            answer="The total was $1,250 in 2024.",
            context="Total amount: $1,250.00. Year: 2024."
        )
        self.assertGreater(result.score, 0.5)

    def test_wrong_number(self):
        from llmevalkit.hallucination import NumericHallucination
        nh = NumericHallucination()
        result = nh.evaluate(
            answer="Revenue was $5 million.",
            context="Company reported revenue of $3 million."
        )
        self.assertLess(result.score, 1.0)

    def test_no_numbers(self):
        from llmevalkit.hallucination import NumericHallucination
        nh = NumericHallucination()
        result = nh.evaluate(
            answer="The weather is sunny.",
            context="It is a sunny day."
        )
        self.assertEqual(result.score, 1.0)


class TestNegationHallucination(unittest.TestCase):

    def test_no_flip(self):
        from llmevalkit.hallucination import NegationHallucination
        nh = NegationHallucination()
        result = nh.evaluate(
            answer="The drug is not approved for children.",
            context="The medication is not approved for pediatric use."
        )
        self.assertEqual(result.score, 1.0)

    def test_negation_flip(self):
        from llmevalkit.hallucination import NegationHallucination
        nh = NegationHallucination()
        result = nh.evaluate(
            answer="The drug is approved for children.",
            context="The drug is not approved for children."
        )
        self.assertLess(result.score, 1.0)


class TestFabricatedInfo(unittest.TestCase):

    def test_supported_text(self):
        from llmevalkit.hallucination import FabricatedInfo
        fi = FabricatedInfo()
        result = fi.evaluate(
            answer="Solar energy is renewable and reduces electricity costs.",
            context="Solar energy is a renewable source that lowers electricity costs."
        )
        self.assertGreater(result.score, 0.5)

    def test_fabricated_text(self):
        from llmevalkit.hallucination import FabricatedInfo
        fi = FabricatedInfo()
        result = fi.evaluate(
            answer="Quantum computing will replace all classical computers by 2025.",
            context="Solar energy is a renewable source."
        )
        self.assertLess(result.score, 1.0)


class TestContradictionDetector(unittest.TestCase):

    def test_no_contradiction(self):
        from llmevalkit.hallucination import ContradictionDetector
        cd = ContradictionDetector()
        result = cd.evaluate(
            answer="The project was successful.",
            context="The project achieved all its goals successfully."
        )
        self.assertEqual(result.score, 1.0)

    def test_contradiction(self):
        from llmevalkit.hallucination import ContradictionDetector
        cd = ContradictionDetector()
        result = cd.evaluate(
            answer="The project was a complete failure.",
            context="The project was a great success."
        )
        self.assertLess(result.score, 1.0)


class TestSelfConsistency(unittest.TestCase):

    def test_consistent(self):
        from llmevalkit.hallucination import SelfConsistency
        sc = SelfConsistency()
        result = sc.evaluate(answer=[
            "Python was created in 1991.",
            "Python was created in 1991 by Guido van Rossum.",
        ])
        self.assertGreater(result.score, 0.5)

    def test_inconsistent(self):
        from llmevalkit.hallucination import SelfConsistency
        sc = SelfConsistency()
        result = sc.evaluate(answer=[
            "Python was created in 1991.",
            "Python was created in 1985.",
            "Python was created in 2001.",
        ])
        self.assertLess(result.score, 0.9)

    def test_needs_list(self):
        from llmevalkit.hallucination import SelfConsistency
        sc = SelfConsistency()
        result = sc.evaluate(answer="single output")
        self.assertEqual(result.score, 0.0)


class TestConfidenceCalibration(unittest.TestCase):

    def test_no_confidence_words(self):
        from llmevalkit.hallucination import ConfidenceCalibration
        cc = ConfidenceCalibration()
        result = cc.evaluate(answer="Python is a programming language.")
        self.assertEqual(result.score, 1.0)

    def test_overconfident(self):
        from llmevalkit.hallucination import ConfidenceCalibration
        cc = ConfidenceCalibration()
        result = cc.evaluate(
            answer="The company definitely earned $5 million in revenue.",
            context="Revenue figures are not yet reported."
        )
        self.assertLess(result.score, 1.0)


class TestInstructionHallucination(unittest.TestCase):

    def test_on_topic(self):
        from llmevalkit.hallucination import InstructionHallucination
        ih = InstructionHallucination()
        result = ih.evaluate(
            question="What are the benefits of solar energy?",
            answer="Solar energy is renewable and reduces electricity bills."
        )
        self.assertGreater(result.score, 0.3)

    def test_off_topic(self):
        from llmevalkit.hallucination import InstructionHallucination
        ih = InstructionHallucination()
        result = ih.evaluate(
            question="What are the benefits of solar energy?",
            answer="The stock market crashed in 2008 due to housing crisis."
        )
        self.assertLess(result.score, 0.5)


# ============================================================
# NEW MULTIMODAL TESTS
# ============================================================

class TestDocumentLayoutAccuracy(unittest.TestCase):

    def test_matching_layout(self):
        from llmevalkit.multimodal import DocumentLayoutAccuracy
        dla = DocumentLayoutAccuracy()
        result = dla.evaluate(
            answer="# Invoice\nItem | Qty | Price\nWidget | 10 | $50",
            reference="# Invoice\nItem | Qty | Price\nWidget | 10 | $50"
        )
        self.assertGreater(result.score, 0.5)

    def test_different_layout(self):
        from llmevalkit.multimodal import DocumentLayoutAccuracy
        dla = DocumentLayoutAccuracy()
        result = dla.evaluate(
            answer="Invoice details paragraph text",
            reference="# Invoice\n## Details\nItem | Qty | Price"
        )
        self.assertLess(result.score, 1.0)


class TestMultimodalConsistency(unittest.TestCase):

    def test_consistent(self):
        from llmevalkit.multimodal import MultimodalConsistency
        mc = MultimodalConsistency()
        result = mc.evaluate(
            answer="A brown dog running in a park.",
            reference="Photo shows a brown dog running through a park."
        )
        self.assertGreater(result.score, 0.3)

    def test_inconsistent(self):
        from llmevalkit.multimodal import MultimodalConsistency
        mc = MultimodalConsistency()
        result = mc.evaluate(
            answer="A car driving on a highway.",
            reference="Photo of a cat sleeping on a couch."
        )
        self.assertLess(result.score, 0.5)


# ============================================================
# V4 PRESET TESTS
# ============================================================

class TestV4Presets(unittest.TestCase):

    def test_hallucination_preset(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination")
        self.assertEqual(len(e.metrics), 8)

    def test_hallucination_quick(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination_quick")
        self.assertEqual(len(e.metrics), 3)

    def test_hallucination_rag(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination_rag")
        self.assertEqual(len(e.metrics), 4)

    def test_hallucination_agent(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination_agent")
        self.assertEqual(len(e.metrics), 3)

    def test_hallucination_medical(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="hallucination_medical")
        self.assertEqual(len(e.metrics), 4)

    def test_multimodal_updated(self):
        from llmevalkit import Evaluator
        e = Evaluator(provider="none", preset="multimodal")
        self.assertEqual(len(e.metrics), 6)

    def test_v1_v2_v3_still_work(self):
        from llmevalkit import Evaluator
        for preset in ["math", "hipaa", "doceval", "security", "governance"]:
            e = Evaluator(provider="none", preset=preset)
            self.assertGreater(len(e.metrics), 0)


if __name__ == "__main__":
    unittest.main()
