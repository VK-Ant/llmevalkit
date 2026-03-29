"""llmevalkit compliance module.

Compliance testing metrics for LLM outputs. Tests for PII exposure,
HIPAA, GDPR, India DPDP Act, and EU AI Act compliance.

All metrics work in two modes:
    Without API (free):  Pattern matching + NLP detection
    With API (deeper):   Pattern + NLP + LLM-based analysis

Usage:
    from llmevalkit.compliance import PIIDetector, HIPAACheck
    from llmevalkit.compliance import GDPRCheck, DPDPCheck, EUAIActCheck
    from llmevalkit.compliance import CustomRule
"""

from llmevalkit.compliance.pii import PIIDetector
from llmevalkit.compliance.hipaa import HIPAACheck
from llmevalkit.compliance.gdpr import GDPRCheck
from llmevalkit.compliance.dpdp import DPDPCheck
from llmevalkit.compliance.eu_ai_act import EUAIActCheck
from llmevalkit.compliance.custom_rule import CustomRule

__all__ = [
    "PIIDetector",
    "HIPAACheck",
    "GDPRCheck",
    "DPDPCheck",
    "EUAIActCheck",
    "CustomRule",
]
