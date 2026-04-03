"""AI governance framework checks.

NISTCheck: NIST AI Risk Management Framework (Govern, Map, Measure, Manage)
CoSAICheck: Coalition for Secure AI security posture
ISO42001Check: AI management system standard
SOC2Check: Security controls for AI systems

All work with keyword matching (offline) or LLM analysis (deeper).
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult


# NIST AI RMF keywords for each function
NIST_KEYWORDS = {
    "govern": [
        "governance", "policy", "policies", "oversight", "accountability",
        "responsible", "ethical", "trustworthy", "risk management",
        "organizational", "roles", "responsibilities",
    ],
    "map": [
        "risk", "identify", "assessment", "threat", "vulnerability",
        "impact", "likelihood", "stakeholder", "context", "boundary",
    ],
    "measure": [
        "metric", "measure", "evaluate", "test", "benchmark", "monitor",
        "performance", "accuracy", "fairness", "reliability", "audit",
    ],
    "manage": [
        "mitigate", "control", "remediate", "respond", "plan",
        "incident", "continuous", "improvement", "update", "review",
    ],
}

COSAI_KEYWORDS = [
    "security", "supply chain", "model integrity", "data poisoning",
    "adversarial", "authentication", "authorization", "encryption",
    "access control", "audit trail", "logging", "monitoring",
    "vulnerability", "threat model", "secure deployment",
]

ISO42001_KEYWORDS = [
    "ai management", "management system", "policy", "objectives",
    "risk assessment", "risk treatment", "competence", "awareness",
    "documented information", "operational planning", "performance evaluation",
    "internal audit", "management review", "continual improvement",
    "interested parties", "scope", "leadership", "context",
]

SOC2_KEYWORDS = [
    "security", "availability", "processing integrity", "confidentiality",
    "privacy", "access control", "encryption", "monitoring", "incident response",
    "change management", "risk assessment", "vendor management",
    "data classification", "backup", "disaster recovery",
]


GOVERNANCE_LLM_PROMPT = """Analyze this text for alignment with the {{ framework }} framework.

{% if framework == "NIST AI RMF" %}
Check alignment with the four NIST AI RMF functions:
1. GOVERN: Is AI governance, policy, and accountability addressed?
2. MAP: Are AI risks identified and assessed?
3. MEASURE: Are AI systems measured, evaluated, and monitored?
4. MANAGE: Are risks mitigated and managed with action plans?
{% elif framework == "CoSAI" %}
Check for AI security posture:
1. Model integrity and protection
2. Supply chain security
3. Adversarial robustness
4. Secure deployment practices
{% elif framework == "ISO 42001" %}
Check alignment with AI management system requirements:
1. Leadership and commitment to responsible AI
2. Risk assessment and treatment
3. Operational controls and procedures
4. Performance evaluation and improvement
{% elif framework == "SOC 2" %}
Check for security controls:
1. Security (access controls, encryption)
2. Availability (uptime, disaster recovery)
3. Processing integrity (accuracy, completeness)
4. Confidentiality and privacy
{% endif %}

Text to analyze:
\"\"\"
{{ answer }}
\"\"\"

Respond with ONLY a valid JSON object:
{
    "alignment": [
        {"area": "area name", "present": true, "evidence": "what was found"}
    ],
    "score": 3,
    "summary": "brief assessment"
}

Score 1-5 where 1 = no alignment, 5 = strong alignment."""


def _check_keywords(text, keywords):
    """Check how many keywords from a list appear in text."""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    return found


def _governance_evaluate(name, framework, keywords_dict_or_list, use_llm, client, text):
    """Common evaluation logic for governance metrics."""
    if not text:
        return MetricResult(name=name, score=0.0, reason="Empty text")

    areas = {}

    if isinstance(keywords_dict_or_list, dict):
        # Multiple areas (like NIST with govern/map/measure/manage)
        for area, keywords in keywords_dict_or_list.items():
            found = _check_keywords(text, keywords)
            areas[area] = {
                "keywords_found": found,
                "count": len(found),
                "total": len(keywords),
                "coverage": len(found) / len(keywords) if keywords else 0.0,
            }
        total_coverage = sum(a["coverage"] for a in areas.values()) / len(areas)
    else:
        # Single keyword list
        found = _check_keywords(text, keywords_dict_or_list)
        areas["overall"] = {
            "keywords_found": found,
            "count": len(found),
            "total": len(keywords_dict_or_list),
            "coverage": len(found) / len(keywords_dict_or_list) if keywords_dict_or_list else 0.0,
        }
        total_coverage = areas["overall"]["coverage"]

    # LLM analysis
    llm_result = None
    if use_llm and client:
        llm_result = _llm_check(client, text, framework)
        if llm_result:
            raw_score = llm_result.get("score", 3)
            if isinstance(raw_score, str):
                try:
                    raw_score = float(raw_score)
                except ValueError:
                    raw_score = 3
            total_coverage = (raw_score - 1) / 4.0
            total_coverage = max(0.0, min(1.0, total_coverage))

    score = round(total_coverage, 4)

    if score >= 0.6:
        reason = "Good {} alignment ({:.0%} coverage)".format(framework, score)
    elif score >= 0.3:
        reason = "Partial {} alignment ({:.0%} coverage)".format(framework, score)
    else:
        reason = "Low {} alignment ({:.0%} coverage)".format(framework, score)

    details = {"areas": areas, "framework": framework}
    if llm_result:
        details["llm_assessment"] = llm_result.get("summary", "")
        details["llm_alignment"] = llm_result.get("alignment", [])

    return MetricResult(name=name, score=score, reason=reason, details=details)


def _llm_check(client, text, framework):
    from jinja2 import Template
    try:
        template = Template(GOVERNANCE_LLM_PROMPT)
        prompt = template.render(answer=text, framework=framework)
        system = "You are an AI governance expert. Respond with valid JSON only."
        return client.generate_json(prompt, system=system)
    except Exception as e:
        print("Governance LLM error: {}".format(e), file=sys.stderr)
        return None


class NISTCheck:
    """Check alignment with NIST AI Risk Management Framework.

    Tests the four functions: Govern, Map, Measure, Manage.
    """
    name = "nist_check"
    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight
    @property
    def required_fields(self):
        return ["answer"]
    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))
    def evaluate(self, client=None, **kwargs):
        return _governance_evaluate(
            self.name, "NIST AI RMF", NIST_KEYWORDS,
            self.use_llm, client, kwargs.get("answer", ""))


class CoSAICheck:
    """Check alignment with Coalition for Secure AI framework."""
    name = "cosai_check"
    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight
    @property
    def required_fields(self):
        return ["answer"]
    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))
    def evaluate(self, client=None, **kwargs):
        return _governance_evaluate(
            self.name, "CoSAI", COSAI_KEYWORDS,
            self.use_llm, client, kwargs.get("answer", ""))


class ISO42001Check:
    """Check alignment with ISO 42001 AI management system standard."""
    name = "iso42001_check"
    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight
    @property
    def required_fields(self):
        return ["answer"]
    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))
    def evaluate(self, client=None, **kwargs):
        return _governance_evaluate(
            self.name, "ISO 42001", ISO42001_KEYWORDS,
            self.use_llm, client, kwargs.get("answer", ""))


class SOC2Check:
    """Check alignment with SOC 2 security controls for AI systems."""
    name = "soc2_check"
    def __init__(self, use_llm=False, weight=1.0):
        self.use_llm = use_llm
        self.weight = weight
    @property
    def required_fields(self):
        return ["answer"]
    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))
    def evaluate(self, client=None, **kwargs):
        return _governance_evaluate(
            self.name, "SOC 2", SOC2_KEYWORDS,
            self.use_llm, client, kwargs.get("answer", ""))
