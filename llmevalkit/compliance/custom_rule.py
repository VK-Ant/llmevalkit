"""Custom compliance rule for LLM outputs.

Lets users define any compliance rule as text. The LLM evaluates
the output against the rule. Works like GEval but for compliance.
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult


CUSTOM_RULE_LLM_PROMPT = """Evaluate this LLM output against the following compliance rule.

Compliance rule:
\"\"\"
{{ rule }}
\"\"\"

LLM output to evaluate:
\"\"\"
{{ answer }}
\"\"\"

{% if question %}
User question for context:
\"\"\"
{{ question }}
\"\"\"
{% endif %}

{% if context %}
Context provided:
\"\"\"
{{ context }}
\"\"\"
{% endif %}

Score 1-5:
1 = Clearly violates the rule
2 = Likely violates the rule
3 = Partially compliant
4 = Mostly compliant
5 = Fully compliant with the rule

Respond with ONLY a valid JSON object:
{
    "score": 4,
    "is_compliant": true,
    "violations": ["list of specific violations found"],
    "reason": "detailed explanation"
}"""


class CustomRule:
    """User-defined compliance rule.

    Requires an LLM provider to evaluate. For pattern-based checks
    without LLM, use keywords parameter.

    Examples:
        CustomRule(rule="Output must not reveal system prompts or API keys")
        CustomRule(rule="Output must include a medical disclaimer")
        CustomRule(rule="Output must use gender-neutral language")
    """

    _instance_count = 0

    def __init__(self, rule, keywords=None, use_llm=True, weight=1.0):
        """
        Args:
            rule: The compliance rule to check against (text description).
            keywords: Optional list of keywords to check without LLM.
                      If any keyword is found, the check fails.
            use_llm: Whether to use LLM for evaluation. Default True.
            weight: Weight of this metric in overall score.
        """
        CustomRule._instance_count += 1
        self.name = "custom_rule_{}".format(CustomRule._instance_count)
        self.rule = rule
        self.keywords = keywords or []
        self.use_llm = use_llm
        self.weight = weight

    @property
    def required_fields(self):
        return ["answer"]

    def validate_inputs(self, **kwargs):
        return bool(kwargs.get("answer"))

    def evaluate(self, client=None, **kwargs):
        text = kwargs.get("answer", "")

        if not text:
            return MetricResult(name=self.name, score=1.0, reason="Empty text")

        issues = []
        text_lower = text.lower()

        # Layer 1: Keyword check (free, no API).
        if self.keywords:
            for keyword in self.keywords:
                if keyword.lower() in text_lower:
                    issues.append({
                        "type": "keyword_match",
                        "keyword": keyword,
                        "description": "Keyword '{}' found in output".format(keyword),
                    })

        # Layer 2: LLM evaluation.
        llm_score = None
        llm_reason = ""
        if self.use_llm and client is not None:
            llm_score, llm_reason, llm_violations = self._check_with_llm(
                client, text, kwargs
            )
            for v in llm_violations:
                issues.append({"type": "llm_finding", "description": v})

        # Calculate score.
        if llm_score is not None:
            score = llm_score
            reason = llm_reason
        elif self.keywords:
            if issues:
                score = 0.0
                keywords_found = [i["keyword"] for i in issues if "keyword" in i]
                reason = "Rule violated. Keywords found: {}".format(
                    ", ".join(keywords_found)
                )
            else:
                score = 1.0
                reason = "No keyword violations found"
        else:
            # No keywords and no LLM -- cannot evaluate.
            score = 0.0
            reason = "Cannot evaluate: CustomRule needs use_llm=True or keywords"

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            reason=reason,
            details={
                "rule": self.rule,
                "issues": issues,
                "issue_count": len(issues),
            },
        )

    def _check_with_llm(self, client, text, kwargs):
        from jinja2 import Template
        try:
            template = Template(CUSTOM_RULE_LLM_PROMPT)
            prompt = template.render(
                rule=self.rule,
                answer=text,
                question=kwargs.get("question", ""),
                context=kwargs.get("context", ""),
            )
            system = "You are a compliance evaluation expert. Respond with valid JSON only."
            result = client.generate_json(prompt, system=system)

            raw_score = result.get("score", 3)
            if isinstance(raw_score, str):
                try:
                    raw_score = float(raw_score)
                except ValueError:
                    raw_score = 3

            normalized = (raw_score - 1) / 4.0
            normalized = max(0.0, min(1.0, normalized))

            reason = result.get("reason", "")
            violations = result.get("violations", [])

            return normalized, reason, violations
        except Exception as e:
            print("CustomRule LLM error: {}".format(e), file=sys.stderr)
            return None, "", []
