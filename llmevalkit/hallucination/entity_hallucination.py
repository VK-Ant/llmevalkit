"""Entity hallucination detection.

Checks if entities (names, places, organizations, dates) in the
output actually exist in the source context. If the output mentions
"Dr. Kumar" but the context only has "Dr. Sharma", that is an
entity hallucination.

Works offline with regex + spaCy NER. With API adds LLM judgment.
"""

from __future__ import annotations

import re
import sys

from llmevalkit.models import MetricResult


def _extract_entities_regex(text):
    """Extract entities using regex patterns."""
    entities = set()
    # Capitalized multi-word names (2-4 words)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text):
        entities.add(m.group(1))
    # Titles + names
    for m in re.finditer(r'\b(?:Dr|Mr|Mrs|Ms|Prof|Rev)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text):
        entities.add(m.group(0))
    # Dates
    for m in re.finditer(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', text):
        entities.add(m.group(0))
    for m in re.finditer(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b', text, re.IGNORECASE):
        entities.add(m.group(0))
    return entities


def _extract_entities_spacy(text):
    """Extract entities using spaCy NER if available."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return set()
        doc = nlp(text)
        return {ent.text for ent in doc.ents if ent.label_ in
                ("PERSON", "ORG", "GPE", "LOC", "DATE", "FAC", "NORP")}
    except ImportError:
        return set()


ENTITY_LLM_PROMPT = """Compare the entities in the output against the source context.

Source context:
\"\"\"
{{ context }}
\"\"\"

Output to check:
\"\"\"
{{ answer }}
\"\"\"

List every entity (person, organization, location, date) in the output.
For each, check if it matches the context.

Respond with ONLY valid JSON:
{
    "entities": [
        {"entity": "Dr. Kumar", "type": "PERSON", "in_context": false, "reason": "context says Dr. Sharma"}
    ],
    "hallucinated_count": 1,
    "total_entities": 3,
    "score": 0.67
}"""


class EntityHallucination:
    """Detect entity hallucinations in LLM output.

    EntityHallucination()              -- regex + spaCy, free
    EntityHallucination(use_llm=True)  -- adds LLM analysis
    """

    name = "entity_hallucination"

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
        context = kwargs.get("context", "")

        if not answer:
            return MetricResult(name=self.name, score=1.0, reason="Empty output")

        if not context:
            return MetricResult(name=self.name, score=1.0,
                                reason="No context provided, cannot check entities",
                                details={"note": "Provide context to enable entity checking"})

        if self.use_llm and client:
            return self._check_with_llm(client, answer, context)

        # Offline: extract entities from both texts
        answer_entities = _extract_entities_regex(answer) | _extract_entities_spacy(answer)
        context_entities = _extract_entities_regex(context) | _extract_entities_spacy(context)
        context_lower = context.lower()

        if not answer_entities:
            return MetricResult(name=self.name, score=1.0,
                                reason="No entities found in output",
                                details={"entities_checked": 0})

        results = []
        for entity in answer_entities:
            in_context = (entity.lower() in context_lower or
                          any(entity.lower() in ce.lower() or ce.lower() in entity.lower()
                              for ce in context_entities))
            results.append({
                "entity": entity,
                "in_context": in_context,
            })

        grounded = sum(1 for r in results if r["in_context"])
        total = len(results)
        hallucinated = total - grounded
        score = grounded / total if total > 0 else 1.0

        if hallucinated == 0:
            reason = "All {} entities found in context".format(total)
        else:
            bad = [r["entity"] for r in results if not r["in_context"]]
            reason = "{} of {} entities not in context: {}".format(
                hallucinated, total, ", ".join(bad[:5]))

        return MetricResult(
            name=self.name, score=round(score, 4), reason=reason,
            details={"entities": results, "total": total,
                      "grounded": grounded, "hallucinated": hallucinated})

    def _check_with_llm(self, client, answer, context):
        from jinja2 import Template
        try:
            template = Template(ENTITY_LLM_PROMPT)
            prompt = template.render(answer=answer, context=context)
            result = client.generate_json(prompt, system="You are a hallucination detection expert. Respond with valid JSON only.")
            score = result.get("score", 1.0)
            if isinstance(score, str):
                score = float(score)
            return MetricResult(
                name=self.name, score=round(max(0, min(1, score)), 4),
                reason="LLM analysis: {} hallucinated of {} entities".format(
                    result.get("hallucinated_count", 0), result.get("total_entities", 0)),
                details=result)
        except Exception as e:
            print("EntityHallucination LLM error: {}".format(e), file=sys.stderr)
            return MetricResult(name=self.name, score=0.0, reason="LLM error: {}".format(e))
