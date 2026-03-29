# llmevalkit

LLM evaluation and compliance testing library for Python.
21 built-in metrics: 15 quality + 6 compliance.
Works with or without an API key.

- 7 local quality metrics: free, instant, runs offline
- 8 API quality metrics: uses any LLM provider to evaluate
- 6 compliance metrics: PII, HIPAA, GDPR, DPDP Act, EU AI Act, Custom Rules
- Parallel execution: API metrics run simultaneously for speed
- 8 providers: OpenAI, Azure, Groq, Anthropic, HuggingFace, Ollama, Custom, None

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/llmevalkit/blob/main/notebooks/llmevalkit_v2_demo.ipynb)

## Install

```
pip install llmevalkit
```

For deeper PII detection with NLP (optional):

```
pip install llmevalkit[nlp]
python -m spacy download en_core_web_sm
```

## Quick Start

### Quality evaluation (free, no API)

```python
from llmevalkit import Evaluator

evaluator = Evaluator(provider="none", preset="math")
result = evaluator.evaluate(
    question="What is Python?",
    answer="Python is a high-level programming language.",
    context="Python is a high-level, interpreted programming language."
)
print(result.overall_score)
print(result.summary())
```

### LLM-as-judge evaluation (needs API key)

```python
from llmevalkit import Evaluator

evaluator = Evaluator(provider="groq", model="llama-3.3-70b-versatile", preset="rag")
result = evaluator.evaluate(
    question="What is Python?",
    answer="Python is a programming language.",
    context="Python is a high-level, interpreted programming language."
)
print(result.summary())
```

### Compliance testing (free, no API)

```python
from llmevalkit import Evaluator

evaluator = Evaluator(provider="none", preset="hipaa")
result = evaluator.evaluate(
    answer="Patient John Smith, SSN 123-45-6789, was admitted on 03/15/1980."
)
print(result.summary())
# Score: 0.0 -- HIPAA identifiers detected
```

### Quality + Compliance together

```python
from llmevalkit import Evaluator, BLEUScore, ROUGEScore
from llmevalkit.compliance import PIIDetector, HIPAACheck

evaluator = Evaluator(
    provider="none",
    metrics=[BLEUScore(), ROUGEScore(), PIIDetector(), HIPAACheck()],
)
result = evaluator.evaluate(
    answer="Solar energy reduces carbon emissions.",
    context="Solar energy is a renewable source."
)
for name, m in result.metrics.items():
    print("{:<22} {:.3f}".format(name, m.score))
```

### Custom metrics (pick and choose)

```python
from llmevalkit import (
    Evaluator, BLEUScore, ROUGEScore, TokenOverlap,
    Faithfulness, Hallucination, GEval,
)

evaluator = Evaluator(
    provider="groq",
    model="llama-3.3-70b-versatile",
    metrics=[
        BLEUScore(), ROUGEScore(), TokenOverlap(),
        Faithfulness(), Hallucination(),
        GEval(criteria="Is this helpful for a beginner?"),
    ],
)
result = evaluator.evaluate(question="...", answer="...", context="...")
```

## All 21 Metrics

### Local quality metrics (no API needed)

| S.No. | Metric | What it measures |
|-------|--------|-----------------|
| 1 | BLEUScore | N-gram precision between answer and reference |
| 2 | ROUGEScore | Recall-oriented overlap (ROUGE-1, 2, L) |
| 3 | TokenOverlap | Word-level F1 with stopword filtering |
| 4 | SemanticSimilarity | Cosine similarity of text embeddings |
| 5 | KeywordCoverage | Percentage of key terms covered |
| 6 | AnswerLength | Whether answer meets min/max word count |
| 7 | ReadabilityScore | Flesch-Kincaid readability grade level |

### API quality metrics (needs provider)

| S.No. | Metric | What it measures |
|-------|--------|-----------------|
| 8 | Faithfulness | Is the answer grounded in the context? |
| 9 | Hallucination | Are there fabricated claims? (works without context) |
| 10 | AnswerRelevance | Does the answer address the question? |
| 11 | ContextRelevance | Is the retrieved context useful? |
| 12 | Coherence | Is the answer logically structured? |
| 13 | Completeness | Does the answer cover all aspects? |
| 14 | Toxicity | Is the content safe and appropriate? |
| 15 | GEval | Custom criteria you define |

### Compliance metrics (works without API or with API for deeper analysis)

| S.No. | Metric | What it checks | Regulation |
|-------|--------|---------------|------------|
| 16 | PIIDetector | Names, SSN, Aadhaar, PAN, email, phone, credit card, IP | Universal |
| 17 | HIPAACheck | All 18 Safe Harbor identifiers (45 CFR 164.514) | US HIPAA |
| 18 | GDPRCheck | Data minimization, consent, right to erasure, transparency | EU GDPR |
| 19 | DPDPCheck | Aadhaar/PAN exposure, consent, children's data, data principal rights | India DPDP Act 2023 |
| 20 | EUAIActCheck | Risk classification (4 levels), transparency, human oversight, prohibited practices | EU AI Act |
| 21 | CustomRule | Any compliance rule you define (keyword-based or LLM-based) | User-defined |

## Quality Metric Examples

### Individual local metrics

```python
from llmevalkit import BLEUScore, ROUGEScore, TokenOverlap, KeywordCoverage

answer = "Python is a high-level programming language for web and data science."
context = "Python is a high-level, interpreted programming language."

bleu = BLEUScore()
r = bleu.evaluate(answer=answer, context=context)
print("BLEU: {:.3f}".format(r.score))
print("Precisions: {}".format(r.details["precisions"]))

rouge = ROUGEScore()
r = rouge.evaluate(answer=answer, context=context)
print("ROUGE: {:.3f}".format(r.score))
print("ROUGE-1 F1: {}".format(r.details["rouge1"]["f1"]))

overlap = TokenOverlap()
r = overlap.evaluate(answer=answer, context=context)
print("Token Overlap: {:.3f}".format(r.score))

kw = KeywordCoverage()
r = kw.evaluate(answer=answer, context=context)
print("Keyword Coverage: {:.3f}".format(r.score))
print("Missing: {}".format(r.details["missing"]))
```

### Semantic similarity and readability

```python
from llmevalkit import SemanticSimilarity, ReadabilityScore, AnswerLength

sim = SemanticSimilarity()
r = sim.evaluate(answer="Python is a coding language.", context="Python is a programming language.")
print("Similarity: {:.3f}".format(r.score))

read = ReadabilityScore()
r = read.evaluate(answer="Python is a simple language for beginners.")
print("Readability: {:.3f}".format(r.score))
print("Grade level: {}".format(r.details.get("flesch_kincaid_grade")))

length = AnswerLength(min_words=10, max_words=200)
r = length.evaluate(answer="Yes.")
print("Length score: {:.3f}, words: {}".format(r.score, r.details.get("word_count")))
```

### LLM-as-judge metrics

```python
from llmevalkit import Evaluator, Faithfulness, Hallucination, AnswerRelevance, GEval

evaluator = Evaluator(
    provider="groq",
    model="llama-3.3-70b-versatile",
    metrics=[Faithfulness(), Hallucination(), AnswerRelevance()],
)
result = evaluator.evaluate(
    question="What are the benefits of solar energy?",
    answer="Solar energy is renewable and reduces electricity bills.",
    context="Solar energy is a renewable source that lowers electricity costs."
)
for name, m in result.metrics.items():
    print("{}: {:.3f} - {}".format(name, m.score, m.reason[:80]))
```

### GEval with custom criteria

```python
from llmevalkit import Evaluator, GEval

evaluator = Evaluator(
    provider="groq",
    model="llama-3.3-70b-versatile",
    metrics=[
        GEval(criteria="Is the response helpful for someone considering solar energy?"),
        GEval(criteria="Does the answer include specific facts or numbers?"),
    ],
)
result = evaluator.evaluate(
    question="What are the benefits of solar energy?",
    answer="Solar panels can last 25-30 years and reduce electricity bills by 50-75%."
)
```

## Compliance Metric Examples

### PIIDetector

```python
from llmevalkit.compliance import PIIDetector

pii = PIIDetector()                # pattern + NLP, free
pii = PIIDetector(use_llm=True)    # pattern + NLP + LLM, deeper

result = pii.evaluate(
    answer="Contact raj@gmail.com or call +91 98765 43210. PAN: ABCDE1234F."
)
print("Score: {}".format(result.score))  # 0.0 = PII found
for item in result.details["pii_found"]:
    print("  {}: {}".format(item["type"], item["value"]))
```

### HIPAACheck

```python
from llmevalkit.compliance import HIPAACheck

hipaa = HIPAACheck()                # pattern + NLP, free
hipaa = HIPAACheck(use_llm=True)    # adds LLM for contextual analysis

result = hipaa.evaluate(
    answer="Patient SSN: 123-45-6789, MRN: 12345678"
)
print("Identifiers found: {}".format(result.details["identifiers_found"]))
# [7, 8] -- SSN is #7, MRN is #8 in HIPAA's 18 identifiers
```

### GDPRCheck

```python
from llmevalkit.compliance import GDPRCheck

gdpr = GDPRCheck()
result = gdpr.evaluate(
    question="How do I delete my data?",
    answer="We store all data securely."
)
# Flags: Article 17 right to erasure not acknowledged
```

### DPDPCheck

```python
from llmevalkit.compliance import DPDPCheck

dpdp = DPDPCheck()
result = dpdp.evaluate(
    answer="We collect student data for targeted advertising to children."
)
# Flags: Section 9 children's data violation
```

### EUAIActCheck

```python
from llmevalkit.compliance import EUAIActCheck

eu = EUAIActCheck()
result = eu.evaluate(
    answer="We calculate a social score for each citizen."
)
print("Risk level: {}".format(result.details["risk_level"]))  # unacceptable
```

### CustomRule

```python
from llmevalkit.compliance import CustomRule

rule = CustomRule(
    rule="Output must not contain API keys or secrets",
    keywords=["api_key", "secret", "password", "sk-"],
    use_llm=False,
)
result = rule.evaluate(answer="Set your api_key=sk-12345")
# Score: 0.0 (keyword matched)
```

## Supported Providers

| S.No. | Provider | Example |
|-------|----------|---------|
| 1 | OpenAI | `Evaluator(provider="openai", model="gpt-4o-mini")` |
| 2 | Azure OpenAI | `Evaluator(provider="azure", model="gpt-4o-mini", api_key="...", base_url="...")` |
| 3 | Groq | `Evaluator(provider="groq", model="llama-3.3-70b-versatile")` |
| 4 | Anthropic | `Evaluator(provider="anthropic", model="claude-sonnet-4-20250514")` |
| 5 | HuggingFace | `Evaluator(provider="huggingface", model="meta-llama/Llama-3.1-8B-Instruct")` |
| 6 | Ollama | `Evaluator(provider="ollama", model="llama3.1")` |
| 7 | Custom | `Evaluator(provider="custom", model="my-model", base_url="http://localhost:8000/v1")` |
| 8 | None (local only) | `Evaluator(provider="none", preset="math")` |

## Presets

### Quality presets

| S.No. | Preset | Metrics included |
|-------|--------|-----------------|
| 1 | math / local | BLEUScore, ROUGEScore, TokenOverlap, KeywordCoverage, AnswerLength, ReadabilityScore |
| 2 | rag | Faithfulness, AnswerRelevance, ContextRelevance, Hallucination |
| 3 | chatbot | AnswerRelevance, Coherence, Toxicity, Hallucination |
| 4 | summarization | Faithfulness, Completeness, Coherence |
| 5 | safety | Toxicity, Hallucination |
| 6 | hybrid_rag | TokenOverlap, BLEU, KeywordCoverage, Faithfulness, Hallucination |

### Compliance presets

| S.No. | Preset | Metrics included |
|-------|--------|-----------------|
| 7 | pii | PIIDetector |
| 8 | hipaa | PIIDetector, HIPAACheck |
| 9 | gdpr | PIIDetector, GDPRCheck |
| 10 | india / dpdp | PIIDetector, DPDPCheck |
| 11 | eu_ai | PIIDetector, GDPRCheck, EUAIActCheck |
| 12 | compliance_all | PIIDetector, HIPAACheck, GDPRCheck, DPDPCheck, EUAIActCheck |

### Combined presets (quality + compliance)

| S.No. | Preset | Metrics included |
|-------|--------|-----------------|
| 13 | rag_hipaa | Faithfulness, Hallucination, AnswerRelevance, PIIDetector, HIPAACheck |
| 14 | rag_gdpr | Faithfulness, Hallucination, AnswerRelevance, PIIDetector, GDPRCheck |
| 15 | rag_india | Faithfulness, Hallucination, AnswerRelevance, PIIDetector, DPDPCheck |

## Batch Evaluation

```python
from llmevalkit import Evaluator

# Quality batch
evaluator = Evaluator(provider="none", preset="math")
batch = evaluator.evaluate_batch([
    {"question": "What is AI?", "answer": "AI is artificial intelligence.", "context": "AI is..."},
    {"question": "What is AI?", "answer": "Yes.", "context": "AI is..."},
])
print("Pass rate: {:.0%}".format(batch.pass_rate))

# Compliance batch
evaluator = Evaluator(provider="none", preset="hipaa")
batch = evaluator.evaluate_batch([
    {"answer": "Recovery rate improved by 20%."},
    {"answer": "Patient John, SSN 123-45-6789."},
])
print("Pass rate: {:.0%}".format(batch.pass_rate))
```

## CLI

```
llmevalkit evaluate --question "What is AI?" --answer "AI is artificial intelligence." --preset math
llmevalkit info
```

## Disclaimer

llmevalkit is a testing and evaluation tool. It helps developers detect potential compliance issues in LLM outputs. It does not provide legal advice, regulatory certification, or compliance guarantees.

HIPAA, GDPR, DPDP Act, EU AI Act, and NIST AI RMF are government regulations and frameworks. llmevalkit is not affiliated with, endorsed by, or certified by any government body.

Using this library does not make your system compliant with any regulation. Consult qualified legal and compliance professionals for compliance decisions.

## License

MIT

## Author

Venkatkumar Rajan - https://linkedin.com/in/venkatkumarvk | https://github.com/VK-Ant
