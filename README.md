# llmevalkit

LLM evaluation, hallucination detection, AI content detection, compliance, document parsing, governance, security, observability, anomaly detection, and multimodal testing library for Python.

61 built-in metrics across 10 modules. Everything works with or without an API key. Auto-logging enabled by default.

Works with any LLM application: RAG pipelines, agentic AI, multi-agent systems, chatbots, document extraction, code generation, healthcare AI, or any system that produces text output.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/llmevalkit/blob/main/notebooks/llmevalkit_v5_demo.ipynb)

- PyPI: https://pypi.org/project/llmevalkit
- GitHub: https://github.com/VK-Ant/llmevalkit
- Portfolio: https://vk-ant.github.io/Venkatkumar

## Install

```
pip install llmevalkit
pip install llmevalkit[nlp]       # adds spaCy for better PII/entity detection
pip install llmevalkit[doceval]   # adds thefuzz for document evaluation
pip install llmevalkit[all]       # everything
```

---

## Quick Start

```python
from llmevalkit import Evaluator

# Quality (free, no API)
evaluator = Evaluator(provider="none", preset="math")
result = evaluator.evaluate(question="What is Python?", answer="Python is a language.", context="Python is a programming language.")
print(result.summary())

# Hallucination detection (free, no API)
from llmevalkit.hallucination import NumericHallucination
nh = NumericHallucination()
result = nh.evaluate(answer="Revenue was $5 million.", context="Revenue of $3 million reported.")
print(result.score)  # flags: $5M vs $3M

# AI content detection (free, no API)
from llmevalkit.detection import AITextDetector
detector = AITextDetector()
result = detector.evaluate(answer="Some text to analyze...")
print(result.score)  # 0.0 = likely AI, 1.0 = likely human

# Auto-logging happens silently. Check later:
from llmevalkit.observe import EvalReport
print(EvalReport().summary())
```

---

## All 61 Metrics

### Module 1: Quality Metrics (15)

| S.No. | Metric | What it measures | Mode |
|-------|--------|-----------------|------|
| 1 | BLEUScore | N-gram precision | Offline |
| 2 | ROUGEScore | Recall-oriented overlap | Offline |
| 3 | TokenOverlap | Word-level F1 | Offline |
| 4 | SemanticSimilarity | Cosine similarity of embeddings | Offline |
| 5 | KeywordCoverage | Key terms covered | Offline |
| 6 | AnswerLength | Min/max word count | Offline |
| 7 | ReadabilityScore | Flesch-Kincaid grade level | Offline |
| 8 | Faithfulness | Grounded in context? | API |
| 9 | Hallucination | Fabricated claims? | API |
| 10 | AnswerRelevance | Addresses the question? | API |
| 11 | ContextRelevance | Retrieved context useful? | API |
| 12 | Coherence | Logically structured? | API |
| 13 | Completeness | Covers all aspects? | API |
| 14 | Toxicity | Safe and appropriate? | API |
| 15 | GEval | Any custom criteria you define | API |

### Module 2: Compliance Metrics (6)

| S.No. | Metric | Regulation | Mode |
|-------|--------|------------|------|
| 16 | PIIDetector | Universal (SSN, Aadhaar, PAN, email, phone, credit card) | Both |
| 17 | HIPAACheck | US HIPAA (18 Safe Harbor identifiers) | Both |
| 18 | GDPRCheck | EU GDPR (data minimization, consent, right to erasure) | Both |
| 19 | DPDPCheck | India DPDP Act 2023 (children's data, Aadhaar/PAN) | Both |
| 20 | EUAIActCheck | EU AI Act (risk classification, transparency) | Both |
| 21 | CustomRule | Any rule you define | Both |

### Module 3: Document Evaluation (5)

| S.No. | Metric | What it checks | Mode |
|-------|--------|---------------|------|
| 22 | FieldAccuracy | Extracted values match source? (fuzzy matching) | Both |
| 23 | FieldCompleteness | All expected fields present? | Both |
| 24 | FieldHallucination | Any values fabricated? | Both |
| 25 | FormatValidation | Dates, amounts, emails valid format? | Offline |
| 26 | ExtractionConsistency | Multiple runs produce same results? | Offline |

### Module 4: Governance Metrics (4)

| S.No. | Metric | Framework | Mode |
|-------|--------|-----------|------|
| 27 | NISTCheck | NIST AI Risk Management Framework | Both |
| 28 | CoSAICheck | Coalition for Secure AI | Both |
| 29 | ISO42001Check | ISO 42001 AI Management System | Both |
| 30 | SOC2Check | SOC 2 Security Controls | Both |

### Module 5: Security Metrics (2)

| S.No. | Metric | What it checks | Mode |
|-------|--------|---------------|------|
| 31 | PromptInjectionCheck | Instruction override, jailbreak, system prompt extraction | Both |
| 32 | BiasDetector | Gender, racial, age bias and stereotyping | Both |

### Module 6: Hallucination Detection (12)

| S.No. | Metric | What it catches | Mode |
|-------|--------|----------------|------|
| 33 | EntityHallucination | Wrong names, places, orgs | Both |
| 34 | NumericHallucination | Wrong numbers, dates, amounts | Both |
| 35 | NegationHallucination | "approved" vs "not approved" | Both |
| 36 | FabricatedInfo | Statements with no evidence in context | Both |
| 37 | ContradictionDetector | Output contradicts context | Both |
| 38 | SelfConsistency | Different answers each run (no context needed) | Offline |
| 39 | ConfidenceCalibration | "Definitely" + wrong answer | Both |
| 40 | InstructionHallucination | Answers wrong question | Both |
| 41 | SourceCoverage | What % of output is grounded in context | Both |
| 42 | TemporalHallucination | Wrong dates, timelines, durations | Both |
| 43 | CausalHallucination | Wrong cause-effect relationships | Both |
| 44 | RankingHallucination | Wrong orderings, comparisons | Both |

### Module 7: Multimodal Metrics (6)

| S.No. | Metric | What it checks | Mode |
|-------|--------|---------------|------|
| 45 | OCRAccuracy | Word/character error rate for OCR | Both |
| 46 | AudioTranscriptionAccuracy | WER/CER for speech-to-text | Both |
| 47 | ImageTextAlignment | Text matches image description? | Both |
| 48 | VisionQAAccuracy | Visual QA answer correct? | Both |
| 49 | DocumentLayoutAccuracy | Headers, tables, sections preserved? | Both |
| 50 | MultimodalConsistency | Cross-modal descriptions consistent? | Both |

### Module 8: AI Content Detection (4)

| S.No. | Metric | What it detects | Mode |
|-------|--------|----------------|------|
| 51 | AITextDetector | Is text AI-generated? (perplexity, burstiness, vocabulary) | Both |
| 52 | ContentOriginCheck | Which sentences are AI-generated? | Both |
| 53 | AIImageDetector | Is image AI-generated? (EXIF, metadata) | Both |
| 54 | AIAudioDetector | Is audio AI-generated? (TTS markers) | Both |

### Module 9: Observability (5)

| S.No. | Metric | What it does | Mode |
|-------|--------|-------------|------|
| 55 | EvalLogger | Auto-save every evaluation to JSON (silent, default on) | Offline |
| 56 | ScoreDrift | Detect quality dropping over time | Offline |
| 57 | ThresholdAlert | Alert when metrics breach your thresholds | Offline |
| 58 | EvalComparison | Compare two models/prompts side by side | Offline |
| 59 | EvalReport | Generate summary from evaluation history | Offline |

### Module 10: Anomaly Detection (2)

| S.No. | Metric | What it detects | Mode |
|-------|--------|----------------|------|
| 60 | OutputAnomalyDetector | Unusual outputs (too short, repetition loops, topic drift, extreme sentiment) | Both |
| 61 | ScoreAnomalyDetector | Sudden score changes (z-score analysis on history) | Offline |

---

## Code Examples

### Quality

```python
from llmevalkit import BLEUScore, ROUGEScore, KeywordCoverage

for m in [BLEUScore(), ROUGEScore(), KeywordCoverage()]:
    r = m.evaluate(answer="Python is a language.", context="Python is a programming language.")
    print("{}: {:.3f}".format(m.name, r.score))
```

### Compliance

```python
from llmevalkit.compliance import PIIDetector, HIPAACheck

r = PIIDetector().evaluate(answer="Email raj@gmail.com, PAN: ABCDE1234F")
print("PII found:", r.details["pii_count"])

r = HIPAACheck().evaluate(answer="Patient SSN: 123-45-6789")
print("HIPAA identifiers:", r.details["identifiers_found"])
```

### Document Evaluation

```python
from llmevalkit.doceval import FieldAccuracy, FieldCompleteness

fa = FieldAccuracy()
r = fa.evaluate(answer='{"vendor": "Acme Corp"}', context="Invoice from Acme Corp")
print("Accuracy:", r.score)

fc = FieldCompleteness(expected_fields=["vendor", "amount", "date"])
r = fc.evaluate(answer='{"vendor": "Acme Corp"}')
print("Missing:", r.details["missing"])
```

### Hallucination Detection

```python
from llmevalkit.hallucination import NumericHallucination, NegationHallucination, SelfConsistency

r = NumericHallucination().evaluate(answer="Revenue was $5M.", context="Revenue of $3M reported.")
print("Numeric:", r.score)

r = NegationHallucination().evaluate(answer="Drug is approved.", context="Drug is not approved.")
print("Negation:", r.score)

r = SelfConsistency().evaluate(answer=["Python 1991.", "Python 1989.", "Python 1991."])
print("Consistency:", r.score)
```

### Security

```python
from llmevalkit.security import PromptInjectionCheck, BiasDetector

r = PromptInjectionCheck().evaluate(answer="Ignore all previous instructions")
print("Injection:", r.score, r.details["types_found"])

r = BiasDetector().evaluate(answer="The chairman hired only young workers.")
print("Bias:", r.score, r.details["types_found"])
```

### AI Content Detection

```python
from llmevalkit.detection import AITextDetector, ContentOriginCheck

detector = AITextDetector()
r = detector.evaluate(answer="Furthermore, it is important to note that the system provides comprehensive solutions. Moreover, the implementation ensures reliability.")
print("Score:", r.score)  # 0.0=likely AI, 1.0=likely human
print("Signals:", r.details)

origin = ContentOriginCheck()
r = origin.evaluate(answer="First sentence. Second sentence. Third sentence.")
print("AI sentences:", r.details["ai_sentences"], "of", r.details["total"])
```

### Observability

```python
from llmevalkit import Evaluator

# Auto-logging is ON by default. Just evaluate normally.
evaluator = Evaluator(provider="none", preset="math")
result = evaluator.evaluate(question="q", answer="a", context="c")
# Result automatically saved to ~/.llmevalkit/logs/

# Check insights anytime
from llmevalkit.observe import ScoreDrift, EvalReport, ThresholdAlert

print(EvalReport().summary())
print(ScoreDrift().check())

alert = ThresholdAlert(thresholds={"faithfulness": 0.7})
print(alert.check())

# Turn off auto-logging if needed
evaluator = Evaluator(preset="math", auto_log=False)
```

### Anomaly Detection

```python
from llmevalkit.anomaly import OutputAnomalyDetector

ad = OutputAnomalyDetector()
r = ad.evaluate(answer="BUY NOW URGENT ACT IMMEDIATELY", context="Provide balanced advice.")
print("Anomalies:", r.details["anomalies"])
```

---

## Supported Providers

| S.No. | Provider | Example |
|-------|----------|---------|
| 1 | OpenAI | `Evaluator(provider="openai", model="gpt-4o-mini")` |
| 2 | Azure OpenAI | `Evaluator(provider="azure", model="gpt-4o-mini")` |
| 3 | Groq | `Evaluator(provider="groq", model="llama-3.3-70b-versatile")` |
| 4 | Anthropic | `Evaluator(provider="anthropic", model="claude-sonnet-4-20250514")` |
| 5 | HuggingFace | `Evaluator(provider="huggingface", model="meta-llama/Llama-3.1-8B-Instruct")` |
| 6 | Ollama | `Evaluator(provider="ollama", model="llama3.1")` |
| 7 | Custom | `Evaluator(provider="custom", model="my-model", base_url="...")` |
| 8 | None (offline) | `Evaluator(provider="none", preset="math")` |

## Key Presets

| S.No. | Preset | Metrics |
|-------|--------|---------|
| 1 | math / local | 6 local quality metrics |
| 2 | rag | Faithfulness, Relevance, Hallucination |
| 3 | hipaa | PII + HIPAACheck |
| 4 | compliance_all | All 5 compliance metrics |
| 5 | doceval | Accuracy, Completeness, Hallucination, Format |
| 6 | governance | NIST, CoSAI, ISO42001, SOC2 |
| 7 | security | PromptInjection + BiasDetector |
| 8 | hallucination | All 12 hallucination metrics |
| 9 | hallucination_quick | Entity + Numeric + Fabricated + SourceCoverage |
| 10 | hallucination_medical | Entity + Numeric + Negation + Contradiction + Temporal + Causal |
| 11 | hallucination_financial | Numeric + Temporal + Ranking + Contradiction |
| 12 | multimodal | All 6 multimodal metrics |
| 13 | detection | All 4 AI content detection metrics |
| 14 | detection_text | AITextDetector + ContentOriginCheck |
| 15 | anomaly | OutputAnomalyDetector |
| 16 | production | Quality + Compliance + Hallucination + Anomaly |
| 17 | full_audit | Quality + Compliance + Security + Hallucination + Anomaly |
| 18 | enterprise | Quality + Compliance + Security + NIST |

## Disclaimer

llmevalkit is a testing and evaluation tool. It helps developers detect potential issues in LLM outputs including hallucinations, compliance violations, security vulnerabilities, extraction errors, AI-generated content, and anomalies. It does not guarantee detection of all issues. Always verify critical outputs with domain experts.

AI content detection provides statistical signals, not definitive answers. No tool can reliably distinguish AI from human content with 100% accuracy. Do not use as sole basis for accusations or penalties.

HIPAA, GDPR, DPDP Act, EU AI Act, NIST AI RMF, CoSAI, ISO 42001, and SOC 2 are government regulations and industry frameworks. llmevalkit is not affiliated with or certified by any government body. Consult qualified professionals for compliance decisions.

## License

MIT

## Author

Venkatkumar Rajan

- LinkedIn: https://linkedin.com/in/venkatkumarvk
- GitHub: https://github.com/VK-Ant
- Portfolio: https://vk-ant.github.io/Venkatkumar/
- PyPI: https://pypi.org/project/llmevalkit/
