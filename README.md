# LLMEVAL

A Python library for evaluating LLM outputs. 15 built-in metrics.
Works with or without an API key.

- 7 math-based metrics: free, instant, runs offline
- 8 LLM-as-judge metrics: uses any LLM provider to evaluate
- Supports: OpenAI, Azure, Anthropic, Groq, Ollama, or no provider at all

## Install

```
pip install llmevalkit
```

## Quick start

### Math evaluation (free, no API key needed)

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

evaluator = Evaluator(provider="groq", model="llama-3.1-70b-versatile", preset="rag")
result = evaluator.evaluate(
    question="What is Python?",
    answer="Python is a programming language.",
    context="Python is a high-level, interpreted programming language."
)
print(result.summary())
```

### Hybrid (math + LLM together)

```python
from llmevalkit import (
    Evaluator, BLEUScore, ROUGEScore, TokenOverlap,
    Faithfulness, Hallucination, GEval,
)

evaluator = Evaluator(
    provider="groq",
    model="llama-3.1-70b-versatile",
    metrics=[
        BLEUScore(), ROUGEScore(), TokenOverlap(),
        Faithfulness(), Hallucination(),
        GEval(criteria="Is this helpful for a beginner?"),
    ],
)
result = evaluator.evaluate(question="...", answer="...", context="...")
```

## All 15 metrics

See [metrics/README.md](llmevalkit/metrics/README.md) for detailed documentation on each metric including what it measures, how it works, the formula, and a code example.

### Math metrics (no API needed)

| # | Metric | What it measures |
|---|--------|-----------------|
| 1 | BLEUScore | N-gram precision between answer and reference |
| 2 | ROUGEScore | Recall-oriented overlap (ROUGE-1, 2, L) |
| 3 | TokenOverlap | Word-level F1 with stopword filtering |
| 4 | SemanticSimilarity | Cosine similarity of text embeddings |
| 5 | KeywordCoverage | Percentage of key terms covered |
| 6 | AnswerLength | Whether answer meets min/max word count |
| 7 | ReadabilityScore | Flesch-Kincaid readability grade level |

### LLM-as-judge metrics (needs API)

| # | Metric | What it measures |
|---|--------|-----------------|
| 8 | Faithfulness | Is the answer grounded in the context? |
| 9 | Hallucination | Are there fabricated claims? (works without context) |
| 10 | AnswerRelevance | Does the answer address the question? |
| 11 | ContextRelevance | Is the retrieved context useful? |
| 12 | Coherence | Is the answer logically structured? |
| 13 | Completeness | Does the answer cover all aspects? |
| 14 | Toxicity | Is the content safe and appropriate? |
| 15 | GEval | Custom criteria you define |

## Providers

```python
Evaluator(provider="openai", model="gpt-4o-mini")
Evaluator(provider="groq", model="llama-3.1-70b-versatile")
Evaluator(provider="anthropic", model="claude-sonnet-4-20250514")
Evaluator(provider="ollama", model="llama3.1")
Evaluator(provider="none", preset="math")   # no API needed
```

## Presets

```python
Evaluator(preset="rag")           # Faithfulness, AnswerRelevance, ContextRelevance, Hallucination
Evaluator(preset="chatbot")       # AnswerRelevance, Coherence, Toxicity, Hallucination
Evaluator(preset="math")          # All 7 math metrics
Evaluator(preset="hybrid_rag")    # Math + LLM combined
```

## Batch evaluation

```python
batch = evaluator.evaluate_batch([
    {"question": "What is AI?", "answer": "AI is artificial intelligence.", "context": "..."},
    {"question": "What is ML?", "answer": "ML uses data to learn.", "context": "..."},
])
print(batch.pass_rate)
df = batch.to_dataframe()  # needs pandas
df.to_csv("results.csv")
```

## License

MIT

## Author

Venkatkumar Rajan - https://linkedin.com/in/venkatkumarvk
