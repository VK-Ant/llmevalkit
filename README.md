# llmevalkit

A Python library for evaluating LLM outputs. 15 built-in metrics.
Works with or without an API key.

- 7 math-based metrics: free, instant, runs offline
- 8 LLM-as-judge metrics: uses any LLM provider to evaluate
- Supports: OpenAI, Azure, Anthropic, Groq, Ollama, or no provider at all

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vk-ant/llmevalkit/blob/main/notebooks/llmevalkit_demo.ipynb)

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

| S.No. | Metric | What it measures |
|-------|--------|-----------------|
| 1 | BLEUScore | N-gram precision between answer and reference |
| 2 | ROUGEScore | Recall-oriented overlap (ROUGE-1, 2, L) |
| 3 | TokenOverlap | Word-level F1 with stopword filtering |
| 4 | SemanticSimilarity | Cosine similarity of text embeddings |
| 5 | KeywordCoverage | Percentage of key terms covered |
| 6 | AnswerLength | Whether answer meets min/max word count |
| 7 | ReadabilityScore | Flesch-Kincaid readability grade level |

### LLM-as-judge metrics (needs API)

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

## Supported providers

| S.No. | Provider | Example |
|-------|----------|---------|
| 1 | OpenAI | `Evaluator(provider="openai", model="gpt-4o-mini")` |
| 2 | Azure OpenAI | `Evaluator(provider="azure", model="gpt-4o-mini", api_key="...", base_url="...")` |
| 3 | Groq | `Evaluator(provider="groq", model="llama-3.1-70b-versatile")` |
| 4 | Anthropic | `Evaluator(provider="anthropic", model="claude-sonnet-4-20250514")` |
| 5 | HuggingFace | `Evaluator(provider="huggingface", model="meta-llama/Llama-3.1-8B-Instruct")` |
| 6 | Ollama | `Evaluator(provider="ollama", model="llama3.1")` |
| 7 | Custom | `Evaluator(provider="custom", model="my-model", base_url="http://localhost:8000/v1")` |
| 8 | None (math only) | `Evaluator(provider="none", preset="math")` |

## Presets

| S.No. | Preset | Metrics included |
|-------|--------|-----------------|
| 1 | rag | Faithfulness, AnswerRelevance, ContextRelevance, Hallucination |
| 2 | chatbot | AnswerRelevance, Coherence, Toxicity, Hallucination |
| 3 | safety | Toxicity, Hallucination |
| 4 | summarization | Faithfulness, Completeness, Coherence |
| 5 | math | All 7 math metrics |
| 6 | math_minimal | TokenOverlap, AnswerLength |
| 7 | hybrid_rag | TokenOverlap, BLEU, KeywordCoverage, Faithfulness, Hallucination |

## Batch evaluation

```python
from llmevalkit import Evaluator

evaluator = Evaluator(provider="none", preset="math")
batch = evaluator.evaluate_batch([
    {"question": "What is AI?", "answer": "AI is artificial intelligence.", "context": "..."},
    {"question": "What is ML?", "answer": "ML uses data to learn.", "context": "..."},
])
print(batch.pass_rate)
df = batch.to_dataframe()  # needs pandas
df.to_csv("results.csv")
```

## CLI

```
llmevalkit evaluate --question "What is AI?" --answer "AI is artificial intelligence." --preset math
llmevalkit evaluate --file test_cases.json --output results.json
llmevalkit info
```

## Project structure

```
llmevalkit/
    __init__.py
    evaluator.py
    models.py
    llm_client.py
    prompts.py
    cli.py
    metrics/
        README.md
        base.py
        faithfulness.py
        hallucination.py
        answer_relevance.py
        context_relevance.py
        coherence.py
        completeness.py
        toxicity.py
        geval.py
        math_metrics.py
    utils/
        token_counter.py
tests/
    test_llmeval.py
examples/
    all_15_metrics.py
```

## License

MIT

## Author

Venkatkumar Rajan(VK) - https://linkedin.com/in/venkatkumarvk
