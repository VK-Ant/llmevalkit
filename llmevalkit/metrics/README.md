# LLMEVAL Metrics Reference

This document explains all 15 metrics: what each one measures, why it matters,
how it works, and how to use it in code.

All algorithms used (BLEU, ROUGE, Flesch-Kincaid, cosine similarity, F1) are
public domain techniques from published academic papers. No copyrighted code
is used. LLMEVAL is 100% original implementation.

---

## Part A: Math Metrics (No API Required)

These 7 metrics use math formulas only. They cost nothing, run instantly,
and work offline. Good for CI/CD pipelines and batch testing.


### 1. BLEUScore

**What it measures:**
How many word sequences (n-grams) from the answer also appear in the reference.
Commonly used in machine translation and text generation.

**Why it matters:**
If the answer shares many word sequences with the reference, it is likely
saying similar things. Low BLEU means the answer uses very different wording.

**How it works:**
1. Split both texts into words.
2. For n = 1, 2, 3, 4: count matching n-grams between answer and reference.
3. Calculate precision at each level.
4. Apply brevity penalty if the answer is shorter than the reference.
5. Final score = brevity_penalty * geometric_mean(precisions).

**Formula:**
BLEU = BP * exp( (1/N) * sum(log(precision_n)) )
BP = min(1, exp(1 - reference_length / answer_length))

**Needs:** answer + context (or reference)

**Score guide:**
- 1.0 = word sequences match perfectly
- 0.5 = moderate overlap
- 0.0 = no matching word sequences

**Code:**
```python
from llmevalkit import BLEUScore

bleu = BLEUScore(max_n=4)
result = bleu.evaluate(
    answer="Solar panels convert sunlight into electricity.",
    context="Solar panels use photovoltaic cells to convert sunlight into electrical energy."
)
print(result.score)                          # 0.412
print(result.details["precisions"])          # [0.71, 0.50, 0.33, 0.20]
print(result.details["brevity_penalty"])     # 0.85
```


### 2. ROUGEScore

**What it measures:**
How much of the reference content is captured in the answer.
BLEU measures precision (how much of your answer is correct).
ROUGE measures recall (how much of the reference did you cover).

**Why it matters:**
For summarization and RAG, you want to know: did the answer include
the important information from the source? ROUGE checks this.

**How it works:**
1. ROUGE-1: unigram overlap F1 (single words).
2. ROUGE-2: bigram overlap F1 (word pairs).
3. ROUGE-L: longest common subsequence F1.
4. Final = 0.4 * ROUGE-1 + 0.3 * ROUGE-2 + 0.3 * ROUGE-L.

**Formula:**
ROUGE-N recall = matching_ngrams / total_reference_ngrams
ROUGE-N F1 = 2 * precision * recall / (precision + recall)

**Needs:** answer + context (or reference)

**Score guide:**
- 1.0 = all reference content captured
- 0.5 = moderate coverage
- 0.0 = no overlap

**Code:**
```python
from llmevalkit import ROUGEScore

rouge = ROUGEScore()
result = rouge.evaluate(
    answer="Python is a popular high-level programming language.",
    context="Python is a high-level, interpreted programming language known for simplicity."
)
print(result.score)
print(result.details["rouge1"])    # {"precision": ..., "recall": ..., "f1": ...}
print(result.details["rouge2"])
print(result.details["rougeL"])
```


### 3. TokenOverlap

**What it measures:**
Simple word-level overlap after removing common stopwords (the, is, a, etc.).
Focuses on content words that carry meaning.

**Why it matters:**
Fast sanity check. If answer and context share no content words,
something is wrong with the generation or retrieval.

**How it works:**
1. Tokenize both texts, convert to lowercase.
2. Remove 100+ English stopwords.
3. Precision = common words / answer words.
4. Recall = common words / reference words.
5. F1 = 2 * precision * recall / (precision + recall).

**Formula:**
F1 = 2 * P * R / (P + R)

**Needs:** answer + context (or reference)

**Code:**
```python
from llmevalkit import TokenOverlap

overlap = TokenOverlap()
result = overlap.evaluate(
    answer="Machine learning enables computers to learn from data.",
    context="Machine learning is a method where computers learn patterns from data."
)
print(result.score)
print(result.details["common_tokens"])
print(result.details["reference_only"])
```


### 4. SemanticSimilarity

**What it measures:**
Meaning similarity using vector embeddings. Two sentences can use different
words but mean the same thing ("cat sleeping" vs "feline resting").

**Why it matters:**
Word overlap misses paraphrases. Semantic similarity catches them.
More accurate than token-level metrics for evaluating meaning.

**How it works:**
1. If sentence-transformers is installed: encode both texts into vectors,
   calculate cosine similarity.
2. If not installed: uses bag-of-words cosine similarity as fallback.
Both run locally, no API call.

**Formula:**
cosine_similarity = dot(A, B) / (norm(A) * norm(B))

**Needs:** answer + context (or reference)

**Code:**
```python
from llmevalkit import SemanticSimilarity

sim = SemanticSimilarity()
result = sim.evaluate(
    answer="The cat is sleeping on the couch.",
    reference="A feline is resting on the sofa."
)
print(result.score)
print(result.details["method"])
```


### 5. KeywordCoverage

**What it measures:**
What percentage of important words from the context appear in the answer.

**Why it matters:**
If the context mentions "photovoltaic", "electricity", "sunlight" but the
answer only mentions "electricity", important information is missing.
Quick proxy for completeness without needing an LLM.

**How it works:**
1. Extract non-stopword tokens from context.
2. Filter out words shorter than 3 characters.
3. Check which keywords appear in the answer.
4. Score = covered / total.

**Needs:** answer + context

**Code:**
```python
from llmevalkit import KeywordCoverage

kw = KeywordCoverage()
result = kw.evaluate(
    answer="Solar panels reduce electricity costs.",
    context="Solar panels use photovoltaic cells to convert sunlight into electricity."
)
print(result.score)
print(result.details["covered"])    # ["solar", "panels", "electricity"]
print(result.details["missing"])    # ["photovoltaic", "cells", "sunlight", "convert"]
```


### 6. AnswerLength

**What it measures:**
Whether the answer is within an acceptable word count range.

**Why it matters:**
Catches two common failures: (1) answers that are too short ("Yes", "I don't know")
and (2) answers that are excessively long and verbose.

**How it works:**
1. Count words.
2. Below minimum: score = word_count / min_words.
3. Above maximum: gradual penalty.
4. Within range: score = 1.0.

**Needs:** answer only

**Code:**
```python
from llmevalkit import AnswerLength

length = AnswerLength(min_words=10, max_words=200)

result = length.evaluate(answer="Yes.")
print(result.score)                        # low (too short)
print(result.details["word_count"])        # 1

result = length.evaluate(answer="Solar energy provides renewable power for homes and businesses.")
print(result.score)                        # 1.0 (good length)
```


### 7. ReadabilityScore

**What it measures:**
How easy the text is to read, using the Flesch-Kincaid formula.

**Why it matters:**
If your AI outputs text at a college-graduate reading level but your
audience is general public, the output is not usable. This metric catches that.

**How it works:**
1. Count sentences, words, and syllables.
2. Flesch Reading Ease = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words).
3. Also calculates grade level.
4. Normalized from 0-100 to 0-1 scale.

**Formula:**
FRE = 206.835 - 1.015 * (W/S) - 84.6 * (Syl/W)

**Needs:** answer only

**Code:**
```python
from llmevalkit import ReadabilityScore

read = ReadabilityScore()
result = read.evaluate(answer="The cat sat on the mat. The dog ran.")
print(result.score)
print(result.details["flesch_reading_ease"])
print(result.details["flesch_kincaid_grade"])
print(result.details["level"])
```

---

## Part B: LLM-as-Judge Metrics (API Required)

These 8 metrics send a structured prompt to an LLM (GPT-4, Llama, Claude, etc.)
and ask it to judge the answer. The LLM returns a JSON score (1-5) which gets
normalized to 0-1. These give deeper, more nuanced evaluation than math metrics.


### 8. Faithfulness

**What it measures:**
Whether every claim in the answer is supported by the provided context.

**Why it matters:**
This is the most important metric for RAG systems. If the answer adds
information not present in the context, users get misleading results.
The LLM breaks the answer into individual claims and checks each one.

**How it works:**
1. LLM extracts claims from the answer.
2. For each claim, checks if the context supports it.
3. Counts supported vs unsupported claims.
4. Score based on the ratio of supported claims.

**Needs:** question + answer + context

**Code:**
```python
from llmevalkit import Evaluator, Faithfulness

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[Faithfulness()])
result = e.evaluate(
    question="What is the refund policy?",
    answer="Full refund within 30 days. After that, store credit.",
    context="Refund policy: full refunds within 30 days. After 30 days, store credit."
)
print(result.metrics["faithfulness"].score)
print(result.metrics["faithfulness"].reason)
```


### 9. Hallucination

**What it measures:**
Whether there are fabricated, incorrect, or unverifiable claims in the answer.

**Why it matters:**
Hallucination is the biggest risk in LLM outputs. This metric catches
made-up facts, incorrect numbers, and confident claims about things
that are not true.

**Key feature:** Works WITHOUT context (reference-free). The LLM uses its
own knowledge to check facts. Score is inverted: higher = fewer hallucinations = better.

**How it works:**
1. LLM identifies fabricated statements.
2. Classifies each as fabrication, contradiction, or unverifiable.
3. Score inverted: 1 (no hallucination) becomes 1.0, 5 (all hallucinated) becomes 0.0.

**Needs:** question + answer (context optional)

**Code:**
```python
from llmevalkit import Evaluator, Hallucination

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[Hallucination()])
result = e.evaluate(
    question="When was Python created?",
    answer="Python was created by Guido van Rossum in 1991."
)
print(result.metrics["hallucination"].score)
```


### 10. AnswerRelevance

**What it measures:**
Whether the answer actually addresses the question that was asked.

**Why it matters:**
An answer can be factually correct but still useless if it does not
answer the question. Example: asked "What is the capital of France?"
and getting a response about French cuisine.

**Needs:** question + answer

**Code:**
```python
from llmevalkit import Evaluator, AnswerRelevance

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[AnswerRelevance()])
result = e.evaluate(
    question="What is the capital of France?",
    answer="Paris is the capital of France."
)
print(result.metrics["answer_relevance"].score)
```


### 11. ContextRelevance

**What it measures:**
Whether the retrieved context contains useful information for the question.

**Why it matters:**
If your retriever returns irrelevant documents, even a perfect LLM will
produce bad answers. This metric diagnoses retrieval quality separately
from generation quality.

**Needs:** question + context

**Code:**
```python
from llmevalkit import Evaluator, ContextRelevance

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[ContextRelevance()])
result = e.evaluate(
    question="How do solar panels work?",
    answer="",
    context="Solar panels contain photovoltaic cells that convert sunlight into electricity."
)
print(result.metrics["context_relevance"].score)
```


### 12. Coherence

**What it measures:**
Whether the answer is logically structured, clear, and easy to follow.

**Why it matters:**
Even accurate information is useless if it is presented in a confusing,
disorganized way. This uses LLM judgment (not just readability formulas)
to assess logical flow and transitions.

**Needs:** question + answer

**Code:**
```python
from llmevalkit import Evaluator, Coherence

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[Coherence()])
result = e.evaluate(
    question="Explain machine learning.",
    answer="ML is a subset of AI. It learns from data. First you provide training data. "
           "Then the algorithm finds patterns. Finally it predicts on new data."
)
print(result.metrics["coherence"].score)
```


### 13. Completeness

**What it measures:**
Whether the answer covers ALL aspects of the question.

**Why it matters:**
A partial answer that only addresses one part of a multi-part question
should score low. Unlike KeywordCoverage (which counts words), this uses
LLM judgment to assess conceptual coverage.

**Needs:** question + answer (context optional)

**Code:**
```python
from llmevalkit import Evaluator, Completeness

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[Completeness()])
result = e.evaluate(
    question="What are the pros AND cons of remote work?",
    answer="Remote work offers flexibility."   # only pros, no cons
)
print(result.metrics["completeness"].score)    # low score
```


### 14. Toxicity

**What it measures:**
Whether the content is safe, professional, and appropriate.

**Why it matters:**
Any customer-facing AI must not produce harmful, biased, or offensive content.
Score is inverted: higher = less toxic = better.

**Needs:** answer only

**Code:**
```python
from llmevalkit import Evaluator, Toxicity

e = Evaluator(provider="groq", model="llama-3.1-70b-versatile",
              metrics=[Toxicity()])
result = e.evaluate(
    answer="Solar energy benefits communities worldwide."
)
print(result.metrics["toxicity"].score)    # high = safe
```


### 15. GEval (Custom Criteria)

**What it measures:**
Whatever YOU want. You write the evaluation criteria as text, and the LLM
judges the answer against your criteria.

**Why it matters:**
This makes LLMEVAL work for any domain. Education, legal, support,
code review, content writing -- you define the rules.

**Needs:** question + answer + your criteria text

**Code:**
```python
from llmevalkit import Evaluator, GEval

e = Evaluator(
    provider="groq",
    model="llama-3.1-70b-versatile",
    metrics=[
        GEval(criteria="Is the response empathetic?"),
        GEval(criteria="Does it provide clear next steps?"),
        GEval(criteria="Is the tone professional?"),
    ],
)
result = e.evaluate(
    question="My order is 2 weeks late!",
    answer="I am sorry for the delay. I have escalated your order. "
           "You will get a tracking update within 24 hours."
)
for name, m in result.metrics.items():
    print(name, m.score)
```

---

## How to run all 15 together

```python
from llmevalkit import (
    Evaluator,
    BLEUScore, ROUGEScore, TokenOverlap, SemanticSimilarity,
    KeywordCoverage, AnswerLength, ReadabilityScore,
    Faithfulness, Hallucination, AnswerRelevance, ContextRelevance,
    Coherence, Completeness, Toxicity, GEval,
)

evaluator = Evaluator(
    provider="groq",
    model="llama-3.1-70b-versatile",
    metrics=[
        BLEUScore(), ROUGEScore(), TokenOverlap(), SemanticSimilarity(),
        KeywordCoverage(), AnswerLength(), ReadabilityScore(),
        Faithfulness(), Hallucination(), AnswerRelevance(), ContextRelevance(),
        Coherence(), Completeness(), Toxicity(),
        GEval(criteria="Is this helpful for a beginner?"),
    ],
)

result = evaluator.evaluate(
    question="What are the benefits of solar energy?",
    answer="Solar energy is renewable and reduces electricity bills.",
    context="Solar energy is a renewable source that lowers electricity costs."
)

print(result.summary())
```
