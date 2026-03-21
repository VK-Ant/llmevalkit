"""
All 15 metrics example for llmevalkit.

This file shows how to use each metric one by one.
Part A runs without any API key (free).
Part B needs a Groq API key (or OpenAI, Azure, etc).

Install:
    pip install llmevalkit

Run:
    python all_15_metrics.py
"""

import os


# Sample data used throughout all examples.
question = "What are the benefits of solar energy?"

answer = (
    "Solar energy is a renewable source that reduces electricity bills "
    "and has low maintenance costs. It helps reduce carbon emissions "
    "and contributes to environmental sustainability. Solar panels "
    "can last 25-30 years with minimal upkeep."
)

context = (
    "Solar energy is a renewable source of energy that reduces reliance "
    "on fossil fuels. It can significantly lower electricity bills and "
    "has relatively low maintenance costs. Solar panels help reduce "
    "carbon emissions and contribute to environmental sustainability. "
    "Modern solar panels have a lifespan of 25-30 years."
)


def show(name, result):
    """Print a metric result in a readable way."""
    print("  {}: {:.3f}".format(name, result.score))
    if result.reason:
        print("  Reason: {}".format(result.reason[:90]))
    print()


# ------------------------------------------------------------------
# Part A: Math metrics (free, no API key, runs instantly)
# ------------------------------------------------------------------

def run_math_metrics():
    print("Part A: Math metrics (free, no API)")
    print("-" * 50)

    # 1. BLEU score
    # Measures how many word sequences from the answer match the reference.
    from llmevalkit import BLEUScore

    bleu = BLEUScore()
    result = bleu.evaluate(answer=answer, context=context)
    show("1. BLEUScore", result)
    print("  Precisions per n-gram: {}".format(result.details.get("precisions")))
    print()

    # 2. ROUGE score
    # Measures how much of the reference content appears in the answer.
    from llmevalkit import ROUGEScore

    rouge = ROUGEScore()
    result = rouge.evaluate(answer=answer, context=context)
    show("2. ROUGEScore", result)
    print("  ROUGE-1 F1: {}".format(result.details["rouge1"]["f1"]))
    print("  ROUGE-2 F1: {}".format(result.details["rouge2"]["f1"]))
    print("  ROUGE-L F1: {}".format(result.details["rougeL"]["f1"]))
    print()

    # 3. Token overlap
    # Simple word overlap after removing common words like "the", "is", "a".
    from llmevalkit import TokenOverlap

    overlap = TokenOverlap()
    result = overlap.evaluate(answer=answer, context=context)
    show("3. TokenOverlap", result)
    print("  Common words: {}".format(result.details.get("common_tokens", [])[:8]))
    print("  Missing from answer: {}".format(result.details.get("reference_only", [])[:5]))
    print()

    # 4. Semantic similarity
    # Checks if two texts mean the same thing, even with different words.
    # Uses sentence-transformers if installed, otherwise bag-of-words.
    from llmevalkit import SemanticSimilarity

    sim = SemanticSimilarity()
    result = sim.evaluate(answer=answer, context=context)
    show("4. SemanticSimilarity", result)
    print("  Method used: {}".format(result.details.get("method")))
    print()

    # 5. Keyword coverage
    # What percentage of important words from the context appear in the answer.
    from llmevalkit import KeywordCoverage

    kw = KeywordCoverage()
    result = kw.evaluate(answer=answer, context=context)
    show("5. KeywordCoverage", result)
    print("  Covered: {}".format(result.details.get("covered", [])))
    print("  Missing: {}".format(result.details.get("missing", [])))
    print()

    # 6. Answer length
    # Is the answer too short or too long?
    from llmevalkit import AnswerLength

    length = AnswerLength(min_words=10, max_words=200)
    result = length.evaluate(answer=answer)
    show("6. AnswerLength", result)
    print("  Word count: {}".format(result.details.get("word_count")))

    # Also test with a very short answer to see the penalty.
    short_result = length.evaluate(answer="Yes.")
    print("  Short answer test: score={:.3f}, words={}".format(
        short_result.score, short_result.details.get("word_count")
    ))
    print()

    # 7. Readability
    # How easy is the text to read? Uses the Flesch-Kincaid formula.
    from llmevalkit import ReadabilityScore

    read = ReadabilityScore()
    result = read.evaluate(answer=answer)
    show("7. ReadabilityScore", result)
    print("  Flesch Reading Ease: {}".format(result.details.get("flesch_reading_ease")))
    print("  Grade level: {}".format(result.details.get("flesch_kincaid_grade")))
    print("  Level: {}".format(result.details.get("level")))
    print()


# ------------------------------------------------------------------
# Part B: LLM-as-judge metrics (needs API key)
# ------------------------------------------------------------------

def run_llm_metrics():
    print("Part B: LLM-as-judge metrics (using Groq)")
    print("-" * 50)

    from llmevalkit import Evaluator, Faithfulness, Hallucination
    from llmevalkit import AnswerRelevance, ContextRelevance
    from llmevalkit import Coherence, Completeness, Toxicity, GEval

    # Change provider and model to whatever you use.
    # provider="openai", model="gpt-4o-mini"
    # provider="groq", model="llama-3.3-70b-versatile"
    # provider="ollama", model="llama3.1"
    provider = "groq"
    model = "llama-3.3-70b-versatile"

    # 8. Faithfulness
    # Is every claim in the answer supported by the context?
    print("8. Faithfulness")
    e = Evaluator(provider=provider, model=model, metrics=[Faithfulness()])
    r = e.evaluate(question=question, answer=answer, context=context)
    m = r.metrics["faithfulness"]
    print("  Score: {:.3f}".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 9. Hallucination
    # Are there fabricated or incorrect claims?
    # Works WITHOUT context (reference-free).
    # Higher score = fewer hallucinations = better.
    print("9. Hallucination")
    e = Evaluator(provider=provider, model=model, metrics=[Hallucination()])
    r = e.evaluate(question=question, answer=answer)
    m = r.metrics["hallucination"]
    print("  Score: {:.3f} (higher = less hallucination)".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 10. Answer relevance
    # Does the answer actually address the question?
    print("10. AnswerRelevance")
    e = Evaluator(provider=provider, model=model, metrics=[AnswerRelevance()])
    r = e.evaluate(question=question, answer=answer)
    m = r.metrics["answer_relevance"]
    print("  Score: {:.3f}".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 11. Context relevance
    # Is the retrieved context useful for this question?
    print("11. ContextRelevance")
    e = Evaluator(provider=provider, model=model, metrics=[ContextRelevance()])
    r = e.evaluate(question=question, answer=answer, context=context)
    m = r.metrics["context_relevance"]
    print("  Score: {:.3f}".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 12. Coherence
    # Is the answer well-structured and easy to follow?
    print("12. Coherence")
    e = Evaluator(provider=provider, model=model, metrics=[Coherence()])
    r = e.evaluate(question=question, answer=answer)
    m = r.metrics["coherence"]
    print("  Score: {:.3f}".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 13. Completeness
    # Does the answer cover all parts of the question?
    print("13. Completeness")
    e = Evaluator(provider=provider, model=model, metrics=[Completeness()])
    r = e.evaluate(question=question, answer=answer, context=context)
    m = r.metrics["completeness"]
    print("  Score: {:.3f}".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 14. Toxicity
    # Is the content safe and appropriate?
    # Higher score = less toxic = better.
    print("14. Toxicity")
    e = Evaluator(provider=provider, model=model, metrics=[Toxicity()])
    r = e.evaluate(question=question, answer=answer)
    m = r.metrics["toxicity"]
    print("  Score: {:.3f} (higher = safer)".format(m.score))
    print("  Reason: {}".format(m.reason[:90]))
    print()

    # 15. GEval (custom criteria)
    # You write the rules. The LLM judges against your rules.
    print("15. GEval (custom criteria)")
    e = Evaluator(
        provider=provider,
        model=model,
        metrics=[
            GEval(criteria="Is the response helpful for someone considering solar energy?"),
            GEval(criteria="Does the answer include specific facts or numbers?"),
        ],
    )
    r = e.evaluate(question=question, answer=answer)
    for name, m in r.metrics.items():
        print("  {}: {:.3f}".format(name, m.score))
        print("  Reason: {}".format(m.reason[:90]))
    print()


# ------------------------------------------------------------------
# Part C: Using the Evaluator with presets
# ------------------------------------------------------------------

def run_preset_examples():
    print("Part C: Using presets")
    print("-" * 50)

    from llmevalkit import Evaluator

    # Math preset runs all 7 math metrics at once, no API needed.
    print("Math preset (all 7 math metrics, free):")
    e = Evaluator(provider="none", preset="math")
    r = e.evaluate(question=question, answer=answer, context=context)
    for name, m in r.metrics.items():
        print("  {:<22} {:.3f}".format(name, m.score))
    print("  Overall: {:.3f}".format(r.overall_score))
    print()


# ------------------------------------------------------------------
# Part D: Batch evaluation
# ------------------------------------------------------------------

def run_batch_example():
    print("Part D: Batch evaluation")
    print("-" * 50)

    from llmevalkit import Evaluator

    e = Evaluator(provider="none", preset="math")

    cases = [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language for web and data science.",
            "context": "Python is a high-level, interpreted programming language.",
        },
        {
            "question": "What is Python?",
            "answer": "Yes.",
            "context": "Python is a high-level, interpreted programming language.",
        },
        {
            "question": "What is Python?",
            "answer": "JavaScript is used for building websites and mobile apps.",
            "context": "Python is a high-level, interpreted programming language.",
        },
    ]

    batch = e.evaluate_batch(cases, show_progress=False)

    for i, r in enumerate(batch.results):
        print("  Case {}: score={:.3f} passed={}".format(i + 1, r.overall_score, r.passed))

    print()
    print("  Average: {:.3f}".format(batch.average_score))
    print("  Pass rate: {:.0%}".format(batch.pass_rate))
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("llmevalkit v1.0.0 - All 15 Metrics Example")
    print("=" * 50)
    print()

    run_math_metrics()
    run_preset_examples()
    run_batch_example()

    if os.getenv("GROQ_API_KEY"):
        run_llm_metrics()
    else:
        print("Part B: Skipped (no GROQ_API_KEY found)")
        print("  To run LLM metrics:")
        print("  export GROQ_API_KEY='gsk_...'")
        print()

    print("Done.")
