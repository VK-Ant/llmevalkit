"""Token counting and cost estimation utilities."""

from __future__ import annotations

from typing import Optional


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for tokenizer selection
    
    Returns:
        Token count
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Rough estimate if tiktoken not available
        return len(text) // 4


# Approximate pricing per 1M tokens (input/output) as of 2025
MODEL_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
    num_metrics: int = 4,
) -> dict[str, float]:
    """Estimate evaluation cost.
    
    Args:
        input_tokens: Tokens in the input (question + context + answer)
        output_tokens: Expected output tokens per metric (default ~200)
        model: Model name
        num_metrics: Number of metrics being evaluated
    
    Returns:
        Dict with cost breakdown
    """
    pricing = MODEL_PRICING.get(model, (1.00, 3.00))  # Default fallback
    
    total_input = input_tokens * num_metrics
    total_output = output_tokens * num_metrics
    
    input_cost = (total_input / 1_000_000) * pricing[0]
    output_cost = (total_output / 1_000_000) * pricing[1]
    
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "model": model,
    }
