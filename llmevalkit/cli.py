"""LLMEVAL Command Line Interface.

Usage:
    llmeval evaluate --question "..." --answer "..." --context "..."
    llmeval evaluate --file test_cases.json
    llmeval info
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(
        prog="llmevalkit",
        description="LLMEVAL — Comprehensive LLM Evaluation Library",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--question", "-q", type=str, help="Question text")
    eval_parser.add_argument("--answer", "-a", type=str, help="Answer text")
    eval_parser.add_argument("--context", "-c", type=str, default="", help="Context text")
    eval_parser.add_argument("--reference", "-r", type=str, default=None, help="Reference answer")
    eval_parser.add_argument("--file", "-f", type=str, help="JSON file with test cases")
    eval_parser.add_argument("--provider", type=str, default="openai", help="LLM provider")
    eval_parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    eval_parser.add_argument("--preset", type=str, default="rag", help="Metric preset")
    eval_parser.add_argument("--threshold", type=float, default=0.5, help="Pass/fail threshold")
    eval_parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    eval_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # info command
    subparsers.add_parser("info", help="Show LLMEVAL info and available metrics")

    args = parser.parse_args()

    if args.command == "info":
        _show_info()
    elif args.command == "evaluate":
        _run_evaluate(args)
    else:
        parser.print_help()


def _show_info():
    from llmeval import __version__
    from llmevalkit.evaluator import METRIC_PRESETS

    console.print(f"\n[bold blue]LLMEVAL[/] v{__version__}")
    console.print("Comprehensive LLM Evaluation Library\n")
    console.print("[bold]Available Presets:[/]")
    for name, metrics in METRIC_PRESETS.items():
        metric_names = ", ".join(m.name if hasattr(m, "name") else m.__name__ for m in metrics)
        console.print(f"  [cyan]{name:<16}[/] → {metric_names}")
    console.print("\n[bold]Supported Providers:[/]")
    console.print("  openai, azure, anthropic, groq, ollama, custom\n")


def _run_evaluate(args):
    from llmeval import Evaluator

    evaluator = Evaluator(
        provider=args.provider,
        model=args.model,
        preset=args.preset,
        threshold=args.threshold,
        verbose=args.verbose,
    )

    if args.file:
        with open(args.file, "r") as f:
            test_cases = json.load(f)
        if not isinstance(test_cases, list):
            test_cases = [test_cases]
        batch_result = evaluator.evaluate_batch(test_cases)

        if args.output:
            with open(args.output, "w") as f:
                json.dump([r.to_dict() for r in batch_result.results], f, indent=2)
            console.print(f"\n[green]Results saved to {args.output}[/]")
        else:
            evaluator.print_report(batch_result)

    elif args.question and args.answer:
        result = evaluator.evaluate(
            question=args.question,
            answer=args.answer,
            context=args.context,
            reference=args.reference,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"\n[green]Result saved to {args.output}[/]")
        else:
            evaluator.print_report(result)
    else:
        console.print("[red]Error: Provide --question and --answer, or --file[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
