"""llmevalkit observability module.

Auto-logging, score drift detection, threshold alerts,
model comparison, and evaluation reports.

All data stored as local JSON files. No server, no database,
no internet needed.
"""

from __future__ import annotations

import json
import os
import math
from datetime import datetime

from llmevalkit.models import MetricResult


DEFAULT_LOG_DIR = os.path.expanduser("~/.llmevalkit/logs")


class EvalLogger:
    """Auto-save evaluation results to JSON files.

    Used internally by Evaluator when auto_log=True (default).
    """

    name = "eval_logger"

    def __init__(self, log_dir=None):
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "eval_history.jsonl")

    def log(self, result_dict):
        """Append one evaluation result to the log file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": result_dict.get("overall_score", 0.0),
            "passed": result_dict.get("passed", False),
            "metrics": {},
        }
        for name, metric_data in result_dict.get("metrics", {}).items():
            if hasattr(metric_data, "score"):
                entry["metrics"][name] = metric_data.score
            elif isinstance(metric_data, dict):
                entry["metrics"][name] = metric_data.get("score", 0.0)

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Silent fail -- logging should never break evaluation

    def read_logs(self, last_n=None):
        """Read evaluation history."""
        if not os.path.exists(self.log_file):
            return []
        entries = []
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            return []
        if last_n:
            entries = entries[-last_n:]
        return entries

    def clear(self):
        """Clear all logs."""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)


class ScoreDrift:
    """Detect if evaluation scores are dropping over time.

    Compares recent evaluations vs earlier evaluations.
    """

    name = "score_drift"

    def __init__(self, log_dir=None, window=50):
        self.logger = EvalLogger(log_dir)
        self.window = window

    def check(self):
        """Check for score drift. Returns dict with drift analysis."""
        entries = self.logger.read_logs()

        if len(entries) < self.window * 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least {} evaluations for drift detection (have {})".format(
                    self.window * 2, len(entries)),
                "total_evaluations": len(entries),
            }

        recent = entries[-self.window:]
        previous = entries[-(self.window * 2):-self.window]

        # Compare overall scores
        recent_avg = sum(e["overall_score"] for e in recent) / len(recent)
        prev_avg = sum(e["overall_score"] for e in previous) / len(previous)
        drift = recent_avg - prev_avg

        # Per-metric drift
        metric_drifts = {}
        all_metrics = set()
        for e in recent + previous:
            all_metrics.update(e.get("metrics", {}).keys())

        for metric in all_metrics:
            r_scores = [e["metrics"].get(metric, None) for e in recent]
            p_scores = [e["metrics"].get(metric, None) for e in previous]
            r_scores = [s for s in r_scores if s is not None]
            p_scores = [s for s in p_scores if s is not None]
            if r_scores and p_scores:
                r_avg = sum(r_scores) / len(r_scores)
                p_avg = sum(p_scores) / len(p_scores)
                metric_drifts[metric] = {
                    "recent_avg": round(r_avg, 4),
                    "previous_avg": round(p_avg, 4),
                    "drift": round(r_avg - p_avg, 4),
                }

        # Find biggest drops
        drops = {k: v for k, v in metric_drifts.items() if v["drift"] < -0.05}

        status = "degrading" if drift < -0.05 else "stable" if abs(drift) < 0.05 else "improving"

        return {
            "status": status,
            "overall_drift": round(drift, 4),
            "recent_avg": round(recent_avg, 4),
            "previous_avg": round(prev_avg, 4),
            "metric_drifts": metric_drifts,
            "biggest_drops": drops,
            "window_size": self.window,
            "total_evaluations": len(entries),
        }


class ThresholdAlert:
    """Alert when metrics drop below defined thresholds."""

    name = "threshold_alert"

    def __init__(self, thresholds=None, log_dir=None, window=50):
        self.thresholds = thresholds or {}
        self.logger = EvalLogger(log_dir)
        self.window = window

    def check(self):
        """Check recent evaluations against thresholds."""
        if not self.thresholds:
            return {"status": "no_thresholds", "message": "No thresholds defined"}

        entries = self.logger.read_logs(last_n=self.window)
        if not entries:
            return {"status": "no_data", "message": "No evaluation history found"}

        alerts = []
        for metric, threshold in self.thresholds.items():
            scores = [e["metrics"].get(metric, None) for e in entries]
            scores = [s for s in scores if s is not None]
            if not scores:
                continue

            below = sum(1 for s in scores if s < threshold)
            avg = sum(scores) / len(scores)

            if below > 0:
                alerts.append({
                    "metric": metric,
                    "threshold": threshold,
                    "avg_score": round(avg, 4),
                    "below_count": below,
                    "total_checked": len(scores),
                    "message": "{} below {} in {} of {} evaluations".format(
                        metric, threshold, below, len(scores)),
                })

        return {
            "status": "alerts" if alerts else "ok",
            "alerts": alerts,
            "thresholds": self.thresholds,
            "evaluations_checked": len(entries),
        }


class EvalComparison:
    """Compare two evaluation results side by side."""

    name = "eval_comparison"

    @staticmethod
    def compare(result_a, result_b, label_a="Model A", label_b="Model B"):
        """Compare two EvaluationResult objects."""
        comparison = {
            "label_a": label_a,
            "label_b": label_b,
            "overall_a": result_a.overall_score,
            "overall_b": result_b.overall_score,
            "overall_diff": round(result_b.overall_score - result_a.overall_score, 4),
            "winner": label_b if result_b.overall_score > result_a.overall_score else label_a,
            "metrics": {},
        }

        all_metrics = set(result_a.metrics.keys()) | set(result_b.metrics.keys())
        for metric in sorted(all_metrics):
            score_a = result_a.metrics[metric].score if metric in result_a.metrics else None
            score_b = result_b.metrics[metric].score if metric in result_b.metrics else None
            diff = round(score_b - score_a, 4) if score_a is not None and score_b is not None else None

            comparison["metrics"][metric] = {
                label_a: score_a,
                label_b: score_b,
                "diff": diff,
                "better": label_b if diff and diff > 0 else label_a if diff and diff < 0 else "tie",
            }

        return comparison


class EvalReport:
    """Generate summary report from evaluation history."""

    name = "eval_report"

    def __init__(self, log_dir=None):
        self.logger = EvalLogger(log_dir)

    def summary(self, last_n=None):
        """Generate evaluation summary."""
        entries = self.logger.read_logs(last_n=last_n)

        if not entries:
            return {"status": "no_data", "message": "No evaluation history found"}

        overall_scores = [e["overall_score"] for e in entries]
        pass_count = sum(1 for e in entries if e.get("passed", False))

        # Per-metric stats
        metric_stats = {}
        all_metrics = set()
        for e in entries:
            all_metrics.update(e.get("metrics", {}).keys())

        for metric in sorted(all_metrics):
            scores = [e["metrics"].get(metric, None) for e in entries]
            scores = [s for s in scores if s is not None]
            if scores:
                metric_stats[metric] = {
                    "avg": round(sum(scores) / len(scores), 4),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                    "count": len(scores),
                }

        # Find worst metric
        worst = min(metric_stats.items(), key=lambda x: x[1]["avg"]) if metric_stats else ("none", {"avg": 0})

        return {
            "total_evaluations": len(entries),
            "pass_rate": round(pass_count / len(entries), 4) if entries else 0,
            "avg_score": round(sum(overall_scores) / len(overall_scores), 4),
            "min_score": round(min(overall_scores), 4),
            "max_score": round(max(overall_scores), 4),
            "worst_metric": worst[0],
            "worst_avg": worst[1]["avg"],
            "metric_stats": metric_stats,
            "first_eval": entries[0].get("timestamp", ""),
            "last_eval": entries[-1].get("timestamp", ""),
        }
