"""Evaluation at Scale / LLMOps — Distributed eval, Cost tracking, Scheduling, Dashboard."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskAssignment:
    """A task assigned to a worker."""
    task_id: str
    worker_id: str
    result: Optional[Any] = None
    completed: bool = False


@dataclass
class ScheduledJob:
    """A scheduled evaluation job."""
    job_id: str
    model: str
    benchmark: str
    cron_expr: str
    active: bool = True


@dataclass
class MetricEntry:
    """A single metric entry."""
    name: str
    value: float
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Distributed Evaluator
# ---------------------------------------------------------------------------

class DistributedEvaluator:
    """Distribute evaluation work across simulated workers."""

    def __init__(self) -> None:
        self._assignments: Dict[str, TaskAssignment] = {}

    def distribute_work(
        self,
        tasks: List[str],
        workers: List[str],
    ) -> List[TaskAssignment]:
        """Distribute tasks across workers (round-robin)."""
        assignments: List[TaskAssignment] = []
        if not workers:
            return assignments
        for i, task_id in enumerate(tasks):
            worker = workers[i % len(workers)]
            ta = TaskAssignment(task_id=task_id, worker_id=worker)
            assignments.append(ta)
            self._assignments[task_id] = ta
        return assignments

    def collect_results(
        self, task_ids: List[str]
    ) -> Dict[str, Any]:
        """Collect results from distributed tasks."""
        results: Dict[str, Any] = {}
        for tid in task_ids:
            ta = self._assignments.get(tid)
            if ta and ta.completed:
                results[tid] = ta.result
            else:
                results[tid] = None
        return results

    def get_aggregate_metrics(
        self, results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute aggregate metrics from distributed results."""
        numeric_results = [
            v for v in results.values() if isinstance(v, (int, float))
        ]
        if not numeric_results:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": len(numeric_results),
            "mean": sum(numeric_results) / len(numeric_results),
            "min": min(numeric_results),
            "max": max(numeric_results),
        }


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Track evaluation costs."""

    def __init__(self) -> None:
        self._runs: Dict[str, float] = {}
        self._estimates: Dict[str, float] = {}

    def track_run(self, run_id: str, cost: float) -> None:
        """Track cost for a run."""
        self._runs[run_id] = cost

    def get_run_cost(self, run_id: str) -> float:
        """Get cost for a specific run."""
        return self._runs.get(run_id, 0.0)

    def get_cost_report(self, period: str = "all") -> Dict[str, Any]:
        """Get cost report for a period."""
        total = sum(self._runs.values())
        return {
            "period": period,
            "total_cost": total,
            "num_runs": len(self._runs),
            "avg_cost": total / len(self._runs) if self._runs else 0.0,
            "runs": dict(self._runs),
        }

    def estimate_cost(
        self, benchmark: str, model: str
    ) -> float:
        """Estimate cost for running a benchmark on a model."""
        key = f"{model}:{benchmark}"
        if key in self._estimates:
            return self._estimates[key]
        # Simple heuristic: estimate based on model name
        base = 0.01
        if "gpt-4" in model.lower():
            base = 0.10
        elif "gpt-3" in model.lower():
            base = 0.02
        elif "claude" in model.lower():
            base = 0.08
        self._estimates[key] = base
        return base


# ---------------------------------------------------------------------------
# Evaluation Scheduler
# ---------------------------------------------------------------------------

class EvaluationScheduler:
    """Schedule evaluation runs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, ScheduledJob] = {}
        self._counter = 0

    def schedule_evaluation(
        self,
        model: str,
        benchmark: str,
        cron_expr: str,
    ) -> ScheduledJob:
        """Schedule an evaluation job."""
        self._counter += 1
        job_id = f"job_{self._counter}"
        job = ScheduledJob(
            job_id=job_id,
            model=model,
            benchmark=benchmark,
            cron_expr=cron_expr,
        )
        self._jobs[job_id] = job
        return job

    def get_scheduled_jobs(self) -> List[ScheduledJob]:
        """Get all scheduled jobs."""
        return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.active = False
        return True


# ---------------------------------------------------------------------------
# Metrics Dashboard
# ---------------------------------------------------------------------------

class MetricsDashboard:
    """Evaluation metrics dashboard for monitoring."""

    def __init__(self) -> None:
        self._metrics: List[MetricEntry] = []

    def add_metric(self, name: str, value: float, timestamp: float = 0.0) -> None:
        """Add a metric entry."""
        self._metrics.append(MetricEntry(name=name, value=value, timestamp=timestamp))

    def get_metrics(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> List[MetricEntry]:
        """Get metrics with optional filters."""
        if not filters:
            return list(self._metrics)
        result = []
        for m in self._metrics:
            match = True
            if "name" in filters and m.name != filters["name"]:
                match = False
            if "min_value" in filters and m.value < filters["min_value"]:
                match = False
            if "max_value" in filters and m.value > filters["max_value"]:
                match = False
            if match:
                result.append(m)
        return result

    def export_metrics(self, format: str = "dict") -> Any:
        """Export metrics in specified format."""
        if format == "dict":
            return [
                {"name": m.name, "value": m.value, "timestamp": m.timestamp}
                for m in self._metrics
            ]
        elif format == "csv":
            lines = ["name,value,timestamp"]
            for m in self._metrics:
                lines.append(f"{m.name},{m.value},{m.timestamp}")
            return "\n".join(lines)
        return self._metrics

    def compute_trends(
        self, metric_name: str, window: int = 5
    ) -> Dict[str, float]:
        """Compute trend metrics for a specific metric name."""
        values = [m.value for m in self._metrics if m.name == metric_name]
        if not values:
            return {"current": 0.0, "moving_avg": 0.0, "trend": 0.0}
        recent = values[-window:]
        moving_avg = sum(recent) / len(recent)
        if len(values) >= 2:
            earlier = values[-2 * window : -window] if len(values) >= 2 * window else values[: len(values) // 2]
            if earlier:
                earlier_avg = sum(earlier) / len(earlier)
                trend = moving_avg - earlier_avg
            else:
                trend = 0.0
        else:
            trend = 0.0
        return {
            "current": values[-1],
            "moving_avg": moving_avg,
            "trend": trend,
        }
