"""Production Evaluation Pipelines — Pipeline, Regression Gate, Release, Golden Datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineStage:
    """A stage in an evaluation pipeline."""
    name: str
    evaluator: Callable[..., Dict[str, Any]]
    enabled: bool = True


@dataclass
class PipelineResult:
    """Result of running an evaluation pipeline."""
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    passed: bool = True
    summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class GoldenDataset:
    """A golden test dataset."""
    name: str
    examples: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

class EvaluationPipeline:
    """Production evaluation pipeline with configurable stages."""

    def __init__(self) -> None:
        self._stages: List[PipelineStage] = []

    def add_stage(
        self, name: str, evaluator: Callable[..., Dict[str, Any]]
    ) -> None:
        """Add a stage to the pipeline."""
        self._stages.append(PipelineStage(name=name, evaluator=evaluator))

    def run_pipeline(
        self, model_fn: Callable[..., Any]
    ) -> PipelineResult:
        """Run all pipeline stages against a model."""
        result = PipelineResult()
        for stage in self._stages:
            if not stage.enabled:
                continue
            stage_result = stage.evaluator(model_fn)
            result.stages[stage.name] = stage_result
            if not stage_result.get("passed", True):
                result.passed = False
        # Summary
        all_scores: List[float] = []
        for sr in result.stages.values():
            if "score" in sr:
                all_scores.append(sr["score"])
        if all_scores:
            result.summary["avg_score"] = sum(all_scores) / len(all_scores)
            result.summary["min_score"] = min(all_scores)
        return result

    def get_pipeline_report(self, results: PipelineResult) -> Dict[str, Any]:
        """Generate a report from pipeline results."""
        return {
            "passed": results.passed,
            "num_stages": len(results.stages),
            "stages": results.stages,
            "summary": results.summary,
        }


# ---------------------------------------------------------------------------
# Regression Gate
# ---------------------------------------------------------------------------

class RegressionGate:
    """CI/CD regression gate for model evaluation."""

    def __init__(self) -> None:
        self._thresholds: Dict[str, float] = {}

    def set_threshold(self, metric: str, threshold: float) -> None:
        """Set threshold for a metric."""
        self._thresholds[metric] = threshold

    def check(self, results: Dict[str, float]) -> Dict[str, Any]:
        """Check results against thresholds."""
        failures: List[str] = []
        details: Dict[str, Dict[str, float]] = {}
        for metric, threshold in self._thresholds.items():
            value = results.get(metric, 0.0)
            passed = value >= threshold
            details[metric] = {
                "value": value,
                "threshold": threshold,
                "passed": passed,
            }
            if not passed:
                failures.append(metric)
        return {
            "passed": len(failures) == 0,
            "failures": failures,
            "details": details,
        }

    def should_block(self, results: Dict[str, float]) -> bool:
        """Return True if results should block deployment."""
        return not self.check(results)["passed"]


# ---------------------------------------------------------------------------
# Release Evaluator
# ---------------------------------------------------------------------------

class ReleaseEvaluator:
    """Pre-release evaluation management."""

    def __init__(self) -> None:
        self._evaluations: Dict[str, Dict[str, Any]] = {}

    def evaluate_release(
        self,
        model_version: str,
        test_suite: Dict[str, float],
    ) -> Dict[str, Any]:
        """Evaluate a model version against a test suite."""
        avg_score = sum(test_suite.values()) / len(test_suite) if test_suite else 0.0
        result = {
            "model_version": model_version,
            "test_suite": test_suite,
            "avg_score": avg_score,
            "ready": avg_score >= 0.8,
        }
        self._evaluations[model_version] = result
        return result

    def compare_versions(
        self, v1: str, v2: str
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        ev1 = self._evaluations.get(v1, {})
        ev2 = self._evaluations.get(v2, {})
        s1 = ev1.get("avg_score", 0.0)
        s2 = ev2.get("avg_score", 0.0)
        return {
            "v1": v1,
            "v2": v2,
            "v1_score": s1,
            "v2_score": s2,
            "delta": s2 - s1,
            "improved": s2 > s1,
        }

    def get_release_readiness(self, model_version: str) -> Dict[str, Any]:
        """Get release readiness status for a model version."""
        ev = self._evaluations.get(model_version)
        if ev is None:
            return {"model_version": model_version, "status": "not_evaluated", "ready": False}
        return {
            "model_version": model_version,
            "status": "ready" if ev["ready"] else "not_ready",
            "ready": ev["ready"],
            "avg_score": ev["avg_score"],
        }


# ---------------------------------------------------------------------------
# Golden Dataset Manager
# ---------------------------------------------------------------------------

class GoldenDatasetManager:
    """Manage golden test datasets for evaluation."""

    def __init__(self) -> None:
        self._datasets: Dict[str, GoldenDataset] = {}

    def create_dataset(
        self, name: str, examples: List[Dict[str, Any]]
    ) -> GoldenDataset:
        """Create a new golden dataset."""
        ds = GoldenDataset(name=name, examples=list(examples))
        self._datasets[name] = ds
        return ds

    def add_example(self, dataset: str, example: Dict[str, Any]) -> bool:
        """Add an example to a dataset. Returns True if successful."""
        ds = self._datasets.get(dataset)
        if ds is None:
            return False
        ds.examples.append(example)
        return True

    def get_dataset(self, name: str) -> Optional[GoldenDataset]:
        """Get a dataset by name."""
        return self._datasets.get(name)

    def validate_dataset(self, name: str) -> Dict[str, Any]:
        """Validate a dataset for completeness."""
        ds = self._datasets.get(name)
        if ds is None:
            return {"valid": False, "error": "Dataset not found"}
        issues: List[str] = []
        for i, ex in enumerate(ds.examples):
            if not isinstance(ex, dict):
                issues.append(f"Example {i} is not a dict")
            elif "input" not in ex and "question" not in ex:
                issues.append(f"Example {i} missing input/question field")
        return {
            "valid": len(issues) == 0,
            "name": name,
            "examples": len(ds.examples),
            "issues": issues,
        }
