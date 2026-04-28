"""Robustness & Reliability — Adversarial testing, calibration, regression."""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PerturbationResult:
    """Result of an adversarial perturbation test."""
    original_text: str
    perturbed_text: str
    method: str
    prediction_changed: bool


@dataclass
class CalibrationBin:
    """A bin in a calibration reliability diagram."""
    lower: float
    upper: float
    mean_confidence: float
    accuracy: float
    count: int


@dataclass
class RegressionReport:
    """Report comparing baseline vs current results."""
    baseline_score: float
    current_score: float
    delta: float
    regressed: bool


# ---------------------------------------------------------------------------
# Adversarial Tester
# ---------------------------------------------------------------------------

class AdversarialTester:
    """Adversarial robustness testing for NLP models."""

    def generate_perturbation(self, text: str, method: str = "typo") -> str:
        """Generate a perturbed version of text using specified method."""
        if not text:
            return text
        if method == "typo":
            chars = list(text)
            # Swap two adjacent characters
            for i in range(len(chars) - 1):
                if chars[i].isalpha() and chars[i + 1].isalpha() and random.random() < 0.15:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
            return "".join(chars)
        elif method == "drop":
            chars = list(text)
            result = [c for c in chars if random.random() > 0.1 or not c.isalpha()]
            return "".join(result)
        elif method == "swap_words":
            words = text.split()
            if len(words) > 1:
                i = random.randint(0, len(words) - 2)
                words[i], words[i + 1] = words[i + 1], words[i]
            return " ".join(words)
        elif method == "repeat":
            words = text.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                words.insert(idx, words[idx])
            return " ".join(words)
        return text

    def run_attack(
        self,
        model_fn: Callable[[str], Any],
        text: str,
        attacks: List[str],
    ) -> List[PerturbationResult]:
        """Run adversarial attacks and check for prediction changes."""
        original_pred = model_fn(text)
        results: List[PerturbationResult] = []
        for method in attacks:
            perturbed = self.generate_perturbation(text, method)
            new_pred = model_fn(perturbed)
            results.append(PerturbationResult(
                original_text=text,
                perturbed_text=perturbed,
                method=method,
                prediction_changed=(original_pred != new_pred),
            ))
        return results

    def get_robustness_score(self, results: List[PerturbationResult]) -> float:
        """Compute robustness score (fraction unchanged)."""
        if not results:
            return 1.0
        unchanged = sum(1 for r in results if not r.prediction_changed)
        return unchanged / len(results)


# ---------------------------------------------------------------------------
# Text Attack Simulator
# ---------------------------------------------------------------------------

_ATTACK_RECIPES = {
    "deepwordbug": {"method": "typo", "intensity": 0.15},
    "textfooler": {"method": "drop", "intensity": 0.1},
    "pwws": {"method": "swap_words", "intensity": 1},
    "pruthi": {"method": "typo", "intensity": 0.2},
    "scpn": {"method": "repeat", "intensity": 1},
}


class TextAttackSimulator:
    """Simulate text-based adversarial attacks."""

    def create_attack_recipe(self, name: str) -> Dict[str, Any]:
        """Create an attack recipe by name."""
        return _ATTACK_RECIPES.get(name.lower(), {"method": "typo", "intensity": 0.1})

    def apply_perturbation(self, text: str, recipe: Dict[str, Any]) -> str:
        """Apply perturbation based on a recipe."""
        tester = AdversarialTester()
        return tester.generate_perturbation(text, recipe.get("method", "typo"))

    def measure_accuracy_drop(
        self,
        original_accuracy: float,
        perturbed_accuracy: float,
    ) -> float:
        """Measure accuracy drop from original to perturbed."""
        return original_accuracy - perturbed_accuracy


# ---------------------------------------------------------------------------
# Calibration Evaluator
# ---------------------------------------------------------------------------

class CalibrationEvaluator:
    """Evaluate model calibration (confidence vs accuracy)."""

    def compute_ece(
        self,
        predictions: List[Tuple[float, int]],
        labels: List[int],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error.

        *predictions* is a list of (confidence, predicted_class).
        *labels* is a list of true labels.
        """
        if not predictions or not labels:
            return 0.0
        bin_size = 1.0 / n_bins
        total = len(predictions)
        ece = 0.0
        for b in range(n_bins):
            lower = b * bin_size
            upper = (b + 1) * bin_size
            indices = [
                i for i, (conf, _) in enumerate(predictions)
                if lower <= conf < upper
            ]
            if not indices:
                continue
            bin_acc = sum(1 for i in indices if predictions[i][1] == labels[i]) / len(indices)
            bin_conf = sum(predictions[i][0] for i in indices) / len(indices)
            ece += len(indices) / total * abs(bin_acc - bin_conf)
        return ece

    def reliability_diagram(
        self,
        predictions: List[Tuple[float, int]],
        labels: List[int],
        n_bins: int = 10,
    ) -> List[CalibrationBin]:
        """Generate reliability diagram data."""
        bin_size = 1.0 / n_bins
        bins: List[CalibrationBin] = []
        for b in range(n_bins):
            lower = b * bin_size
            upper = (b + 1) * bin_size
            indices = [
                i for i, (conf, _) in enumerate(predictions)
                if lower <= conf < upper
            ]
            if not indices:
                bins.append(CalibrationBin(lower=lower, upper=upper,
                                           mean_confidence=0.0, accuracy=0.0, count=0))
                continue
            bin_acc = sum(1 for i in indices if predictions[i][1] == labels[i]) / len(indices)
            bin_conf = sum(predictions[i][0] for i in indices) / len(indices)
            bins.append(CalibrationBin(
                lower=lower, upper=upper,
                mean_confidence=bin_conf, accuracy=bin_acc, count=len(indices),
            ))
        return bins

    def expected_calibration_error(
        self,
        predictions: List[Tuple[float, int]],
        labels: List[int],
        n_bins: int = 10,
    ) -> float:
        """Alias for compute_ece."""
        return self.compute_ece(predictions, labels, n_bins)


# ---------------------------------------------------------------------------
# Regression Tester
# ---------------------------------------------------------------------------

class RegressionTester:
    """Regression testing for CI/CD model evaluation."""

    def __init__(self) -> None:
        self._baselines: Dict[str, Dict[str, float]] = {}

    def create_baseline(
        self, model_name: str, results: Dict[str, float]
    ) -> None:
        """Create a baseline for a model."""
        self._baselines[model_name] = dict(results)

    def check_regression(
        self, model_name: str, new_results: Dict[str, float]
    ) -> RegressionReport:
        """Check if new results regress against baseline.

        Compares the average of metric values.
        """
        baseline = self._baselines.get(model_name)
        if baseline is None:
            return RegressionReport(0.0, 0.0, 0.0, False)

        base_avg = sum(baseline.values()) / len(baseline) if baseline else 0.0
        new_avg = sum(new_results.values()) / len(new_results) if new_results else 0.0
        delta = new_avg - base_avg
        return RegressionReport(
            baseline_score=base_avg,
            current_score=new_avg,
            delta=delta,
            regressed=delta < -0.05,
        )

    def get_regression_report(
        self, baseline: Dict[str, float], current: Dict[str, float]
    ) -> RegressionReport:
        """Generate a regression report comparing two result sets."""
        base_avg = sum(baseline.values()) / len(baseline) if baseline else 0.0
        curr_avg = sum(current.values()) / len(current) if current else 0.0
        delta = curr_avg - base_avg
        return RegressionReport(
            baseline_score=base_avg,
            current_score=curr_avg,
            delta=delta,
            regressed=delta < -0.05,
        )
