"""Human-in-the-Loop Evaluation — Annotations, RLHF, Quality Control."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AnnotationTask:
    """A human annotation task."""
    task_id: str
    example: str
    guidelines: str
    annotation: Optional[str] = None
    completed: bool = False


@dataclass
class PreferencePair:
    """A preference pair for RLHF."""
    prompt: str
    response_a: str
    response_b: str
    preferred: str  # "a" or "b"
    annotator: Optional[str] = None


@dataclass
class EvalSession:
    """A human evaluation session."""
    session_id: str
    evaluators: List[str]
    examples: List[str]
    results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Annotation Manager
# ---------------------------------------------------------------------------

class AnnotationManager:
    """Manage human annotations for evaluation."""

    def __init__(self) -> None:
        self._tasks: Dict[str, AnnotationTask] = {}
        self._counter = 0

    def create_task(self, example: str, guidelines: str) -> AnnotationTask:
        """Create a new annotation task."""
        self._counter += 1
        task_id = f"task_{self._counter}"
        task = AnnotationTask(
            task_id=task_id,
            example=example,
            guidelines=guidelines,
        )
        self._tasks[task_id] = task
        return task

    def submit_annotation(self, task_id: str, annotation: str) -> bool:
        """Submit an annotation for a task."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.annotation = annotation
        task.completed = True
        return True

    def get_inter_annotator_agreement(
        self, annotations: List[List[str]]
    ) -> float:
        """Compute inter-annotator agreement (simple percent agreement)."""
        if not annotations or len(annotations) < 2:
            return 0.0
        n_items = len(annotations[0])
        agreements = 0
        for i in range(n_items):
            labels = [ann[i] for ann in annotations if i < len(ann)]
            if len(set(labels)) == 1:
                agreements += 1
        return agreements / n_items if n_items > 0 else 0.0


# ---------------------------------------------------------------------------
# RLHF Quality Checker
# ---------------------------------------------------------------------------

class RLHFQualityChecker:
    """Check quality of RLHF preference data."""

    def check_preference_quality(self, preference: PreferencePair) -> Dict[str, Any]:
        """Check the quality of a single preference annotation."""
        issues: List[str] = []
        if preference.preferred not in ("a", "b"):
            issues.append("Invalid preference value")
        if preference.response_a == preference.response_b:
            issues.append("Identical responses")
        if not preference.prompt.strip():
            issues.append("Empty prompt")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def compute_agreement_rate(
        self, preferences: List[PreferencePair]
    ) -> float:
        """Compute agreement rate among preference annotations.

        Groups by prompt and checks consistency.
        """
        if not preferences:
            return 0.0
        by_prompt: Dict[str, List[str]] = {}
        for p in preferences:
            by_prompt.setdefault(p.prompt, []).append(p.preferred)
        total = 0
        consistent = 0
        for prompt, prefs in by_prompt.items():
            if len(prefs) < 2:
                continue
            total += 1
            if len(set(prefs)) == 1:
                consistent += 1
        return consistent / total if total > 0 else 1.0

    def flag_inconsistent(
        self, preferences: List[PreferencePair]
    ) -> List[PreferencePair]:
        """Flag inconsistent preference pairs."""
        by_prompt: Dict[str, List[PreferencePair]] = {}
        for p in preferences:
            by_prompt.setdefault(p.prompt, []).append(p)
        flagged: List[PreferencePair] = []
        for prompt, prefs in by_prompt.items():
            if len(prefs) < 2:
                continue
            if len(set(p.preferred for p in prefs)) > 1:
                flagged.extend(prefs)
        return flagged


# ---------------------------------------------------------------------------
# Human Evaluator
# ---------------------------------------------------------------------------

class HumanEvaluator:
    """Manage human evaluation sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[str, EvalSession] = {}
        self._counter = 0

    def create_eval_session(
        self,
        evaluators: List[str],
        examples: List[str],
    ) -> EvalSession:
        """Create a new evaluation session."""
        self._counter += 1
        session_id = f"session_{self._counter}"
        session = EvalSession(
            session_id=session_id,
            evaluators=evaluators,
            examples=examples,
        )
        self._sessions[session_id] = session
        return session

    def collect_results(
        self, session_id: str
    ) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Collect results from an evaluation session."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        return session.results if session.results else None

    def compute_human_metrics(
        self, results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Compute aggregate human evaluation metrics."""
        if not results:
            return {"avg_score": 0.0, "num_evaluators": 0}
        all_scores: List[float] = []
        for evaluator, evals in results.items():
            for ev in evals:
                if "score" in ev:
                    all_scores.append(ev["score"])
        return {
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "num_evaluators": len(results),
            "total_evaluations": len(all_scores),
        }


# ---------------------------------------------------------------------------
# Quality Controller
# ---------------------------------------------------------------------------

class QualityController:
    """Annotation quality control metrics."""

    def compute_cohens_kappa(
        self, annotations_a: List[str], annotations_b: List[str]
    ) -> float:
        """Compute Cohen's Kappa for two annotators."""
        if not annotations_a or not annotations_b:
            return 0.0
        n = min(len(annotations_a), len(annotations_b))
        if n == 0:
            return 0.0

        # Observed agreement
        agreements = sum(1 for i in range(n) if annotations_a[i] == annotations_b[i])
        po = agreements / n

        # Expected agreement
        categories = set(annotations_a + annotations_b)
        pe = 0.0
        for cat in categories:
            pa = sum(1 for i in range(n) if annotations_a[i] == cat) / n
            pb = sum(1 for i in range(n) if annotations_b[i] == cat) / n
            pe += pa * pb

        if pe == 1.0:
            return 1.0
        return (po - pe) / (1 - pe)

    def compute_fleiss_kappa(
        self, annotations: List[List[str]]
    ) -> float:
        """Compute Fleiss' Kappa for multiple annotators."""
        if not annotations or len(annotations) < 2:
            return 0.0

        n_items = len(annotations[0])
        n_annotators = len(annotations)
        categories = set()
        for ann in annotations:
            categories.update(ann)

        cat_list = sorted(categories)
        cat_idx = {c: i for i, c in enumerate(cat_list)}
        n_cats = len(cat_list)

        # Build agreement table
        table: List[List[int]] = []
        for i in range(n_items):
            row = [0] * n_cats
            for ann in annotations:
                if i < len(ann):
                    row[cat_idx[ann[i]]] += 1
            table.append(row)

        # Compute P_i for each item
        p_i_sum = 0.0
        for row in table:
            total = sum(row)
            if total <= 1:
                continue
            p_i = (sum(r * r for r in row) - total) / (total * (total - 1))
            p_i_sum += p_i

        p_bar = p_i_sum / n_items if n_items > 0 else 0.0

        # Compute P_e
        p_e = 0.0
        for j in range(n_cats):
            col_sum = sum(row[j] for row in table)
            p_j = col_sum / (n_items * n_annotators) if n_items * n_annotators > 0 else 0.0
            p_e += p_j ** 2

        if p_e == 1.0:
            return 1.0
        return (p_bar - p_e) / (1 - p_e)

    def detect_spammers(
        self, annotations: Dict[str, List[str]]
    ) -> List[str]:
        """Detect potential spammer annotators.

        Flags annotators with > 80% same label.
        """
        spammers: List[str] = []
        for annotator, labels in annotations.items():
            if not labels:
                continue
            counter = Counter(labels)
            most_common_ratio = counter.most_common(1)[0][1] / len(labels)
            if most_common_ratio > 0.8:
                spammers.append(annotator)
        return spammers
