"""Multimodal Evaluation — VQA, Image Captioning, Object Hallucination."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VQAResult:
    """Result of a VQA evaluation."""
    question: str
    predicted: str
    ground_truth: str
    correct: bool
    category: Optional[str] = None


@dataclass
class CaptionResult:
    """Result of a caption evaluation."""
    image_id: str
    candidate: str
    references: List[str]
    cider_score: float = 0.0
    spice_score: float = 0.0
    meteor_score: float = 0.0


# ---------------------------------------------------------------------------
# VQA Evaluator
# ---------------------------------------------------------------------------

class VQAEvaluator:
    """Visual Question Answering evaluation."""

    def __init__(self) -> None:
        self._results: List[VQAResult] = []

    def evaluate_answer(
        self,
        predicted: str,
        ground_truth: str,
        question: str = "",
        category: Optional[str] = None,
    ) -> VQAResult:
        """Evaluate a VQA answer."""
        pred_lower = predicted.strip().lower()
        gt_lower = ground_truth.strip().lower()
        correct = pred_lower == gt_lower
        result = VQAResult(
            question=question,
            predicted=predicted,
            ground_truth=ground_truth,
            correct=correct,
            category=category,
        )
        self._results.append(result)
        return result

    def compute_accuracy(self, results: Optional[List[VQAResult]] = None) -> float:
        """Compute accuracy across VQA results."""
        data = results or self._results
        if not data:
            return 0.0
        correct = sum(1 for r in data if r.correct)
        return correct / len(data)

    def get_category_breakdown(
        self, results: Optional[List[VQAResult]] = None
    ) -> Dict[str, float]:
        """Get accuracy breakdown by category."""
        data = results or self._results
        by_cat: Dict[str, List[bool]] = {}
        for r in data:
            cat = r.category or "default"
            by_cat.setdefault(cat, []).append(r.correct)
        return {
            cat: sum(vals) / len(vals)
            for cat, vals in by_cat.items()
        }


# ---------------------------------------------------------------------------
# Image Caption Scorer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer."""
    return text.lower().split()


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Get n-gram Counter."""
    c: Counter = Counter()
    for i in range(len(tokens) - n + 1):
        c[tuple(tokens[i:i+n])] += 1
    return c


class ImageCaptionScorer:
    """Image captioning metrics: CIDEr, SPICE, METEOR."""

    def compute_cider(
        self, candidate: str, references: List[str], n: int = 4
    ) -> float:
        """Compute CIDEr score."""
        if not candidate or not references:
            return 0.0
        cand_tokens = _tokenize(candidate)
        cand_ngrams = _get_ngrams(cand_tokens, n)

        ref_ngrams_list = [_get_ngrams(_tokenize(r), n) for r in references]
        n_refs = len(references)

        # TF for candidate
        cand_tf: Counter = Counter()
        total = max(sum(cand_ngrams.values()), 1)
        for ng, count in cand_ngrams.items():
            cand_tf[ng] = count / total

        # IDF for n-grams
        doc_freq: Counter = Counter()
        for ref_ng in ref_ngrams_list:
            for ng in ref_ng:
                doc_freq[ng] += 1

        # CIDEr: cosine similarity with TF-IDF weighting
        score = 0.0
        for ng in cand_ngrams:
            tf = cand_tf[ng]
            idf = math.log((n_refs + 1) / (doc_freq.get(ng, 0) + 1)) + 1
            cand_val = tf * idf
            ref_vals = []
            for ref_ng in ref_ngrams_list:
                ref_total = max(sum(ref_ng.values()), 1)
                ref_tf = ref_ng.get(ng, 0) / ref_total
                ref_vals.append(ref_tf * idf)
            avg_ref = sum(ref_vals) / len(ref_vals)
            score += cand_val * avg_ref

        # Normalize
        norm_cand = math.sqrt(sum(v ** 2 for v in cand_tf.values())) or 1.0
        return min(score / norm_cand, 10.0) / 10.0  # Normalize to [0, 1]

    def compute_spice(
        self, candidate: str, references: List[str]
    ) -> float:
        """Compute SPICE score (simplified: F1 on tuples)."""
        if not candidate or not references:
            return 0.0
        cand_tuples = set()
        cand_tokens = _tokenize(candidate)
        for i in range(len(cand_tokens) - 1):
            cand_tuples.add((cand_tokens[i], cand_tokens[i + 1]))

        ref_tuples: set = set()
        for ref in references:
            tokens = _tokenize(ref)
            for i in range(len(tokens) - 1):
                ref_tuples.add((tokens[i], tokens[i + 1]))

        if not cand_tuples and not ref_tuples:
            return 0.0
        if not cand_tuples or not ref_tuples:
            return 0.0

        overlap = cand_tuples & ref_tuples
        precision = len(overlap) / len(cand_tuples)
        recall = len(overlap) / len(ref_tuples)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_meteor(
        self, candidate: str, reference: str
    ) -> float:
        """Compute METEOR score (simplified)."""
        cand_tokens = _tokenize(candidate)
        ref_tokens = _tokenize(reference)
        if not cand_tokens or not ref_tokens:
            return 0.0

        cand_set = set(cand_tokens)
        ref_set = set(ref_tokens)
        matches = cand_set & ref_set

        if not matches:
            return 0.0

        precision = len(matches) / len(cand_tokens)
        recall = len(matches) / len(ref_tokens)

        # F-mean with recall weighted higher
        fmean = 10 * precision * recall / (9 * precision + recall) if (9 * precision + recall) > 0 else 0.0

        # Penalty for fragmentation (simplified)
        chunks = 0
        prev_matched = False
        for ct in cand_tokens:
            if ct in ref_set:
                if not prev_matched:
                    chunks += 1
                prev_matched = True
            else:
                prev_matched = False

        fragmentation = chunks / len(matches) if matches else 0
        penalty = 0.5 * fragmentation

        return fmean * (1 - penalty)


# ---------------------------------------------------------------------------
# Object Hallucination Detector
# ---------------------------------------------------------------------------

class ObjectHallucinationDetector:
    """Detect object hallucination in image captions."""

    def detect_hallucination(
        self, caption: str, objects: List[str]
    ) -> Dict[str, Any]:
        """Detect hallucinated objects in caption.

        Returns objects mentioned in caption but NOT in ground-truth objects.
        """
        caption_lower = caption.lower()
        caption_words = set(caption_lower.split())
        obj_set = set(o.lower() for o in objects)

        mentioned = [o for o in objects if o.lower() in caption_words]
        hallucinated = [w for w in caption_words if w not in obj_set and len(w) > 2]

        return {
            "mentioned_objects": mentioned,
            "potential_hallucinations": hallucinated[:5],
            "hallucination_detected": len(hallucinated) > 0,
        }

    def compute_hallucination_rate(
        self, results: List[Dict[str, Any]]
    ) -> float:
        """Compute hallucination rate across results."""
        if not results:
            return 0.0
        hallucinated = sum(1 for r in results if r.get("hallucination_detected"))
        return hallucinated / len(results)

    def get_object_f1(
        self,
        predicted: List[str],
        ground_truth: List[str],
    ) -> float:
        """Compute F1 score for predicted vs ground-truth objects."""
        pred_set = set(o.lower() for o in predicted)
        gt_set = set(o.lower() for o in ground_truth)

        if not pred_set and not gt_set:
            return 1.0
        if not pred_set or not gt_set:
            return 0.0

        overlap = pred_set & gt_set
        precision = len(overlap) / len(pred_set)
        recall = len(overlap) / len(gt_set)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
