"""NLG Metrics Engine — BLEU, ROUGE, BERTScore, COMET, Perplexity, TruthfulQA."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BLEUResult:
    """Result of a BLEU score computation."""
    score: float
    precisions: List[float]
    brevity_penalty: float
    effective_order: int


@dataclass
class ROUGEResult:
    """Result of a ROUGE score computation."""
    precision: float
    recall: float
    fmeasure: float


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Return length of the longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ---------------------------------------------------------------------------
# BLEU Scorer
# ---------------------------------------------------------------------------

class BLEUScorer:
    """BLEU score computation for machine translation / NLG evaluation."""

    def compute_ngrams(self, text: str, n: int) -> Counter:
        """Compute n-gram frequency counts for *text*."""
        tokens = _tokenize(text)
        ngrams: Counter = Counter()
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i : i + n])] += 1
        return ngrams

    def brevity_penalty(self, candidate_len: int, reference_len: int) -> float:
        """Compute the BLEU brevity penalty."""
        if candidate_len == 0:
            return 0.0
        if candidate_len >= reference_len:
            return 1.0
        return math.exp(1 - reference_len / candidate_len)

    def compute_bleu(
        self,
        candidate: str,
        references: List[str],
        max_n: int = 4,
    ) -> BLEUResult:
        """Compute BLEU score for a single candidate against references."""
        cand_tokens = _tokenize(candidate)
        ref_tokens_list = [_tokenize(r) for r in references]

        if not cand_tokens:
            return BLEUResult(0.0, [0.0] * max_n, 0.0, 0)

        # Closest reference length
        ref_lens = [len(r) for r in ref_tokens_list]
        closest_ref_len = min(ref_lens, key=lambda rl: abs(rl - len(cand_tokens)))

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            cand_ngrams = self.compute_ngrams(candidate, n)
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            clipped = 0
            total = max(sum(cand_ngrams.values()), 1)
            for ngram, count in cand_ngrams.items():
                max_ref = max(
                    (self.compute_ngrams(r, n).get(ngram, 0) for r in references),
                    default=0,
                )
                clipped += min(count, max_ref)
            precisions.append(clipped / total)

        # Geometric mean of precisions (skip zeros)
        log_avg = 0.0
        effective = 0
        for p in precisions:
            if p > 0:
                log_avg += math.log(p)
                effective += 1

        if effective == 0:
            return BLEUResult(0.0, precisions, 0.0, 0)

        bp = self.brevity_penalty(len(cand_tokens), closest_ref_len)
        score = bp * math.exp(log_avg / effective)
        return BLEUResult(score, precisions, bp, effective)

    def corpus_bleu(
        self,
        candidates: List[str],
        references_list: List[List[str]],
        max_n: int = 4,
    ) -> float:
        """Compute corpus-level BLEU score."""
        if not candidates:
            return 0.0
        scores = [
            self.compute_bleu(c, r, max_n).score
            for c, r in zip(candidates, references_list)
        ]
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# ROUGE Scorer
# ---------------------------------------------------------------------------

class ROUGEScorer:
    """ROUGE metrics for summarization evaluation."""

    def compute_rouge_n(
        self, hypothesis: str, reference: str, n: int = 1
    ) -> ROUGEResult:
        """Compute ROUGE-N score."""
        hyp_ngrams = BLEUScorer().compute_ngrams(hypothesis, n)
        ref_ngrams = BLEUScorer().compute_ngrams(reference, n)

        if not ref_ngrams or not hyp_ngrams:
            return ROUGEResult(0.0, 0.0, 0.0)

        overlap = 0
        for ngram, count in hyp_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))

        precision = overlap / sum(hyp_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())
        if precision + recall == 0:
            return ROUGEResult(0.0, 0.0, 0.0)
        fmeasure = 2 * precision * recall / (precision + recall)
        return ROUGEResult(precision, recall, fmeasure)

    def compute_rouge_l(self, hypothesis: str, reference: str) -> ROUGEResult:
        """Compute ROUGE-L (longest common subsequence) score."""
        hyp_tokens = _tokenize(hypothesis)
        ref_tokens = _tokenize(reference)

        if not hyp_tokens or not ref_tokens:
            return ROUGEResult(0.0, 0.0, 0.0)

        lcs_len = _lcs_length(hyp_tokens, ref_tokens)
        precision = lcs_len / len(hyp_tokens)
        recall = lcs_len / len(ref_tokens)
        if precision + recall == 0:
            return ROUGEResult(0.0, 0.0, 0.0)
        fmeasure = 2 * precision * recall / (precision + recall)
        return ROUGEResult(precision, recall, fmeasure)

    def summary_level_rouge(
        self, summary: str, reference: str
    ) -> Dict[str, ROUGEResult]:
        """Compute summary-level ROUGE (1, 2, L) scores."""
        return {
            "rouge-1": self.compute_rouge_n(summary, reference, 1),
            "rouge-2": self.compute_rouge_n(summary, reference, 2),
            "rouge-l": self.compute_rouge_l(summary, reference),
        }


# ---------------------------------------------------------------------------
# BERTScore Simulator (simple lexical overlap proxy)
# ---------------------------------------------------------------------------

class BERTScoreSimulator:
    """Semantic similarity scoring (lexical-overlap proxy)."""

    def compute_similarity(self, candidate: str, reference: str) -> float:
        """Compute similarity score between candidate and reference."""
        cand_tokens = set(_tokenize(candidate))
        ref_tokens = set(_tokenize(reference))
        if not cand_tokens or not ref_tokens:
            return 0.0
        overlap = cand_tokens & ref_tokens
        if not overlap:
            return 0.0
        precision = len(overlap) / len(cand_tokens)
        recall = len(overlap) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def batch_score(
        self, candidates: List[str], references: List[str]
    ) -> List[float]:
        """Score a batch of candidate-reference pairs."""
        return [
            self.compute_similarity(c, r)
            for c, r in zip(candidates, references)
        ]

    def get_idf_weights(self, documents: List[str]) -> Dict[str, float]:
        """Compute IDF weights for tokens across documents."""
        n_docs = len(documents)
        if n_docs == 0:
            return {}
        doc_freq: Counter = Counter()
        for doc in documents:
            tokens = set(_tokenize(doc))
            for t in tokens:
                doc_freq[t] += 1
        return {
            t: math.log((n_docs + 1) / (df + 1)) + 1
            for t, df in doc_freq.items()
        }


# ---------------------------------------------------------------------------
# COMET Scorer (simplified proxy)
# ---------------------------------------------------------------------------

class COMETScorer:
    """Translation evaluation scorer (simplified proxy)."""

    def score(
        self, source: str, translation: str, reference: str
    ) -> float:
        """Score a single translation against source and reference."""
        sim = BERTScoreSimulator().compute_similarity(translation, reference)
        # small bonus for source-reference overlap
        src_sim = BERTScoreSimulator().compute_similarity(source, reference)
        return 0.8 * sim + 0.2 * src_sim

    def corpus_score(
        self,
        sources: List[str],
        translations: List[str],
        references: List[str],
    ) -> float:
        """Score a corpus of translations."""
        if not sources:
            return 0.0
        scores = [
            self.score(s, t, r)
            for s, t, r in zip(sources, translations, references)
        ]
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Perplexity Calculator
# ---------------------------------------------------------------------------

class PerplexityCalculator:
    """Language model evaluation via perplexity and cross-entropy."""

    def compute_perplexity(self, log_probs: List[float]) -> float:
        """Compute perplexity from a list of log probabilities."""
        if not log_probs:
            return float("inf")
        avg_neg_log = -sum(log_probs) / len(log_probs)
        return math.exp(avg_neg_log)

    def compute_cross_entropy(
        self, predictions: List[List[float]], targets: List[int]
    ) -> float:
        """Compute cross-entropy loss from prediction distributions and targets."""
        if not predictions:
            return 0.0
        total = 0.0
        for dist, target in zip(predictions, targets):
            if target < 0 or target >= len(dist):
                total += 10.0  # large penalty for invalid
            else:
                prob = max(dist[target], 1e-10)
                total -= math.log(prob)
        return total / len(predictions)


# ---------------------------------------------------------------------------
# TruthfulQA Evaluator
# ---------------------------------------------------------------------------

class TruthfulQAEvaluator:
    """Truthfulness evaluation for QA systems."""

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        correct_answers: List[str],
    ) -> Dict[str, object]:
        """Evaluate a single answer for truthfulness."""
        best_sim = 0.0
        answer_lower = answer.lower().strip()
        for ref in correct_answers:
            ref_lower = ref.lower().strip()
            if answer_lower == ref_lower:
                best_sim = 1.0
                break
            sim = BERTScoreSimulator().compute_similarity(answer, ref)
            best_sim = max(best_sim, sim)
        return {
            "question": question,
            "answer": answer,
            "similarity": best_sim,
            "truthful": best_sim >= 0.5,
        }

    def is_truthful(self, answer: str, reference: str) -> bool:
        """Check if an answer is truthful against a reference."""
        sim = BERTScoreSimulator().compute_similarity(answer, reference)
        return sim >= 0.5

    def get_score(
        self, answers: List[str], references: List[str]
    ) -> float:
        """Compute overall truthfulness score."""
        if not answers:
            return 0.0
        results = [
            self.is_truthful(a, r)
            for a, r in zip(answers, references)
        ]
        return sum(results) / len(results)
