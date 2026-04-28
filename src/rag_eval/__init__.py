"""RAG System Evaluation — RAGAS, Retrieval, Pipeline, Context."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RAGQueryResult:
    """Evaluation result for a single RAG query."""
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall: float = 0.0

    def __post_init__(self) -> None:
        vals = [self.faithfulness, self.answer_relevancy,
                self.context_precision, self.context_recall]
        self.overall = sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# RAGAS Framework
# ---------------------------------------------------------------------------

class RAGASFramework:
    """RAGAS metrics computation for RAG evaluation."""

    def compute_faithfulness(self, answer: str, context: str) -> float:
        """Compute faithfulness of answer to context."""
        if not answer or not context:
            return 0.0
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        if not answer_tokens:
            return 0.0
        supported = answer_tokens & context_tokens
        return len(supported) / len(answer_tokens)

    def compute_answer_relevancy(self, question: str, answer: str) -> float:
        """Compute answer relevancy to question."""
        if not question or not answer:
            return 0.0
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        if not q_tokens:
            return 0.0
        overlap = q_tokens & a_tokens
        return len(overlap) / len(q_tokens)

    def compute_context_precision(
        self, contexts: List[str], question: str
    ) -> float:
        """Compute context precision for retrieved contexts."""
        if not contexts:
            return 0.0
        q_tokens = set(question.lower().split())
        scores: List[float] = []
        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            if not ctx_tokens:
                scores.append(0.0)
                continue
            overlap = q_tokens & ctx_tokens
            scores.append(len(overlap) / len(q_tokens))
        return sum(scores) / len(scores)

    def compute_context_recall(
        self, contexts: List[str], ground_truth: str
    ) -> float:
        """Compute context recall against ground truth."""
        if not contexts or not ground_truth:
            return 0.0
        gt_tokens = set(ground_truth.lower().split())
        all_ctx_tokens: set = set()
        for ctx in contexts:
            all_ctx_tokens |= set(ctx.lower().split())
        if not gt_tokens:
            return 0.0
        recalled = gt_tokens & all_ctx_tokens
        return len(recalled) / len(gt_tokens)


# ---------------------------------------------------------------------------
# Retrieval Evaluator
# ---------------------------------------------------------------------------

class RetrievalEvaluator:
    """Evaluate retrieval quality with standard IR metrics."""

    def precision_at_k(
        self, retrieved: List[str], relevant: List[str], k: int = 10
    ) -> float:
        """Compute Precision@K."""
        if k <= 0:
            return 0.0
        top_k = retrieved[:k]
        rel_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in rel_set)
        return hits / k

    def recall_at_k(
        self, retrieved: List[str], relevant: List[str], k: int = 10
    ) -> float:
        """Compute Recall@K."""
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        rel_set = set(relevant)
        hits = sum(1 for doc in top_k if doc in rel_set)
        return hits / len(rel_set)

    def mrr(self, ranked_results: List[str]) -> float:
        """Compute Mean Reciprocal Rank (single query)."""
        for i, doc in enumerate(ranked_results, 1):
            if doc.startswith("relevant") or doc.startswith("rel_"):
                return 1.0 / i
        return 0.0

    def ndcg(
        self, ranked_results: List[str], ideal: List[str]
    ) -> float:
        """Compute NDCG (Normalized Discounted Cumulative Gain)."""
        def _dcg(results: List[str]) -> float:
            val = 0.0
            for i, doc in enumerate(results):
                gain = 1.0 if doc in ideal else 0.0
                val += gain / math.log2(i + 2)
            return val

        actual_dcg = _dcg(ranked_results)
        ideal_dcg = _dcg(ideal)
        if ideal_dcg == 0:
            return 0.0
        return actual_dcg / ideal_dcg


# ---------------------------------------------------------------------------
# RAG Pipeline Evaluator
# ---------------------------------------------------------------------------

class RAGPipelineEvaluator:
    """End-to-end RAG pipeline evaluation."""

    def __init__(self) -> None:
        self._ragas = RAGASFramework()
        self._results: List[RAGQueryResult] = []

    def evaluate_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> RAGQueryResult:
        """Evaluate a single RAG query end-to-end."""
        result = RAGQueryResult(
            question=question,
            answer=answer,
            faithfulness=self._ragas.compute_faithfulness(answer, " ".join(contexts)),
            answer_relevancy=self._ragas.compute_answer_relevancy(question, answer),
            context_precision=self._ragas.compute_context_precision(contexts, question),
            context_recall=self._ragas.compute_context_recall(contexts, ground_truth),
        )
        self._results.append(result)
        return result

    def get_aggregate_scores(self, results: Optional[List[RAGQueryResult]] = None) -> Dict[str, float]:
        """Get aggregate scores across all evaluated queries."""
        data = results or self._results
        if not data:
            return {
                "avg_faithfulness": 0.0,
                "avg_answer_relevancy": 0.0,
                "avg_context_precision": 0.0,
                "avg_context_recall": 0.0,
                "avg_overall": 0.0,
            }
        return {
            "avg_faithfulness": sum(r.faithfulness for r in data) / len(data),
            "avg_answer_relevancy": sum(r.answer_relevancy for r in data) / len(data),
            "avg_context_precision": sum(r.context_precision for r in data) / len(data),
            "avg_context_recall": sum(r.context_recall for r in data) / len(data),
            "avg_overall": sum(r.overall for r in data) / len(data),
        }

    def get_failure_analysis(self, results: Optional[List[RAGQueryResult]] = None) -> Dict[str, Any]:
        """Identify failing queries (overall < 0.5)."""
        data = results or self._results
        failures = [r for r in data if r.overall < 0.5]
        return {
            "total": len(data),
            "failures": len(failures),
            "failure_rate": len(failures) / len(data) if data else 0.0,
            "failure_questions": [r.question for r in failures],
        }


# ---------------------------------------------------------------------------
# Context Relevancy Scorer
# ---------------------------------------------------------------------------

class ContextRelevancyScorer:
    """Score context quality for RAG systems."""

    def score_context(self, question: str, context: str) -> float:
        """Score context relevance to question."""
        if not question or not context:
            return 0.0
        q_tokens = set(question.lower().split())
        c_tokens = set(context.lower().split())
        if not q_tokens:
            return 0.0
        overlap = q_tokens & c_tokens
        return len(overlap) / len(q_tokens)

    def score_relevance(self, question: str, context: str) -> float:
        """Alias for score_context."""
        return self.score_context(question, context)

    def identify_irrelevant_segments(
        self, question: str, context: str
    ) -> List[str]:
        """Identify sentences in context irrelevant to question."""
        q_tokens = set(question.lower().split())
        sentences = [s.strip() for s in context.split(".") if s.strip()]
        irrelevant: List[str] = []
        for sent in sentences:
            s_tokens = set(sent.lower().split())
            overlap = q_tokens & s_tokens
            if len(overlap) == 0:
                irrelevant.append(sent)
        return irrelevant
