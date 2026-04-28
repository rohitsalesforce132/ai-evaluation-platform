# AI Evaluation Platform

A complete, production-grade AI evaluation and testing platform built in pure Python (stdlib only). Covers all major evaluation paradigms from _The AI Engineer's AI Evaluation and Testing Interview Guide_.

## Architecture

```
ai-evaluation-platform/
├── src/
│   ├── metrics/          # NLG Metrics Engine (Ch4) — BLEU, ROUGE, BERTScore, COMET, Perplexity, TruthfulQA
│   ├── rag_eval/         # RAG System Evaluation (Ch6) — RAGAS, Retrieval, Pipeline, Context
│   ├── agent_eval/       # Agentic AI Evaluation (Ch7) — Task, Tool, Benchmark, Trajectory
│   ├── safety_eval/      # Safety & Alignment (Ch9) — Toxicity, Bias, Red-teaming, Factuality
│   ├── robustness/       # Robustness & Reliability (Ch10) — Adversarial, Calibration, Regression
│   ├── benchmark/        # Benchmark Management (Ch5) — Runner, HumanEval, Math, Multilingual
│   ├── multimodal_eval/  # Multimodal Evaluation (Ch8) — VQA, Captioning, Hallucination
│   ├── pipeline/         # Production Pipelines (Ch12) — Pipeline, Regression Gate, Release, Golden Datasets
│   ├── human_eval/       # Human-in-the-Loop (Ch13) — Annotations, RLHF, Quality Control
│   └── scaling/          # Evaluation at Scale (Ch14) — Distributed, Cost, Scheduling, Dashboard
├── tests/
│   └── test_all.py       # 150+ deterministic tests
├── README.md
└── STAR.md
```

## 10 Subsystems

| # | Subsystem | Chapter | Classes | Purpose |
|---|-----------|---------|---------|---------|
| 1 | `metrics` | Ch4 | BLEUScorer, ROUGEScorer, BERTScoreSimulator, COMETScorer, PerplexityCalculator, TruthfulQAEvaluator | NLG metrics |
| 2 | `rag_eval` | Ch6 | RAGASFramework, RetrievalEvaluator, RAGPipelineEvaluator, ContextRelevancyScorer | RAG evaluation |
| 3 | `agent_eval` | Ch7 | TaskCompletionEvaluator, ToolUseEvaluator, AgentBenchSimulator, TrajectoryAnalyzer | Agentic AI |
| 4 | `safety_eval` | Ch9 | ToxicityDetector, BiasEvaluator, RedTeamRunner, FActScorer | Safety & alignment |
| 5 | `robustness` | Ch10 | AdversarialTester, TextAttackSimulator, CalibrationEvaluator, RegressionTester | Robustness |
| 6 | `benchmark` | Ch5 | BenchmarkRunner, HumanEvalRunner, MathReasoningEvaluator, MultilingualEvaluator | Benchmarks |
| 7 | `multimodal_eval` | Ch8 | VQAEvaluator, ImageCaptionScorer, ObjectHallucinationDetector | Multimodal |
| 8 | `pipeline` | Ch12 | EvaluationPipeline, RegressionGate, ReleaseEvaluator, GoldenDatasetManager | Production |
| 9 | `human_eval` | Ch13 | AnnotationManager, RLHFQualityChecker, HumanEvaluator, QualityController | Human eval |
| 10 | `scaling` | Ch14 | DistributedEvaluator, CostTracker, EvaluationScheduler, MetricsDashboard | LLMOps |

## Quick Start

```bash
# Run all tests
pytest tests/test_all.py -v

# Run a specific subsystem's tests
pytest tests/test_all.py -k "TestBLEUScorer" -v
```

## Design Principles

- **Zero dependencies** — pure Python stdlib, runs anywhere
- **Deterministic** — all tests produce identical results on every run
- **Production patterns** — real evaluation workflows, not toys
- **Comprehensive** — 150+ tests covering edge cases across all subsystems
- **Type-annotated** — full type hints on all public APIs
- **Dataclass-driven** — structured data throughout

## Key Metrics Implemented

- **BLEU** (corpus & sentence-level with brevity penalty)
- **ROUGE** (ROUGE-1, ROUGE-2, ROUGE-L)
- **CIDEr / SPICE / METEOR** (image captioning)
- **NDCG / MRR / Precision@K / Recall@K** (retrieval)
- **ECE** (Expected Calibration Error)
- **Cohen's Kappa / Fleiss' Kappa** (inter-annotator agreement)
- **pass@k** (code generation)
- **BBQ score** (bias benchmark)
