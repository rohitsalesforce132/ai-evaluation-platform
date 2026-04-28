# GitHub Copilot Instructions

## Project Overview
AI Evaluation & Testing Platform — complete evaluation stack from "The AI Engineer's AI Evaluation and Testing Interview Guide".

## Architecture
```
src/
├── metrics/           # NLG metrics (BLEU, ROUGE, BERTScore, COMET, Perplexity, TruthfulQA)
├── rag_eval/          # RAG evaluation (RAGAS, retrieval quality, context scoring)
├── agent_eval/        # Agentic AI evaluation (task completion, tool use, trajectory analysis)
├── safety_eval/       # Safety & alignment (toxicity, bias, red-teaming, factuality)
├── robustness/        # Robustness (adversarial, TextAttack, calibration, regression)
├── benchmark/         # Benchmark management (HumanEval, math reasoning, multilingual)
├── multimodal_eval/   # Multimodal evaluation (VQA, image captioning, object hallucination)
├── pipeline/          # Production pipelines (CI/CD gates, release evaluation, golden datasets)
├── human_eval/        # Human-in-the-loop (annotation, RLHF quality, inter-annotator agreement)
└── scaling/           # Evaluation at scale (distributed eval, cost tracking, scheduling, dashboards)
```

## Conventions
- Pure Python stdlib only — zero external dependencies
- Type hints on all public methods
- Dataclasses for structured data, Enums for vocabularies
- Tests in tests/test_all.py using pytest

## Running
```bash
pytest tests/test_all.py -v  # Run all 219 tests
```
