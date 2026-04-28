"""Agentic AI Evaluation — Task completion, tool use, benchmarks, trajectories."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskStep:
    """A single step in an agent trajectory."""
    action: str
    tool: Optional[str] = None
    result: Optional[str] = None
    success: bool = True


@dataclass
class ToolCall:
    """Represents a tool call made by an agent."""
    tool_name: str
    arguments: Dict[str, Any]
    expected_tool: Optional[str] = None
    correct: bool = False


@dataclass
class BenchTask:
    """A benchmark task definition."""
    task_id: str
    description: str
    expected_tools: List[str] = field(default_factory=list)
    success_criteria: Optional[Dict[str, Any]] = None


@dataclass
class BenchResult:
    """Result of a benchmark run for a single agent."""
    agent_name: str
    task_id: str
    completed: bool
    score: float


# ---------------------------------------------------------------------------
# Task Completion Evaluator
# ---------------------------------------------------------------------------

class TaskCompletionEvaluator:
    """Evaluate agent task completion quality."""

    def evaluate_task(
        self, plan: List[str], execution: List[TaskStep]
    ) -> Dict[str, Any]:
        """Evaluate task execution against plan."""
        if not plan:
            return {"completion_rate": 0.0, "steps_completed": 0, "plan_size": 0}
        successful = sum(1 for s in execution if s.success)
        completion_rate = successful / len(plan) if plan else 0.0
        return {
            "completion_rate": min(completion_rate, 1.0),
            "steps_completed": successful,
            "plan_size": len(plan),
        }

    def compute_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Compute aggregate success rate."""
        if not results:
            return 0.0
        rates = [r.get("completion_rate", 0.0) for r in results]
        return sum(rates) / len(rates)

    def get_trajectory_score(self, steps: List[TaskStep]) -> float:
        """Score trajectory based on successful steps."""
        if not steps:
            return 0.0
        return sum(1 for s in steps if s.success) / len(steps)


# ---------------------------------------------------------------------------
# Tool Use Evaluator
# ---------------------------------------------------------------------------

class ToolUseEvaluator:
    """Evaluate agent tool usage quality."""

    def evaluate_tool_call(self, call: ToolCall, expected: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single tool call."""
        expected_tool = expected or call.expected_tool
        correct = call.tool_name == expected_tool if expected_tool else call.correct
        return {
            "tool": call.tool_name,
            "expected": expected_tool,
            "correct": correct,
        }

    def compute_tool_accuracy(self, calls: List[ToolCall]) -> float:
        """Compute overall tool selection accuracy."""
        if not calls:
            return 0.0
        correct = sum(1 for c in calls if c.correct or c.tool_name == c.expected_tool)
        return correct / len(calls)

    def get_tool_distribution(self, calls: List[ToolCall]) -> Dict[str, int]:
        """Get distribution of tools used."""
        dist: Counter = Counter()
        for c in calls:
            dist[c.tool_name] += 1
        return dict(dist)


# ---------------------------------------------------------------------------
# AgentBench Simulator
# ---------------------------------------------------------------------------

class AgentBenchSimulator:
    """Benchmark simulation for agentic systems."""

    def __init__(self) -> None:
        self._tasks: Dict[str, BenchTask] = {}
        self._results: List[BenchResult] = []

    def register_task(self, task_id: str, task: BenchTask) -> None:
        """Register a benchmark task."""
        self._tasks[task_id] = task

    def run_benchmark(
        self,
        agent: Callable[[BenchTask], bool],
        tasks: Optional[List[str]] = None,
    ) -> List[BenchResult]:
        """Run benchmark tasks against an agent function."""
        task_ids = tasks or list(self._tasks.keys())
        results: List[BenchResult] = []
        for tid in task_ids:
            task = self._tasks.get(tid)
            if task is None:
                continue
            completed = agent(task)
            score = 1.0 if completed else 0.0
            result = BenchResult(
                agent_name="agent",
                task_id=tid,
                completed=completed,
                score=score,
            )
            results.append(result)
            self._results.append(result)
        return results

    def get_leaderboard(
        self, results: Optional[List[BenchResult]] = None
    ) -> List[Dict[str, Any]]:
        """Get sorted leaderboard from results."""
        data = results or self._results
        by_agent: Dict[str, List[BenchResult]] = {}
        for r in data:
            by_agent.setdefault(r.agent_name, []).append(r)
        board = []
        for name, agent_results in by_agent.items():
            avg_score = sum(r.score for r in agent_results) / len(agent_results)
            board.append({"agent": name, "avg_score": avg_score, "tasks": len(agent_results)})
        board.sort(key=lambda x: x["avg_score"], reverse=True)
        return board


# ---------------------------------------------------------------------------
# Trajectory Analyzer
# ---------------------------------------------------------------------------

class TrajectoryAnalyzer:
    """Analyze agent trajectories for quality and efficiency."""

    def analyze_trajectory(self, steps: List[TaskStep]) -> Dict[str, Any]:
        """Analyze a trajectory for basic statistics."""
        if not steps:
            return {"total_steps": 0, "success_rate": 0.0, "unique_tools": 0}
        success_rate = sum(1 for s in steps if s.success) / len(steps)
        tools = set(s.tool for s in steps if s.tool)
        return {
            "total_steps": len(steps),
            "success_rate": success_rate,
            "unique_tools": len(tools),
        }

    def detect_loops(self, steps: List[TaskStep]) -> List[List[int]]:
        """Detect repeating action sequences (loops)."""
        loops: List[List[int]] = []
        actions = [s.action for s in steps]
        n = len(actions)
        for window in range(2, n // 2 + 1):
            for i in range(n - 2 * window + 1):
                seq1 = actions[i : i + window]
                seq2 = actions[i + window : i + 2 * window]
                if seq1 == seq2:
                    loops.append(list(range(i, i + 2 * window)))
                    if len(loops) >= 5:
                        return loops
        return loops

    def compute_efficiency(self, steps: List[TaskStep]) -> float:
        """Compute trajectory efficiency (successful steps / total steps)."""
        if not steps:
            return 0.0
        successful = sum(1 for s in steps if s.success)
        return successful / len(steps)
