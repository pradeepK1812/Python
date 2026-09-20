"""
Evaluator Layer Contract
========================

Purpose:
    The evaluator layer determines whether the agent's current progress
    satisfies a specific evaluation condition.

Design principles:
    1. The LLM decides what action to take.
    2. Tools execute actions deterministically.
    3. Evaluators independently assess the outcome of those actions.
    4. Evaluators must not decide which tool the agent should use.
    5. Evaluators return a standardized EvaluationResult.
    6. Multiple specialized evaluators can be composed by the top-level
       evaluation layer.

Architecture:

    Agent
      |
      v
    LLM
      |
      v
    Tool
      |
      v
    Environment / Tool Result
      |
      v
    Evaluator
      |
      +----> TestEvaluator
      +----> AcceptanceCriteriaEvaluator
      +----> FileChangeEvaluator
      |
      v
    EvaluationResult
      |
      +----> satisfied
      +----> reason
      +----> details

Important distinction:
    An evaluator determines whether its specific condition is satisfied.
    A successful individual evaluation does not necessarily mean that the
    overall agent objective has been completed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationContext:
    """Context provided to an evaluator."""

    objective: str
    tool_name: str | None
    tool_result: Any
    state: dict[str, Any]


@dataclass
class EvaluationResult:
    """Standard result returned by an evaluator."""

    satisfied: bool
    reason: str
    details: dict[str, Any]


class Evaluator(ABC):
    """Base contract for all concrete evaluators."""

    @abstractmethod
    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        """Evaluate the current agent state."""
        pass
