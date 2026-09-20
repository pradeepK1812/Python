"""
Task Specification Contract
===========================

Purpose:
    Defines the user-provided task that the agent must accomplish.

Responsibilities:
    1. Store the task objective.
    2. Store the acceptance criteria.
    3. Validate that the task specification is usable.
    4. Provide a stable interface for the agent and evaluator layers.

The task specification does not:
    - Select tools.
    - Execute tools.
    - Implement evaluation logic.
    - Control the LLM.

Architecture:

    Task File
        |
        v
    TaskSpecification
        |
        +----> objective
        |
        +----> acceptance_criteria
        |
        +--------------------+
                             |
                  +----------+----------+
                  |                     |
                  v                     v
                 Agent              Evaluator
"""


from dataclasses import dataclass


@dataclass
class TaskSpecification:
    """Validated representation of a user-defined agent task."""

    objective: str
    acceptance_criteria: list[str]

    def __post_init__(self):
        if not self.objective.strip():
            raise ValueError("Objective cannot be empty.")

        if not self.acceptance_criteria:
            raise ValueError(
                "At least one acceptance criterion is required."
            )

        if any(not criterion.strip() for criterion in self.acceptance_criteria):
            raise ValueError(
                "Acceptance criteria cannot contain empty values."
            )


