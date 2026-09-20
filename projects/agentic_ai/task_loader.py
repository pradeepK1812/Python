"""
Task Loader Contract
====================

Purpose:
    Load a task definition from an external task file and convert it into
    a validated TaskSpecification.

Responsibilities:
    1. Accept the task file path supplied by the caller.
    2. Read the task file.
    3. Parse the file contents.
    4. Construct a TaskSpecification.
    5. Return the validated TaskSpecification.

The task loader does not:
    - Execute the task.
    - Select tools.
    - Invoke the LLM.
    - Perform evaluation.
    - Decide whether the task is complete.

Architecture:

    Command Line
         |
         | task file path
         v
    Task Loader
         |
         | parsed task
         v
    TaskSpecification
         |
         v
       Agent
"""

from task_spec import TaskSpecification
import yaml
from task_spec import TaskSpecification


def load_task(path: str) -> TaskSpecification:
    """
    Load and validate a task definition from a YAML file.

    Args:
        path: Path to the task definition file.

    Returns:
        A validated TaskSpecification.

    Raises:
        FileNotFoundError: If the task file does not exist.
        ValueError: If the task file is invalid.
    """

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Task file must contain a YAML mapping.")

    if "objective" not in data:
        raise ValueError("Task file is missing 'objective'.")

    if "acceptance_criteria" not in data:
        raise ValueError("Task file is missing 'acceptance_criteria'.")

    return TaskSpecification(
        objective=data["objective"],
        acceptance_criteria=data["acceptance_criteria"],
    )


