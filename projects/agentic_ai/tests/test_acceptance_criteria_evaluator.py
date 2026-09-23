from evaluator import EvaluationContext,EvaluationStatus
from evaluators.acceptance_criteria_evaluator import (
    AcceptanceCriteriaEvaluator,
)


def test_acceptance_criteria_not_applicable_for_non_test_tool():
    evaluator = AcceptanceCriteriaEvaluator()

    context = EvaluationContext(
        objective="Create calculator.py",
        tool_name="list_files",
        tool_result="calculator.py\ntests",
        state={
            "acceptance_criteria": [
                "All unit tests must pass."
            ]
        },
    )

    result = evaluator.evaluate(context)

    assert result.status == EvaluationStatus.NOT_APPLICABLE



def test_acceptance_criteria_satisfied():
    context = EvaluationContext(
        objective="Create calculator.py.",
        tool_name="run_tests",
        tool_result="5 passed in 0.03s",
        state={
            "acceptance_criteria": [
                "All unit tests must pass.",
            ],
        },
    )

    result = AcceptanceCriteriaEvaluator().evaluate(context)

    assert result.status == EvaluationStatus.SATISFIED
    assert result.reason == "All acceptance criteria are satisfied."


def test_acceptance_criteria_not_satisfied():
    context = EvaluationContext(
        objective="Create calculator.py.",
        tool_name="run_tests",
        tool_result="1 failed, 4 passed",
        state={
            "acceptance_criteria": [
                "All unit tests must pass.",
            ],
        },
    )

    result = AcceptanceCriteriaEvaluator().evaluate(context)

    assert result.status == EvaluationStatus.FAILED
    assert (
        result.reason
        == "Acceptance criterion not satisfied: "
        "all unit tests must pass."
    )


def test_acceptance_criteria_missing():
    context = EvaluationContext(
        objective="Create calculator.py.",
        tool_name="run_tests",
        tool_result="5 passed",
        state={},
    )

    result = AcceptanceCriteriaEvaluator().evaluate(context)

    assert result.status == EvaluationStatus.FAILED
    assert result.reason == "No acceptance criteria provided."
