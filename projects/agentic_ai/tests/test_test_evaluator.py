from evaluator import EvaluationContext,EvaluationStatus
from evaluators.test_evaluator import TestEvaluator


def test_test_evaluator_passed():
    context = EvaluationContext(
        objective="All tests must pass.",
        tool_name="run_tests",
        tool_result="5 passed in 0.03s",
        state={},
    )

    result = TestEvaluator().evaluate(context)

    assert result.status == EvaluationStatus.SATISFIED
    assert result.reason == "All tests passed."


def test_test_evaluator_failed():
    context = EvaluationContext(
        objective="All tests must pass.",
        tool_name="run_tests",
        tool_result="1 failed, 4 passed",
        state={},
    )

    result = TestEvaluator().evaluate(context)

    assert result.status == EvaluationStatus.FAILED
    assert result.reason == "Test suite contains failures."


def test_test_evaluator_without_test_result():
    context = EvaluationContext(
        objective="All tests must pass.",
        tool_name="read_file",
        tool_result="some file content",
        state={},
    )

    result = TestEvaluator().evaluate(context)

    assert result.status == EvaluationStatus.NOT_APPLICABLE
    assert result.reason == "TestEvaluator does not apply to this tool."
