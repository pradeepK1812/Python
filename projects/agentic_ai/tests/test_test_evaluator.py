from evaluator import EvaluationContext
from evaluators.test_evaluator import TestEvaluator


def test_test_evaluator_passed():
    context = EvaluationContext(
        objective="All tests must pass.",
        tool_name="run_tests",
        tool_result="5 passed in 0.03s",
        state={},
    )

    result = TestEvaluator().evaluate(context)

    assert result.satisfied is True
    assert result.reason == "All tests passed."


def test_test_evaluator_failed():
    context = EvaluationContext(
        objective="All tests must pass.",
        tool_name="run_tests",
        tool_result="1 failed, 4 passed",
        state={},
    )

    result = TestEvaluator().evaluate(context)

    assert result.satisfied is False
    assert result.reason == "Test suite contains failures."


def test_test_evaluator_without_test_result():
    context = EvaluationContext(
        objective="All tests must pass.",
        tool_name="read_file",
        tool_result="some file content",
        state={},
    )

    result = TestEvaluator().evaluate(context)

    assert result.satisfied is False
    assert result.reason == "No test result is available."
