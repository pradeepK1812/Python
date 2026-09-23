
from evaluator import (
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
)


class TestEvaluator(Evaluator):
    """Evaluate whether the project's test suite passed."""

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        if context.tool_name != "run_tests":
            return EvaluationResult(
                status=EvaluationStatus.NOT_APPLICABLE,
                reason="TestEvaluator does not apply to this tool.",
                details={},
            )

        result = str(context.tool_result)

        if "failed" in result.lower():
            return EvaluationResult(
                status=EvaluationStatus.FAILED,
                reason="Test suite contains failures.",
                details={
                    "test_output": result,
                },
            )

        if "passed" in result.lower():
            return EvaluationResult(
                status=EvaluationStatus.SATISFIED,
                reason="All tests passed.",
                details={
                    "test_output": result,
                },
            )

        return EvaluationResult(
            status=EvaluationStatus.FAILED,
            reason="Unable to determine test result.",
            details={
                "test_output": result,
            },
        )


