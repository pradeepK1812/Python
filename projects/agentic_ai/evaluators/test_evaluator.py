from evaluator import (
    EvaluationContext,
    EvaluationResult,
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
                satisfied=False,
                reason="No test result is available.",
                details={},
            )

        result = str(context.tool_result)

        if "failed" in result.lower():
            return EvaluationResult(
                satisfied=False,
                reason="Test suite contains failures.",
                details={
                    "test_output": result,
                },
            )

        if "passed" in result.lower():
            return EvaluationResult(
                satisfied=True,
                reason="All tests passed.",
                details={
                    "test_output": result,
                },
            )

        return EvaluationResult(
            satisfied=False,
            reason="Unable to determine test result.",
            details={
                "test_output": result,
            },
        )
