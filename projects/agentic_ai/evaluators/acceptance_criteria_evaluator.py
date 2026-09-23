
from evaluator import (
    EvaluationContext,
    EvaluationResult,
    EvaluationStatus,
    Evaluator,
)

class AcceptanceCriteriaEvaluator(Evaluator):
    """Evaluate whether the task acceptance criteria are satisfied."""

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        acceptance_criteria = context.state.get(
            "acceptance_criteria",
            [],
        )

        if not acceptance_criteria:
            return EvaluationResult(
                status=EvaluationStatus.FAILED,
                reason="No acceptance criteria provided.",
                details={},
            )
        if context.tool_name != "run_tests":
            return EvaluationResult(
                status=EvaluationStatus.NOT_APPLICABLE,
                reason="Acceptance criteria cannot be evaluated from this tool result.",
                details={},
            )
        test_result = str(context.tool_result)

        for criterion in acceptance_criteria:
            criterion_lower = criterion.lower()

            if "all unit tests must pass" in criterion_lower:
                if "failed" in test_result.lower():
                    return EvaluationResult(
                        status=EvaluationStatus.FAILED,
                        reason=(
                            "Acceptance criterion not satisfied: "
                            "all unit tests must pass."
                        ),
                        details={
                            "criterion": criterion,
                            "test_output": test_result,
                        },
                    )

                if "passed" not in test_result.lower():
                    return EvaluationResult(
                        status=EvaluationStatus.FAILED,
                        reason=(
                            "Unable to verify acceptance criterion: "
                            "all unit tests must pass."
                        ),
                        details={
                            "criterion": criterion,
                            "test_output": test_result,
                        },
                    )

        return EvaluationResult(
            status=EvaluationStatus.SATISFIED,
            reason="All acceptance criteria are satisfied.",
            details={
                "criteria": acceptance_criteria,
            },
        )
