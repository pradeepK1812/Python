from evaluator import EvaluationContext, EvaluationResult, Evaluator, EvaluationStatus


class EvaluationEngine:
    """Coordinate multiple evaluators."""

    def __init__(self, evaluators: list[Evaluator]):
        self.evaluators = evaluators
    

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        has_satisfied_evaluator = False

        for evaluator in self.evaluators:
            result = evaluator.evaluate(context)

            if result.status == EvaluationStatus.FAILED:
                return result

            if result.status == EvaluationStatus.SATISFIED:
                has_satisfied_evaluator = True

        if has_satisfied_evaluator:
            return EvaluationResult(
                status=EvaluationStatus.SATISFIED,
                reason="All applicable evaluators are satisfied.",
                details={},
            )

        return EvaluationResult(
            status=EvaluationStatus.NOT_APPLICABLE,
            reason="No evaluator was applicable to the current tool result.",
            details={},
        )
    
