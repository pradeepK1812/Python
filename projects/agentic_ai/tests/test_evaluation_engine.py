from evaluator import EvaluationContext, EvaluationResult, Evaluator,EvaluationStatus
from evaluators.evaluation_engine import EvaluationEngine


class AlwaysSatisfiedEvaluator(Evaluator):
    """Test evaluator that always succeeds."""

    def evaluate(self, context):
        return EvaluationResult(
            status=EvaluationStatus.SATISFIED,
            reason="Evaluator satisfied.",
            details={},
        )


class AlwaysFailingEvaluator(Evaluator):
    """Test evaluator that always fails."""

    def evaluate(self, context):
        return EvaluationResult(
            status=EvaluationStatus.FAILED,
            reason="Evaluator failed.",
            details={},
        )



class AlwaysNotApplicableEvaluator(Evaluator):
    """Test evaluator that always not applicable."""

    def evaluate(self, context):
        return EvaluationResult(
            status=EvaluationStatus.NOT_APPLICABLE,
            reason="Not relevent.",
            details={},
        )

class TrackingEvaluator(Evaluator):
    """Test evaluator that records whether it was executed."""

    def __init__(self):
        self.called = False

    def evaluate(self, context):
        self.called = True

        return EvaluationResult(
            status=EvaluationStatus.SATISFIED,
            reason="Tracking evaluator executed.",
            details={},
        )


def test_evaluation_engine_all_evaluators_satisfied():
    engine = EvaluationEngine(
        [
            AlwaysSatisfiedEvaluator(),
            AlwaysSatisfiedEvaluator(),
        ]
    )

    context = EvaluationContext(
        objective="Test objective.",
        tool_name=None,
        tool_result=None,
        state={},
    )

    result = engine.evaluate(context)

    assert result.status == EvaluationStatus.SATISFIED
    assert result.reason == "All applicable evaluators are satisfied."


def test_evaluation_engine_returns_first_failure():
    engine = EvaluationEngine(
        [
            AlwaysSatisfiedEvaluator(),
            AlwaysFailingEvaluator(),
        ]
    )

    context = EvaluationContext(
        objective="Test objective.",
        tool_name=None,
        tool_result=None,
        state={},
    )

    result = engine.evaluate(context)

    assert result.status == EvaluationStatus.FAILED
    assert result.reason == "Evaluator failed."


def test_evaluation_engine_ignores_not_applicable():
    engine = EvaluationEngine(
        [
            AlwaysNotApplicableEvaluator(),
            AlwaysSatisfiedEvaluator(),
        ]
    )

    context = EvaluationContext(
        objective="test",
        tool_name="list_files",
        tool_result="",
        state={},
    )

    result = engine.evaluate(context)

    assert result.status == EvaluationStatus.SATISFIED


def test_evaluation_engine_all_evaluators_not_applicable():

    engine = EvaluationEngine(
            [
                AlwaysNotApplicableEvaluator(),
                AlwaysNotApplicableEvaluator(),
            ]
        )

    context = EvaluationContext(
            objective="test",
            tool_name="list_files",
            tool_result="",
            state={},
        )

    result = engine.evaluate(context)

    assert result.status == EvaluationStatus.NOT_APPLICABLE





def test_evaluation_engine_stops_after_failure():
    tracking_evaluator = TrackingEvaluator()

    engine = EvaluationEngine(
        [
            AlwaysFailingEvaluator(),
            tracking_evaluator,
        ]
    )

    context = EvaluationContext(
        objective="Test objective.",
        tool_name=None,
        tool_result=None,
        state={},
    )

    result = engine.evaluate(context)

    assert result.status == EvaluationStatus.FAILED
    assert tracking_evaluator.called is False
