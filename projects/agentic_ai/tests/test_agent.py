from unittest.mock import Mock
import pytest
import agent
from task_spec import TaskSpecification


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1

        if self.calls == 1:
            raise RuntimeError("temporary LLM failure")

        return Mock(
            content="Task completed successfully.",
            tool_calls=None,
        )


def test_run_agent_retries_after_llm_failure(monkeypatch):
    fake_llm = FakeLLM()

    monkeypatch.setattr(agent, "GroqLLM", lambda: fake_llm)

    task = TaskSpecification(
        objective="Test agent retry behavior.",
        acceptance_criteria=["Task completes successfully."],
    )

    result = agent.run_agent(task)

    assert result == "Task completed successfully."
    assert fake_llm.calls == 2




class AlwaysFailingLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools=None):
        self.calls += 1
        raise RuntimeError("LLM unavailable")


def test_run_agent_stops_after_llm_retry_failure(monkeypatch):
    fake_llm = AlwaysFailingLLM()

    monkeypatch.setattr(agent, "GroqLLM", lambda: fake_llm)

    task = TaskSpecification(
        objective="Test agent retry failure.",
        acceptance_criteria=["Task completes successfully."],
    )

    with pytest.raises(
        RuntimeError,
        match="LLM request failed after 1 retry",
    ):
        agent.run_agent(task)

    assert fake_llm.calls == 2
