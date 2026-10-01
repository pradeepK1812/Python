import pytest

from llm import GroqLLM, LLMError


class FakeCompletions:
    def create(self, **kwargs):
        raise RuntimeError("provider failure")


class FakeChat:
    completions = FakeCompletions()


class FakeClient:
    chat = FakeChat()


def test_groq_llm_converts_provider_error_to_llm_error():
    llm = GroqLLM.__new__(GroqLLM)
    llm.client = FakeClient()
    llm.model = "test-model"

    with pytest.raises(LLMError, match="provider failure"):
        llm.generate(
            messages=[{"role": "user", "content": "test"}]
        )
