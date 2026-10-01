import pytest

from llm import GroqLLM, LLMError



class FakeMessage:
    content = "test response"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class SuccessfulCompletions:
    def create(self, **kwargs):
        return FakeResponse()


class SuccessfulChat:
    completions = SuccessfulCompletions()


class SuccessfulClient:
    chat = SuccessfulChat()

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



def test_groq_llm_returns_provider_message():
    llm = GroqLLM.__new__(GroqLLM)
    llm.client = SuccessfulClient()
    llm.model = "test-model"

    message = llm.generate(
        messages=[{"role": "user", "content": "test"}]
    )

    assert message.content == "test response"
