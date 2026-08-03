"""
Tests for the DefaultGenerator implementation.
"""

from ml.rag.domain import GeneratedAnswer, Metadata, RetrievedContext
from ml.rag.generators.default_generator import DefaultGenerator
from ml.rag.llms.base import LLM


class DummyLLM(LLM):
    """
    Simple LLM implementation used for unit testing.
    """

    @property
    def model_name(self) -> str:
        return "dummy"

    def generate(
        self,
        prompt: str,
    ) -> str:
        return "Dummy response"


def test_import() -> None:
    """
    Verify the DefaultGenerator can be imported.
    """
    from ml.rag.generators.default_generator import DefaultGenerator

    assert DefaultGenerator is not None


def test_constructor() -> None:
    """
    Verify constructor dependency injection.
    """
    llm = DummyLLM()

    generator = DefaultGenerator(llm)

    assert generator._llm is llm


def test_build_prompt() -> None:
    """
    Verify prompt construction from retrieved contexts.
    """
    generator = DefaultGenerator(DummyLLM())

    contexts = [
        RetrievedContext(
            content="TCP is a connection-oriented protocol.",
            metadata={"source": "unit-test",},
        ),
        RetrievedContext(
            content="UDP is connectionless.",
            metadata={"source": "unit-test",},
        ),
    ]

    prompt = generator._build_prompt(
        query="Which protocol guarantees delivery?",
        retrieved_contexts=contexts,
    )

    assert "Context 1:" in prompt
    assert "Context 2:" in prompt

    assert "TCP is a connection-oriented protocol." in prompt
    assert "UDP is connectionless." in prompt

    assert "Question:" in prompt
    assert "Which protocol guarantees delivery?" in prompt
    assert "Answer:" in prompt


def test_generate() -> None:
    """
    Verify GeneratedAnswer is returned.
    """
    generator = DefaultGenerator(DummyLLM())

    contexts = [
        RetrievedContext(
            content="TCP is reliable.",
            metadata={"source": "unit-test",},
        ),
    ]

    answer = generator.generate(
        query="Which protocol is reliable?",
        retrieved_contexts=contexts,
    )

    assert isinstance(answer, GeneratedAnswer)
    assert answer.answer == "Dummy response"


def test_generate_with_empty_context() -> None:
    """
    Verify generation succeeds even when no context is retrieved.
    """
    generator = DefaultGenerator(DummyLLM())

    answer = generator.generate(
        query="What is TCP?",
        retrieved_contexts=[],
    )

    assert isinstance(answer, GeneratedAnswer)
    assert answer.answer == "Dummy response"
