"""
Module: generator_demo.py

Demonstrates the DefaultGenerator using a real Ollama LLM.

This example manually creates RetrievedContext objects,
constructs a DefaultGenerator, and generates an answer
without involving the Retriever or VectorStore.

Purpose
-------
- Smoke test the Generator abstraction.
- Validate prompt construction.
- Validate integration with OllamaLLM.
"""

from ml.rag.domain import RetrievedContext
from ml.rag.generators.default_generator import DefaultGenerator
from ml.rag.llms.ollama import OllamaLLM


def main() -> None:

    llm = OllamaLLM(
        model_name="llama3.2:1b",
    )

    generator = DefaultGenerator(
        llm=llm,
    )

    retrieved_contexts = [
        RetrievedContext(
            content=(
                "TCP (Transmission Control Protocol) is a "
                "connection-oriented transport protocol that "
                "provides reliable, ordered delivery of data."
            ),
            metadata={},
            
        ),
        RetrievedContext(
            content=(
                "TCP performs retransmission, flow control "
                "and congestion control."
            ),
            metadata={},
           
        ),
    ]
    print(f"Model: {llm.model_name}")
    print()

    print("Query:")
    print("What is TCP?")
    print()
    print("Answer:")
    answer = generator.generate(
        query="What is TCP?",
        retrieved_contexts=retrieved_contexts,
    )

    print(answer.answer)


if __name__ == "__main__":
    main()
