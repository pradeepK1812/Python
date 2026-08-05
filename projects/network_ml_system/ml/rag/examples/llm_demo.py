"""
Module: llm_demo.py

Demonstrates the direct use of the LLM abstraction.

This example creates an OllamaLLM instance and generates text
from a simple prompt without involving the Generator,
Retriever, or any other RAG components.

Purpose
-------
- Smoke test the LLM abstraction.
- Verify communication with a running Ollama server.
- Demonstrate the simplest usage of OllamaLLM.
"""


from ml.rag.llms.ollama import OllamaLLM


def main() -> None:

    llm = OllamaLLM(
        model_name="llama3.2:1b",
    )

    print(f"Model: {llm.model_name}")
    response = llm.generate(
        "Reply with exactly: Hello Network ML Framework!"
    )

    print(response)


if __name__ == "__main__":
    main()
