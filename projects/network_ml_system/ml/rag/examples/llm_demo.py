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
