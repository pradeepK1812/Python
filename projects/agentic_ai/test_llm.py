from llm import LLM


llm = LLM()

messages = [
    {
        "role": "user",
        "content": "Explain Python decorators in 3 sentences."
    }
]

response = llm.generate(messages)

print(response)
