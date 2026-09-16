from llm import LLM


read_file_tool = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read and return the contents of a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the text file to read.",
                }
            },
            "required": ["path"],
        },
    },
}


llm = LLM()

messages = [
    {
        "role": "system",
        "content": (
            "You are a Python coding assistant. "
            "Use the available tools when you need information "
            "from the user's project."
        ),
    },
    {
        "role": "user",
        "content": (
            "Please inspect the file agent_architecture.md "
            "and tell me what the main objective of our agent project is."
        ),
    },
]

message = llm.generate(
    messages,
    tools=[read_file_tool],
)

print("CONTENT:")
print(message.content)

print("\nTOOL CALLS:")
print(message.tool_calls)
