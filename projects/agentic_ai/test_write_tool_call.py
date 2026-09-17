from llm import LLM


tools = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the text file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


llm = LLM()

messages = [
    {
        "role": "system",
        "content": (
            "You are a Python coding assistant. "
            "Use the available tools when necessary."
        ),
    },
    {
        "role": "user",
        "content": (
            "Create a file called agent_test.py containing exactly this "
            "Python code:\n\n"
            "print('Hello Agent')"
        ),
    },
]


message = llm.generate(
    messages,
    tools=tools,
)


print("CONTENT:")
print(message.content)

print("\nTOOL CALLS:")
print(message.tool_calls)
