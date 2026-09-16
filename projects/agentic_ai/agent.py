import json
from llm import LLM
from tools import read_file

TOOLS = [
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
    }
]


def execute_tool(tool_call):
    """Execute a tool requested by the LLM."""

    function_name = tool_call.function.name
    #arguments = tool_call.function.arguments
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "read_file":
        return read_file(arguments["path"])

    raise ValueError(f"Unknown tool: {function_name}")


def run_agent(objective: str):
    """Run the basic agent loop."""

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
            "content": objective,
        },
    ]

    while True:
        message = llm.generate(
            messages,
            tools=TOOLS,
        )

        # Add the assistant's response to the conversation.
        messages.append(message)

        # No tool call means the LLM has produced the final answer.
        if not message.tool_calls:
            return message.content

        # Execute requested tools.
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "content": result,
                }
            )
