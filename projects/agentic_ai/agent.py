import json
from llm import LLM
from tools import read_file
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY


def execute_tool(tool_call):
    """Execute a tool requested by the LLM."""

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    tool = TOOL_REGISTRY.get(function_name)

    if tool is None:
        raise ValueError(f"Unknown tool: {function_name}")

    return tool(**arguments)

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

    MAX_ITERATIONS = 10

    for iteration in range(MAX_ITERATIONS):
        
        print(f"\n--- Agent iteration {iteration + 1} ---")
        message = llm.generate(
            messages,
            tools = TOOL_DEFINITIONS,
        )

        # Add the assistant's response to the conversation.
        messages.append(message)

        # No tool call means the LLM has produced the final answer.
        if not message.tool_calls:
            return message.content

        # Execute requested tools.
        for tool_call in message.tool_calls:

            print(f"Tool requested: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
            result = execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "content": result,
                }
            )
    
    raise RuntimeError(
        f"Agent stopped after reaching the maximum "
        f"of {MAX_ITERATIONS} iterations."
    )

