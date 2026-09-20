import json
from llm import HuggingFaceLLM
from llm import GroqLLM
from tools import read_file
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY
from task_spec import TaskSpecification

def execute_tool(tool_call):
    """Execute a tool requested by the LLM."""

    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    tool = TOOL_REGISTRY.get(function_name)

    if tool is None:
        raise ValueError(f"Unknown tool: {function_name}")
    
    try:
        return tool(**arguments)

    except Exception as exc:
        return (
            f"Tool error while executing '{function_name}': "
            f"{type(exc).__name__}: {exc}"
        ) 



def run_agent(task: TaskSpecification):
    """Run the basic agent loop."""

    #llm = LLM()
    #llm = HuggingFaceLLM()
    llm = GroqLLM()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding assistant. "
                "Use only the tools explicitly provided to you. "
                "Do not invent, assume, or call tools that are not provided. "
                "When creating tests for this project, place them under "
                "the tests/ directory."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Objective:\n{task.objective}\n\n"
                "Acceptance criteria:\n"
                + "\n".join(
                    f"- {criterion}"
                    for criterion in task.acceptance_criteria
                )
            ),
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
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )
    
    raise RuntimeError(
        f"Agent stopped after reaching the maximum "
        f"of {MAX_ITERATIONS} iterations."
    )



if __name__ == "__main__":
    import sys
    from task_loader import load_task

    if len(sys.argv) != 2:
        print("Usage: python agent.py <task-file>")
        sys.exit(1)

    task_file = sys.argv[1]

    task = load_task(task_file)

    result = run_agent(task)

    print("\n=== Agent Result ===")
    print(result)
