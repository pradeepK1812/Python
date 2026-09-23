#json
import json
#LLM 
from llm import HuggingFaceLLM
from llm import GroqLLM
#tools
from tools import read_file
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY
#tasks
from task_spec import TaskSpecification
#evaluators
from evaluator import EvaluationContext,EvaluationStatus
from evaluators.evaluation_engine import EvaluationEngine
from evaluators.test_evaluator import TestEvaluator
from evaluators.acceptance_criteria_evaluator import (
    AcceptanceCriteriaEvaluator,
)

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
    
    #LLM initialization
    #llm = LLM()
    #llm = HuggingFaceLLM()
    llm = GroqLLM()
    #initialization of evaluation engine
    evaluation_engine = EvaluationEngine(
        [
            TestEvaluator(),
            AcceptanceCriteriaEvaluator(),
        ]
    )
    messages = [
        {
            "role": "system",
            "content": (
                            " You are a Python coding agent working on the current repository. "
                            "Use ONLY the tools provided in the tools parameter. "
                            "The only available tools are: "
                            "list_files, read_file, write_file, and run_tests. "
                            "Never call or reference tools named search, repo_browser.search, "
                            "repo_browser.print_tree, or any other tool that is not explicitly "
                            "provided. "
                            "If you need to inspect the repository, use list_files and read_file. "
                            "If you need to create or modify a file, use write_file. "
                            "If you need to verify the implementation, use run_tests. "
                            "Do not assume that any other repository or filesystem tool exists. "
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
            
            evaluation_context = EvaluationContext(
                objective=task.objective,
                tool_name=tool_call.function.name,
                tool_result=result,
                state={
                    "acceptance_criteria": task.acceptance_criteria,
                },
            )

            evaluation = evaluation_engine.evaluate(evaluation_context)

            print(f"Evaluation: {evaluation.reason}")

            #if evaluation.satisfied:
            if evaluation.status == EvaluationStatus.SATISFIED:
                return (
                    f"Objective satisfied.\n"
                    f"Evaluation: {evaluation.reason}"
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
