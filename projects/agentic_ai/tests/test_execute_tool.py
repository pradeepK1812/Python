import json
from agent import execute_tool
import sys
print("\n--> SYS PATH:", sys.path)




class FailingToolCall:
    class Function:
        name = "read_file"
        arguments = '{"path": "does_not_exist.py"}'

    function = Function()


def test_execute_tool_tool_failure_returns_error():
    result = execute_tool(FailingToolCall())

    assert result.startswith(
        "Tool error while executing 'read_file': FileNotFoundError:"
    )
class MalformedToolCall:
    class Function:
        name = "read_file"
        arguments = '{"path": '

    function = Function()


def test_execute_tool_malformed_arguments_returns_error():
    result = execute_tool(MalformedToolCall())

    assert result.startswith(
        "Tool error: invalid JSON arguments for tool 'read_file':"
    )

class FakeToolCall:
    class Function:
        name = "repo_browser.list_files"
        arguments = json.dumps({"path": ""})

    function = Function()


def test_execute_tool_unknown_tool_returns_error():
    result = execute_tool(FakeToolCall())

    assert result == (
        "Tool error: unknown tool 'repo_browser.list_files'."
    )
