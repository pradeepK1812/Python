import subprocess 
from pathlib import Path
import sys

def read_file(path: str) -> str:
    """Read and return the contents of a text file."""
    return Path(path).read_text(encoding="utf-8")



def write_file(path: str, content: str) -> str:
    """Write content to a text file and return a confirmation."""
    Path(path).write_text(content, encoding="utf-8")
    return f"Successfully wrote file: {path}"

def run_tests() -> str:
    """Run the project's pytest test suite and return the result."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/"],
        capture_output=True,
        text=True,
    )

    return (
        f"Exit code: {result.returncode}\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


TOOL_REGISTRY = {
    "read_file": read_file,
    "write_file": write_file,
    "run_tests": run_tests,
}

# Describes tools that are exposed to the LLM.
TOOL_DEFINITIONS = [
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
    
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the project's pytest test suite and return the test results.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
