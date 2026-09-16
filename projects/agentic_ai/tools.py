from pathlib import Path


def read_file(path: str) -> str:
    """Read and return the contents of a text file."""
    return Path(path).read_text(encoding="utf-8")


