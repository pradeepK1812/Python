"""
Parser Contract
---------------

The parser converts an unstructured Document into a
StructuredDocument.

The parser guarantees that:

1. Every semantic section is identified.

2. Every piece of business knowledge belongs to exactly one
   Section.

3. Original content is preserved.

4. No knowledge is duplicated.

5. No knowledge is discarded.

The parser analyzes structure only.

It never performs chunking, embedding generation,
or retrieval.
"""

def _is_major_separator(line: str) -> bool:
    """
    Returns True if the line is a major section separator.
    """

    line = line.strip()

    return (
        len(line) >= 10
        and set(line) == {"="}
    )


def _is_minor_separator(line: str) -> bool:
    return (
      len(line) >= 10
      and set(line) == {"-"}
    )

def _is_blank(line: str) -> bool:
    """
    Returns True if the line contains only whitespace.
    """
    return not line.strip()

def _is_major_heading(lines: list[str], index: int) -> bool:
    
    if((index+2) < len(lines):
       if (set(lines[index]) == {"="} and not(_is_blank(lines[index+1]) and set(lines[index+2]) == {"="}):
           return True
       else:
           return False

    else:
          return False

    
