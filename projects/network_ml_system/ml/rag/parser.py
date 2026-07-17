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
    
    if((index+2) < len(lines)):
       if (_is_major_separator(lines[index])  and not(_is_blank(lines[index+1])) and _is_major_separator(lines[index+2])):
           return True
       else:
           return False

    else:
          return False

def _read_major_heading(lines: list[str], index: int) -> str:
    return(lines[index+1]).strip()


def _read_minor_heading(lines: list[str], index: int) -> str:
    return _read_major_heading(lines, index)


def _is_minor_heading(lines: list[str], index: int) -> bool:

    if((index+2) < len(lines)):
       if (_is_minor_separator(lines[index])  and not(_is_blank(lines[index+1])) and _is_minor_separator(lines[index+2])):
           return True
       else:
           return False

    else:
          return False


    ### main parser logic ############################################

def _read_section_content(lines, index):

           content_lines = []
           numlines = len(lines)
           while(index < numlines and not _is_minor_heading(lines[index]):

                 currentline = lines[index]
                 index = index+1
                 if( _is_blank(currentline):
                    continue
                 else:
                    #read line
                    content_lines.append(currentline.strip())
                 
           return content_lines, index
#Main parser state machine#########################################

def parse(document: Document) -> StructuredDocument:

    lines = document.content.splitlines()
    sections = []
    contentlen = len(lines) 
    index = 0
    if not _is_major_heading(lines, 0):
        raise ValueError("Document does not start with a major heading.")

    title = _read_major_heading(lines,index)
    index += 3

    while index < contentlen:

        if _is_minor_heading(lines,index):

            heading = _read_minor_heading(lines,index)
            index += 3

            content, index = _read_section_content(lines,index)
            section = Section(title=heading,level=1,content="\n".join(content),)
            sections.append(section)
        else :
             index+=1

    return StructuredDocument(source_document=document,title=title,sections=sections)
