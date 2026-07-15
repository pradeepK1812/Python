"""
Document reader for the RAG subsystem.

Responsibility
--------------
Read knowledge documents from the filesystem and convert
them into Document domain objects.
"""

from pathlib import Path

from ml.rag.domain import Document

__all__ = ["read_documents"]


def read_documents(root_directory: str) -> list[Document]:
    """
    Read every Markdown document recursively.

    Parameters
    ----------
    root_directory : str
        Root knowledge directory.

    Returns
    -------
    list[Document]
        Loaded knowledge documents.
    """

    root = Path(root_directory)

    documents: list[Document] = []

    for file_path in sorted(root.rglob("*.md")):

        content = file_path.read_text(encoding="utf-8")

        document = Document(
            name=file_path.name,
            path=str(file_path),
            content=content,
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    docs = read_documents("ml/docs")

    print(f"Loaded {len(docs)} documents")

    for document in docs:
        print("-" * 60)
        print(document.name)
        print(document.path)
        print(f"{len(document.content)} characters")
