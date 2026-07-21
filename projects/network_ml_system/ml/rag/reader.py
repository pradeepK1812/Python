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
    if not root.exists():
       raise FileNotFoundError(f"Root directory does not exist: {root.resolve()}")
    if not root.is_dir():
       raise NotADirectoryError(
        f"'{root}' is not a valid directory."
    )

    documents: list[Document] = []

    for file_path in sorted(root.rglob("*.md")):

        try:
            content = file_path.read_text(
            encoding="utf-8"
        )
        except OSError as exc:
            raise RuntimeError(
            f"Failed to read '{file_path}'."
        ) from exc

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
