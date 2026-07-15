


"""
Core domain models for the RAG subsystem.

This module defines the business entities that flow through
the RAG pipeline.

Represents a single knowledge document in the RAG knowledge base.

A Document is the fundamental unit produced by the Reader
component and consumed by the Chunker.

Engineering Principle
---------------------
Model business concepts explicitly rather than using generic
Python dictionaries.



Current Version
---------------
Document

Future Versions
---------------
Chunk
Embedding
SearchResult
RetrievedContext
Prompt
"""

from dataclasses import dataclass

__all__ = ["Document"]


@dataclass(slots=True, frozen=True)
class Document:
    """
    Represents one knowledge document.

    Attributes
    ----------
    name
        File name.

    path
        Relative or absolute path.

    content
        Complete textual content.
    """

    name: str
    path: str
    content: str


@dataclass(slots=True, frozen=True)
class Chunk:
    """
    Represents one semantic chunk extracted from a document.

    A Chunk is the fundamental unit that will later be
    converted into an embedding and stored in the vector
    database.
    """

    chunk_id: str
    source_document: str
    section: str
    content: str


@dataclass(slots=True, frozen=True)
class Section:
    """
    Represents one semantic section inside a document.
    """

    title: str

    level: int

    content: str


@dataclass(slots=True, frozen=True)
class StructuredDocument:
    """
    Parsed representation of a knowledge document.
    """

    source_document: Document

    title: str

    sections: list[Section]


