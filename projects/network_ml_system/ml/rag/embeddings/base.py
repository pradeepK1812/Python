"""
Module: base.py

Defines the abstraction for all embedding model implementations.

Overview
--------
An EmbeddingModel converts textual content into a semantic vector
representation. It defines the contract that every concrete embedding
provider (SentenceTransformer, OpenAI, Ollama, etc.) must implement.

Responsibilities
----------------
- Accept textual input.
- Generate a semantic embedding vector.
- Expose the identity of the embedding model.

Guarantees
----------
- Produces a dense numeric embedding vector.
- Does not modify the input text.
- Produces embeddings using the configured model.
- Raises an exception if embedding generation fails.

Does NOT
---------
- Know about domain objects (Document, Chunk, EmbeddedChunk).
- Store embeddings.
- Perform similarity search.
- Interact with vector databases.
- Perform retrieval or ranking.
- Manage model selection or orchestration.

Design Principles
-----------------
- Infrastructure layer component.
- Stateless from the perspective of the RAG pipeline.
- Reusable outside this project.
- Independent of any specific embedding provider.



--------------------------------------------------
Transformation
--------------------------------------------------
Input:
    str

Output:
    list[float] (Embedding vector)
------------------------------------------------------

Dependencies
------------
This abstraction depends only on Python standard library modules and
must not depend on the RAG domain or pipeline layers.
---------------------------------------------------------
"""



from abc import ABC, abstractmethod
from typing import TypeAlias

# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------

# Represents a semantic embedding vector.
EmbeddingVector: TypeAlias = list[float]



class EmbeddingModel(ABC):
    """
    Defines the contract for all embedding model implementations.

    Concrete implementations convert textual input into semantic embedding
    vectors while exposing a stable, provider-independent interface to the
    pipeline layer.
    """
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the name of the embedding model.
        """
        ...

    @abstractmethod
    def embed(self, text: str) -> EmbeddingVector:
        """
        Generates a semantic embedding for the supplied text.
        """
        ...


