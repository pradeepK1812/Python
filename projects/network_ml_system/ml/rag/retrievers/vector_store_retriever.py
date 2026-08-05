"""
vector_store_retriever.py

Retriever implementation based on dense vector similarity search.
"""

from __future__ import annotations

from ..domain import RetrievedContext,VectorStore
from ..embeddings.base import EmbeddingModel
from .retriever import Retriever
#from .vector_store import VectorStore


class VectorStoreRetriever(Retriever):
    """
    Retrieves contextual knowledge using an EmbeddingModel and a VectorStore.

    Workflow:

        Query
          │
          ▼
    EmbeddingModel
          │
          ▼
    Query Embedding
          │
          ▼
    VectorStore
          │
          ▼
    RetrievedContext[]
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """
        Retrieve the most relevant contexts for the supplied query.
        """

        query_embedding = self._embedding_model.embed(query)

        embedded_chunks = self._vector_store.search(
            embedding=query_embedding,
            top_k=top_k,
        )

        contexts: list[RetrievedContext] = []

        for embedded_chunk in embedded_chunks:

             metadata = {
                "chunk_id": embedded_chunk.chunk.chunk_id,
                "document_name": embedded_chunk.chunk.source_document.name,
                "document_path": embedded_chunk.chunk.source_document.path,
                "section": embedded_chunk.chunk.section.title,
                "chunk_index": embedded_chunk.chunk.chunk_index,
             }
             contexts.append(
                    RetrievedContext(
                        content=embedded_chunk.chunk.content,
                        metadata=metadata,
                        #score is  default none for now
                    )
             )

        return contexts
