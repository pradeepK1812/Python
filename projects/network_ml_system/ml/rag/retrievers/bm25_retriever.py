"""
Module: bm25_retriever.py

Defines a BM25-based implementation of the Retriever abstraction.

Overview
--------
BM25Retriever performs lexical retrieval over the textual content of
knowledge chunks using the BM25 ranking algorithm.

Unlike dense vector retrieval, BM25 does not use embeddings. It matches
query terms against chunk text and ranks chunks according to their lexical
relevance.

Architecture
------------
The retriever operates directly on Chunk objects:

    Chunk[]
       │
       ▼
   Tokenization
       │
       ▼
    BM25 Index
       │
       │ query
       ▼
  Ranked Chunks
       │
       ▼
RetrievedContext[]

Responsibilities
----------------
- Maintain a BM25 index over knowledge chunks.
- Tokenize chunk content for BM25 indexing.
- Tokenize incoming queries.
- Calculate BM25 relevance scores.
- Return the highest-ranked chunks as RetrievedContext objects.
- Preserve chunk identity and metadata in the returned contexts.

Does NOT
---------
- Generate embeddings.
- Perform vector similarity search.
- Store chunks in a vector database.
- Perform hybrid retrieval.
- Perform RRF fusion.
- Generate answers.
- Invoke an LLM.

Design Principles
-----------------
- Implements the Retriever abstraction.
- Retrieval-strategy specific.
- Independent of vector databases and embedding models.
- Uses Chunk as the indexed knowledge unit.
- Returns RetrievedContext domain objects.
- Keeps BM25 implementation details hidden from callers.

Retrieval Semantics
-------------------
- Results are ordered by decreasing BM25 relevance score.
- At most top_k contexts are returned.
- An empty corpus produces an empty result.
- An empty query produces no meaningful lexical matches.
- The retriever never returns None.

Future Integration
-------------------
This retriever may later participate in hybrid retrieval together with
dense vector retrieval. Hybrid score/rank fusion, such as Reciprocal
Rank Fusion (RRF), is intentionally outside the responsibility of this
class.
"""

from __future__ import annotations

from ml.rag.domain import Chunk, RetrievedContext
from ml.rag.retrievers.retriever import Retriever
import re
from rank_bm25 import BM25Okapi

class BM25Retriever(Retriever):
    """
    BM25-based lexical retriever.

    Retrieves relevant knowledge using BM25 lexical ranking.

    The retriever builds an in-memory BM25 index over Chunk.content and
    maps ranked BM25 results back to RetrievedContext domain objects.
    """

    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:

        self._chunks = chunks
        if chunks:
            tokenized_chunks = [
                self._tokenize(chunk.content)
                for chunk in chunks
            ]
            self._bm25 = BM25Okapi(tokenized_chunks)
        else:
            self._bm25 = None
        
       

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedContext]:
        """
        Retrieve the most relevant contexts for the supplied query.
        """

        if top_k < 1:
              raise ValueError("top_k must be greater than zero.")

        if self._bm25 is None:
           return []

        # 1. Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
           return []


        # 2. Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # 3. Rank chunk indexes by score
        ranked_indices = sorted(
               range(len(scores)),
               key=lambda i: scores[i],
               reverse=True,
        )[:top_k]

        # 4. Select top_k chunks
        selected_indices = ranked_indices[:top_k]

        # 5. Convert chunks to RetrievedContext
        contexts: list[RetrievedContext] = []
        
        for index in selected_indices:
            chunk = self._chunks[index]

            metadata = {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.source_document.name,
                "document_path": chunk.source_document.path,
                "section": chunk.section.title,
                "chunk_index": chunk.chunk_index,
            }

            contexts.append(
                RetrievedContext(
                    content=chunk.content,
                    metadata=metadata,
                    score=float(scores[index]),
                )
            )

        return contexts
   

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text for BM25 retrieval.
        """
        # implementation
        if not text:
            return []

        # Find all sequences of word characters (letters, digits, underscores)
        return re.findall(r"\w+", text.lower())
