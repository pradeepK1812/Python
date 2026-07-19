"""
Strategy

Converts textual chunks into semantic vector representations using an embedding model.

Contract

Input:

Chunk

Output:

EmbeddedChunk

Guarantees:

Generates exactly one embedding.
Creates an immutable snapshot of the input Chunk.
Records the embedding model used.
Does not modify the original Chunk.

Never performs:

Vector database storage.
Similarity search.
Retrieval.
Ranking.
LLM inference.
"""

