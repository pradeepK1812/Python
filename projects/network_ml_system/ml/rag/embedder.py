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

class Embedder:
    """
    Transform Chunk objects into EmbeddedChunk objects.
    The Embedder bridges the RAG domain model (Chunk) and the embedding
    infrastructure (EmbeddingModel) by producing EmbeddedChunk objects.

    Responsibilities
    ----------------
    - Accept Chunk objects.
    - Generate semantic embeddings.
    - Construct EmbeddedChunk objects.

    Does NOT
    --------
    - Read files.
    - Parse documents.
    - Store embeddings.
    - Retrieve embeddings.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._embedding_model = embedding_model

    def embed(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
         """Embed a sequence of chunks.""" 
        embedded_chunks: list[EmbeddedChunk] = []

        for chunk in chunks:
            embedded_chunks.append(
                self._embed_chunk(chunk)
            )

        return embedded_chunks


    def _embed_chunk(
         self,
         chunk: Chunk,
    ) -> EmbeddedChunk:
        """create embedding vector from the chunk"""
        embedding_vector = self._embedding_model.embed(
           chunk.text
        )

        return EmbeddedChunk(
            chunk=chunk,
            embedding=embedding_vector,
            embedding_model=self._embedding_model.model_name,
        )
