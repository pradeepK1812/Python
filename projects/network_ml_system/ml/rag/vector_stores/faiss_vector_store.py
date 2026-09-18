import faiss
import numpy as np

from ml.rag.domain import EmbeddedChunk
from ml.rag.domain import VectorStore


class FAISSVectorStore(VectorStore):
    """
    FAISS implementation of the VectorStore interface.

    Uses FAISS IndexFlatIP for exact inner-product search.
    Stored and query vectors will be L2-normalized so that
    inner product is equivalent to cosine similarity.
    """

    def __init__(
        self,
        dimension: int,
    ) -> None:

        if dimension < 1:
            raise ValueError("dimension must be greater than zero.")

        self._dimension = dimension

        self._index = faiss.IndexFlatIP(
            self._dimension
        )

        self._chunks: list[EmbeddedChunk] = []


    def add(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:

        if not chunks:
            return

        vectors = np.array(
            [chunk.embedding for chunk in chunks],
            dtype=np.float32,
        )

        faiss.normalize_L2(vectors)

        self._index.add(vectors)

        self._chunks.extend(chunks)


    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[EmbeddedChunk]:

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if self._index.ntotal == 0:
            return []

        query = np.array(
            [embedding],
            dtype=np.float32,
        )

        faiss.normalize_L2(query)
        
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(
            query,
            k,
        )

        results = []

        for index in indices[0]:
            results.append(
                self._chunks[index]
            )

        return results
