import faiss
import numpy as np
from ml.rag.domain import EmbeddedChunk
from ml.rag.domain import VectorStore


class FAISSHNSWVectorStore(VectorStore):
    """
    FAISS HNSW implementation of the VectorStore interface.
    """

    def __init__(
        self,
        dimension: int,
        M: int,
        ef_search: int = 10,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be greater than zero.")

        if M < 1:
            raise ValueError("M must be greater than zero.")

        if ef_search < 1:
            raise ValueError("ef_search must be greater than zero.")

        self._dimension = dimension
        self._M = M
        self._ef_search = ef_search

        self._index = faiss.IndexHNSWFlat(
            self._dimension,
            self._M,
            faiss.METRIC_INNER_PRODUCT,
        )

        self._index.hnsw.efSearch = self._ef_search

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
            if index == -1:
                continue

            results.append(self._chunks[index])

        return results
