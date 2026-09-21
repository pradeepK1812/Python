import faiss
import numpy as np
from ml.rag.domain import EmbeddedChunk
from ml.rag.domain import VectorStore


class FAISSIVFVectorStore(VectorStore):
    """
    FAISS IVF implementation of the VectorStore interface.

    Uses IndexIVFFlat with inner-product similarity.
    """

    def __init__(
        self,
        dimension: int,
        nlist: int,
        nprobe: int = 1,
    ) -> None:

        if dimension < 1:
            raise ValueError("dimension must be greater than zero.")

        if nlist < 1:
            raise ValueError("nlist must be greater than zero.")

        if nprobe < 1:
            raise ValueError("nprobe must be greater than zero.")

        self._dimension = dimension
        self._nlist = nlist
        self._nprobe = nprobe

        self._quantizer = faiss.IndexFlatIP(
            self._dimension
        )

        self._index = faiss.IndexIVFFlat(
            self._quantizer,
            self._dimension,
            self._nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        self._index.nprobe = self._nprobe  # To apply it to FAISS

        self._chunks: list[EmbeddedChunk] = []
    
    def train(
    self,
    embeddings: list[list[float]],
    ) -> None:

        if not embeddings:
            raise ValueError("embeddings cannot be empty.")

        vectors = np.array(
            embeddings,
            dtype=np.float32,
        )

        faiss.normalize_L2(vectors)

        self._index.train(vectors)

    

    def add(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:

        if not chunks:
            return

        if not self._index.is_trained:
            raise RuntimeError(
                "FAISS IVF index must be trained before adding vectors."
            )

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

        if not self._index.is_trained:
            raise RuntimeError(
                "FAISS IVF index must be trained before searching."
            )

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

            results.append(
                self._chunks[index]
            )

        return results




