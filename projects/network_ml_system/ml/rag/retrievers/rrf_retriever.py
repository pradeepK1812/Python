from __future__ import annotations

from ml.rag.domain import Chunk, RetrievedContext
from ml.rag.retrievers.retriever import Retriever
import re



class RRFHybridRetriever(Retriever):

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = 60,
    ):
        self._retrievers = retrievers
        if not retrievers:
            raise ValueError("At least one retriever is required.")
        self._rrf_k = rrf_k
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:

        rrf_scores: dict[str, float] = {}
        if top_k < 1:
             raise ValueError("top_k must be greater than zero.")
        contexts_by_chunk: dict[str, RetrievedContext] = {}
        for retriever in self._retrievers:
            contexts = retriever.retrieve(
             query=query,
             top_k=top_k,
            )
            for rank, context in enumerate(contexts, start=1):
                chunk_id = context.metadata["chunk_id"]
                rrf_score = 1 / (self._rrf_k + rank)
                rrf_scores[chunk_id] = (rrf_scores.get(chunk_id, 0.0) + rrf_score)
                if chunk_id not in contexts_by_chunk:
                    contexts_by_chunk[chunk_id] = context
        
        ranked_chunk_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )
        ranked_chunk_ids = ranked_chunk_ids[:top_k]
        contexts: list[RetrievedContext] = []

        for chunk_id in ranked_chunk_ids:
            context = contexts_by_chunk[chunk_id]
            context.score = rrf_scores[chunk_id]
            contexts.append(context)

        return contexts
