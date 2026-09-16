"""
Cross-Encoder based reranker.

Contract:
- Accept a query and a list of RetrievedContext candidates.
- Score each candidate using a Cross-Encoder model.
- Return candidates in decreasing relevance order.
- Return at most top_k contexts.
- Preserve candidate content and metadata.
- Replace RetrievedContext.score with the reranker score.
- Return [] when no candidates are provided.
- top_k must be greater than zero.
- Does not perform candidate retrieval.
- Does not generate the final answer.
"""

from sentence_transformers import CrossEncoder

from ml.rag.domain import RetrievedContext
from ml.rag.rerankers.reranker import Reranker


class CrossEncoderReranker(Reranker):
    """
    Reranker implementation using a Cross-Encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self._model = CrossEncoder(model_name)

   
    def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
        top_k: int = 5,
    ) -> list[RetrievedContext]:

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if not contexts:
            return []

        pairs = [
            (query, context.content)
            for context in contexts
        ]

        scores = self._model.predict(pairs)

        scored_contexts = []

        for context, score in zip(contexts, scores):
            context.score = float(score)
            scored_contexts.append(context)

        scored_contexts.sort(
            key=lambda context: context.score,
            reverse=True,
        )

        return scored_contexts[:top_k]
