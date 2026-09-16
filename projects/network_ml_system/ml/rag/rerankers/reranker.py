"""
Reranker abstraction.

Contract:
- Accept a query and a list of RetrievedContext candidates.
- Re-rank the candidates according to their relevance to the query.
- Return RetrievedContext objects in decreasing relevance order.
- Return at most top_k contexts.
- Return an empty list when no candidates are provided.
- Never return None.
- Preserve candidate content and metadata.
- The returned score represents the reranker's relevance score.
- Must not perform candidate retrieval.
- Must not generate the final answer.
- top_k must be greater than zero.
"""


from abc import ABC, abstractmethod

from ml.rag.domain import RetrievedContext


class Reranker(ABC):
    """
    Abstract interface for second-stage ranking of retrieved contexts.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """
        Re-rank retrieved candidate contexts according to their
        relevance to the query.

        Args:
            query: User query used to evaluate candidate relevance.
            contexts: Candidate contexts produced by a Retriever.
            top_k: Maximum number of contexts to return.

        Returns:
            Retrieved contexts ordered from most relevant to
            least relevant.

        Raises:
            ValueError: If top_k is less than 1.
        """
        ...
