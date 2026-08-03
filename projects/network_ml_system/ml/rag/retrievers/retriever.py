"""
Module: retriever.py

Defines the abstraction for retrieving relevant knowledge in response to a
user query.

Overview
--------
The Retriever represents the business capability of knowledge retrieval.
It remains independent of the underlying retrieval strategy.

Concrete implementations may retrieve knowledge using dense vector search,
keyword search, hybrid retrieval, knowledge graphs, SQL queries, or other
techniques.

Responsibilities
----------------
- Accept a natural language query.
- Retrieve the most relevant contextual knowledge.
- Return results ordered by decreasing relevance.

Guarantees
----------
- Returns RetrievedContext domain objects.
- Results are ordered by relevance.
- Returns an empty list when no relevant context is found.
- Never returns None.

Does NOT
---------
- Generate embeddings.
- Perform vector similarity calculations.
- Store embeddings.
- Generate answers.
- Invoke an LLM.

Design Principles
-----------------
- Business capability abstraction.
- Strategy-independent.
- Depends only on framework domain objects.
"""



from __future__ import annotations

from abc import ABC, abstractmethod

from ml.rag.domain import RetrievedContext


class Retriever(ABC):
   
    """
    Defines the contract for retrieving relevant contextual knowledge.

    Concrete implementations are responsible for performing retrieval using
    their chosen strategy while honoring the framework's retrieval contract.

    Contract
    --------
    - Returns RetrievedContext domain objects.
    - Results are ordered by decreasing relevance.
    - Returns an empty list if no context is found.
    - Never returns None.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """
        Retrieve the most relevant contextual knowledge for the supplied query.

        Args:
            query:
                User query.

            top_k:
                Maximum number of contexts to return.

        Returns:
            A list of RetrievedContext objects ordered by decreasing relevance.

        Returns an empty list when no relevant context is found.
        Raises:
            NotImplementedError
        """
        ...
