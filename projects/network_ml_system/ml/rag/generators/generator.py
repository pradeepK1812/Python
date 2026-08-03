"""
Module: generator.py

Defines the abstraction for answer generation within the RAG framework.

Overview
--------
A Generator represents the business capability of transforming a user
query and retrieved contextual knowledge into a final answer.

Concrete implementations may use one or more Large Language Models (LLMs),
different prompt construction strategies, or other generation techniques,
while presenting a stable, provider-independent interface to the rest of
the framework.

Responsibilities
----------------
- Accept a user query.
- Accept retrieved contextual knowledge.
- Generate an answer.
- Return the generated answer as a domain object.

Guarantees
----------
- Returns exactly one GeneratedAnswer.
- Never modifies the supplied RetrievedContext objects.
- Remains independent of any specific LLM provider.
- Never returns None.

Does NOT
---------
- Retrieve contextual knowledge.
- Perform similarity search.
- Generate embeddings.
- Parse documents.
- Read files.
- Know about any specific LLM implementation.

Design Principles
-----------------
- Business layer abstraction.
- Provider-independent.
- Defines the contract for answer generation.
- Delegates text generation to infrastructure-layer LLM implementations.
- Extensible to multiple generation strategies.

--------------------------------------------------
Transformation
--------------------------------------------------

Input:
    Query (str)
    RetrievedContext[]

Output:
    GeneratedAnswer

--------------------------------------------------
Dependencies
--------------------------------------------------

This abstraction depends only on the RAG domain model and must not depend
on any concrete LLM implementation.
"""

from ..domain import RetrievedContext,GeneratedAnswer 
from abc import ABC, abstractmethod

class Generator(ABC):
    """
    Defines the business capability of answer generation.

    A Generator transforms a user query and the retrieved contextual
    knowledge into a generated answer.

    Concrete implementations may use different generation strategies
    while remaining independent of the underlying LLM provider.
    """
   
    @abstractmethod
    def generate(
        self,
        query: str,
        retrieved_contexts: list[RetrievedContext],
    ) -> GeneratedAnswer:
    
        """
        Generate an answer for the supplied query using the retrieved
        contextual knowledge.

        Args:
            query:
                User query.

            retrieved_contexts:
                Contexts retrieved by the Retriever, ordered by relevance.

        Returns:
            A GeneratedAnswer containing the generated response.
        """
        ...
