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
