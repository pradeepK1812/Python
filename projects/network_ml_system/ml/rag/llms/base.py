
"""
Module: base.py

Defines the abstraction for all Large Language Model (LLM) implementations.

Overview
--------
An LLM converts a textual prompt into generated text.
It defines the contract that every concrete LLM provider
(OpenAI, Ollama, Claude, Gemini, etc.) must implement.

Responsibilities
----------------
- Accept textual prompts.
- Generate text.
- Expose the configured model identity.

Guarantees
----------
- Returns generated text.
- Does not modify the supplied prompt.
- Uses the configured model.
- Raises an exception if text generation fails.

Does NOT
---------
- Retrieve contextual knowledge.
- Construct prompts.
- Search vector stores.
- Generate embeddings.
- Know about the Generator abstraction.
- Know about the RAG pipeline.

Design Principles
-----------------
- Infrastructure layer component.
- Stateless from the perspective of the RAG pipeline.
- Reusable outside this project.
- Independent of any specific LLM provider.

--------------------------------------------------
Transformation
--------------------------------------------------

Input:
    str (Prompt)

Output:
    str (Generated text)

--------------------------------------------------
Dependencies
--------------------------------------------------

This abstraction depends only on Python standard library modules
and must not depend on the RAG domain or pipeline layers.
"""


from abc import ABC, abstractmethod



class LLM(ABC):
    
    """
    Defines the contract for all Large Language Model implementations.

    Concrete implementations generate text from textual prompts while
    exposing a stable, provider-independent interface to the framework.
    """
    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the configured language model name.
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> str:

        """
        Generates text for the supplied prompt.

        Args:
            prompt:
                The textual prompt supplied to the language model.

        Returns:
            The generated text.

        """
        ...

