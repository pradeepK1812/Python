"""
Module: default_generator.py

Defines the default implementation of the Generator abstraction.

Overview
--------
The DefaultGenerator orchestrates the answer generation workflow by
constructing a prompt from the user query and retrieved contextual
knowledge, invoking the configured LLM, and transforming the generated
text into a GeneratedAnswer.

Responsibilities
----------------
- Accept a user query.
- Accept retrieved contextual knowledge.
- Construct a prompt for the configured LLM.
- Invoke the configured LLM.
- Transform generated text into a GeneratedAnswer.
- Return the generated answer.

Guarantees
----------
- Produces exactly one GeneratedAnswer.
- Uses the configured LLM to generate text.
- Does not modify the supplied RetrievedContext objects.
- Returns the generated answer produced by the configured LLM.

Does NOT
---------
- Retrieve contextual knowledge.
- Perform similarity search.
- Generate embeddings.
- Know about any specific LLM provider.
- Store prompts.
- Manage conversations.

Design Principles
-----------------
- Business component.
- Default implementation of the Generator abstraction.
- Delegates text generation to the configured LLM.
- Keeps prompt construction internal in Version 1.
- Independent of any specific LLM provider.
"""
class DefaultGenerator(Generator):

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        self._llm = llm

    def generate(
        self,
        query: str,
        retrieved_contexts: list[RetrievedContext],
    ) -> GeneratedAnswer:
        
        prompt = self._build_prompt(
            query,
            retrieved_contexts,
        )

        generated_text = self._llm.generate(
            prompt,
        )

        return GeneratedAnswer(
            answer=generated_text,
        )
    
    
        def _build_prompt(
        self,
        query: str,
        retrieved_contexts: list[RetrievedContext],
        ) -> str:
        """
        Construct the prompt supplied to the configured LLM.

        Formats the retrieved contextual knowledge into a readable prompt
        and appends the user's question, producing the final prompt used
        for answer generation.
        """

        # ------------------------------------------------------------------
        # Format each retrieved context as a separate context block.
        # ------------------------------------------------------------------
        context_blocks = [
            f"Context {index}:\n{context.content}"
            for index, context in enumerate(
                retrieved_contexts,
                start=1,
            )
        ]

        # ------------------------------------------------------------------
        # Combine all context blocks into a single context section.
        # ------------------------------------------------------------------
        context_text = "\n\n".join(context_blocks)

        # ------------------------------------------------------------------
        # Build the final prompt.
        # ------------------------------------------------------------------
        return f"""You are a helpful AI assistant.

        Answer the user's question using only the provided context.

        If the answer cannot be determined from the provided context,
        respond that the information is not available.

        Context:
        ----------------------------------------

        {context_text}

        ----------------------------------------

        Question:
        {query}

        Answer:"""
    
