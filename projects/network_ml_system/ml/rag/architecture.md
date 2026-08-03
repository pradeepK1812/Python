==========================================================
DOCUMENT GRAMMAR V1
==========================================================

Purpose

Defines the semantic organization of documents stored
inside the enterprise knowledge base.

The parser relies on this grammar to identify logical
sections before chunk generation.

----------------------------------------------------------
Level 0 : Document
----------------------------------------------------------

A document represents one complete knowledge artifact.

Examples

AI_SYSTEMS_ENGINEERING_PRINCIPLES.md

RAG_Architecture.md

AI_ML_roadmap.md

----------------------------------------------------------
Level 1 : Document Header
----------------------------------------------------------

Pattern

==========================================================
TITLE
==========================================================

Represents

The overall document title.

Examples

AI SYSTEMS ENGINEERING PRINCIPLES

RAG ARCHITECTURE

----------------------------------------------------------
Level 2 : Major Section
----------------------------------------------------------

Pattern

==========================================================
SECTION TITLE
==========================================================

Represents

A major logical division of the document.

Examples

PRINCIPLE 0

RAG ARCHITECTURE

EVOLUTION OF RAG ARCHITECTURE

----------------------------------------------------------
Level 3 : Semantic Section
----------------------------------------------------------

Pattern

----------------------------------------------------------
SECTION TITLE
----------------------------------------------------------

Represents

A semantically independent knowledge unit.

Examples

Stage 1 : Why RAG?

Stage 2 : Embeddings

Principle 1

Principle 2

----------------------------------------------------------
Level 4 : Named Subsection
----------------------------------------------------------

Pattern

A short title followed by content.

Examples

Problem

Solution

Benefits

Examples

Responsibilities

Architecture

Purpose

Key Learning

Engineering Principle

Knowledge Sources

----------------------------------------------------------
Content
----------------------------------------------------------

Everything following a section belongs to that section
until another semantic boundary is encountered.

Content may include

• Paragraphs

• Bullet Lists

• Tables

• ASCII Diagrams

• Unicode Diagrams

• Blank Lines

All formatting should be preserved.

----------------------------------------------------------
Parser Output
----------------------------------------------------------

The parser converts

Document

↓

StructuredDocument

↓

Sections

without modifying the document content.

==================================================================================================================


Flow diagram: 




Knowledge Base
       │
       ▼
Reader
       │
       ▼
Document
       │
       ▼
Parser
       │
       ▼
StructuredDocument
       │
       ▼
Chunker
       │
       ▼
Chunk
       │
       ▼
Embedding
       │
       ▼
Vector Store
       │
       ▼
Retriever
       │
       ▼
Prompt Builder
       │
       ▼
LLM

=============================================================================================================

Each pipeline stage consumes one immutable domain object and produces a new immutable domain object.
An embedding is a numerical representation of semantic meaning in a vector space, where semantically similar pieces of text are located close to one another.
"An EmbeddedChunk contains an immutable snapshot of the Chunk that was embedded."
----------------------------------------------------------------------------------------------------
Refined architecture :

                  DOMAIN
────────────────────────────────────

Document

StructuredDocument

Chunk

EmbeddedChunk


────────────────────────────────────
          PIPELINE

Reader

Parser

Chunker

Embedder


────────────────────────────────────
      INFRASTRUCTURE

EmbeddingModel (ABC)

SentenceTransformerEmbeddingModel

OpenAIEmbeddingModel

OllamaEmbeddingModel
------------------------------------------------------------------------------------------------------------------

Version 1 uses eager model loading during object construction. A SentenceTransformerEmbeddingModel instance is considered fully initialized and ready for use only after the configured model has been successfully loaded. This provides fail-fast behavior, simplifies the object lifecycle, and keeps the embedding operation focused solely on inference. Alternative loading strategies (e.g., lazy loading) remain implementation details that can be introduced in future versions without changing the EmbeddingModel contract.
------------------------------------------------------------------------------------------------------------------------------------

V1 	                                               V2+
Prompt construction inside DefaultGenerator	      PromptBuilder abstraction
No filtering	                                  Filtering abstraction
No RetrievalRequest	                              RetrievalRequest
Inline mapping	                                  Mapper abstraction
One retrieval strategy	                          Multiple retrieval strategies

----------------------------------------------------------------------------------------------------------------------------------
## Version 1

Prompt construction is implemented internally within DefaultGenerator.

This keeps the Generator implementation simple while supporting the
current requirement of a single prompt construction strategy.

## Future Evolution

If multiple prompt construction strategies emerge (for example,
chat prompts, summarization prompts, few-shot prompts, or
agent prompts), prompt construction should be extracted into a
dedicated PromptBuilder abstraction.

Proposed architecture:

Generator
    │
    ▼
DefaultGenerator
    ├── PromptBuilder
    └── LLM



                     Generator (ABC)
                        ▲
                        │
               DefaultGenerator
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 Prompt Construction                 LLM (ABC)
                                        ▲
                                        │
                      OpenAI / Ollama / Claude / ...

## Future Evolution

GeneratedAnswer may evolve to include additional information such as:

- citations
- generation metadata
- usage statistics
- safety annotations

These are intentionally excluded from Version 1 to keep the domain model focused on the primary business requirement: returning a generated answer.
-------------------------------------------------------------------------------------------------------------------------------
## Future Evolution

If the framework requires provider-specific information such as:

- token usage
- finish reason
- safety annotations
- latency
- cost

the LLM abstraction may evolve to return an `LLMResponse` domain object instead of a plain string.

Version 1 intentionally returns `str` to keep the abstraction focused on its primary business responsibility: generating text.
------------------------------------------------------------------------------------------------------------------------------------

Generator architecture:

Query
RetrievedContext[]
        │
        ▼
DefaultGenerator
        │
        ├── Build Prompt
        │
        ├── Invoke LLM
        │
        ├── Receive generated text
        │
        └── Construct GeneratedAnswer
                │
                ▼
         GeneratedAnswer


                          Generator (ABC)
                        ▲
                        │
               DefaultGenerator
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  _build_prompt()                    LLM (ABC)
        │                               │
        ▼                               ▼
      Prompt                    Generated Text
                \               /
                 \             /
                  ▼           ▼
                 GeneratedAnswer

------------------------------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------------------------

