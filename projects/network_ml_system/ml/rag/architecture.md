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
