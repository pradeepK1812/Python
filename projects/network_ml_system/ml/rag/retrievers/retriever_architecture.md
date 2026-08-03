
# Architecture Principles

**Project:** Python RAG Framework

---

# Purpose

This document captures the architectural principles that govern the design of the framework.

Every component, abstraction, implementation, and interaction within the framework should adhere to these principles.

The purpose is to ensure:

- Consistency
- Extensibility
- Maintainability
- Testability
- Low coupling
- High cohesion

These principles take precedence over implementation convenience.

---

# Principle 1
## Abstractions Represent Business Capabilities

Framework abstractions should describe **what** capability the framework provides rather than **how** that capability is implemented.

### Good Examples

| Abstraction | Business Capability |
|-------------|---------------------|
| Reader | Read documents |
| Parser | Parse content |
| Chunker | Split content |
| EmbeddingModel | Generate embeddings |
| VectorStore | Store and retrieve knowledge |
| Retriever | Retrieve relevant knowledge |
| Generator | Generate answers |
| LLM | Produce language completions |

These abstractions remain stable even if implementations change.

---

# Principle 2
## Implementations Represent Strategies

Concrete classes describe **how** a capability is achieved.

Examples:

| Abstraction | Implementation |
|-------------|----------------|
| Reader | PDFReader |
| Reader | DOCXReader |
| Parser | MarkdownParser |
| Chunker | RecursiveChunker |
| EmbeddingModel | SentenceTransformerEmbeddingModel |
| EmbeddingModel | OpenAIEmbeddingModel |
| VectorStore | ChromaVectorStore |
| VectorStore | QdrantVectorStore |
| Retriever | VectorStoreRetriever |
| LLM | OpenAILLM |
| LLM | OllamaLLM |

Changing a strategy must never require changing the abstraction.

---

# Principle 3
## Depend on Abstractions

Components communicate through framework abstractions.

Never depend directly on concrete implementations.

Good

```python
vector_store: VectorStore
```

Bad

```python
vector_store: ChromaVectorStore
```

Benefits:

- Loose coupling
- Easier testing
- Easier replacement
- Extensibility

This follows the Dependency Inversion Principle (DIP).

---

# Principle 4
## Constructor Injection

Collaborators are supplied through constructors.

Example

```python
class VectorStoreRetriever(Retriever):

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        ...
```

Avoid passing collaborators into every method call.

Constructor injection:

- expresses required dependencies
- simplifies APIs
- improves readability
- supports dependency inversion
- improves testing

---

# Principle 5
## Single Responsibility Principle

Every component should have one clearly defined responsibility.

Examples

Reader

> Read documents.

Parser

> Parse content.

Chunker

> Split text.

EmbeddingModel

> Generate embeddings.

Retriever

> Retrieve knowledge.

Generator

> Generate answers.

If a class requires "and" to describe its responsibility, it probably has multiple responsibilities.

---

# Principle 6
## High Cohesion

A component should contain only logic directly related to its responsibility.

Example

Retriever

Responsible for

- retrieval coordination

Not responsible for

- parsing
- embedding generation algorithms
- prompt creation
- answer generation

Responsibilities should remain focused.

---

# Principle 7
## Low Coupling

Components should know as little as possible about one another.

Example

Retriever knows only:

- EmbeddingModel
- VectorStore

It does not know:

- Chroma
- FAISS
- cosine similarity
- SQL
- sentence transformers

Lower coupling makes replacing implementations straightforward.

---

# Principle 8
## Orchestrators Coordinate Work

Some components primarily coordinate other components rather than implementing complex algorithms themselves.

Examples

Embedder

Coordinates

Chunk
→ EmbeddingModel
→ EmbeddedChunk

Retriever

Coordinates

Query
→ EmbeddingModel
→ VectorStore
→ EmbeddedChunks

Generator (expected)

Coordinates

Query
→ Retrieved Knowledge
→ Prompt Builder
→ LLM
→ Answer

Orchestrators represent business workflows.

---

# Principle 9
## Processing Components Transform Data

Processing components perform one transformation.

Examples

PDF
→ Reader
→ Text

Text
→ Parser
→ ParsedDocument

ParsedDocument
→ Chunker
→ Chunks

Chunk
→ EmbeddingModel
→ Embedding

Prompt
→ LLM
→ Response

These components encapsulate algorithms.

---

# Principle 10
## Hide Implementation Details

Public interfaces expose business concepts.

They should never expose implementation details.

Example

Good

```python
retrieve(query)
```

Bad

```python
cosine_similarity_search(...)
```

Business language produces stable APIs.

---

# Principle 11
## Favor Composition Over Inheritance

Components collaborate through composition.

Example

```python
Retriever
 ├── EmbeddingModel
 └── VectorStore
```

Avoid deep inheritance hierarchies.

Composition improves flexibility and testability.

---

# Principle 12
## Open for Extension, Closed for Modification

Framework abstractions should support adding new implementations without modifying existing code.

Examples

New Reader

```
HTMLReader
```

New Vector Store

```
MilvusVectorStore
```

New LLM

```
ClaudeLLM
```

Existing abstractions remain unchanged.

---

# Principle 13
## Consistent Naming

Abstract classes should describe capabilities.

Concrete classes should describe implementation strategies.

Examples

Good

```
Retriever
VectorStoreRetriever
```

Good

```
EmbeddingModel
SentenceTransformerEmbeddingModel
```

Avoid implementation-specific names for abstractions.

---

# Principle 14
## Business Language First

The public API should reflect the language used by application developers rather than infrastructure terminology.

Example

Good

```
retrieve()
```

Bad

```
nearest_neighbors()
```

Example

Good

```
generate()
```

Bad

```
chat_completion()
```

Business terminology makes the framework easier to understand and more resilient to technology changes.

---

# Principle 15
## Consistency Across the Framework

Equivalent concepts should follow equivalent designs.

Examples

Embedder owns an EmbeddingModel.

Retriever owns an EmbeddingModel and a VectorStore.

Generator should own its collaborators in the same way.

Consistency is preferred over introducing special cases.

---

# Emerging Architectural Layers

The framework naturally separates into three layers.

```
+------------------------------------------------+
| Business Orchestrators                         |
|------------------------------------------------|
| RAGPipeline                                    |
| Retriever                                      |
| Generator                                      |
| Embedder                                       |
+------------------------------------------------+

+------------------------------------------------+
| Domain Services                                |
|------------------------------------------------|
| Reader                                         |
| Parser                                         |
| Chunker                                        |
| EmbeddingModel                                 |
| VectorStore                                    |
| LLM                                            |
+------------------------------------------------+

+------------------------------------------------+
| Infrastructure Implementations                 |
|------------------------------------------------|
| PDFReader                                      |
| MarkdownParser                                 |
| RecursiveChunker                               |
| SentenceTransformerEmbeddingModel              |
| ChromaVectorStore                              |
| OpenAILLM                                      |
+------------------------------------------------+
```

Each layer depends only on the layer immediately below it.

---

# Architectural Decision Process

Every new component should be designed by answering the following questions:

1. What business capability does it represent?
2. What is its single responsibility?
3. Is it an abstraction or an implementation?
4. What collaborators does it require?
5. Should it orchestrate or process?
6. Does it depend only on abstractions?
7. Does it follow constructor injection?
8. Can a new implementation be added without modifying it?

If any answer is "No", the design should be revisited.

---

# Summary

The framework is built around one central philosophy:

> **Abstractions describe business capabilities. Implementations describe strategies. Components collaborate through abstractions using composition and constructor injection. Orchestrators coordinate business workflows, while processing components encapsulate algorithms. Every component has a single, focused responsibility and remains open for extension while closed for modification.**

These principles define the architectural identity of the framework and should guide every future design decision.



Principle 17 - Abstract Implementation Metrics

Implementation-specific metrics should be exposed only through framework-level abstractions.

Example

Good

score: float

Bad

cosine_similarity: float

===========================================================================================================
