# Retrieval Architecture

---

## Status

**Current Phase:** Architecture Design

**Implementation Status:** Not Started

**Approved ADRs**

- ADR-001 – Storage Independence
- ADR-002 – Embedding Model Independence
- ADR-003 – Search Execution Ownership
- ADR-004 – Search Contract

---

# Table of Contents

1. Purpose
2. First Principles
3. Design Principles
4. Responsibilities
5. Component Architecture
6. Public Interfaces
7. Domain Objects
8. Retrieval Pipeline
9. Future Extensions
10. Design Decisions (ADR)

---

# Section 1 — Purpose

The Retrieval subsystem is responsible for locating the most relevant business knowledge for a user query and supplying it to downstream components.

The subsystem is independent of:

- Storage Engine
- Embedding Model
- Language Model

The Retrieval subsystem focuses solely on finding relevant knowledge and does not concern itself with document ingestion, prompt construction, or response generation.

---

# Section 2 — First Principles

These principles are derived before considering any concrete implementation such as FAISS, Chroma, Milvus, Qdrant, or pgvector.

## Principle 1 — Knowledge Retrieval

The primary objective of the Retrieval subsystem is to return the most relevant business knowledge for a given query.

---

## Principle 2 — Storage Independence

Retrieval depends only on an abstract storage interface rather than any concrete vector database implementation.

---

## Principle 3 — Embedding Independence

Retrieval is independent of any embedding provider such as:

- Sentence Transformers
- OpenAI Embeddings
- BGE
- E5

Instead, it depends only on an abstract embedding interface.

---

## Principle 4 — Single Responsibility

Each component has exactly one primary responsibility.

| Component | Responsibility |
|-----------|----------------|
| Retriever | Retrieval orchestration |
| VectorStore | Storage and Search |
| PromptBuilder | Prompt Construction |
| LLM | Response Generation |

No component should assume responsibilities belonging to another component.

---

# Section 3 — Design Principles

The Retrieval subsystem follows the following architectural principles:

- Separation of Concerns
- Dependency Inversion
- Storage Independence
- Embedding Independence
- Declarative APIs
- Extensibility through Abstraction

---

# Section 4 — Responsibilities

## Retriever

### Responsible For

- Accepting retrieval requests
- Obtaining query embeddings
- Constructing RetrievalRequest
- Invoking the VectorStore
- Applying retrieval strategies
- Returning relevant knowledge

### Not Responsible For

- Storing vectors
- Executing similarity search
- Building indexes
- Parsing documents
- Building prompts
- Calling the LLM

---

## VectorStore

### Responsible For

- Persisting embeddings
- Maintaining indexes
- Executing similarity search
- Returning search results

### Not Responsible For

- Embedding queries
- Prompt construction
- Calling LLMs
- Business logic

---

# Section 5 — Component Architecture

```
                    User Query
                         │
                         ▼
                   Query Embedder
                         │
                         ▼
                 RetrievalRequest
                         │
                         ▼
                     Retriever
                         │
                         ▼
             VectorStore.search(request)
                         │
          ┌──────────────┴──────────────┐
          │                             │
      FAISS Store                 Qdrant Store
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  RetrievalResult
                         │
                         ▼
                     Retriever
                         │
                         ▼
                  Prompt Builder
```

---

# Section 6 — Public Interfaces

(To be completed)

This section will define the public contracts exposed by:

- Retriever
- VectorStore
- RetrievalRequest
- RetrievalResult

No implementation details will be documented here.

---

# Section 7 — Domain Objects

(To be completed)

Expected domain objects include:

- RetrievalRequest
- RetrievalResult
- SearchResult
- SearchMetadata

These are conceptual objects only. Their implementation will be discussed later.

---

# Section 8 — Retrieval Pipeline

```
User Query
      │
      ▼
Generate Query Embedding
      │
      ▼
Create RetrievalRequest
      │
      ▼
Retriever
      │
      ▼
VectorStore.search()
      │
      ▼
Retrieve Candidate Knowledge
      │
      ▼
(Optional)
Ranking
Filtering
Reranking
      │
      ▼
Return Relevant Knowledge
```

---

# Section 9 — Future Extensions

## Retrieval Strategies

- Hybrid Search
- BM25
- Multi-vector Retrieval

---

## Ranking

- Cross Encoder
- MMR
- Learned Ranking

---

## Query Processing

- Query Expansion
- Query Rewriting

---

## Filtering

- Metadata Filtering
- Namespace Filtering
- Tenant Isolation

---

## Performance

- Caching
- Distributed Retrieval
- Sharding

---

# Section 10 — Architectural Decision Records

---

## ADR-001 — Storage Independence

### Status

Accepted

### Decision

The Retrieval subsystem depends on an abstract `VectorStore` interface rather than any concrete vector database implementation.

### Rationale

This enables support for multiple storage backends including FAISS, Chroma, Milvus, Qdrant, pgvector, and future implementations without changing retrieval logic.

### Alternatives Considered

Retriever coupled directly to FAISS.

### Decision

Rejected.

### Design Conclusion

Concrete vector databases are implementation details hidden behind the `VectorStore` abstraction.

---

## ADR-002 — Embedding Model Independence

### Status

Accepted

### Decision

The Retriever depends on an abstract embedding interface capable of generating query embeddings.

### Rationale

The Retrieval subsystem should remain independent of specific embedding providers.

### Design Conclusion

Embedding generation is delegated to the embedding abstraction.

The Retriever consumes embeddings without knowledge of how they are produced.

---

## ADR-003 — Search Execution Ownership

### Status

Accepted

### Decision

The `VectorStore` is responsible for executing search operations and determining how those operations are carried out.

### Rationale

Search execution is a storage concern.

Similarity metrics, index structures, ANN algorithms, distributed execution, and optimization strategies belong entirely to the storage layer.

### Design Conclusion

The Retriever delegates search execution to the `VectorStore` and remains independent of storage-specific search algorithms.

---

## ADR-004 — Search Contract

### Status

Accepted

### Decision

The Retriever communicates with the `VectorStore` exclusively through an abstract search contract (for example, a `RetrievalRequest`).

### Rationale

The Retriever specifies **what** knowledge is required rather than **how** it should be retrieved.

The `VectorStore` interprets the request and executes the appropriate search.

### Design Conclusion

The retrieval interface is declarative rather than procedural, allowing storage implementations to evolve independently.

RetrievalRequest

1. Search Representation
   (the semantic representation of the query)

2. Result Count
   (the maximum number of results requested)
-------------------------------------------------------------------------------------------------------------

# Glossary

| Term | Description |
|------|-------------|
| Chunk | Smallest retrievable unit of knowledge |
| EmbeddedChunk | Chunk together with its vector embedding |
| Embedding | Numerical representation of semantic meaning |
| Retriever | Orchestrates the retrieval workflow |
| VectorStore | Stores embeddings and executes search |
| RetrievalRequest | Declarative search request |
| RetrievalResult | Result returned from the VectorStore |
| Search | Finding semantically relevant knowledge |
| Ranking | Ordering retrieved knowledge by relevance |

----------------------------------------------------------------------------------------------------------------------
