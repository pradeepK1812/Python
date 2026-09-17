# FAISS Introduction — First Principles

## 1. Why do we need FAISS?

In a RAG system, documents are converted into embedding vectors and stored for similarity search.

For a query:

```text
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Vector Search
    ↓
Top-K Relevant Chunks

The simplest way to perform vector search is exact nearest-neighbor search.

For every query vector, we compare it against every stored vector:

Query Vector
     ↓
 ┌───┴─────────────────────────────┐
 ↓     ↓     ↓     ↓              ↓
 V1    V2    V3    V4    ...      VN
 └─────┴─────┴─────┴───────────────┘
                 ↓
          Similarity / Distance
                 ↓
              Top-K

This is essentially brute-force search.

If there are N vectors and each vector has dimension D, the approximate computational cost of comparing a query against the complete collection is:

Cost ≈ N × D

As the number of vectors becomes very large, searching every vector for every query becomes increasingly expensive.

2. Exact Nearest-Neighbor Search

Suppose we have:

N = 1,000,000 vectors
D = 384 dimensions

For each query, exact search potentially compares the query against all one million vectors.

The advantage is that the result is exact:

Query
  ↓
Compare with ALL vectors
  ↓
Calculate similarity/distance
  ↓
Rank vectors
  ↓
Top-K exact nearest neighbors

The disadvantage is search cost at large scale.

This leads to the fundamental question:

Can we find the nearest neighbors without examining every vector?

This is the motivation for Approximate Nearest Neighbor (ANN) search.

3. What is FAISS?

FAISS = Facebook AI Similarity Search.

FAISS is a library/toolkit for efficient similarity search and clustering of dense vectors.

The key idea is that FAISS provides multiple indexing and search strategies rather than implementing only one search mechanism.

Our mental model:

FAISS
  = toolkit/library
  + multiple indexing strategies
  + similarity search
  + exact → approximate spectrum

Therefore:

FAISS does not mean ANN only.

FAISS can also perform exact nearest-neighbor search.

4. Indexing

The central concept in FAISS is the index.

Instead of simply keeping a collection of vectors and scanning them directly, we build an index that determines how those vectors are represented and searched.

Conceptually:

Vectors
   ↓
Build Index
   ↓
Indexed representation
   ↓
Query Index
   ↓
Nearest Neighbors

Different index types use different strategies.

Examples include:

FAISS
  │
  ├── IndexFlat
  │
  ├── IVF
  │
  ├── HNSW
  │
  └── Other index combinations

These strategies provide different trade-offs between:

Search speed
Search accuracy
Memory consumption
Index construction cost
5. IndexFlat — Our First FAISS Index

IndexFlat is the simplest FAISS index and provides an exact search baseline.

Conceptually:

                 FAISS
                   │
              IndexFlat
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
       V1         V2         V3
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
             Compare all
                   ↓
                 Top-K

IndexFlat performs exhaustive search over the stored vectors.

This makes it useful as our baseline before introducing approximate indexing techniques.

We can then compare:

IndexFlat
    ↓
Exact Search

with:

IVF
    ↓
Approximate Search

and:

HNSW
    ↓
Approximate Search

This allows us to understand the actual speed/accuracy trade-offs rather than treating ANN as a black box.

6. Similarity and Distance

Vector search can be based on different similarity/distance metrics.

Two important concepts in FAISS are:

L2 Distance
Inner Product

RAG systems commonly use cosine similarity for normalized embedding vectors.

Cosine similarity is:

cos(q, v) = (q · v) / (||q|| ||v||)

If both vectors are normalized:

||q|| = 1
||v|| = 1

then:

cos(q, v) = q · v

There is also a useful relationship between normalized vectors and squared L2 distance:

||q - v||² = 2 - 2(q · v)

Therefore, for normalized vectors:

Higher cosine similarity
        ↕
Higher inner product
        ↕
Lower L2 distance

This relationship allows cosine-style retrieval to be implemented using an appropriate FAISS metric after vector normalization.

7. FAISS and our RAG Architecture

Our RAG framework already separates the vector-search capability from its implementation:

Retriever
    ↓
VectorStore
    ↓
ChromaVectorStore

Because the VectorStore abstraction already exists, we can introduce FAISS without changing the Retriever architecture:

                    VectorStore
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
      ChromaVectorStore      FAISSVectorStore
                                    │
                               FAISS Index
                                    │
                         ┌──────────┼──────────┐
                         ↓          ↓          ↓
                    IndexFlat      IVF        HNSW

The Retriever should not need to know which indexing strategy is being used.

It simply interacts with the business-level VectorStore interface.

This gives us an opportunity to validate one of our core engineering principles:

Abstractions represent business capabilities; implementations represent implementation strategies.

8. FAISS Learning Progression

We will study FAISS in the following order:

FAISS Fundamentals
        ↓
IndexFlat
        ↓
FAISS Implementation
        ↓
Exact Search Baseline
        ↓
IVF
        ↓
HNSW
        ↓
Vector Database Comparison

The first implementation goal is:

Embedding Vectors
       ↓
FAISS IndexFlat
       ↓
Add Vectors
       ↓
Search Query Vector
       ↓
Top-K Results

We will then compare the FAISS result with our existing Chroma-based retrieval.

Key Takeaways
FAISS is a vector similarity-search library/toolkit.
FAISS supports multiple indexing strategies.
FAISS is not limited to ANN; it also supports exact nearest-neighbor search.
IndexFlat provides an exact/brute-force search baseline.
IVF and HNSW provide approximate search strategies.
Different indexing strategies trade off:
Search speed
Accuracy
Memory
Index construction cost
The central problem is:

How can we search a very large vector collection efficiently without comparing every query against every vector?

Our RAG architecture already has a VectorStore abstraction, allowing us to experiment with FAISS implementations without changing the Retriever.
FAISS Mental Model
                         FAISS
                           │
             Vector indexing + search
                           │
              ┌────────────┴────────────┐
              │                         │
          Exact Search              Approximate Search
              │                         │
          IndexFlat              ┌──────┴──────┐
              │                  │             │
        Brute Force             IVF           HNSW
                             partitioning     graph

Core mental model:

FAISS
  = toolkit/library
  + multiple indexing strategies
  + similarity search
  + exact → approximate spectrum
================================================================================================

IN FAISS:



IndexFlat
    =
Exact nearest-neighbor search
    +
Store vectors directly
    +
Compare query against all vectors
    +
Return nearest K

And:

IndexFlatL2
    → smaller L2 distance = better

IndexFlatIP
    → larger inner product = better

====================================================================
