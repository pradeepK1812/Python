# Reranker

## 1. Purpose

The Reranker is a second-stage retrieval component responsible for
reordering an already retrieved set of candidate contexts according to
their relevance to a query.

The Reranker does not discover or retrieve candidates from a knowledge
base. Candidate generation is the responsibility of a Retriever.

The Reranker improves the ordering of retrieved candidates before they
are passed to downstream components such as the Generator.

---

## 2. Responsibility

The Reranker is responsible for:

- Accepting a query and a set of retrieved candidate contexts.
- Evaluating the relevance of each candidate to the query.
- Assigning a business-level relevance score to candidates.
- Ordering candidates from most relevant to least relevant.
- Returning the requested number of top-ranked candidates.
- Preserving the candidate context and metadata.
- Handling empty candidate lists.

---

## 3. Non-Responsibilities

The Reranker does not:

- Retrieve documents from a knowledge base.
- Perform vector search.
- Perform lexical/BM25 search.
- Generate embeddings for the retrieval pipeline.
- Perform RRF or other rank-fusion algorithms.
- Generate the final natural-language answer.
- Modify the underlying knowledge base or vector store.

Candidate generation and candidate ranking are intentionally separate
responsibilities.

---

## 4. Processing Flow

The expected retrieval flow is:

    Query
      |
      v
    Retriever
      |
      v
    Candidate RetrievedContext[]
      |
      v
    Reranker
      |
      v
    Re-ranked RetrievedContext[]
      |
      v
    Generator

A Retriever should generally retrieve more candidates than the final
number required by the Generator.

For example:

    Retriever -> top 10 candidates
                     |
                     v
                 Reranker
                     |
                     v
                 final top 2

This allows the Retriever to optimize for candidate recall while the
Reranker optimizes the ordering of those candidates.

---

## 5. Abstraction

The Reranker abstraction represents the business capability of
re-ranking retrieved contexts.

Conceptually:

    class Reranker(ABC):

        @abstractmethod
        def rerank(
            self,
            query: str,
            contexts: list[RetrievedContext],
            top_k: int = 5,
        ) -> list[RetrievedContext]:
            ...

The abstraction must not expose implementation-specific concepts such
as:

- Cross-Encoder
- Transformer model
- model logits
- tokenizer details
- framework-specific objects

---

## 6. Input

The Reranker accepts:

### Query

A user query represented as a string.

### Candidate contexts

A list of `RetrievedContext` objects produced by a Retriever.

Each candidate contains:

- content
- metadata
- optional retrieval score

The Reranker evaluates the candidate content in relation to the query.

---

## 7. Output

The Reranker returns:

    list[RetrievedContext]

The returned contexts must be ordered from highest relevance to lowest
relevance.

The returned list must contain at most `top_k` contexts.

The original context content and metadata must be preserved.

The `score` field represents the Reranker's relevance score after
reranking.

The score is a business-level relevance score and must not expose
implementation-specific score semantics.

---

## 8. Score Semantics

The Reranker owns the meaning of the returned `RetrievedContext.score`.

The score represents the relevance of a candidate to the query according
to the Reranker's ranking strategy.

Consumers must not assume that the score is:

- a cosine similarity
- a BM25 score
- an RRF score
- a probability

unless explicitly documented by the concrete implementation.

The ordering of the returned contexts is the primary contract.

---

## 9. Top-K Semantics

`top_k` represents the maximum number of contexts returned after
reranking.

If fewer candidates are supplied than `top_k`, all available candidates
are returned.

Example:

    candidates = 3
    top_k = 5

Result:

    3 contexts

If:

    candidates = 10
    top_k = 3

Result:

    3 highest-ranked contexts

`top_k` must be greater than zero.

An invalid `top_k` should raise `ValueError`.

---

## 10. Empty Input

If the candidate list is empty, the Reranker returns:

    []

No model inference is required.

The Reranker must not return `None`.

---

## 11. Candidate Identity

The Reranker operates on existing `RetrievedContext` objects.

It must preserve the identity of each candidate through its metadata,
including the existing `chunk_id`.

The Reranker must not create a new chunk identity.

---

## 12. Ordering

The returned candidates must be ordered by decreasing relevance.

If two candidates receive equal relevance scores, the implementation
should use a deterministic tie-breaking strategy.

The tie-breaking strategy should not depend on nondeterministic model
or collection behavior.

---

## 13. Initial Implementation Strategy

The first concrete implementation will use a Cross-Encoder model.

Unlike a standard embedding model, a Cross-Encoder evaluates the query
and candidate context together.

Conceptually:

    Query ------------------+
                            |
                            v
                      Cross-Encoder
                            ^
                            |
    Candidate Context ------+
                            |
                            v
                      Relevance Score

For N candidates, the Cross-Encoder evaluates N query-context pairs.

The resulting relevance scores are then used to order the candidates.

---

## 14. Cross-Encoder Implementation

The Cross-Encoder implementation is responsible for:

- Loading the selected Cross-Encoder model.
- Creating query-context pairs.
- Computing relevance scores.
- Associating scores with the corresponding `RetrievedContext`.
- Sorting candidates by score.
- Returning the requested top-k candidates.

The concrete implementation must hide model-library-specific details
behind the `Reranker` abstraction.

---

## 15. Candidate Count vs Final Top-K

The number of candidates supplied to the Reranker and the final `top_k`
are separate concepts.

Example:

    Retriever:
        top_k = 10

    Reranker:
        top_k = 2

The Retriever produces a candidate pool of 10 contexts.

The Reranker evaluates those 10 contexts and returns the best 2.

This separation is important because retrieving only the final number of
contexts may prevent the Reranker from recovering a relevant context
that was ranked slightly lower by the initial Retriever.

---

## 16. Reranking Pipeline Variants

The architecture should support reranking after different retrieval
strategies.

### Dense retrieval

    Query
      |
      v
    Dense Retriever
      |
      v
    Candidates
      |
      v
    Reranker
      |
      v
    Final Contexts

### BM25 retrieval

    Query
      |
      v
    BM25 Retriever
      |
      v
    Candidates
      |
      v
    Reranker
      |
      v
    Final Contexts

### Hybrid retrieval

    Query
      |
      +----> Dense Retriever ----+
      |                          |
      +----> BM25 Retriever -----+
                                 |
                                 v
                                RRF
                                 |
                                 v
                             Candidates
                                 |
                                 v
                              Reranker
                                 |
                                 v
                           Final Contexts

The hybrid + reranker pipeline is a later experiment. The initial
reranker evaluation should be performed independently to measure the
effect of reranking.

---

## 17. Error Handling

The Reranker should validate:

- `top_k >= 1`
- required query input
- valid candidate collection

Invalid `top_k` should raise `ValueError`.

Model-specific failures should be allowed to propagate rather than being
silently converted into empty retrieval results.

---

## 18. Evaluation Strategy

Reranking should be evaluated independently from RRF initially.

The first experiment should compare:

    Dense Retriever
        vs
    Dense Retriever + Reranker

The Retriever should produce a candidate pool larger than the final
evaluation K.

Example:

    Candidate K = 5
    Final K = 2

The existing retrieval evaluation metrics can then be used:

- Precision@K
- Recall@K
- MRR

The purpose of the experiment is to determine whether reranking improves
the relevance of the final retrieved contexts.

After the standalone reranker experiment, a second experiment can
evaluate:

    Dense + BM25
        |
        v
       RRF
        |
        v
    Candidate pool
        |
        v
    Reranker

This separates the effects of hybrid retrieval and reranking.

---

## 19. Architectural Principle

The Reranker is an independent second-stage ranking capability.

The architecture intentionally separates:

    Candidate Generation
            from
    Candidate Ranking

Retrievers determine which candidates are considered.

Rerankers determine the relative relevance of those candidates.

This separation allows different retrieval and reranking strategies to
be composed without coupling their implementations.

