# RAG Retrieval Evaluation Metrics — Quick Reference

## Purpose

This document is a recall/reference note for evaluating the quality of a Retrieval-Augmented Generation (RAG) retriever.

> A retriever should not be judged only by whether it returns an answer. We should measure how much relevant information it retrieves, how much irrelevant information it retrieves, and how highly it ranks useful information.

---

# 1. Confusion Matrix

For binary relevance:

- **Positive** = relevant chunk
- **Negative** = irrelevant chunk
- **TP** = relevant chunk retrieved
- **FP** = irrelevant chunk retrieved
- **FN** = relevant chunk exists but was not retrieved
- **TN** = irrelevant chunk correctly not retrieved

| | Actual Positive | Actual Negative |
|---|---|---|
| Predicted Positive | TP | FP |
| Predicted Negative | FN | TN |

---

# 2. Accuracy

Accuracy asks:

> Out of all predictions, how many were correct?

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)

----------------------------------------------------------------------------------------------------

Accuracy is usually not the primary RAG retrieval metric because the number of irrelevant chunks can greatly exceed the number of relevant chunks.

Example:

1000 chunks
10 relevant
990 irrelevant

A retriever that retrieves nothing could appear to have very high accuracy because it correctly avoids 990 irrelevant chunks, while finding zero useful information.

Therefore:

High accuracy can hide a terrible retrieval system when the negative class dominates.

3. Precision

Precision asks:

When the system retrieves something as relevant, how often is it actually relevant?

Precision = TP / (TP + FP)

High precision means retrieved context contains a high proportion of useful information.

Low precision means too much irrelevant context is being returned.

Irrelevant context can:

consume context-window space
increase LLM cost and latency
potentially confuse the LLM
potentially increase the chance of an incorrect or poorly grounded answer

Example:

10 retrieved
8 relevant
2 irrelevant

Precision = 8 / 10 = 80%
4. Recall

Recall asks:

Of all relevant information available, how much did the retriever find?

Recall = TP / (TP + FN)

High recall means the retriever is less likely to miss information needed to answer the query.

Example:

10 relevant chunks exist
8 retrieved

Recall = 8 / 10 = 80%
5. Precision vs Recall

Think of the trade-off as:

Precision
    ↓
"Don't give me junk."

Recall
    ↓
"Don't miss useful information."

Conservative retrieval may have:

Precision = high
Recall = low

Broad retrieval may have:

Precision = lower
Recall = high

The goal is not automatically to maximize one metric. The appropriate balance depends on the application.

6. Precision@K

RAG systems normally retrieve only the top K results.

Precision@K asks:

Of the first K retrieved chunks, how many are relevant?

Precision@K =
|Retrieved_K ∩ Relevant| / K

Example:

Relevant:
chunk_0002
chunk_0000
chunk_0004

Retrieved@2:
chunk_0002
chunk_0000

Therefore:

Precision@2 = 2 / 2 = 1.0 = 100%

Precision@K is highly useful for RAG because it measures the quality of the context actually passed downstream.

7. Recall@K

Recall@K asks:

Of all relevant chunks, how many were found in the first K results?

Recall@K =
|Retrieved_K ∩ Relevant| / |Relevant|

Example:

Relevant:
chunk_0002
chunk_0000
chunk_0004

Retrieved@2:
chunk_0002
chunk_0000

Therefore:

Recall@2 = 2 / 3 ≈ 66.7%

This gives:

Precision@2 = 100%
Recall@2    = 66.7%

So the retrieval is clean but incomplete.

8. F1 Score

F1 balances Precision and Recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Example:

Precision = 1.0
Recall = 0.667

F1 ≈ 0.80

F1 is useful when a single balanced number is required.

For our RAG work, Precision@K and Recall@K are more directly useful because they describe top-K retrieval behavior.

9. MRR — Mean Reciprocal Rank

Precision and Recall do not fully capture ranking position.

MRR asks:

How highly ranked is the first relevant result?

For one query:

Reciprocal Rank = 1 / rank_of_first_relevant_result

Examples:

Relevant at rank 1 → 1.0
Relevant at rank 2 → 0.5
Relevant at rank 3 → 0.333
Relevant at rank 4 → 0.25

For multiple queries:

MRR = (1 / N) × Σ(1 / rank_i)

MRR is useful when finding a relevant result early in the ranked list is important.

10. NDCG — Normalized Discounted Cumulative Gain

NDCG is useful when relevance is graded, rather than simply relevant/not-relevant.

Example relevance levels:

3 → highly relevant
2 → relevant
1 → somewhat relevant
0 → irrelevant

For our HTTP/TCP example, we identified:

chunk_0002 → Primary
chunk_0000 → Supporting
chunk_0004 → Supporting

That distinction could eventually be represented using graded relevance.

NDCG considers both:

relevance level
ranking position

We do not need to implement NDCG yet.

11. Our Current RAG Ground Truth

For:

Why does HTTP use TCP?

we defined:

chunk_0002 → Primary
chunk_0000 → Supporting
chunk_0004 → Supporting

For our initial Precision@K and Recall@K implementation, all three are treated as relevant:

Relevant:
chunk_0002
chunk_0000
chunk_0004

Non-relevant:

chunk_0001
chunk_0003

This is binary relevance even though the semantic relevance is actually graded.

12. Current Example

Suppose the retriever returns:

Top-2:
chunk_0002
chunk_0000

Ground truth:

Relevant:
chunk_0002
chunk_0000
chunk_0004

Then:

Relevant retrieved = 2
Retrieved = 2
Relevant total = 3

Therefore:

Precision@2 = 2 / 2 = 1.0 = 100%

Recall@2 = 2 / 3 ≈ 0.667 = 66.7%

If the first result is relevant:

MRR = 1 / 1 = 1.0
13. Metric Summary
Metric	           Main Question	                                     RAG Importance
Accuracy	How many predictions are correct overall?	                 Low
Precision	How much retrieved information is relevant?	                 High
Recall	    How much relevant information was found?	                 High
F1	        How well are Precision and Recall balanced?	                 Medium
Precision@K	How good are the first K results?	                         Very High
Recall@K	How much relevant knowledge is in the first K? 	             Very High
MRR	        How early is the first relevant result?	                     High
NDCG@K	    Are highly relevant results ranked higher?	                 Very High for mature evaluation

14. Why This Matters Architecturally

The purpose of metrics is not to collect numbers.

The purpose is to support engineering and architecture decisions.

Business Requirement
        ↓
Technical KPI
        ↓
Architecture Choice
        ↓
Experiment
        ↓
Measurement
        ↓
Trade-off
        ↓
Business Decision

For example, increasing K may:

K increases
    ↓
Recall may increase
    ↓
More context is retrieved
    ↓
Precision may decrease
    ↓
Context size increases
    ↓
LLM latency/cost may increase

Therefore:

More retrieved chunks are not automatically better.

The architect chooses the appropriate trade-off according to the business requirement.

15. Engineer vs Architect View
Engineer

Does retrieval work?

Architect

How good is retrieval?
How do we measure it?
What happens as the corpus grows?
What happens when retrieval quality degrades?
What latency does it introduce?
What does it cost?
How does K affect quality?
What happens when the vector database fails?
How fresh must the knowledge be?
Which optimization best aligns with the business requirement?

The architectural objective is:

Optimize the system to meet business requirements, not simply to maximize an isolated technical metric.

16. Our RAG Learning Progression
Implementation
      ↓
Measurement
      ↓
Optimization
      ↓
Trade-offs
      ↓
Architecture

Our working engineering loop is:

Understand
    ↓
Verify
    ↓
Measure
    ↓
Analyze
    ↓
Decide
    ↓
Move forward

Current status:

Precision@K  → COMPLETED
Recall@K     → COMPLETED
MRR          → NEXT
NDCG         → Later, if needed

The next major experiment is to compare our different chunking strategies using the same evaluation ground truth and metrics.

==========================================================================================================================

The Architect's 10 Questions
How good is the system?
What does "good" mean for this particular system?
How do we measure it?
Which KPIs/metrics objectively tell us whether it is good?
What happens as the system scales?
What happens when the corpus, traffic, users, data, or workload grows?
What happens when quality degrades?
How do we detect degradation, diagnose it, and recover?
What latency does it introduce?
Where are the latency bottlenecks and what are the latency targets?
What does it cost?
Compute, storage, network, LLM/API calls, operational and maintenance cost.
How do configuration parameters affect quality?
In our current RAG example: How does K affect precision, recall, latency and cost?
What happens when a dependency fails?
For RAG: vector DB failure, embedding-model failure, LLM failure, network failure, etc.
How fresh must the data be?
What is the required freshness/consistency model, and how do we maintain it?
Which optimization best aligns with the business requirement?
This is the most important one.
==================================================================================================================
