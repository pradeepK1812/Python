==========================================================
RAG ARCHITECTURE V1
(Planning & Knowledge Retrieval)

==========================================================
An AI system is composed of multiple specialized models and components/services,
each optimized for a specific objective. 
These components/services are orchestrated together to solve a business problem.

==========================================================
RAG ARCHITECTURE
==========================================================

1. Purpose of this Document

2. Evolution of RAG Architecture
      Stage 1
      Stage 2
      ...
      Stage 16

3. Final Enterprise RAG Architecture

4. Component Responsibilities

5. Engineering Principles

6. Future Evolution
      ↓
      AI Agents
      ↓
      Multi-Agent Systems
      ↓
      Enterprise AI Platform




--------------------------------------------------------------------------------
# ==========================================================
# EVOLUTION OF RAG ARCHITECTURE
# ==========================================================

The architecture below was not copied from any framework.
It was derived incrementally from first principles by
analyzing the responsibilities required to solve an
enterprise business problem.

----------------------------------------------------------
Stage 1 : Why RAG?
----------------------------------------------------------

Problem

LLMs are trained once and cannot continuously learn new
enterprise knowledge.

Solution

Separate Knowledge from Reasoning.

Architecture

Documents
      │
      ▼
     LLM

----------------------------------------------------------
Stage 2 : Embeddings
----------------------------------------------------------

Instead of keyword search, represent text using semantic
vectors.

Sentence
      │
      ▼
Embedding Model
      │
      ▼
Semantic Vector

Key Learning

Embedding models are optimized for semantic representation,
not language generation.

----------------------------------------------------------
Stage 3 : Chunking
----------------------------------------------------------

Instead of embedding entire documents, divide them into
semantically meaningful chunks.

Document
      │
      ▼
Semantic Chunks
      │
      ▼
Embeddings

Engineering Principle

Start with semantic boundaries (chapters, sections, etc.)
and iteratively optimize chunk size based on retrieval
quality.

----------------------------------------------------------
Stage 4 : Vector Search
----------------------------------------------------------

Chunks
      │
      ▼
Embeddings
      │
      ▼
Vector Database
      │
      ▼
Top-K Similar Results

Concepts Learned

• Approximate Nearest Neighbor (ANN)
• FAISS
• K-Means
• Vector Indexing

----------------------------------------------------------
Stage 5 : Retrieval is not enough
----------------------------------------------------------

Similarity does not necessarily imply usefulness.

Architecture

Retrieve
      │
      ▼
Re-rank
      │
      ▼
LLM

Engineering Principle

Retrieval and ranking are different responsibilities.

----------------------------------------------------------
Stage 6 : Prompt Construction
----------------------------------------------------------

The retrieved context should not be blindly forwarded
to the LLM.

Instead, construct an optimized prompt using only the
most relevant information.

Architecture

Top-K
      │
      ▼
Prompt Constructor
      │
      ▼
LLM

----------------------------------------------------------
Stage 7 : Enterprise RAG
----------------------------------------------------------

Enterprise RAG indexes only business-specific knowledge.

Examples

Ericsson

• OMS Documentation
• Deployment Guides
• Knowledge Base
• Runbooks

Hospital

• Medical Research
• Drug Information
• Clinical Guidelines

Engineering Principle

One knowledge base per business domain.

----------------------------------------------------------
Stage 8 : Multiple Knowledge Sources
----------------------------------------------------------

Not every answer comes from enterprise documents.

Knowledge Sources

• Enterprise RAG
• Internet Search
• APIs
• Databases

Architecture

Multiple Knowledge Sources
          │
          ▼
     Unified Retrieval

----------------------------------------------------------
Stage 9 : Retrieval Orchestrator
----------------------------------------------------------

Multiple retrieval services require coordination.

Responsibilities

• Merge results
• Remove duplicates
• Normalize responses
• Handle failures
• Coordinate retrieval

Architecture

Knowledge Sources
         │
         ▼
Retrieval Orchestrator
         │
         ▼
Unified Candidates

----------------------------------------------------------
Stage 10 : Intent Router
----------------------------------------------------------

Not every query should be sent to every knowledge source.

Architecture

User Query
      │
      ▼
Intent Router
      │
      ▼
Relevant Knowledge Sources

Engineering Principle

Business objectives determine retrieval strategy.

----------------------------------------------------------
Stage 11 : Tool Calls
----------------------------------------------------------

Some knowledge is dynamic and cannot be stored inside RAG.

Examples

• uname -r
• rpm -q openssl
• systemctl status
• Database Queries
• REST APIs

Engineering Principle

Use tools to observe live system state.

----------------------------------------------------------
Stage 12 : Planner
----------------------------------------------------------

Complex business objectives require multiple dependent
actions.

Architecture

Goal
     │
     ▼
Plan
     │
     ▼
Execute
     │
     ▼
Observe
     │
     ▼
Plan Again

Engineering Principle

Planning is iterative.

----------------------------------------------------------
Stage 13 : Memory
----------------------------------------------------------

The LLM should not own long-term memory.

Separate memory into dedicated components.

Examples

• Conversation History
• User Preferences
• Previous Plans
• Business Context

----------------------------------------------------------
Stage 14 : Investigation State
----------------------------------------------------------

Enterprise AI should preserve structured investigation
state rather than simple conversation history.

State Contains

• Goal
• Completed Steps
• Evidence
• Tool Outputs
• Commands Executed
• Remaining Tasks
• Confidence
• Decision Rationale

Engineering Principle

Store evidence together with conclusions.

----------------------------------------------------------
Stage 15 : Executor
----------------------------------------------------------

Planning and execution are separate responsibilities.

Planner

Determines WHAT to do.

Executor

Determines HOW to perform it.

Engineering Principle

Separate Planning from Execution.

----------------------------------------------------------
Stage 16 : Closed-Loop Goal-Driven Execution
----------------------------------------------------------

Enterprise AI systems should continuously execute,
observe, and adapt until the business objective has
been achieved.

Architecture

Business Goal
       │
       ▼
Create Plan
       │
       ▼
Execute Step
       │
       ▼
Observe Result
       │
       ▼
Goal Achieved?
       │
   ┌───┴────┐
   │        │
 Yes        No
 │          │
 ▼          ▼
Finish   Revise Plan
              │
              ▼
        Execute Next Step

Engineering Principle

AI systems should be goal-driven rather than
workflow-driven.

The business objective remains constant while the
execution plan evolves dynamically.




------------------------------------------------------------------------------------------------

                    User Goal
                         │
                         ▼
                  Intent Router
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                  ▼
 Enterprise RAG     Internet Search      Tool Calls
       │                 │                  │
       └─────────────────┼──────────────────┘
                         ▼
            Retrieval Orchestrator
                         ▼
                   Unified Candidates
                         ▼
                    Re-ranker
                         ▼
               Prompt Constructor
                         ▼
                    Planner
                         │
                 Reads/Writes State
                         │
       ┌─────────────────┼─────────────────┐
       ▼                                   ▼
 Investigation State                Long-term Memory
       │                                   │
       └─────────────────┬─────────────────┘
                         ▼
                        LLM
                         ▼
                  Final Response

==================================================================================================


| Component              | Primary Responsibility                                                    |
| ---------------------- | ------------------------------------------------------------------------- |
| Intent Router          | Decide which knowledge sources or services should be consulted            |
| Enterprise RAG         | Retrieve business/domain-specific knowledge                               |
| Internet Search        | Retrieve external and up-to-date knowledge                                |
| Tool Calls             | Collect live system state and execute actions                             |
| Retrieval Orchestrator | Merge, normalize, and coordinate results from multiple sources            |
| Re-ranker              | Select the highest-value context for reasoning                            |
| Prompt Constructor     | Assemble an optimized prompt for the LLM                                  |
| Planner                | Decide the next action based on the current objective and system state    |
| Investigation State    | Maintain structured state, evidence, completed actions, and pending tasks |
| Long-term Memory       | Store persistent user and organizational knowledge                        |
| LLM                    | Perform reasoning and generate responses                                  |
|                                                                                                    |
-----------------------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------------------------------------------

Agent Loop:


                 Business Goal
                       │
                       ▼
                  Create Plan
                       │
                       ▼
                 Execute Step
                       │
                       ▼
              Observe Outcome
                       │
          ┌────────────┴────────────┐
          │                         │
     Success                   Failure
          │                         │
          ▼                         ▼
  Update State              Update State
          │                         │
          └────────────┬────────────┘
                       ▼
                Goal Achieved?
                       │
              ┌────────┴────────┐
              │                 │
             Yes                No
              │                 │
              ▼                 ▼
      Final Conclusion     Revise Plan
                                   │
                                   ▼
                            Execute Next Step
---------------------------------------------------------------------------------------------------------------------------------
