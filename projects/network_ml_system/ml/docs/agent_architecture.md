=============================================================================================
RAG Architecture

↓

Need for Planning

↓

Need for Memory

↓

Need for Tools

↓

Need for Runtime

↓

Enterprise Agent Architecture


==============================================================================================

                      User Goal
                           │
                           ▼
                    Agent Runtime
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
     Planner          Investigation      Long-term
                           State            Memory
        │                                      ▲
        ▼                                      │
 Execution Plan                                │
        │                                      │
        ▼                                      │
     Executor──────────────────────────────────┘
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
RAG   Internet       Tool Calls
 │      │               │
 └──────┼───────────────┘
        ▼
Retrieval Orchestrator
        ▼
    Re-ranker
        ▼
Prompt Constructor
        ▼
       LLM
        │
        ▼
  Observations
        │
        └──────────────► Agent Runtime


================================================================================================================








