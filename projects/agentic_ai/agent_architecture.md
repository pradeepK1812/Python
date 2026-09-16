# Agentic AI — Coding Agent Architecture

## 1. Objective

Build a Python coding agent that can:

- Accept a coding objective
- Inspect an existing repository
- Understand the relevant code
- Modify source code
- Create/update unit tests
- Run tests
- Analyze failures
- Iterate on the implementation
- Validate success criteria
- Stop when the objective is satisfied

---

## 2. Core Agent Architecture

```text
              Objective
                  |
                  v
           +-------------+
           |     LLM     |
           |   "Brain"   |
           +------+------+
                  |
             choose action
                  |
       +----------+----------+
       |          |          |
       v          v          v
  read_file  write_file  run_tests
       |          |          |
       +----------+----------+
                  |
                  v
               Result
                  |
                  v
                 LLM
                  |
           Objective met?
             /       \
           NO         YES
           |           |
           +-- loop    v
                    DONE



3. Fundamental Agent Loop
OBSERVE
   ↓
REASON
   ↓
ACT
   ↓
OBSERVE
   ↓
REASON
   ↓
ACT
   ↓
...
   ↓
DONE
4. Components
LLM

The reasoning/decision-making component.

Tools

Capabilities exposed to the agent:

read_file()
write_file()
search_code()
run_command()
run_tests()
State

Maintains information about:

objective
plan
files inspected
files modified
commands executed
test results
errors
Environment

The actual software repository and execution environment.

Evaluator

Determines whether the objective and acceptance criteria have been satisfied.

5. Key Principle

Let the LLM decide what to do next; let deterministic tools execute and verify the action.

6. Important Distinction

LLM ≠ Agent

Tools ≠ Agent

The agent is the complete system combining:

LLM + Tools + State + Environment + Objective + Control Loop


Roadmap:
----------------------

LLM + Tools + Agent Loop
        ↓
Next
Tool Calling
        ↓
Planning
        ↓
State / Memory
        ↓
RAG
        ↓
Evaluation
        ↓
Guardrails
        ↓
Multi-Agent
        ↓
Production Architecture



=================================================================================
LLM vs Agent boundaries
----------------------------------------------------------------------------------
              LLM SIDE
        ┌─────────────────┐
        │                 │
        │  Reasoning      │
        │  Decision       │
        │  Tool selection │
        │                 │
        └────────┬────────┘
                 │
           TOOL CALL
                 │
═════════════════╪══════════════════
          SYSTEM BOUNDARY
═════════════════╪══════════════════
                 │
        ┌────────▼────────┐
        │ Python Agent    │
        │                 │
        │ Validate call   │
        │ Execute tool    │
        │ Capture result  │
        └────────┬────────┘
                 │
                 ▼
             Environment
---------------------------------------------------------------------------

LLM
 │
 │ decides
 ▼
Tool Calling
 │
 │ enables actions
 ▼
Agent Loop
 │
 │ enables iterative behaviour
 ▼
Agent

--------------------------------------------------------------
Seperation of concerns in Agent architecture
┌──────────────────────────────────────┐
│              AGENT                   │
│                                      │
│  "What should I do next?"            │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│              TOOLS                   │
│                                      │
│  "How do I perform that operation?"  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│           ENVIRONMENT                │
│                                      │
│  Repository / OS / compiler / tests  │
└──────────────────────────────────────┘



