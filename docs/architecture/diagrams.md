# Architecture Diagrams (v1.0)

Mermaid diagrams for interviews, README embeds, and onboarding.
Architecture is **frozen** at Version 1.0 — no MCP / Memory / Multi-Agent here.

## Component Diagram

```mermaid
flowchart TB
  subgraph Presentation
    API[FastAPI /health + /api/v1/rag]
    CLI[scripts/demo.py + evaluation CLIs]
  end

  subgraph Application
    RAG[RAGService]
    Agent[AgentService]
    Auto[EvaluationAutomationService]
    Cont[ContinuousEvaluationService]
  end

  subgraph Domain
    Retriever[Retriever]
    Prompt[PromptBuilder]
    LLM[LLMProvider]
    Planner[Planner]
    Router[ToolRouter]
    Metrics[Metric Registry]
    Gates[QualityGateEngine]
    Tools[ToolRegistry]
  end

  subgraph Adapters
    ST[SentenceTransformerEmbedding]
    Chroma[ChromaVectorStore]
    OpenAI[OpenAIProvider]
    DE[DeepEvalMetricAdapter]
    LS[LangSmithTracer / NoOpTracer]
  end

  API --> RAG
  CLI --> Agent
  CLI --> Cont
  Agent --> Planner
  Agent --> Router
  Router --> Tools
  Tools --> RAG
  RAG --> Retriever
  RAG --> Prompt
  RAG --> LLM
  Retriever --> ST
  Retriever --> Chroma
  LLM --> OpenAI
  Agent --> Auto
  Auto --> Metrics
  Metrics --> DE
  Agent --> LS
  Auto --> Cont
  Cont --> Gates
```

## Sequence Diagram — Canonical Demo

```mermaid
sequenceDiagram
  participant Demo as scripts/demo.py
  participant DS as GoldenDatasetLoader
  participant Agent as AgentService
  participant Graph as LangGraph
  participant Eval as EvaluationService
  participant QG as ContinuousEvaluationService

  Demo->>DS: load agent-tools-foundation-001
  Demo->>Agent: run(question, expected_*)
  Agent->>Graph: planner → router → finalize
  Graph-->>Agent: ToolExecution[] + answer
  Agent-->>Demo: AgentRunResult + tool_validation + trace ids
  Demo->>Eval: evaluate(RAGResponse from context)
  Eval-->>Demo: EvaluationReport + DeepEval metrics
  Demo->>QG: evaluate([report])
  QG-->>Demo: QualityDecision + quality_report.* + history
```

## Deployment Diagram

```mermaid
flowchart LR
  subgraph Host["Developer / CI / Server"]
    Make[Makefile / GitHub Actions]
    UV[Python 3.12 + uv]
    Make --> UV
  end

  subgraph Docker["Docker Compose"]
    API[rag-api container]
    Vol[(Named volumes<br/>chroma / reports / benchmarks)]
    API --> Vol
  end

  subgraph External["Optional vendors"]
    OAI[OpenAI API]
    LSmith[LangSmith]
  end

  UV -->|uvicorn app.main:app| API
  API -.->|live mode| OAI
  API -.->|ENABLE_LANGSMITH| LSmith
  Client([Browser / curl / recruiter]) --> API
```

## Evaluation Flow

```mermaid
flowchart TD
  A[Question + GoldenExample] --> B[RAG or Agent execution]
  B --> C[EvaluationContext]
  C --> D[DeepEval adapters]
  C --> E[ToolValidator]
  C --> F[LangSmith Tracer]
  D --> G[EvaluationReport]
  E --> G
  F --> G
  G --> H[QualityGateEngine]
  H --> I{Decision}
  I -->|PASS| J[Append BenchmarkHistory]
  I -->|WARNING| J
  I -->|FAIL| K[Block release / fail CI]
  J --> L[quality_report JSON/CSV/HTML]
```

## Agent Flow

```mermaid
stateDiagram-v2
  [*] --> Planner
  Planner --> Router: PlannerDecision + ToolContracts
  Router --> search_docs: order 0
  search_docs --> summarize: order 1
  summarize --> Finalize
  Finalize --> [*]: EvaluationContext + final_answer

  note right of Planner
    Rule-based by default
    LLM-pluggable later (v2)
  end note
```

## Tool Validation Flow

```mermaid
flowchart LR
  Contract[ToolContract from Planner / Golden] --> Validator[ToolValidator]
  Exec[ToolExecution from Agent] --> Mapper[ToolTraceMapper]
  Trace[LangSmith-like payload] --> Mapper
  Mapper --> ExecNorm[Normalized ToolExecution / ActualToolCall]
  ExecNorm --> Validator
  Validator --> Report[ToolValidationReport]
  Report --> EvalReport[EvaluationReport.tool_validation]
  EvalReport --> Gate[Quality gates / max_tool_failures]
```
