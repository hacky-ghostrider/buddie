# Architecture decisions

This directory holds **Architecture Decision Records (ADRs)** and Mermaid diagrams.

**v1.0 status:** Architecture frozen. See [diagrams.md](diagrams.md) and [../../ROADMAP.md](../../ROADMAP.md).

## Index

| ID | Decision |
|----|----------|
| [ADR-001](ADR-001-document-loader-abstraction.md) | `DocumentLoader` abstraction (Strategy), service layer, domain exceptions |
| [ADR-002](ADR-002-loader-factory-and-registry.md) | `DocumentLoaderFactory` + extension registry + `MetadataKeys` |
| [ADR-003](ADR-003-chunking-strategy.md) | Chunking under `app/ingestion/chunking`, recursive Strategy, deterministic ids |
| [ADR-004](ADR-004-embedding-layer.md) | `EmbeddingModel` + Sentence-Transformers; separate from vector stores |
| [ADR-005](ADR-005-vector-store-layer.md) | `VectorStore` + Chroma persistence; search added in Sprint 6 |
| [ADR-006](ADR-006-retrieval-layer.md) | `Retriever` + query embedding + similarity search |
| [ADR-007](ADR-007-generation-layer.md) | `PromptBuilder` + `LLMProvider`; OpenAI Strategy |
| [ADR-008](ADR-008-RAG-Orchestration.md) | `RAGService` orchestration; thin FastAPI; prompt templates |
| [ADR-009](ADR-009-Evaluation-Framework.md) | Tool-independent `Metric` + registry + `EvaluationService` + reports |
| [ADR-010](ADR-010-AI-Evaluation-Automation.md) | DeepEval / LangSmith adapters, tool validation, regression, benchmarks |
| [ADR-010.2](ADR-010.2-Agent-Evaluation-Preparation.md) | Agent eval prep: EvaluationContext, ToolContract, ToolTraceMapper |
| [ADR-011](ADR-011-LangGraph-Agent.md) | LangGraph agent; RAG as tools; Planner/Router; ToolExecutionStatus |
| [ADR-012](ADR-012-AI-Quality-Gates.md) | Quality gates; continuous evaluation; benchmark history |

## Target architecture (v1.0)

```text
ingestion → embeddings → vectorstore → retrieval → generation → orchestration → evaluation → agent → quality gates
```

See [diagrams.md](diagrams.md) for Mermaid component / sequence / deployment / evaluation / agent / tool-validation flows.
