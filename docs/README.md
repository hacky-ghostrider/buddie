# Documentation index

Sprint-wise notes and Architecture Decision Records (ADRs) for the RAG Evaluation Platform.

**Current release:** [v1.0.0](../RELEASE_NOTES.md) — [Sprint 13](sprints/sprint-13-production-release.md)  
**Latest agent sprint:** [Sprint 14 — Multi-Tool Agent](sprints/sprint-14-multi-tool-agent.md)

## Sprints

| Sprint | Doc | Focus |
|--------|-----|--------|
| 1 | [sprint-01-foundation.md](sprints/sprint-01-foundation.md) | Project foundation, FastAPI, settings, logging |
| 2 | [sprint-02-document-ingestion.md](sprints/sprint-02-document-ingestion.md) | DocumentLoader ABC + PDF loading |
| 2.1 | [sprint-02.1-ingestion-architecture-hardening.md](sprints/sprint-02.1-ingestion-architecture-hardening.md) | Factory, registry, MetadataKeys, ADRs |
| 3 | [sprint-03-chunking-engine.md](sprints/sprint-03-chunking-engine.md) | Chunker Strategy, RecursiveChunker, ingestion package layout |
| 4 | [sprint-04-embedding-engine.md](sprints/sprint-04-embedding-engine.md) | EmbeddingModel, SentenceTransformer, EmbeddedDocument |
| 5 | [sprint-05-vector-store.md](sprints/sprint-05-vector-store.md) | VectorStore ABC, Chroma persistence, no search yet |
| 6 | [sprint-06-retrieval.md](sprints/sprint-06-retrieval.md) | Retriever, query embedding, similarity search, Top-K |
| 7 | [sprint-07-generation.md](sprints/sprint-07-generation.md) | PromptBuilder, LLMProvider, OpenAI, GeneratedAnswer |
| 8 | [sprint-08-orchestration.md](sprints/sprint-08-orchestration.md) | RAGService, thin API, prompt templates, latencies |
| 9 | [sprint-09-evaluation-framework.md](sprints/sprint-09-evaluation-framework.md) | Metric ABC, registry, EvaluationService, reports |
| 10 | [sprint-10-evaluation-automation.md](sprints/sprint-10-evaluation-automation.md) | DeepEval + LangSmith adapters, tool validation, regression, benchmarks |
| 10.2 | [sprint-10.2-agent-evaluation-preparation.md](sprints/sprint-10.2-agent-evaluation-preparation.md) | Architecture hardening for Sprint 11 agents |
| 11 | [sprint-11-langgraph-agent.md](sprints/sprint-11-langgraph-agent.md) | Production LangGraph agent; RAG as tools |
| 12 | [sprint-12-quality-gates.md](sprints/sprint-12-quality-gates.md) | Quality gates, continuous evaluation, benchmark history |
| 13 | [sprint-13-production-release.md](sprints/sprint-13-production-release.md) | v1.0 release: Docker, CI, demo, docs |
| **14** | [sprint-14-multi-tool-agent.md](sprints/sprint-14-multi-tool-agent.md) | **Multi-tool workflows, HITL writes, MULTI_TOOL metadata** |

## Demos & interview

| Doc | Description |
|-----|-------------|
| [demo/chatbot-demo.md](demo/chatbot-demo.md) | **5-minute Streamlit chatbot interview demo** |
| [demo/agent-tools-foundation.md](demo/agent-tools-foundation.md) | Canonical tool-validation scenario |
| [demo/langgraph-agent-demo.md](demo/langgraph-agent-demo.md) | LangGraph walkthrough |
| [demo/quality-gates-demo.md](demo/quality-gates-demo.md) | Continuous evaluation demo |
| [interview/90-minute-interview-walkthrough.md](interview/90-minute-interview-walkthrough.md) | Timed interview script |
| [interview/questions.md](interview/questions.md) | Architecture / DeepEval / LangGraph Q&A bank |
| [design/tradeoffs.md](design/tradeoffs.md) | Why LangGraph, Chroma, DeepEval, … |

## Architecture decisions

| ADR | Title |
|-----|--------|
| [ADR-001](architecture/ADR-001-document-loader-abstraction.md) | DocumentLoader abstraction |
| [ADR-002](architecture/ADR-002-loader-factory-and-registry.md) | Loader factory, registry, MetadataKeys |
| [ADR-003](architecture/ADR-003-chunking-strategy.md) | Chunking under ingestion |
| [ADR-004](architecture/ADR-004-embedding-layer.md) | Embedding layer |
| [ADR-005](architecture/ADR-005-vector-store-layer.md) | VectorStore + Chroma |
| [ADR-006](architecture/ADR-006-retrieval-layer.md) | Retriever |
| [ADR-007](architecture/ADR-007-generation-layer.md) | PromptBuilder + LLMProvider |
| [ADR-008](architecture/ADR-008-RAG-Orchestration.md) | RAGService orchestration |
| [ADR-009](architecture/ADR-009-Evaluation-Framework.md) | Tool-independent evaluation |
| [ADR-010](architecture/ADR-010-AI-Evaluation-Automation.md) | DeepEval / LangSmith / tools |
| [ADR-010.2](architecture/ADR-010.2-Agent-Evaluation-Preparation.md) | EvaluationContext before LangGraph |
| [ADR-011](architecture/ADR-011-LangGraph-Agent.md) | LangGraph agent |
| [ADR-012](architecture/ADR-012-AI-Quality-Gates.md) | Quality gates |

Diagrams: [architecture/diagrams.md](architecture/diagrams.md) · Index: [architecture/README.md](architecture/README.md)
