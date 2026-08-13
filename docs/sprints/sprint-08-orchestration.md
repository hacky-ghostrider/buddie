# Sprint 8 — End-to-End RAG Orchestrator



## Goal



Connect existing layers through a dedicated orchestration service.

No new AI capabilities — no DeepEval, RAGAS, streaming, hybrid search,

re-ranking, agents, or prompt-injection defense.



```text

Question

   ↓

RAGService (Orchestrator)

   ↓

Retriever → PromptBuilder → LLMProvider

   ↓

Generated Answer (RAGResponse)

```



## First principles (callouts)



| # | Topic | Callout |

|---|-------|---------|

| 1 | Why orchestrate? | One workflow owner sequences layers |

| 2 | Thin APIs | Controllers validate + delegate only |

| 3 | Controller vs Service | HTTP vs business use-case |

| 4 | No logic in endpoints | Endpoints must stay swappable |

| 5 | Testing | Mock Retriever / Prompt / LLM |

| 6 | Future evaluation | Eval harness calls `RAGService.query` |

| 7 | Future agents | Agent tools wrap the same service |

| 8 | SoC | Each package owns one job |

| 9 | DI | Inject abstractions at the edge |

| 10 | Open/Closed | Add providers without editing orchestrator |

| 11 | Scale | Same service behind API, batch, workers |

| 12 | Interviews | “Why not put RAG in the FastAPI route?” |



## What we tried to achieve



- `RAGService` orchestration (validate → retrieve → prompt → generate → respond)

- `POST /api/v1/rag/query` thin endpoint

- File-based prompt templates under `app/prompts/templates/`

- Soft `MAX_CONTEXT_TOKENS` warning (no truncation yet)

- Per-stage latency + correlation id logging

- Unit tests with fully mocked collaborators



## Architecture



```text

Client

  │

  ▼

FastAPI  (/api/v1/rag/query)

  │  validation + DI + error mapping

  ▼

RAGService

  │

  ├─► Retriever          → RetrievedDocument[]

  ├─► PromptBuilder      → BuiltPrompt  (templates from disk)

  └─► LLMProvider        → GeneratedAnswer

  │

  ▼

RAGResponse { answer, evidence, latencies, metadata }

```



## Configuration



| Variable | Default | Validation |

|----------|---------|------------|

| `MAX_CONTEXT_TOKENS` | `8000` | Must be `> 0` (warn only) |

| `PROMPT_TEMPLATE_DIRECTORY` | `prompts/templates` | Non-empty path |

| `RAG_DEFAULT_TOP_K` | `5` | Must be `> 0` |

| `RAG_DEFAULT_SCORE_THRESHOLD` | `0.0` | Must be in `[0, 1]` |



## Explicitly out of scope



DeepEval, RAGAS, evaluation metrics, red teaming, streaming,

prompt-injection defense, agents, multi-hop retrieval, hybrid search,

re-ranking, context truncation.



## Exit criteria



- API never calls Retriever / LLMProvider directly  

- Orchestrator depends only on abstractions  

- Prompt templates loaded from files  

- Latencies populated in response metadata  

- Unit tests mock all three collaborators  

- No evaluation / streaming / hybrid search added  



## Interview soundbite



> "I keep FastAPI thin and put the RAG use-case in `RAGService`. It sequences

> Retriever → PromptBuilder → LLMProvider behind abstractions, so I can unit

> test the pipeline without OpenAI or Chroma, and later plug the same service

> into evaluation harnesses or agent tools."


