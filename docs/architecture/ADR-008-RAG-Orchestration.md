# ADR-008: RAG orchestration service layer



- **Status:** Accepted  

- **Date:** 2026-08-03  

- **Sprint:** 8  



## Context



Sprints 1–7 delivered ingestion, embeddings, vector store, retrieval, and

generation as independent layers. Callers still had to manually sequence

Retriever → PromptBuilder → LLMProvider. Putting that sequence inside FastAPI

handlers would couple HTTP to business logic and make evaluation / agents hard

to reuse later.



## Decision



1. Introduce `app/orchestration/RAGService` as the application service /

   composition root for one RAG query.

2. Keep FastAPI endpoints thin: validate, inject `RAGService`, map errors.

3. Keep `PromptBuilder` separate (template loading + formatting only).

4. Depend only on abstractions (`Retriever`, `PromptBuilder`, `LLMProvider`).

5. Externalize prompts under `app/prompts/templates/`.

6. Record per-stage latencies and token estimates in the response metadata.

7. Defer evaluation (DeepEval / RAGAS), streaming, hybrid search, and agents.



## Why RAGService exists



It owns the **use-case workflow** — not HTTP, not OpenAI, not Chroma.

Like a Spring `@Service` called by a thin `@RestController`.



## Why PromptBuilder stays separate



Prompt formatting is pure and versionable. Separating it lets non-engineers

iterate on templates and lets tests assert grounding format without LLM calls.



## Why the API remains thin



HTTP concerns (status codes, DI, serialization) differ from RAG concerns

(retrieve → prompt → generate). Thin controllers stay swappable (CLI, batch,

eval harness) without duplicating pipeline logic.



## Why orchestration improves maintainability



| Concern | Without orchestrator | With `RAGService` |

|---------|----------------------|-------------------|

| Testing | Mock HTTP + vendors together | Mock three collaborators |

| Evaluation | Re-implement the pipeline | Call `query()` directly |

| Agents | Duplicate sequencing | Treat RAG as one tool |

| Vendor swap | Touch every endpoint | Swap one Strategy |



## Consequences



- **Positive:** Single entry point for RAG; CI tests without OpenAI/Chroma.  

- **Positive:** Latency / correlation-id observability built into the contract.  

- **Positive:** Prompt files enable versioning without code redeploys.  

- **Negative:** One more package to navigate for newcomers.  

- **Deferred:** Evaluation metrics, streaming, truncation on context overflow.


