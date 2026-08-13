# ADR-007: Generation layer with PromptBuilder + LLMProvider

- **Status:** Accepted  
- **Date:** 2026-08-03  
- **Sprint:** 7  

## Context

Sprint 6 returns scored `RetrievedDocument`s. Callers need a reusable way to
turn question + context into a grounded LLM answer **without** coupling to
FastAPI, full RAG orchestration, streaming, or evaluation frameworks.

Hardcoding OpenAI Chat Completions calls into services would leak vendor APIs
and make Gemini / Claude / Azure / Ollama swaps invasive.

## Decision

1. Place generation in top-level `app/generation/`.
2. Keep **PromptBuilder** pure: question + retrieved docs → `BuiltPrompt`.
3. Define `LLMProvider` with `generate(prompt) → GeneratedAnswer`.
4. Implement `OpenAIProvider` via the official OpenAI Python SDK.
5. Configure via settings: `OPENAI_API_KEY`, `OPENAI_MODEL`, `TEMPERATURE`,
   `MAX_TOKENS`, `OPENAI_TIMEOUT_SECONDS`.
6. Capture usage tokens on `GeneratedAnswer` for cost, monitoring, evaluation.
7. Keep Q&A endpoints, RAG chains, streaming, and DeepEval out of this layer.

## Why this split?

| Piece | Responsibility |
|-------|----------------|
| `PromptBuilder` | Format system + context + question (no network) |
| `LLMProvider` | Vendor-agnostic generate contract |
| `OpenAIProvider` | OpenAI Chat Completions Strategy |
| `GeneratedAnswer` | Structured text + usage + finish reason |

Analogy: recipe card (prompt) vs oven brand (provider) vs plated dish (answer).

## Consequences

- **Positive:** Swap providers without changing PromptBuilder or callers.  
- **Positive:** Prompt formatting is unit-testable without API keys.  
- **Positive:** Token usage enables cost/eval instrumentation early.  
- **Negative:** Chat Completions only for now; tool-calling / streaming deferred.  
- **Deferred:** Full RAG orchestration, FastAPI Q&A, prompt-injection defense.
