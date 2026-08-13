# Sprint 7 — Generation Layer

## Goal

Turn a question plus retrieved documents into a grounded LLM answer.
No FastAPI Q&A endpoint, full RAG orchestration, streaming, prompt-injection
defense, or DeepEval in this sprint.

```text
Question + RetrievedDocument[]  →  PromptBuilder  →  LLMProvider  →  GeneratedAnswer
```

## First principles (callouts)

| # | Topic | Callout |
|---|-------|---------|
| 1 | What is generation? | LLM writes the answer from provided context |
| 2 | Retrieval vs generation | Fetch books first; write the essay later |
| 3 | No direct DB access from LLMs | LLM is the writer, not the librarian |
| 4 | Prompt engineering | Craft instructions + context so the model behaves |
| 5 | System vs user prompt | Role/rules vs the actual ask + evidence |
| 6 | Context window | Finite “desk space” for tokens |
| 7 | Prompt templates | Stable formatting → reproducible answers |
| 8 | Hallucination | Confident inventing when evidence is missing |
| 9 | Grounded generation | Answer only from retrieved context |
| 10 | Temperature | Randomness dial (0 ≈ deterministic) |
| 11 | Max tokens | Hard cap on answer length / cost |
| 12 | Deterministic generation | Low temp + fixed prompt for eval |
| 13 | Provider abstraction | Swap OpenAI / Gemini / Claude without rewrites |
| 14 | Independent testability | Mock LLM; assert prompt + parsing |
| 15 | Interview staples | Grounding, temperature, why abstract providers? |

### Analogies

- Generation ≈ writing a book report from books the librarian fetched.
- Prompt template ≈ a fill-in-the-blank exam form (same structure every time).
- System prompt ≈ teacher rules on the board; user prompt ≈ your question + handouts.
- Context window ≈ a desk that only holds so many pages.
- Temperature 0 ≈ always pick the most likely next word (exam mode).
- Provider ABC ≈ a power socket; OpenAI / Gemini are different plugs.

## What we tried to achieve

- `PromptBuilder` (no LLM calls)
- `LLMProvider` ABC (`generate`)
- `OpenAIProvider` via official OpenAI SDK
- `BuiltPrompt` + `GeneratedAnswer` (+ nested `TokenUsage`)
- Settings: `OPENAI_API_KEY`, `OPENAI_MODEL`, `TEMPERATURE`, `MAX_TOKENS`
- Domain exceptions + centralized logging (no `print`)
- Pytest with mocked OpenAI client (no live API)
- Explicitly **no** FastAPI Q&A / RAG pipeline / eval / streaming

## Architecture

```text
Question + List[RetrievedDocument]
      │
      ▼
PromptBuilder                      app/generation/
      │  BuiltPrompt { system, user, … }
      ▼
LLMProvider (ABC)
      │
      ▼
OpenAIProvider  ──Chat Completions──► OpenAI API
      │
      ▼
GeneratedAnswer { answer, model, usage, finish_reason, tokens… }
```

### Why usage information matters

| Concern | Why tokens matter |
|---------|-------------------|
| Cost | Providers bill per token |
| Monitoring | Spot prompt bloat / runaway completions |
| Evaluation | Compare quality vs token budget across models |

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/generation/prompt_builder.py` | Context + question formatting | Pure, testable prompts |
| `app/generation/llm_provider.py` | `LLMProvider` ABC | Vendor-agnostic contract |
| `app/generation/base.py` | Re-export ABC | Matches other layers' import style |
| `app/generation/openai_provider.py` | OpenAI Strategy | First concrete provider |
| `app/generation/models.py` | `BuiltPrompt`, `GeneratedAnswer` | Explicit I/O shapes |
| `app/generation/exceptions.py` | Domain errors | Clear failure modes |
| `app/config/settings.py` | Model / temp / tokens | No hardcoded knobs |
| `tests/test_generation.py` | Suite | Prompt + mock OpenAI |
| `.env.example` / `README.md` | Docs | Configure generation |
| `docs/architecture/ADR-007-*.md` | ADR | Why PromptBuilder + provider split |

## Configuration

| Variable | Default | Validation |
|----------|---------|------------|
| `OPENAI_API_KEY` | `""` | Required at call time (blank → `MissingAPIKeyError`) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Non-empty |
| `TEMPERATURE` | `0.0` | Must be in `[0, 2]` |
| `MAX_TOKENS` | `1024` | Must be `> 0` |
| `OPENAI_TIMEOUT_SECONDS` | `60` | `None` or `> 0` |

## Why mock LLMs in unit tests

Live LLM calls are **non-deterministic**, **expensive**, **slow**, and require
secrets. Unit tests assert prompt formatting and response parsing with a fake
client so CI stays fast and hermetic. Integration tests against a real model
belong in a later, opt-in suite.

## Explicitly out of scope

FastAPI Q&A endpoint, full RAG orchestration, streaming responses,
prompt-injection defense, DeepEval / evaluation metrics.

## Exit criteria

- PromptBuilder injects context + question consistently  
- Empty question / empty context handled  
- OpenAIProvider returns `GeneratedAnswer` with usage  
- Settings validate temperature and max tokens  
- Provider errors map to domain exceptions  
- Unit tests mock OpenAI (no live calls)  
- No Q&A endpoint or RAG pipeline added  

## Interview soundbite

> "I keep generation separate from retrieval. `PromptBuilder` formats a grounded
> system+user prompt from retrieved chunks; `LLMProvider` is a Strategy so
> OpenAI today can become Gemini or Claude tomorrow. `GeneratedAnswer` carries
> usage tokens for cost and evaluation — and unit tests mock the SDK so CI
> never spends money or depends on model non-determinism."
