# Chatbot UI Demo

**UI:** Streamlit (`frontend/app.py`) → FastAPI → platform services  
**Canonical scenario:** `agent-tools-foundation-001`

---

## Architecture one-liner

> The chatbot is a **thin presentation layer**. Streamlit never runs RAG, agents,
> DeepEval, or LangSmith — it only calls FastAPI, which delegates to the frozen
> production services.

```text
Browser → Streamlit → FastAPI → RAGService / AgentService / Demo runner
                              → Evaluation / Tool Validation / Quality Gates
```

---

## Prerequisites (local)

Two terminals:

```bash
# Terminal 1 — backend
uv run uvicorn app.main:app --reload

# Terminal 2 — UI
uv run streamlit run frontend/app.py
```

Optional: `make demo` still works offline without the UI.

If RAG chat says the collection is missing, run:

```bash
make seed
# or: uv run python scripts/seed_vectorstore.py
```

On Windows without working `torch`/Chroma native add, the API auto-falls back to
hashing embeddings + a JSON vector file under `data/chroma/`.

---

## 5-minute script

### 1. Start the application (30s)

- Show `/health` is green in the Streamlit sidebar.
- Point at `API_BASE_URL` (default `http://127.0.0.1:8000`).
- Emphasize: **no secrets in the frontend** — OpenAI / LangSmith keys stay on FastAPI.

### 2. Ask a RAG question (30s)

- Mode: **RAG**
- Example: `What is the leave policy?` (or any ingested-doc question if live)
- Show the assistant answer from `POST /api/v1/rag/query`.

### 3. Show retrieved documents (30s)

- Expand **Debug**.
- Point out: question, model, latency breakdown, retrieved documents/chunks,
  correlation id, token usage.
- Stress: these fields come from `RAGResponse` — UI does not retrieve anything.

### 4. Switch to Agent mode (15s)

- Mode selector: **AGENT**
- Explain Planner → Tool Router → tools (`search_docs`, `summarize`, `calculator`, `search`).

### 5. Ask an agent question requiring tools (45s)

- Example: `Summarize the leave policy from the employee handbook.`
- Or calculator: `What is 25 * 4?`
- Answer comes from `POST /api/v1/agent/query` → `AgentService`.

### 6. Show Planner (20s)

- Expand **Agent Flow**.
- Point at planner rationale / selected tools.

### 7. Show tool calls (20s)

- Same panel: tool execution order, arguments, status, latency (Debug).

### 8. Show Tool Validation (20s)

- Expand **Evaluation** (after Run Demo for full report, or Agent response for validation).
- Expected vs actual tools · PASS / FAIL.

### 9. Show DeepEval metrics (20s)

- Click **Run Demo** (offline by default).
- Evaluation panel: Faithfulness, Hallucination, Answer Relevancy,
  Context Precision / Recall, overall score.

### 10. Show Quality Gate (20s)

- Same Evaluation panel: **PASS / WARNING / FAIL**, failed rules, recommendations.

### 11. Open LangSmith Trace (20s)

- Expand **LangSmith Trace**.
- Show Trace ID / Run ID.
- If `ENABLE_LANGSMITH=true`, click **Open LangSmith Trace**.
- Offline: NoOpTracer — explain that the UI never talks to LangSmith directly.

### 12. Explain the architecture (45s)

Talking points:

1. **Separation of concerns** — UI ≠ business logic ≠ infrastructure.
2. **FastAPI as the contract** — React (or anything) can replace Streamlit later.
3. **Evaluation as CI for AI** — tools under contracts, traces in LangSmith,
   language quality via DeepEval adapters, release policy via quality gates.
4. **Canonical demo** — `agent-tools-foundation-001` proves
   `search_docs → summarize → PASS` end-to-end.

---

## Expected demo outcome

| Signal | Expected |
|--------|----------|
| Expected tools | `search_docs`, `summarize` |
| Actual tools | `search_docs`, `summarize` |
| Tool validation | PASS |
| Quality gate | PASS (offline deterministic scores) |
| Artifacts | `data/demo/reports`, `data/demo/quality`, `data/demo/benchmarks` |

---

## Discussion topics

- Why keep the UI separate from `RAGService` / `AgentService`?
- Why call HTTP instead of importing services into Streamlit?
- How would you replace Streamlit with another client without rewriting evaluation?
- Where do secrets live, and why?
- How do quality gates differ from DeepEval metric scores?
