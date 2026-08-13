# Interview Questions — RAG Evaluation Platform v1.0

Use with [`90-minute-interview-walkthrough.md`](90-minute-interview-walkthrough.md) and [`../design/tradeoffs.md`](../design/tradeoffs.md).

---

## Architecture Questions

1. Why is `RAGService` a separate package from FastAPI routes?
2. How do package boundaries enforce dependency direction (domain ← adapters)?
3. What would break if DeepEval types leaked into `RAGService`?
4. Explain Strategy vs Factory vs Registry in this codebase with one example each.
5. Where does composition root live for production wiring vs tests?
6. How does Sprint 10.2 `EvaluationContext` enable Sprint 11 without rewriting evaluation?

---

## DeepEval Questions

1. What problem does `DeepEvalMetricAdapter` solve?
2. Why invert the hallucination score?
3. How do you run CI without paying for LLM-as-judge calls?
4. Which metrics require an expected answer, and why?
5. How would you swap DeepEval for RAGAS later without rewriting `EvaluationService`?

---

## LangGraph Questions

1. Why LangGraph instead of a hand-rolled `if/else` tool loop?
2. What does the Planner own vs the Router?
3. How are RAG capabilities exposed as tools without duplicating retrieval/generation?
4. What is `PlannerDecision` and why was it added in Sprint 12?
5. How would you add a new tool without changing the graph topology?

---

## LangSmith Questions

1. Traces vs metrics — what question does each answer?
2. What is `NoOpTracer` for?
3. How do reports link to a LangSmith run?
4. Why must `ToolValidator` never parse LangSmith objects directly?
5. What would you show an interviewer in the LangSmith UI for `agent-tools-foundation-001`?

---

## Tool Validation Questions

1. What is a `ToolContract`?
2. Difference between `ToolExecution` and `ActualToolCall`?
3. How does `ToolTraceMapper` protect the validator from vendor lock-in?
4. What does the canonical demo assert about tool order?
5. How do quality gates consume tool validation failures?

---

## Quality Gate Questions

1. PASS vs WARNING vs FAIL — policy difference?
2. Where do thresholds live, and why not hard-code them?
3. How does regression detection differ from a single-run gate?
4. What artifacts prove a gate decision in an audit?
5. Give an example recommendation when faithfulness fails.

---

## Tradeoff Questions

1. Why Chroma for v1 instead of Pinecone/Weaviate?
2. Why rule-based planner first?
3. Why freeze architecture at v1.0?
4. Cost vs quality: when do you disable DeepEval in CI?
5. Why quality gates instead of only looking at `overall_score`?

---

## Scenario Questions

1. Faithfulness drops 15% after a prompt change — walk the debug path.
2. Tools run out of order but the answer looks correct — do you ship?
3. LangSmith is down — does evaluation still work?
4. Recruiter runs `make demo` with no keys — what must still succeed?
5. Product wants multi-agent next quarter — what do you refuse to rush into v1?
