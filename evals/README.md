# Buddie Evaluation Layer

Sprint 17–19 foundation for offline AI evaluation (DeepEval + deterministic checks).

## Layout

```text
evals/
  golden_dataset/          # Source-of-truth annotations / goldens (Sprint 17)
    buddie_golden_cases.json
    models.py
    loader.py
  runners/                 # Runtime collection + evaluation suite
    deepeval_case.py       # DeepEvalCompatibleCase (EXPECTED vs ACTUAL fields)
    runtime_collector.py   # AgentService → runtime evidence only
    deepeval_suite.py      # Full suite: DeepEval + retrieval + agent checks
    allure_reporting.py    # Allure attachments for per-case results
    run_buddie_deepeval.py # CLI entrypoint
  metrics/
    config.py              # Thresholds + Gemini judge resolution
    standard.py            # Faithfulness, Relevancy, Hallucination, Contextual*
    g_eval.py              # FINAL_RESPONSE_CORRECTNESS
    retrieval.py           # Precision@K / Recall@K / Hit@K / MRR
    agent_checks.py        # Tool / HITL / argument / task completion
    annotations.py         # Annotation coverage report
    results.py             # Unified CaseEvaluationResult + suite report
```

## Golden dataset (annotation / ground truth)

| Field | Role |
|-------|------|
| `user_query` | Input |
| `expected_answer` | Annotated reference answer |
| `expected_context` | Annotated relevant evidence (retrieval ground truth) |
| `expected_tool` / `expected_tools` | Annotated tool expectation |
| `expected_behavior` | Behavior contract |
| `evaluation_notes` | Human notes |
| `category` | Bucket |

**Baseline:** 28 cases. Do not change existing meaning or expected behavior.

| Category | Count |
|----------|------:|
| leave_hr | 8 |
| holidays | 4 |
| benefits_policies | 3 |
| rag_knowledge | 4 |
| multi_tool | 5 |
| negative_unknown | 4 |

## EXPECTED vs ACTUAL

| Annotated (golden) | Actual (runtime) |
|--------------------|------------------|
| `expected_answer` | `actual_output` / final answer |
| `expected_context` | `retrieval_context` (RAG chunks + tool evidence) |
| `expected_tool(s)` | `selected_tools` / `tool_execution_order` |

`retrieval_context` is **never** filled from `expected_context`. Empty runtime evidence → `[]`.

## Metrics

**DeepEval (LLM judge):** Faithfulness, Answer Relevancy, Hallucination (higher-is-better), Contextual Precision/Recall/Relevancy, G-Eval Final Response Correctness.

**Deterministic retrieval:** Precision@1/3/5, Recall@1/3/5, Hit@1/3/5, MRR — using annotated `expected_context` vs runtime `retrieval_context`.

**Deterministic agent:** tool_correctness, argument_correctness, hitl_correctness, task_completion.

Null/NA when a metric does not apply.

## Run

```bash
# Live evaluation CLI (Gemini judge when GOOGLE_API_KEY or GEMINI_API_KEY is set)
python -m evals.runners.run_buddie_deepeval
python -m evals.runners.run_buddie_deepeval --output data/reports/buddie_eval_suite.json
```

Optional env for the DeepEval LLM judge:

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` | **Canonical** Gemini API key for DeepEval `GeminiModel` |
| `GEMINI_API_KEY` | Legacy alias (read once if `GOOGLE_API_KEY` unset; not kept in env) |
| `GEMINI_MODEL_NAME` | Override judge model (default: `gemini-3.1-flash-lite`) |
| `EVAL_GEMINI_MAX_RETRIES` | Additional 429 retries after first failure (default: `3`) |
| `EVAL_GEMINI_RETRY_DELAY_SEC` | Fallback sleep when 429 omits retry delay (default: `5.0`) |

Report statuses: `passed`, `failed` (threshold/agent), `rate_limited` (Gemini quota), `error` (infrastructure/other LLM failure).

The CLI prints whether Gemini is configured **without** exposing the key.

### Allure report (reporting UI only)

```bash
# Generate Allure results (28 golden cases as individual tests)
pytest tests/test_buddie_eval_allure.py --alluredir=data/reports/allure-results

# Optional: render a prior live JSON report into Allure instead of the
# deterministic in-test measure_fn path
# Windows PowerShell:
#   $env:BUDDIE_EVAL_REPORT="data/reports/buddie_eval_suite.json"
pytest tests/test_buddie_eval_allure.py --alluredir=data/reports/allure-results

# View HTML report (requires Allure CLI)
allure serve data/reports/allure-results
# or: allure generate data/reports/allure-results -o data/reports/allure-report --clean
```

JSON evaluation output remains at `data/reports/buddie_eval_suite.json`.

Deterministic pytest:

- `tests/test_buddie_golden_dataset.py`
- `tests/test_buddie_eval_runtime_collector.py`
- `tests/test_buddie_deepeval_suite.py`
- `tests/test_buddie_eval_sprint19.py`
- `tests/test_buddie_eval_allure.py`
- `tests/test_buddie_rate_limit.py`
- `tests/test_buddie_gemini_judge.py`

## Relation to `datasets/`

`datasets/golden_dataset.json` remains the Sprint 9–12 automation schema. Buddie goldens under `evals/` are a separate evaluation-layer artifact.
