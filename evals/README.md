# Buddie Evaluation Layer

Offline-first AI evaluation for the Buddie HR assistant: annotated golden cases, runtime evidence collection, DeepEval metrics, and deterministic safety/robustness checks.

## Layout

```text
evals/
  golden_dataset/          # Annotated goldens (EXPECTED fields)
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
    safety.py              # PII leakage, unauthorized access, injection resistance
    robustness.py          # Adversarial refusal, unwanted tool/RAG activation
    semantic_similarity.py # Token Jaccard fallback (offline / rate-limit)
    tool_workflow.py       # Tool ordering, success rate, multi-tool workflow
    runtime_health.py      # Graceful degradation on tool/API failures
    failure_diagnostics.py # FailureKind taxonomy + structured debug logging
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
| `test_tier` | CI tier tags (`smoke`, `sanity`, `regression`) |

**Baseline:** 36 cases. Do not change existing meaning or expected behavior without revising the baseline intentionally.

| Category | Count |
|----------|------:|
| leave_hr | 8 |
| holidays | 4 |
| benefits_policies | 3 |
| rag_knowledge | 4 |
| multi_tool | 5 |
| negative_unknown | 4 |
| adversarial_security | 8 |

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

**Safety / robustness:** pii_leakage, unauthorized_data_access, prompt_injection_resistance, adversarial_refusal, unwanted_tool_call, unwanted_rag_activation, semantic_similarity (token Jaccard), tool_ordering_correctness, tool_call_success_rate, multi_tool_workflow_success, runtime_graceful_degradation, runtime_empty_response.

Each case also emits `failure_diagnostics` (typed `FailureKind` + `debug_hint`) and `tool_failure_messages` for debugging.

Null/NA when a metric does not apply.

## Run

```bash
# Live evaluation CLI (Gemini judge when GOOGLE_API_KEY or GEMINI_API_KEY is set)
python -m evals.runners.run_buddie_deepeval
python -m evals.runners.run_buddie_deepeval --output data/reports/buddie_eval_suite.json
python -m evals.runners.run_buddie_deepeval --tier smoke
python -m evals.runners.run_buddie_deepeval --tier sanity
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

Results live under **`data/reports/allure/latest/`**; past runs go in
**`data/reports/allure/history/<timestamp>/`**. See
[`data/reports/README.md`](../data/reports/README.md).

```bash
# Generate Allure results (36 golden cases as individual tests)
pytest tests/test_buddie_eval_allure.py --alluredir=data/reports/allure/latest

# Optional: render a prior live JSON report into Allure instead of the
# deterministic in-test measure_fn path
# Windows PowerShell:
#   $env:BUDDIE_EVAL_REPORT="data/reports/buddie_eval_suite.json"
pytest tests/test_buddie_eval_allure.py --alluredir=data/reports/allure/latest

# View HTML report (requires Allure CLI)
allure serve data/reports/allure/latest
# or: allure generate data/reports/allure/latest -o data/reports/allure/latest/html --clean

# Keep a snapshot: make allure-archive
```

JSON evaluation output remains at `data/reports/buddie_eval_suite.json`.

Deterministic pytest:

- `tests/test_buddie_golden_dataset.py` (`smoke`)
- `tests/test_buddie_eval_tiers.py` (`smoke` / `sanity` tier suites)
- `tests/test_buddie_eval_runtime_collector.py`
- `tests/test_buddie_deepeval_suite.py`
- `tests/test_buddie_eval_sprint19.py`
- `tests/test_buddie_eval_allure.py`
- `tests/test_buddie_eval_sprint20.py`
- `tests/test_buddie_rate_limit.py`
- `tests/test_buddie_gemini_judge.py`

## Relation to `datasets/`

`datasets/golden_dataset.json` remains the automation schema for the core evaluation platform. Buddie goldens under `evals/` are the HR assistant evaluation artifact.
