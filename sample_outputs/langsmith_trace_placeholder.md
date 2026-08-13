# LangSmith Trace Screenshot Placeholder

Replace this file with a real screenshot after running the live demo:

```bash
# .env
ENABLE_LANGSMITH=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=rag-evaluation

python scripts/demo.py --live
```

## What the trace should show

```text
Planner
  ↓
search_docs
  ↓
summarize
  ↓
Final answer
```

Scenario id: `agent-tools-foundation-001`

Suggested filename when you add a real image:

`sample_outputs/langsmith_trace_agent-tools-foundation-001.png`

Then link it from the README Screenshots section.
