# Screenshots placeholder

Add interview / UI screenshots here (not committed by default):

- `01-rag-chat.png` — RAG mode answer + Debug retrieved docs
- `02-agent-flow.png` — Agent Flow panel (Planner → tools → answer)
- `03-evaluation.png` — DeepEval metrics + quality gate after Run Demo
- `04-langsmith.png` — LangSmith Trace panel with clickable URL

Capture locally after:

```bash
uv run uvicorn app.main:app --reload
uv run streamlit run frontend/app.py
```
