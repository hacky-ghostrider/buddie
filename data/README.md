# Sample documents (local only)

Place small sample PDFs here for manual experimentation.

Committed demo seed text lives in ``data/sample/`` (used by
``make seed`` / API auto-seed for the Streamlit RAG chat).

Structured employee HR data (profiles, leave, payroll, attendance) lives in
``data/employees/employees.json`` and is created by
``make seed-employees`` / ``scripts/seed_employee_data.py``. That store is
**not** the RAG vector database.

## Guidelines

- Do **not** commit large PDFs or proprietary documents to git.
- Prefer tiny fixtures under `tests/fixtures/` for automated tests.
- Filenames should be descriptive, e.g. `sample_policy_1page.pdf`.

## Quick local check

```bash
# Seed Chroma for live RAG / Agent chat (creates collection rag_documents)
uv run python scripts/seed_vectorstore.py
# or: make seed

# After placing a PDF in this folder:
uv run python -c "from app.services import DocumentIngestionService; print(len(DocumentIngestionService().load('data/your_file.pdf')))"
```

Evaluation and Allure reports are written under [`data/reports/`](reports/README.md)
(latest Allure: `reports/allure/latest/`, history: `reports/allure/history/`).

Binary contents of `data/` (except this README / `.gitkeep` / `sample/` / `reports/README.md`
and Allure placeholders) are gitignored.
