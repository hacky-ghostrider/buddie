# RAG Evaluation Platform v1.0 — developer task runner
# Requires: Python 3.12+, uv (preferred) or pip, GNU Make
#
# Windows: use Git Bash, WSL, or `choco install make`.

PYTHON ?= python
UV ?= uv
PYTEST ?= $(UV) run pytest
RUFF ?= $(UV) run ruff

.PHONY: help setup test test-smoke test-sanity test-regression test-buddie lint benchmark evaluate demo docker ci clean quality-gate ui api seed seed-employees allure allure-serve allure-archive

help:
	@echo "RAG Evaluation Platform — available targets"
	@echo "  make setup       Install package + dev dependencies"
	@echo "  make test        Run PyTest (mocked vendors, no live APIs)"
	@echo "  make test-smoke  Run smoke-tier PR checks (fast)"
	@echo "  make test-sanity Run smoke + sanity-tier PR checks"
	@echo "  make test-regression Run full Buddie regression (Allure suite)"
	@echo "  make test-buddie Run all Buddie eval pytest modules"
	@echo "  make lint        Run Ruff linter"
	@echo "  make evaluate    Dry-run evaluation automation"
	@echo "  make benchmark   Aggregate latest evaluation reports"
	@echo "  make quality-gate Validate quality gates (CI gate)"
	@echo "  make demo        One-command offline demo"
	@echo "  make seed        Seed vectorstore + employee dataset"
	@echo "  make seed-employees  Seed deterministic employee JSON store"
	@echo "  make api         Start FastAPI (uvicorn --reload)"
	@echo "  make ui          Start Streamlit chatbot UI"
	@echo "  make docker      Build Docker image"
	@echo "  make ci          Lint + test + evaluate + quality-gate"
	@echo "  make allure      Run Buddie eval Allure tests → data/reports/allure/latest"
	@echo "  make allure-serve Open Allure UI for latest results (requires Allure CLI)"
	@echo "  make allure-archive Copy latest Allure run to data/reports/allure/history/"
	@echo "  make clean       Remove caches and local artifacts"

setup:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) sync --extra dev; \
		$(UV) pip install -e ".[dev]"; \
	else \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi
	@test -f .env || cp .env.example .env
	@mkdir -p data/chroma data/reports data/reports/allure/latest data/reports/allure/history data/benchmarks data/quality_reports data/employees
	@echo "Setup complete. Copy secrets into .env if you need live OpenAI/LangSmith."

test:
	$(PYTEST) -q --tb=short

test-smoke:
	$(PYTEST) -m smoke -q --tb=short

test-sanity:
	$(PYTEST) -m "smoke or sanity" -q --tb=short

test-regression:
	$(PYTEST) -m regression -q --tb=short

test-buddie:
	$(PYTEST) tests/test_buddie_*.py -q --tb=short

lint:
	@if command -v $(UV) >/dev/null 2>&1; then \
		$(UV) run ruff check app scripts tests || $(PYTHON) -m ruff check app scripts tests; \
	else \
		$(PYTHON) -m ruff check app scripts tests; \
	fi

evaluate:
	$(UV) run $(PYTHON) scripts/run_evaluation.py --batch --dry-run --run-name ci_evaluation \
		|| $(PYTHON) scripts/run_evaluation.py --batch --dry-run --run-name ci_evaluation

benchmark:
	@REPORT=data/reports/ci_evaluation.json; \
	if [ ! -f "$$REPORT" ]; then \
		$(MAKE) evaluate; \
	fi; \
	$(UV) run $(PYTHON) scripts/run_benchmark.py --reports data/reports/ci_evaluation.json --run-name ci_benchmark \
		|| $(PYTHON) scripts/run_benchmark.py --reports data/reports/ci_evaluation.json --run-name ci_benchmark

quality-gate:
	$(UV) run $(PYTHON) scripts/validate_quality_gates.py \
		|| $(PYTHON) scripts/validate_quality_gates.py

demo:
	$(UV) run $(PYTHON) scripts/demo.py \
		|| $(PYTHON) scripts/demo.py

seed: seed-employees
	$(UV) run $(PYTHON) scripts/seed_vectorstore.py \
		|| $(PYTHON) scripts/seed_vectorstore.py

seed-employees:
	$(UV) run $(PYTHON) scripts/seed_employee_data.py \
		|| $(PYTHON) scripts/seed_employee_data.py

api:
	$(UV) run uvicorn app.main:app --reload --port 8000 \
		|| uvicorn app.main:app --reload --port 8000

ui:
	$(UV) run streamlit run frontend/app.py --server.port 8501 \
		|| streamlit run frontend/app.py --server.port 8501

docker:
	docker compose build api

ci: lint test evaluate quality-gate
	@echo "CI checks passed."

allure:
	@mkdir -p data/reports/allure/latest
	$(PYTEST) tests/test_buddie_eval_allure.py --alluredir=data/reports/allure/latest

allure-serve:
	allure serve data/reports/allure/latest

allure-archive:
	@TS=$$(date +%Y-%m-%d_%H%M); \
	DEST=data/reports/allure/history/$$TS; \
	if [ ! -d data/reports/allure/latest ] || [ -z "$$(ls -A data/reports/allure/latest 2>/dev/null)" ]; then \
		echo "Nothing to archive: data/reports/allure/latest is empty."; exit 1; \
	fi; \
	mkdir -p "$$DEST"; \
	cp -r data/reports/allure/latest/. "$$DEST/"; \
	echo "Archived to $$DEST"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	rm -rf data/reports/* data/benchmarks/* data/quality_reports/* 2>/dev/null || true
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
