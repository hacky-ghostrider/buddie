# Local evaluation reports

Runtime artifacts from eval runs land here (gitignored except this file and
Allure folder placeholders). Committed examples for README browsing live under
[`sample_outputs/`](../../sample_outputs/).

## Where to look

| What | Path |
|------|------|
| **Latest Allure results** | [`allure/latest/`](allure/latest/) |
| **Past Allure runs** | [`allure/history/<timestamp>/`](allure/history/) |
| Buddie eval JSON | `buddie_eval_suite.json` |
| CI / dry-run eval | `ci_evaluation.{json,csv,html}` |

## Allure (Buddie golden cases)

**Windows (no `make` required):**

```powershell
uv run python scripts/run_allure.py
uv run python scripts/run_allure.py --serve
uv run python scripts/run_allure.py --html      # static HTML if serve fails
uv run python scripts/run_allure.py --archive
```

Generate results (36 cases as individual pytest tests; optional tier via `BUDDIE_EVAL_TIER`):

```bash
uv run pytest tests/test_buddie_eval_allure.py --alluredir=data/reports/allure/latest
```

View in the browser (requires [Allure CLI](https://allurereport.org/docs/install/)):

```bash
allure serve data/reports/allure/latest
```

Or build static HTML next to the results:

```bash
allure generate data/reports/allure/latest -o data/reports/allure/latest/html --clean
```

### Archive a run

After a run you want to keep, copy `latest/` into `history/`:

```bash
make allure-archive
```

Windows PowerShell:

```powershell
$ts = Get-Date -Format "yyyy-MM-dd_HHmm"
Copy-Item -Recurse data/reports/allure/latest "data/reports/allure/history/$ts"
```

Makefile shortcuts (Git Bash / WSL / Linux): `make allure`, `make allure-serve`, `make allure-archive`.
Windows without `make`: `uv run python scripts/run_allure.py` (see above).

See [`evals/README.md`](../../evals/README.md) for env vars and live JSON replay.
