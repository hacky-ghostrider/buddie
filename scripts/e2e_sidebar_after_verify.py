"""One-shot E2E: verify employee workspace sidebar after verification.

Uses Streamlit AppTest (same runtime as the UI) to drive:
  home → unverified (no workspace) → set verified session → assert sidebar options.

Writes a JSON report under sample_outputs/ui_polish/ and exits non-zero on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "app.py"
OUT = ROOT / "sample_outputs" / "ui_polish" / "e2e_sidebar_after_verify.json"

EXPECTED_QUICK = [
    "My Leave",
    "Leave History",
    "Upcoming Holidays",
    "Pending Actions",
]
EXPECTED_OPTIONAL = ["Employee Profile"]


def _session_get(at: AppTest, key: str, default=None):
    try:
        return at.session_state[key]
    except Exception:  # noqa: BLE001 — AppTest SessionState has no .get()
        return default


def _button_labels(at: AppTest) -> list[str]:
    labels: list[str] = []
    for btn in at.sidebar.button:
        label = (btn.label or "").strip()
        if label:
            labels.append(label)
    return labels


def _markdown_blobs(at: AppTest) -> str:
    parts: list[str] = []
    for block in at.sidebar.markdown:
        val = getattr(block, "value", None)
        if val is None:
            val = getattr(block, "body", None)
        if val is not None:
            parts.append(str(val))
    for block in at.sidebar.caption:
        val = getattr(block, "value", None)
        if val is None:
            val = getattr(block, "body", None)
        if val is not None:
            parts.append(str(val))
    return "\n".join(parts)


def main() -> int:
    report: dict = {
        "app": str(APP),
        "steps": [],
        "passed": False,
    }

    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    report["steps"].append(
        {
            "name": "load_home_unverified",
            "exception": None if not at.exception else str(at.exception),
            "sidebar_buttons": _button_labels(at),
            "verified_employee_id": _session_get(at, "verified_employee_id"),
        }
    )
    if at.exception:
        report["error"] = f"Home load raised: {at.exception}"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    # Before verification the employee workspace must not render sidebar actions.
    before_labels = set(_button_labels(at))
    leaked = [label for label in EXPECTED_QUICK if label in before_labels]
    report["steps"].append(
        {
            "name": "assert_no_workspace_before_verify",
            "leaked_quick_access": leaked,
            "ok": leaked == [],
        }
    )
    if leaked:
        report["error"] = f"Workspace leaked before verify: {leaked}"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    # Simulate successful verification (same session key the UI sets).
    at.session_state["verified_employee_id"] = "E-1101"
    at.session_state["awaiting_verification"] = False
    at.run()

    if at.exception:
        report["error"] = f"Post-verify rerun raised: {at.exception}"
        report["steps"].append({"name": "post_verify_rerun", "exception": str(at.exception)})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    labels = _button_labels(at)
    blob = _markdown_blobs(at)
    missing_quick = [label for label in EXPECTED_QUICK if label not in labels]
    missing_optional = [label for label in EXPECTED_OPTIONAL if label not in labels]
    has_brand = "Buddie" in blob or "Employee workspace" in blob
    has_eid = "E-1101" in blob
    has_verified = "Verified" in blob
    has_sections = ("QUICK ACCESS" in blob) and ("OPTIONAL" in blob or "Employee" in blob)

    step = {
        "name": "assert_workspace_after_verify",
        "sidebar_buttons": labels,
        "has_brand_or_caption": has_brand,
        "has_employee_id": has_eid,
        "has_verified_marker": has_verified,
        "has_section_markers": has_sections,
        "missing_quick_access": missing_quick,
        "missing_optional": missing_optional,
        "markdown_excerpt": blob[:500],
    }
    report["steps"].append(step)

    ok = (
        not missing_quick
        and not missing_optional
        and has_eid
        and has_verified
        and has_brand
    )
    report["passed"] = ok
    if not ok:
        report["error"] = (
            "Sidebar missing expected employee workspace content after verify"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
