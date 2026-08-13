"""Unit tests for chat session helpers (incl. auto-scroll)."""

from __future__ import annotations

from frontend.components import chat as chat_mod


class _FakeSession(dict):
    """Minimal stand-in for ``st.session_state`` (attr + mapping access)."""

    def __getattr__(self, key: str):  # noqa: ANN001
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:  # noqa: ANN001
        self[key] = value

    def get(self, key, default=None):  # noqa: ANN001
        return dict.get(self, key, default)

    def pop(self, key, default=None):  # noqa: ANN001
        return dict.pop(self, key, default)


def test_request_chat_scroll_sets_force_and_nonce(monkeypatch) -> None:
    state = _FakeSession(_buddie_scroll_nonce=0, _buddie_scroll_force=False)
    monkeypatch.setattr(chat_mod.st, "session_state", state)

    chat_mod.request_chat_scroll(force=True)

    assert state["_buddie_scroll_force"] is True
    assert state["_buddie_scroll_nonce"] == 1


def test_render_chat_autoscroll_targets_streamlit_scroller(monkeypatch) -> None:
    state = _FakeSession(_buddie_scroll_nonce=3, _buddie_scroll_force=True)
    monkeypatch.setattr(chat_mod.st, "session_state", state)

    captured: dict[str, object] = {}

    def _fake_html(html: str, *, height: int = 0, width: int = 0) -> None:
        captured["html"] = html
        captured["height"] = height
        captured["width"] = width

    monkeypatch.setattr(chat_mod.components, "html", _fake_html)

    chat_mod.render_chat_autoscroll()

    html = str(captured["html"])
    assert 'data-testid="stAppScrollToBottomContainer"' in html
    assert "force = true" in html
    assert "scrollTo" in html
    assert "window.scrollTo" not in html
    assert captured["height"] == 0
    # Force is consumed so later soft rerenders do not yank the reader.
    assert state.get("_buddie_scroll_force") is None or state.get(
        "_buddie_scroll_force"
    ) is False


def test_render_chat_autoscroll_noop_without_nonce(monkeypatch) -> None:
    state = _FakeSession(_buddie_scroll_nonce=0, _buddie_scroll_force=True)
    monkeypatch.setattr(chat_mod.st, "session_state", state)

    called = {"n": 0}

    def _fake_html(*_args, **_kwargs) -> None:
        called["n"] += 1

    monkeypatch.setattr(chat_mod.components, "html", _fake_html)
    chat_mod.render_chat_autoscroll()
    assert called["n"] == 0
