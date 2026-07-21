"""Tests for Hermes OS session-scoped native file tool jail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.session_fs_guard import (
    SessionFsJailError,
    active_session_root,
    jail_path,
    reset_session_id,
    set_session_id,
)


@pytest.fixture
def session_env(tmp_path, monkeypatch):
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setenv("HERMES_SESSION_FILES_ROOT", str(root))
    return root


def _with_session(sid: str):
    token = set_session_id(sid)
    return token


def test_write_lands_under_session_dir(session_env, monkeypatch):
    import tools.file_tools as ft

    token = _with_session("chat-1")
    try:
        result = json.loads(
            ft.write_file_tool("out.txt", "hello from session", task_id="default")
        )
        assert not result.get("error"), result
        dest = session_env / "chat-1" / "out.txt"
        assert dest.is_file()
        assert dest.read_text() == "hello from session"
        assert "chat-1" in result.get("resolved_path", "")
    finally:
        reset_session_id(token)


def test_read_relative_path_uses_session_scratch(session_env, monkeypatch):
    """Regression: jail resolved under /tmp/sessions, but shell read used /opt/data cwd."""
    import tools.file_tools as ft

    staged = session_env / "chat-1" / "ATTACHED_DOCUMENTS.md"
    staged.parent.mkdir(parents=True)
    staged.write_text("# resumes\nAlice\n")

    token = _with_session("chat-1")
    try:
        result = json.loads(
            ft.read_file_tool("ATTACHED_DOCUMENTS.md", task_id="default")
        )
        assert not result.get("error"), result
        assert "Alice" in result.get("content", "")
        assert "chat-1" in result.get("resolved_path", "")
    finally:
        reset_session_id(token)


def test_search_relative_path_uses_session_scratch(session_env, monkeypatch):
    import tools.file_tools as ft

    staged = session_env / "chat-1" / "note.txt"
    staged.parent.mkdir(parents=True)
    staged.write_text("findme-token-xyz\n")

    token = _with_session("chat-1")
    try:
        result = json.loads(
            ft.search_tool(
                pattern="findme-token-xyz",
                target="content",
                path=".",
                task_id="default",
            )
        )
        assert not result.get("error"), result
        blob = json.dumps(result)
        assert "findme-token-xyz" in blob
        assert "state.db" not in blob
    finally:
        reset_session_id(token)


def test_read_opt_data_denied(session_env, tmp_path):
    import tools.file_tools as ft

    secret = tmp_path / "opt" / "data" / "config.yaml"
    secret.parent.mkdir(parents=True)
    secret.write_text("SECRET=1")

    token = _with_session("chat-1")
    try:
        result = json.loads(
            ft.read_file_tool(str(secret), task_id="default")
        )
        assert "error" in result
        assert "Access denied" in result["error"] or "outside" in result["error"].lower()
        assert "SECRET=1" not in json.dumps(result)
    finally:
        reset_session_id(token)


def test_read_other_session_denied(session_env):
    import tools.file_tools as ft

    other = session_env / "other-sess" / "private.txt"
    other.parent.mkdir(parents=True)
    other.write_text("other-session-secret")

    token = _with_session("chat-1")
    try:
        # Absolute path into another session
        result = json.loads(ft.read_file_tool(str(other), task_id="default"))
        assert "error" in result
        assert "other-session-secret" not in json.dumps(result)

        # Relative traversal
        result2 = json.loads(
            ft.read_file_tool("../other-sess/private.txt", task_id="default")
        )
        assert "error" in result2
        assert "other-session-secret" not in json.dumps(result2)
    finally:
        reset_session_id(token)


def test_no_session_id_denied_in_os_mode(session_env):
    import tools.file_tools as ft

    result = json.loads(ft._handle_write_file({"path": "x.txt", "content": "no"}, task_id="t"))
    assert "error" in result
    assert "session" in result["error"].lower()


def test_legacy_mode_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_FILES_ROOT", raising=False)
    token = set_session_id(None)
    try:
        assert active_session_root() is None
    finally:
        reset_session_id(token)


def test_jail_path_helper(session_env):
    token = _with_session("s1")
    try:
        p = jail_path("a/b.txt", expand_tilde=lambda x: x)
        assert p == (session_env / "s1" / "a" / "b.txt").resolve()
        with pytest.raises(SessionFsJailError):
            jail_path("../../etc/passwd", expand_tilde=lambda x: x)
        with pytest.raises(SessionFsJailError):
            jail_path("/etc/passwd", expand_tilde=lambda x: x)
    finally:
        reset_session_id(token)


def test_handler_passes_session_id(session_env):
    import tools.file_tools as ft

    out = ft._handle_write_file(
        {"path": "via-handler.txt", "content": "ok"},
        task_id="default",
        session_id="chat-handler",
    )
    data = json.loads(out)
    assert not data.get("error"), data
    assert (session_env / "chat-handler" / "via-handler.txt").read_text() == "ok"


def test_widget_py_allowed_when_scratch_inside_safe_root(tmp_path, monkeypatch):
    """Scratch nested under HERMES_WRITE_SAFE_ROOT (/opt/data/temp/sessions) is writable.

    Mirrors the container layout: HERMES_HOME=HERMES_WRITE_SAFE_ROOT=/opt/data and
    session scratch lives under /opt/data/temp/sessions/{session_id}/.
    """
    import tools.file_tools as ft

    safe_root = tmp_path / "opt" / "data"
    scratch_root = safe_root / "temp" / "sessions"
    scratch_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(safe_root))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    monkeypatch.setenv("HERMES_SESSION_FILES_ROOT", str(scratch_root))

    token = _with_session("api_1784626892_3e471d93")
    try:
        result = json.loads(
            ft._handle_write_file(
                {"path": "widget.py", "content": "class Widget: pass\n"},
                task_id="default",
                session_id="api_1784626892_3e471d93",
            )
        )
        assert not result.get("error"), result
        dest = scratch_root / "api_1784626892_3e471d93" / "widget.py"
        assert dest.is_file()
        assert "class Widget" in dest.read_text()
    finally:
        reset_session_id(token)


def test_opt_data_root_still_blocked_in_session_mode(tmp_path, monkeypatch):
    """A bare /opt/data write (outside the session scratch subtree) stays jailed."""
    import tools.file_tools as ft

    safe_root = tmp_path / "opt" / "data"
    scratch_root = safe_root / "temp" / "sessions"
    scratch_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(safe_root))
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    monkeypatch.setenv("HERMES_SESSION_FILES_ROOT", str(scratch_root))

    token = _with_session("chat-1")
    try:
        result = json.loads(
            ft._handle_write_file(
                {"path": str(safe_root / "widget.py"), "content": "nope"},
                task_id="default",
                session_id="chat-1",
            )
        )
        assert "error" in result
        assert "outside the session directory" in result["error"]
        assert not (safe_root / "widget.py").exists()
    finally:
        reset_session_id(token)
