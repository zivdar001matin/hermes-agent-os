"""Session-scoped filesystem jail for native file tools (Hermes OS).

When ``HERMES_SESSION_FILES_ROOT`` is set (profile containers), native
``read_file`` / ``write_file`` / ``patch`` / ``search_files`` are restricted
to ``{HERMES_SESSION_FILES_ROOT}/{session_id}/``. Team/shared MinIO paths
remain via MCP ``hermes_os_files``.

Upstream Hermes (env unset) keeps legacy path resolution unchanged.
"""

from __future__ import annotations

import contextvars
import os
import re
from pathlib import Path
from typing import Optional

_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_os_session_id", default=None
)

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


class SessionFsJailError(Exception):
    """Raised when a native file path escapes the session scratch directory."""


def session_files_root_env() -> Optional[str]:
    raw = os.environ.get("HERMES_SESSION_FILES_ROOT")
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip()


def hermes_os_session_mode() -> bool:
    """True when running in a Hermes OS profile container (scratch root configured)."""
    return session_files_root_env() is not None


def set_session_id(session_id: Optional[str]) -> contextvars.Token:
    return _SESSION_ID.set(session_id or None)


def reset_session_id(token: contextvars.Token) -> None:
    _SESSION_ID.reset(token)


def get_session_id() -> Optional[str]:
    return _SESSION_ID.get()


def _validate_session_id(session_id: str) -> None:
    if not session_id or not _SESSION_ID_RE.match(session_id):
        raise SessionFsJailError(
            "Native file tools require a valid chat session_id. "
            "Use hermes_os_files MCP tools for team/shared MinIO paths."
        )


def active_session_root() -> Optional[Path]:
    """Return the session scratch directory, or None for legacy (non-OS) mode.

    Raises SessionFsJailError when Hermes OS mode is on but session_id is missing
    or malformed.
    """
    root_env = session_files_root_env()
    if root_env is None:
        return None
    sid = get_session_id()
    if not sid:
        raise SessionFsJailError(
            "Native file tools require a chat session. "
            "Use hermes_os_files MCP tools for team/shared MinIO paths, "
            "or send the prompt via a Hermes chat session."
        )
    _validate_session_id(sid)
    base = Path(root_env).expanduser().resolve()
    session_dir = (base / sid).resolve()
    if session_dir.parent != base:
        raise SessionFsJailError("Invalid session_id for native file tools.")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def jail_path(filepath: str, *, expand_tilde) -> Path:
    """Resolve *filepath* under the active session root; raise on escape.

    *expand_tilde* is the caller's tilde expander (file_tools._expand_tilde).
    """
    session_root = active_session_root()
    if session_root is None:
        raise RuntimeError("jail_path called outside Hermes OS session mode")

    if not filepath or not str(filepath).strip():
        raise SessionFsJailError("Path is required.")

    raw = expand_tilde(str(filepath).strip())
    p = Path(raw)

    # Reject traversal segments before resolve.
    parts = p.parts
    if any(part == ".." for part in parts):
        raise SessionFsJailError(
            f"Access denied: path {filepath!r} escapes the session directory. "
            "Native file tools are limited to this chat session's scratch files. "
            "Use hermes_os_files MCP tools for team/shared MinIO paths."
        )

    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (session_root / p).resolve()

    try:
        resolved.relative_to(session_root)
    except ValueError as exc:
        raise SessionFsJailError(
            f"Access denied: path {filepath!r} is outside the session directory "
            f"({session_root}). Native file tools cannot read /opt/data or other "
            "sessions. Use hermes_os_files MCP tools for team/shared MinIO paths."
        ) from exc

    # Symlink escape: if the path exists, realpath must stay inside.
    if resolved.exists() or resolved.is_symlink():
        real = Path(os.path.realpath(str(resolved)))
        try:
            real.relative_to(session_root)
        except ValueError as exc:
            raise SessionFsJailError(
                f"Access denied: path {filepath!r} resolves outside the session directory."
            ) from exc
        return real
    return resolved
