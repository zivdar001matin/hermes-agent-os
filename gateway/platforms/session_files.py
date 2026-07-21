"""Internal session-files HTTP API for Hermes OS chat scratch.

Django pushes attachments into ``{HERMES_SESSION_FILES_ROOT}/{session_id}/``
and pulls agent-created files out after each prompt turn. These routes are
authenticated with the same Bearer key as the api-server (``API_SERVER_KEY``).

Paths under the session root are strictly jailable: no ``..``, no absolute
paths, no symlink escape.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional


try:
    from aiohttp import web
except ImportError:  # pragma: no cover
    web = None  # type: ignore[assignment]


DEFAULT_SESSION_FILES_ROOT = "/opt/data/temp/sessions"
DEFAULT_MAX_FILE_BYTES = 10_000_000  # align with api_server MAX_REQUEST_BYTES
DEFAULT_MAX_SESSION_BYTES = 200_000_000

# Session ids must be filesystem-safe and non-traversing.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")


def session_files_root() -> Path:
    raw = os.environ.get("HERMES_SESSION_FILES_ROOT") or DEFAULT_SESSION_FILES_ROOT
    return Path(raw).expanduser().resolve()


def max_file_bytes() -> int:
    raw = os.environ.get("HERMES_SESSION_MAX_FILE_BYTES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_FILE_BYTES


def max_session_bytes() -> int:
    raw = os.environ.get("HERMES_SESSION_MAX_TOTAL_BYTES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_SESSION_BYTES


def validate_session_id(session_id: str) -> Optional[str]:
    """Return an error message if *session_id* is malformed, else None."""
    if not session_id or not _SESSION_ID_RE.match(session_id):
        return "Invalid session ID"
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        return "Invalid session ID"
    return None


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_session_dir(session_id: str, *, root: Optional[Path] = None) -> Path:
    err = validate_session_id(session_id)
    if err:
        raise ValueError(err)
    base = (root or session_files_root()).resolve()
    session_dir = (base / session_id).resolve()
    if session_dir.parent != base:
        raise ValueError("Invalid session ID")
    return session_dir



def resolve_session_path(
    session_id: str,
    rel_path: str,
    *,
    root: Optional[Path] = None,
) -> Path:
    """Resolve *rel_path* under the session dir; raise ValueError if unsafe."""
    if not rel_path or not rel_path.strip():
        raise ValueError("Invalid path")
    if rel_path.startswith(("/", "\\")) or Path(rel_path).is_absolute():
        raise ValueError("Absolute paths are not allowed")

    parts = Path(rel_path).parts
    if any(p in ("..", "") for p in parts):
        raise ValueError("Path traversal is not allowed")

    session_dir = resolve_session_dir(session_id, root=root)
    candidate = session_dir.joinpath(*parts)
    try:
        candidate.relative_to(session_dir)
    except ValueError as exc:
        raise ValueError("Path escapes session directory") from exc

    # Every existing ancestor (including symlinks) must realpath inside session_dir.
    check = candidate
    while True:
        if check.exists() or check.is_symlink():
            real = Path(os.path.realpath(str(check)))
            if real != session_dir and not _is_under(real, session_dir):
                raise ValueError("Symlink escape is not allowed")
        if check == session_dir or check.parent == check:
            break
        check = check.parent

    if candidate.exists() or candidate.is_symlink():
        final = Path(os.path.realpath(str(candidate)))
        if final != session_dir and not _is_under(final, session_dir):
            raise ValueError("Symlink escape is not allowed")
        return final
    return candidate


def _dir_total_size(session_dir: Path) -> int:
    total = 0
    if not session_dir.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(session_dir, followlinks=False):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                if fp.is_symlink():
                    continue
                total += fp.stat().st_size
            except OSError:
                continue
    return total


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_session_files(session_id: str, *, root: Optional[Path] = None) -> list[dict[str, Any]]:
    session_dir = resolve_session_dir(session_id, root=root)
    if not session_dir.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for dirpath, _dirnames, filenames in os.walk(session_dir, followlinks=False):
        for name in filenames:
            fp = Path(dirpath) / name
            if fp.is_symlink():
                continue
            try:
                st = fp.stat()
            except OSError:
                continue
            rel = fp.relative_to(session_dir).as_posix()
            entries.append(
                {
                    "path": rel,
                    "size": st.st_size,
                    "sha256": _sha256_file(fp),
                    "mtime": st.st_mtime,
                }
            )
    entries.sort(key=lambda e: e["path"])
    return entries


def write_session_file(
    session_id: str,
    rel_path: str,
    data: bytes,
    *,
    root: Optional[Path] = None,
    max_file: Optional[int] = None,
    max_total: Optional[int] = None,
) -> Path:
    max_file = max_file if max_file is not None else max_file_bytes()
    max_total = max_total if max_total is not None else max_session_bytes()
    if len(data) > max_file:
        raise OverflowError("File exceeds max size")

    session_dir = resolve_session_dir(session_id, root=root)
    session_dir.mkdir(parents=True, exist_ok=True)

    dest = resolve_session_path(session_id, rel_path, root=root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to replace a symlink (could point outside).
    if dest.is_symlink() or (dest.exists() and not dest.is_file()):
        raise ValueError("Refusing to write through non-regular file")

    existing = dest.stat().st_size if dest.is_file() else 0
    current_total = _dir_total_size(session_dir)
    projected = current_total - existing + len(data)
    if projected > max_total:
        raise OverflowError("Session directory exceeds max total size")

    # Write via temp file in the same directory, then replace.
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


def read_session_file(
    session_id: str,
    rel_path: str,
    *,
    root: Optional[Path] = None,
) -> bytes:
    path = resolve_session_path(session_id, rel_path, root=root)
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise FileNotFoundError(rel_path)
    real = Path(os.path.realpath(path))
    session_dir = resolve_session_dir(session_id, root=root)
    if not _is_under(real, session_dir):
        raise ValueError("Symlink escape is not allowed")
    return real.read_bytes()


def delete_session_dir(session_id: str, *, root: Optional[Path] = None) -> None:
    session_dir = resolve_session_dir(session_id, root=root)
    if session_dir.exists():
        shutil.rmtree(session_dir)


def register_session_file_routes(
    app: "web.Application",
    check_auth: Callable[["web.Request"], Optional["web.Response"]],
    *,
    root: Optional[Path] = None,
) -> None:
    """Attach ``/internal/session-files/...`` routes to *app*."""
    if web is None:  # pragma: no cover
        raise RuntimeError("aiohttp is required for session-files routes")

    def _root() -> Path:
        return root if root is not None else session_files_root()

    async def handle_list(request: "web.Request") -> "web.Response":
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        session_id = request.match_info["session_id"]
        err = validate_session_id(session_id)
        if err:
            return web.json_response({"error": err}, status=400)
        try:
            files = list_session_files(session_id, root=_root())
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(files)

    async def handle_delete(request: "web.Request") -> "web.Response":
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        session_id = request.match_info["session_id"]
        err = validate_session_id(session_id)
        if err:
            return web.json_response({"error": err}, status=400)
        try:
            delete_session_dir(session_id, root=_root())
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.Response(status=204)

    async def handle_put(request: "web.Request") -> "web.Response":
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        session_id = request.match_info["session_id"]
        rel_path = request.match_info["path"]
        err = validate_session_id(session_id)
        if err:
            return web.json_response({"error": err}, status=400)
        try:
            data = await request.read()
        except Exception:
            return web.json_response({"error": "Failed to read body"}, status=400)
        try:
            write_session_file(session_id, rel_path, data, root=_root())
        except OverflowError as exc:
            return web.json_response({"error": str(exc)}, status=413)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.Response(status=204)

    async def handle_get_file(request: "web.Request") -> "web.Response":
        auth_err = check_auth(request)
        if auth_err is not None:
            return auth_err
        session_id = request.match_info["session_id"]
        rel_path = request.match_info["path"]
        err = validate_session_id(session_id)
        if err:
            return web.json_response({"error": err}, status=400)
        try:
            data = read_session_file(session_id, rel_path, root=_root())
        except FileNotFoundError:
            return web.json_response({"error": "Not found"}, status=404)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.Response(body=data, content_type="application/octet-stream")

    app.router.add_get("/internal/session-files/{session_id}", handle_list)
    app.router.add_delete("/internal/session-files/{session_id}", handle_delete)
    app.router.add_put(
        "/internal/session-files/{session_id}/{path:.+}", handle_put
    )
    app.router.add_get(
        "/internal/session-files/{session_id}/{path:.+}", handle_get_file
    )
