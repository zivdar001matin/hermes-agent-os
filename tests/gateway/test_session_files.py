"""Tests for /internal/session-files endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.session_files import (
    register_session_file_routes,
    resolve_session_path,
    write_session_file,
)


def _make_adapter(api_key: str = "sk-test") -> APIServerAdapter:
    config = PlatformConfig(enabled=True, extra={"key": api_key})
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter, root: Path) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    register_session_file_routes(app, adapter._check_auth, root=root)
    return app


AUTH = {"Authorization": "Bearer sk-test"}


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions"
    root.mkdir()
    return root


@pytest.mark.asyncio
async def test_round_trip(session_root):
    app = _create_app(_make_adapter(), session_root)
    async with TestClient(TestServer(app)) as client:
        sid = "sess-roundtrip"
        body = b"hello session files"
        put = await client.put(
            f"/internal/session-files/{sid}/notes/hello.txt",
            data=body,
            headers=AUTH,
        )
        assert put.status == 204

        listing = await client.get(
            f"/internal/session-files/{sid}", headers=AUTH
        )
        assert listing.status == 200
        files = await listing.json()
        assert len(files) == 1
        assert files[0]["path"] == "notes/hello.txt"
        assert files[0]["size"] == len(body)
        assert files[0]["sha256"]
        assert "mtime" in files[0]

        got = await client.get(
            f"/internal/session-files/{sid}/notes/hello.txt",
            headers=AUTH,
        )
        assert got.status == 200
        assert await got.read() == body

        deleted = await client.delete(
            f"/internal/session-files/{sid}", headers=AUTH
        )
        assert deleted.status == 204

        listing2 = await client.get(
            f"/internal/session-files/{sid}", headers=AUTH
        )
        assert listing2.status == 200
        assert await listing2.json() == []
        assert not (session_root / sid).exists()


@pytest.mark.asyncio
async def test_path_traversal_rejected_via_helpers(session_root, tmp_path):
    """HTTP clients collapse '../' in URLs; the jail is enforced in helpers."""
    sid = "sess-trav"
    for bad in ("../../etc/passwd", "/etc/passwd", "foo/../../../etc/passwd", ".."):
        with pytest.raises(ValueError):
            write_session_file(sid, bad, b"x", root=session_root)

    # Symlink-as-path-component already covered by HTTP test; ensure no leak.
    outside = tmp_path / "leaked.txt"
    assert not outside.exists()
    assert list(session_root.rglob("*")) == [session_root] or not any(
        p.is_file() for p in session_root.rglob("*")
    )


@pytest.mark.asyncio
async def test_symlink_escape_rejected(session_root, tmp_path):
    sid = "sess-sym"
    session_dir = session_root / sid
    session_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (session_dir / "escape").symlink_to(outside)

    app = _create_app(_make_adapter(), session_root)
    async with TestClient(TestServer(app)) as client:
        resp = await client.put(
            f"/internal/session-files/{sid}/escape",
            data=b"overwrite",
            headers=AUTH,
        )
        assert resp.status == 400
    assert outside.read_text() == "secret"


@pytest.mark.asyncio
async def test_sessions_are_isolated(session_root):
    app = _create_app(_make_adapter(), session_root)
    async with TestClient(TestServer(app)) as client:
        a, b = "sess-a", "sess-b"
        await client.put(
            f"/internal/session-files/{a}/only-a.txt",
            data=b"aaa",
            headers=AUTH,
        )
        await client.put(
            f"/internal/session-files/{b}/only-b.txt",
            data=b"bbb",
            headers=AUTH,
        )
        la = await (
            await client.get(f"/internal/session-files/{a}", headers=AUTH)
        ).json()
        lb = await (
            await client.get(f"/internal/session-files/{b}", headers=AUTH)
        ).json()
        assert [e["path"] for e in la] == ["only-a.txt"]
        assert [e["path"] for e in lb] == ["only-b.txt"]


@pytest.mark.asyncio
async def test_oversized_file_413(session_root, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_MAX_FILE_BYTES", "16")
    from gateway.platforms import session_files as sf

    assert sf.max_file_bytes() == 16

    app = _create_app(_make_adapter(), session_root)
    async with TestClient(TestServer(app)) as client:
        resp = await client.put(
            "/internal/session-files/sess-big/big.bin",
            data=b"x" * 32,
            headers=AUTH,
        )
        assert resp.status == 413


@pytest.mark.asyncio
async def test_auth_required(session_root):
    app = _create_app(_make_adapter(), session_root)
    async with TestClient(TestServer(app)) as client:
        resp = await client.put(
            "/internal/session-files/sess-auth/f.txt",
            data=b"x",
        )
        assert resp.status == 401

        resp = await client.put(
            "/internal/session-files/sess-auth/f.txt",
            data=b"x",
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status == 401


@pytest.mark.asyncio
async def test_malformed_session_id_400(session_root):
    app = _create_app(_make_adapter(), session_root)
    async with TestClient(TestServer(app)) as client:
        # Space is not allowed in session_id pattern for the list route.
        resp = await client.get(
            "/internal/session-files/bad id",
            headers=AUTH,
        )
        assert resp.status == 400

        resp = await client.put(
            "/internal/session-files/bad id/f.txt",
            data=b"x",
            headers=AUTH,
        )
        assert resp.status == 400


def test_resolve_rejects_absolute(session_root):
    with pytest.raises(ValueError):
        resolve_session_path("s1", "/etc/passwd", root=session_root)


def test_write_helper_auto_creates_dir(session_root):
    path = write_session_file("s1", "a/b.txt", b"data", root=session_root)
    assert path.read_bytes() == b"data"
    assert path.is_relative_to(session_root / "s1")


@pytest.mark.asyncio
async def test_put_handler_rejects_traversal_match_info(session_root):
    """Simulate a path the router would pass through without URL collapsing."""
    from unittest.mock import AsyncMock, MagicMock

    from gateway.platforms.session_files import register_session_file_routes

    adapter = _make_adapter()
    app = web.Application()
    register_session_file_routes(app, adapter._check_auth, root=session_root)

    # Find the PUT handler
    put_handler = None
    for route in app.router.routes():
        if route.method == "PUT":
            put_handler = route.handler
            break
    assert put_handler is not None

    request = MagicMock()
    request.headers = {"Authorization": "Bearer sk-test"}
    request.match_info = {
        "session_id": "sess-h",
        "path": "../../etc/passwd",
    }
    request.read = AsyncMock(return_value=b"x")

    resp = await put_handler(request)
    assert resp.status == 400
