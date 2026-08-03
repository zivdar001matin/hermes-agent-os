"""Tests for MEDIA: basename → session-scratch absolute path rewrite."""

from __future__ import annotations

from pathlib import Path

from gateway.platforms.base import (
    resolve_session_scratch_file,
    rewrite_session_media_basenames,
)


def test_rewrite_with_session_id(tmp_path: Path):
    s1 = tmp_path / "sess-1"
    s2 = tmp_path / "sess-2"
    s1.mkdir()
    s2.mkdir()
    (s1 / "a.docx").write_bytes(b"1")
    (s2 / "a.docx").write_bytes(b"2")

    out = rewrite_session_media_basenames(
        "MEDIA:a.docx",
        session_id="sess-1",
        root=tmp_path,
    )
    assert out == f"MEDIA:{(s1 / 'a.docx').resolve()}"


def test_rewrite_basename_when_unique(tmp_path: Path):
    session = tmp_path / "api_abc"
    session.mkdir()
    target = session / "report.docx"
    target.write_bytes(b"x")

    out = rewrite_session_media_basenames(
        "Here you go\nMEDIA:report.docx\nThanks",
        root=tmp_path,
    )
    assert f"MEDIA:{target.resolve()}" in out
    assert "MEDIA:report.docx\n" not in out


def test_no_rewrite_when_ambiguous(tmp_path: Path):
    for name in ("s1", "s2"):
        d = tmp_path / name
        d.mkdir()
        (d / "dup.docx").write_bytes(b"x")

    text = "MEDIA:dup.docx"
    assert rewrite_session_media_basenames(text, root=tmp_path) == text


def test_no_rewrite_absolute_paths(tmp_path: Path):
    text = "MEDIA:/opt/data/temp/sessions/s/a.docx"
    assert rewrite_session_media_basenames(text, root=tmp_path) == text


def test_no_rewrite_missing_file(tmp_path: Path):
    (tmp_path / "s").mkdir()
    text = "MEDIA:missing.docx"
    assert rewrite_session_media_basenames(text, root=tmp_path) == text


def test_resolve_rejects_traversal(tmp_path: Path):
    assert resolve_session_scratch_file("../etc/passwd", root=tmp_path) is None
    assert (
        resolve_session_scratch_file(
            "a.docx", session_id="../x", root=tmp_path
        )
        is None
    )
