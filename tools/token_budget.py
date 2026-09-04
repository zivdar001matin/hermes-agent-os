"""Estimated-token budgets for tool results.

Uses ``estimate_tokens_rough`` (script-aware) so caps treat English and
Arabic-script text fairly — a raw character budget under-caps Persian and
over-punishes Latin.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_READ_FILE_MAX_TOKENS = 4000
DEFAULT_SESSION_SEARCH_MAX_TOKENS = 4000
DEFAULT_SESSION_SEARCH_SNIPPET_MAX_TOKENS = 800

_read_file_max_tokens_cached: Optional[int] = None
_session_search_max_tokens_cached: Optional[int] = None


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv <= 0:
        return default
    return iv


def _tools_section() -> dict:
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        section = cfg.get("tools") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def get_read_file_max_tokens() -> int:
    """Return ``tools.read_file.max_tokens`` (default 4000)."""
    global _read_file_max_tokens_cached
    if _read_file_max_tokens_cached is not None:
        return _read_file_max_tokens_cached
    section = _tools_section().get("read_file")
    val = section.get("max_tokens") if isinstance(section, dict) else None
    _read_file_max_tokens_cached = _coerce_positive_int(val, DEFAULT_READ_FILE_MAX_TOKENS)
    return _read_file_max_tokens_cached


def get_session_search_max_tokens() -> int:
    """Return ``tools.session_search.max_tokens`` (default 4000)."""
    global _session_search_max_tokens_cached
    if _session_search_max_tokens_cached is not None:
        return _session_search_max_tokens_cached
    section = _tools_section().get("session_search")
    val = section.get("max_tokens") if isinstance(section, dict) else None
    _session_search_max_tokens_cached = _coerce_positive_int(
        val, DEFAULT_SESSION_SEARCH_MAX_TOKENS
    )
    return _session_search_max_tokens_cached


def reset_token_budget_caches() -> None:
    """Clear cached tool token budgets (tests / config reload)."""
    global _read_file_max_tokens_cached, _session_search_max_tokens_cached
    _read_file_max_tokens_cached = None
    _session_search_max_tokens_cached = None


def estimate_tokens(text: str) -> int:
    from agent.model_metadata import estimate_tokens_rough

    return estimate_tokens_rough(text or "")


def truncate_text_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Trim ``text`` so ``estimate_tokens`` fits ``max_tokens``.

    Returns ``(text, truncated)``. Prefer a code-point boundary; when
    truncated, append an ellipsis.
    """
    if not text or max_tokens <= 0:
        return text or "", False
    if estimate_tokens(text) <= max_tokens:
        return text, False

    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if estimate_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1

    # Prefer breaking on whitespace when we have room.
    if best and not best[-1].isspace():
        ws = max(best.rfind(" "), best.rfind("\n"), best.rfind("\t"))
        if ws > len(best) // 2 and estimate_tokens(best[:ws]) <= max_tokens:
            best = best[:ws]

    best = best.rstrip()
    if best:
        best = best + "…"
    return best, True


def truncate_lines_to_tokens(
    content: str,
    max_tokens: int,
    *,
    start_line: int = 1,
    total_lines: Any = None,
) -> tuple[str, int, bool, Optional[int]]:
    """Trim line-numbered ``read_file`` content to a token budget.

    Truncates on a complete-line boundary. When truncated, appends::

        [truncated at line N of M — call read_file with offset=N to continue]

    where N is the first unread line (continuation offset).

    Returns ``(text, lines_kept, truncated, next_offset)``.
    """
    if not content:
        return "", 0, False, None
    if estimate_tokens(content) <= max_tokens:
        return content, content.count("\n") + 1, False, None

    lines = content.split("\n")
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens("\n".join(lines[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1

    if lo == 0:
        # First line alone exceeds the budget — keep it so the read is never
        # empty; the continuation offset still advances past it.
        lo = 1

    kept = lines[:lo]
    next_offset = start_line + lo
    shown = "\n".join(kept)
    # Leave room for the continuation marker when possible.
    marker_m = total_lines if total_lines is not None else "?"
    marker = (
        f"[truncated at line {next_offset} of {marker_m} — "
        f"call read_file with offset={next_offset} to continue]"
    )
    # If body + marker still fits, append; otherwise append anyway (marker is
    # small and more valuable than a few trailing tokens of content).
    if estimate_tokens(shown + "\n" + marker) > max_tokens and lo > 1:
        # Drop lines until body+marker fits, keeping at least one line.
        while lo > 1:
            lo -= 1
            kept = lines[:lo]
            shown = "\n".join(kept)
            next_offset = start_line + lo
            marker = (
                f"[truncated at line {next_offset} of {marker_m} — "
                f"call read_file with offset={next_offset} to continue]"
            )
            if estimate_tokens(shown + "\n" + marker) <= max_tokens:
                break

    return shown + "\n" + marker, lo, True, next_offset
