#!/usr/bin/env python3
"""Scan Hermes session history and report actual tool invocation counts.

Sessions live in ``$HERMES_HOME/state.db`` (SQLite), the same store
``session_search`` and the conversation loop persist through
:class:`hermes_state.SessionDB`.

By default scans every profile container (``~/.hermes`` plus
``~/.hermes/profiles/*``). For each session started within the lookback
window (default 90 days), counts how many times each tool was invoked,
how many distinct sessions used it, and when it was last called. MCP tools
are labeled with their server name (``mcp/<server>/<tool>``) so they are
distinguishable from native tools.

Usage:
    python scripts/tool_usage.py
    python scripts/tool_usage.py --profile coder --days 30
    python scripts/tool_usage.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SECONDS_PER_DAY = 86400


def _parse_early_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", "-p", default=None)
    return parser.parse_known_args(argv)[0]


def _apply_profile(profile_name: str) -> None:
    from hermes_cli.profiles import resolve_profile_env

    os.environ["HERMES_HOME"] = resolve_profile_env(profile_name)


def format_tool_display_name(stored_name: str) -> str:
    """Label a stored tool name; MCP tools include server disambiguation."""
    from tools.mcp_tool import MCP_TOOL_NAME_PREFIX

    if not stored_name:
        return stored_name
    if stored_name.startswith(MCP_TOOL_NAME_PREFIX):
        rest = stored_name[len(MCP_TOOL_NAME_PREFIX) :]
        if "__" in rest:
            server, tool = rest.split("__", 1)
            return f"mcp/{server}/{tool}"
        return f"mcp/{rest}"
    if stored_name.startswith("mcp_"):
        return f"mcp/{stored_name[4:]}"
    return stored_name


@dataclass
class ToolUsageStats:
    stored_name: str
    call_count: int = 0
    sessions: Set[str] = field(default_factory=set)
    last_used: float = 0.0

    @property
    def display_name(self) -> str:
        return format_tool_display_name(self.stored_name)

    @property
    def session_count(self) -> int:
        return len(self.sessions)


def _extract_tool_calls(calls_raw: Any) -> List[str]:
    if not calls_raw:
        return []
    calls = calls_raw
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except json.JSONDecodeError:
            return []
    if not isinstance(calls, list):
        return []
    names: List[str] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        func = call.get("function")
        if isinstance(func, dict):
            name = func.get("name")
        else:
            name = call.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _session_key(profile_name: str, session_id: str) -> str:
    return f"{profile_name}:{session_id}"


def _record(
    stats: Dict[str, ToolUsageStats],
    name: str,
    profile_name: str,
    session_id: str,
    ts: float,
) -> None:
    row = stats.get(name)
    if row is None:
        row = ToolUsageStats(stored_name=name)
        stats[name] = row
    row.call_count += 1
    row.sessions.add(_session_key(profile_name, session_id))
    if ts > row.last_used:
        row.last_used = ts


def _merge_tool_stats(
    primary: Dict[str, ToolUsageStats],
    secondary: Dict[str, ToolUsageStats],
) -> Dict[str, ToolUsageStats]:
    """Merge two counters without double-counting overlapping invocations.

    When both sources have data for a tool, keep whichever source recorded
    more calls (same strategy as ``agent.insights.InsightsEngine``). Session
    sets and last-used timestamps are unioned / maxed so metadata stays
    accurate even when counts are reconciled.
    """
    merged: Dict[str, ToolUsageStats] = {}
    for name in set(primary) | set(secondary):
        a = primary.get(name)
        b = secondary.get(name)
        if a and b:
            chosen = a if a.call_count >= b.call_count else b
            other = b if chosen is a else a
            merged[name] = ToolUsageStats(
                stored_name=name,
                call_count=chosen.call_count,
                sessions=set(chosen.sessions) | set(other.sessions),
                last_used=max(chosen.last_used, other.last_used),
            )
        elif a:
            merged[name] = ToolUsageStats(
                stored_name=name,
                call_count=a.call_count,
                sessions=set(a.sessions),
                last_used=a.last_used,
            )
        elif b:
            merged[name] = ToolUsageStats(
                stored_name=name,
                call_count=b.call_count,
                sessions=set(b.sessions),
                last_used=b.last_used,
            )
    return merged


def _merge_profile_stats(
    primary: Dict[str, ToolUsageStats],
    secondary: Dict[str, ToolUsageStats],
) -> Dict[str, ToolUsageStats]:
    """Sum call counts across profiles; union session keys and last-used."""
    merged = {
        name: ToolUsageStats(
            stored_name=name,
            call_count=stat.call_count,
            sessions=set(stat.sessions),
            last_used=stat.last_used,
        )
        for name, stat in primary.items()
    }
    for name, stat in secondary.items():
        row = merged.get(name)
        if row is None:
            merged[name] = ToolUsageStats(
                stored_name=name,
                call_count=stat.call_count,
                sessions=set(stat.sessions),
                last_used=stat.last_used,
            )
        else:
            row.call_count += stat.call_count
            row.sessions |= stat.sessions
            row.last_used = max(row.last_used, stat.last_used)
    return merged


def _scan_tool_messages(
    conn,
    cutoff: float,
    profile_name: str,
) -> Dict[str, ToolUsageStats]:
    stats: Dict[str, ToolUsageStats] = {}
    cursor = conn.execute(
        """
        SELECT m.session_id, m.tool_name, m.timestamp
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE s.started_at >= ?
          AND m.role = 'tool'
          AND m.tool_name IS NOT NULL
          AND TRIM(m.tool_name) != ''
        """,
        (cutoff,),
    )
    for row in cursor.fetchall():
        name = row["tool_name"]
        ts = float(row["timestamp"] or 0.0)
        _record(stats, name, profile_name, row["session_id"], ts)
    return stats


def _scan_assistant_tool_calls(
    conn,
    cutoff: float,
    profile_name: str,
) -> Dict[str, ToolUsageStats]:
    stats: Dict[str, ToolUsageStats] = {}
    cursor = conn.execute(
        """
        SELECT m.session_id, m.tool_calls, m.timestamp
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE s.started_at >= ?
          AND m.role = 'assistant'
          AND m.tool_calls IS NOT NULL
          AND TRIM(m.tool_calls) != ''
        """,
        (cutoff,),
    )
    for row in cursor.fetchall():
        session_id = row["session_id"]
        ts = float(row["timestamp"] or 0.0)
        for name in _extract_tool_calls(row["tool_calls"]):
            _record(stats, name, profile_name, session_id, ts)
    return stats


def _session_count(conn, cutoff: float) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM sessions WHERE started_at >= ?",
        (cutoff,),
    ).fetchone()
    return int(row["n"] if row else 0)


def _stats_to_rows(stats: Dict[str, ToolUsageStats]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, stat in stats.items():
        if stat.call_count <= 0:
            continue
        rows.append(
            {
                "tool": stat.display_name,
                "stored_name": stat.stored_name,
                "calls": stat.call_count,
                "sessions": stat.session_count,
                "last_used": stat.last_used,
                "last_used_date": (
                    datetime.fromtimestamp(stat.last_used, tz=timezone.utc).strftime("%Y-%m-%d")
                    if stat.last_used
                    else None
                ),
            }
        )
    rows.sort(key=lambda item: (-item["calls"], item["tool"]))
    return rows


def collect_tool_usage(
    db_path: Path,
    *,
    days: int,
    profile_name: str,
) -> Dict[str, Any]:
    import sqlite3

    if not db_path.is_file():
        raise FileNotFoundError(f"Session database not found: {db_path}")

    cutoff = time.time() - (days * SECONDS_PER_DAY)
    empty = {
        "profile": profile_name,
        "days": days,
        "cutoff_unix": cutoff,
        "db_path": str(db_path),
        "hermes_home": str(db_path.parent),
        "sessions_scanned": 0,
        "tools": [],
        "total_calls": 0,
        "_stats": {},
    }
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("SELECT 1 FROM sessions LIMIT 1")
    except sqlite3.OperationalError:
        conn.close()
        return empty
    try:
        session_total = _session_count(conn, cutoff)
        from_tool_rows = _scan_tool_messages(conn, cutoff, profile_name)
        from_assistant = _scan_assistant_tool_calls(conn, cutoff, profile_name)

        if not from_tool_rows and from_assistant:
            merged = from_assistant
        elif from_tool_rows and from_assistant:
            merged = _merge_tool_stats(from_tool_rows, from_assistant)
        else:
            merged = from_tool_rows
    finally:
        conn.close()

    rows = _stats_to_rows(merged)
    return {
        "profile": profile_name,
        "days": days,
        "cutoff_unix": cutoff,
        "db_path": str(db_path),
        "hermes_home": str(db_path.parent),
        "sessions_scanned": session_total,
        "tools": rows,
        "total_calls": sum(item["calls"] for item in rows),
        "_stats": merged,
    }


def collect_all_profiles_tool_usage(*, days: int) -> Dict[str, Any]:
    import sqlite3

    from hermes_cli.profiles import list_profiles

    cutoff = time.time() - (days * SECONDS_PER_DAY)
    profiles_report: List[Dict[str, Any]] = []
    combined: Dict[str, ToolUsageStats] = {}
    profiles_scanned: List[str] = []
    profiles_missing_db: List[str] = []
    total_sessions = 0

    for pinfo in list_profiles():
        db_path = pinfo.path / "state.db"
        if not db_path.is_file():
            profiles_missing_db.append(pinfo.name)
            continue
        try:
            profile_data = collect_tool_usage(
                db_path,
                days=days,
                profile_name=pinfo.name,
            )
        except sqlite3.Error as exc:
            profiles_report.append(
                {
                    "profile": pinfo.name,
                    "hermes_home": str(pinfo.path),
                    "db_path": str(db_path),
                    "sessions_scanned": 0,
                    "total_calls": 0,
                    "tool_count": 0,
                    "error": str(exc),
                }
            )
            profiles_scanned.append(pinfo.name)
            continue

        profiles_scanned.append(pinfo.name)
        total_sessions += int(profile_data["sessions_scanned"])
        profiles_report.append(
            {
                "profile": pinfo.name,
                "hermes_home": profile_data["hermes_home"],
                "db_path": profile_data["db_path"],
                "sessions_scanned": profile_data["sessions_scanned"],
                "total_calls": profile_data["total_calls"],
                "tool_count": len(profile_data["tools"]),
            }
        )
        combined = _merge_profile_stats(combined, profile_data["_stats"])

    rows = _stats_to_rows(combined)
    return {
        "scope": "all_profiles",
        "profiles_scanned": profiles_scanned,
        "profiles_missing_db": profiles_missing_db,
        "profiles": profiles_report,
        "days": days,
        "cutoff_unix": cutoff,
        "sessions_scanned": total_sessions,
        "tools": rows,
        "total_calls": sum(item["calls"] for item in rows),
    }


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def render_table(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    if data.get("scope") == "all_profiles":
        profile_names = data.get("profiles_scanned") or []
        profile_label = ", ".join(profile_names) if profile_names else "(none)"
        lines.append(
            f"Tool usage (all profiles: {profile_label}, last {data['days']} days, "
            f"{int(data.get('sessions_scanned') or 0):,} sessions)"
        )
        missing = data.get("profiles_missing_db") or []
        if missing:
            lines.append(f"Profiles without state.db: {', '.join(missing)}")
    else:
        profile = data.get("profile") or "default"
        lines.append(
            f"Tool usage (profile={profile}, last {data['days']} days, "
            f"{int(data.get('sessions_scanned') or 0):,} sessions)"
        )
        if data.get("hermes_home"):
            lines.append(f"HERMES_HOME: {data['hermes_home']}")
        if data.get("db_path"):
            lines.append(f"Database: {data['db_path']}")
    lines.append("")

    tools = data.get("tools") or []
    if not tools:
        lines.append("No tool invocations found in the lookback window.")
        return "\n".join(lines)

    headers = ("Tool", "Calls", "Sessions", "Last used")
    widths = [
        max(len(headers[0]), max(len(row["tool"]) for row in tools)),
        max(len(headers[1]), max(len(_fmt_int(row["calls"])) for row in tools)),
        max(len(headers[2]), max(len(_fmt_int(row["sessions"])) for row in tools)),
        len(headers[3]),
    ]

    def _row(cells: Iterable[str]) -> str:
        tool, calls, sessions, last_used = cells
        return (
            f"  {tool:<{widths[0]}}  {calls:>{widths[1]}}  "
            f"{sessions:>{widths[2]}}  {last_used}"
        )

    lines.append(_row(headers))
    lines.append(
        f"  {'-' * widths[0]}  {'-' * widths[1]}  {'-' * widths[2]}  {'-' * widths[3]}"
    )
    for row in tools:
        lines.append(
            _row(
                (
                    row["tool"],
                    _fmt_int(row["calls"]),
                    _fmt_int(row["sessions"]),
                    row["last_used_date"] or "unknown",
                )
            )
        )

    lines.append("")
    lines.append(f"Total tool calls: {_fmt_int(int(data.get('total_calls') or 0))}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    early = _parse_early_args(argv or sys.argv[1:])
    if early.profile:
        _apply_profile(early.profile)

    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(project_env=os.path.join(_PROJECT_ROOT, ".env"))

    parser = argparse.ArgumentParser(
        description="Report tool invocation counts from Hermes session history.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=early.profile,
        help="Scan a single profile (default: all profiles)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Lookback window in days (default: 90)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    args = parser.parse_args(argv)

    if args.profile:
        if args.profile != (early.profile or ""):
            _apply_profile(args.profile)

    if args.days <= 0:
        print("--days must be a positive integer", file=sys.stderr)
        return 2

    import sqlite3

    try:
        if args.profile:
            from hermes_constants import get_hermes_home

            db_path = get_hermes_home() / "state.db"
            data = collect_tool_usage(db_path, days=args.days, profile_name=args.profile)
            data.pop("_stats", None)
        else:
            data = collect_all_profiles_tool_usage(days=args.days)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print(f"Could not read session database: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_table(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
