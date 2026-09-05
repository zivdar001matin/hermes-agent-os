"""X-Hermes-Request correlation header for llama.cpp telemetry join."""

from __future__ import annotations

import os
from typing import Any, Optional

HERMES_REQUEST_HEADER = "X-Hermes-Request"


def telemetry_profile_name() -> str:
    env = (os.environ.get("HERMES_PROFILE_NAME") or "").strip()
    if env:
        return env
    try:
        from hermes_constants import get_hermes_home

        name = get_hermes_home().name
        if name and name not in {"data", "opt"}:
            return name
    except Exception:
        pass
    return "default"


def build_hermes_request_id(
    session_db: Any,
    session_id: Optional[str],
    *,
    call_index: Optional[int] = None,
) -> tuple[str, int]:
    """Return ``({profile}:{session}:{call_index}, call_index)``."""
    sid = session_id or "none"
    if call_index is None:
        try:
            call_index = int(session_db.next_api_call_index(sid)) if session_db else 1
        except Exception:
            call_index = 1
    return f"{telemetry_profile_name()}:{sid}:{call_index}", call_index


def attach_hermes_request_header(api_kwargs: dict, request_id: str) -> dict:
    extra = dict(api_kwargs.get("extra_headers") or {})
    extra[HERMES_REQUEST_HEADER] = request_id
    api_kwargs["extra_headers"] = extra
    return api_kwargs


def stamp_agent_request_header(agent: Any, api_kwargs: dict) -> str:
    """Allocate a request id on *agent* (once per API call) and attach the header."""
    existing = getattr(agent, "_telemetry_request_id", None)
    if existing:
        attach_hermes_request_header(api_kwargs, existing)
        return existing
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None) or ""
    request_id, call_index = build_hermes_request_id(session_db, session_id)
    agent._telemetry_request_id = request_id
    agent._telemetry_call_index = call_index
    attach_hermes_request_header(api_kwargs, request_id)
    return request_id
