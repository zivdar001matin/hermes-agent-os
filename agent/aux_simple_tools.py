"""Continue simple read-only tool gathering on the auxiliary model.

After the main model emits a batch of cheap, read-only tools (read_file,
search_files, MCP file lists, web_search, …) the next "what else should I
look up?" turn is often another simple tool, not a reasoning step. Sending
that follow-up to the main model wastes the slow/expensive loop.

When ``auxiliary.simple_tools.enabled`` is true and the aux endpoint is
distinct from the main chat model, this module runs a bounded tool-calling
loop on the auxiliary model with a compact prompt (the 4B aux window is
much smaller than the main 27B context). Extra assistant+tool messages are
appended to the live transcript so the main model sees the gather trail on
its next turn. Failures are swallowed — the main loop always continues.
"""

from __future__ import annotations

import json
import logging
import uuid
from types import SimpleNamespace
from typing import AbstractSet, Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

TASK_KEY = "simple_tools"

# Read-only tools the aux model may keep calling. ``write_file`` / ``patch`` /
# ``terminal`` stay on the main model. MCP names are ``mcp__<server>__<tool>``;
# matching uses the trailing segment so ``mcp__hermes_os_files__read_file``
# counts as ``read_file``.
DEFAULT_SIMPLE_TOOLS = frozenset({
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "session_search",
    "skills_list",
    "skill_view",
    "list_files",
    "list_available_files",
    "list_buckets",
    "find_files",
})

_MAX_ITERATIONS_DEFAULT = 4
_MAX_ITERATIONS_CAP = 12
_USER_REQUEST_CHARS = 2000
_TOOL_RESULT_CHARS = 2500
_AUX_NOTE_CHARS = 1500
_AUX_OUTPUT_TOKENS = 1024

_SYSTEM_PROMPT = (
    "You are a fast auxiliary gatherer for a larger coding agent. "
    "You may only call the provided read-only tools to collect more context. "
    "Call tools when another lookup would clearly help. "
    "When you have enough, reply with a short note of what you found (no tools). "
    "Never write, patch, delete, run shell, or address the user as if you were "
    "the main assistant."
)


def _task_config() -> Dict[str, Any]:
    from agent.auxiliary_client import _get_auxiliary_task_config

    cfg = _get_auxiliary_task_config(TASK_KEY)
    return cfg if isinstance(cfg, dict) else {}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _norm_url(url: str) -> str:
    text = (url or "").strip().rstrip("/").lower()
    if text.endswith("/v1"):
        text = text[:-3].rstrip("/")
    return text


def _mcp_suffix(name: str) -> str:
    if name.startswith("mcp__") and "__" in name[5:]:
        return name.rsplit("__", 1)[-1]
    return name


def is_simple_tool(name: str, allowlist: AbstractSet[str]) -> bool:
    """Return True when *name* is in *allowlist*, including MCP-prefixed forms."""
    if not name:
        return False
    if name in allowlist:
        return True
    suffix = _mcp_suffix(name)
    return bool(suffix) and suffix in allowlist


def resolve_allowlist(cfg: Optional[Dict[str, Any]] = None) -> frozenset[str]:
    cfg = cfg if cfg is not None else _task_config()
    raw = cfg.get("tools")
    if isinstance(raw, (list, tuple)) and raw:
        names = {str(item).strip() for item in raw if str(item).strip()}
        if names:
            return frozenset(names)
    return DEFAULT_SIMPLE_TOOLS


def _aux_endpoint_is_distinct(agent: Any, cfg: Dict[str, Any]) -> bool:
    """Skip when aux would hit the same server/model as the main agent."""
    provider = str(cfg.get("provider") or "auto").strip().lower()
    model = str(cfg.get("model") or "").strip()
    aux_url = _norm_url(str(cfg.get("base_url") or ""))
    agent_url = _norm_url(str(getattr(agent, "base_url", "") or ""))
    agent_model = str(getattr(agent, "model", "") or "").strip()
    agent_provider = str(getattr(agent, "provider", "") or "").strip().lower()

    if provider in {"auto", "main"} and not model and not aux_url:
        return False
    if aux_url and agent_url and aux_url != agent_url:
        return True
    if model and agent_model and model != agent_model:
        return True
    if provider not in {"auto", "main", "", agent_provider}:
        return True
    return False


def _tool_names(tool_calls: Sequence[Any]) -> List[str]:
    names: List[str] = []
    for tc in tool_calls or []:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn is not None else None
        if not name and isinstance(tc, dict):
            fn_dict = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn_dict.get("name") or tc.get("name")
        if name:
            names.append(str(name))
    return names


def should_run_aux_simple_followup(
    agent: Any,
    tool_calls: Sequence[Any],
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    if getattr(agent, "_in_aux_simple_tools", False):
        return False
    if getattr(agent, "_interrupt_requested", False):
        return False
    cfg = cfg if cfg is not None else _task_config()
    if not _is_truthy(cfg.get("enabled")):
        return False
    if not _aux_endpoint_is_distinct(agent, cfg):
        return False
    names = _tool_names(tool_calls)
    if not names:
        return False
    allowlist = resolve_allowlist(cfg)
    return all(is_simple_tool(name, allowlist) for name in names)


def _truncate(text: Any, limit: int) -> str:
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 1)].rstrip() + "…"


def _message_text(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    try:
        from agent.message_content import flatten_message_text

        return flatten_message_text(msg.get("content")) or ""
    except Exception:
        content = msg.get("content")
        return content if isinstance(content, str) else ""


def _last_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _message_text(msg).strip()
            if text:
                return _truncate(text, _USER_REQUEST_CHARS)
    return ""


def _recent_tool_digest(
    messages: Sequence[Dict[str, Any]],
    tool_calls: Sequence[Any],
    *,
    limit: int = 8,
) -> str:
    wanted_ids = {
        str(getattr(tc, "id", "") or "").strip()
        for tc in (tool_calls or [])
        if str(getattr(tc, "id", "") or "").strip()
    }
    blocks: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or "").strip()
        if wanted_ids and call_id and call_id not in wanted_ids:
            continue
        name = str(msg.get("name") or "tool")
        body = _truncate(_message_text(msg), _TOOL_RESULT_CHARS)
        blocks.append(f"### {name}\n{body}")
    if not blocks:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                name = str(msg.get("name") or "tool")
                body = _truncate(_message_text(msg), _TOOL_RESULT_CHARS)
                blocks.append(f"### {name}\n{body}")
    return "\n\n".join(blocks[-limit:])


def _simple_tool_schemas(agent: Any, allowlist: AbstractSet[str]) -> List[Dict[str, Any]]:
    valid = getattr(agent, "valid_tool_names", None) or set()
    schemas: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for tool in getattr(agent, "tools", None) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        if not name or name in seen:
            continue
        if valid and name not in valid:
            continue
        if not is_simple_tool(name, allowlist):
            continue
        seen.add(name)
        schemas.append(tool)
    return schemas


def _max_iterations(cfg: Dict[str, Any]) -> int:
    raw = cfg.get("max_iterations", _MAX_ITERATIONS_DEFAULT)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _MAX_ITERATIONS_DEFAULT
    return max(1, min(_MAX_ITERATIONS_CAP, value))


def _parse_tool_calls(response: Any) -> List[Any]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    if message is None:
        return []
    raw = getattr(message, "tool_calls", None) or []
    if isinstance(message, dict):
        raw = message.get("tool_calls") or []
    return list(raw) if raw else []


def _response_text(response: Any) -> str:
    from agent.auxiliary_client import extract_content_or_reasoning

    try:
        return (extract_content_or_reasoning(response) or "").strip()
    except Exception:
        return ""


def _adapt_tool_calls(raw_calls: Iterable[Any], allowlist: AbstractSet[str]) -> List[Any]:
    adapted: List[Any] = []
    for idx, tc in enumerate(raw_calls):
        if isinstance(tc, dict):
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name") or tc.get("name") or "")
            args = fn.get("arguments", tc.get("arguments", "{}"))
            tc_id = tc.get("id")
        else:
            fn = getattr(tc, "function", None)
            name = str(getattr(fn, "name", "") or "")
            args = getattr(fn, "arguments", "{}") if fn is not None else "{}"
            tc_id = getattr(tc, "id", None)
        if not name or not is_simple_tool(name, allowlist):
            continue
        if not isinstance(args, str):
            try:
                args = json.dumps(args)
            except (TypeError, ValueError):
                args = "{}"
        call_id = str(tc_id).strip() if isinstance(tc_id, str) else ""
        if not call_id:
            call_id = f"aux_simple_{idx}_{uuid.uuid4().hex[:10]}"
        adapted.append(
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=args),
            )
        )
    return adapted


def _assistant_dict(adapted: Sequence[Any], content: str = "") -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content or "",
        "finish_reason": "tool_calls",
        "_aux_simple_tools": True,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in adapted
        ],
    }


def _compact_copy_tool_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": "tool",
        "name": msg.get("name") or "",
        "tool_call_id": msg.get("tool_call_id") or "",
        "content": _truncate(_message_text(msg), _TOOL_RESULT_CHARS),
    }


def _fold_aux_note(messages: List[Dict[str, Any]], note: str) -> None:
    text = _truncate((note or "").strip(), _AUX_NOTE_CHARS)
    if not text:
        return
    block = f"\n\n--- auxiliary follow-up ---\n{text}"
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = content + block
            return
        if isinstance(content, dict):
            summary = content.get("text_summary")
            if isinstance(summary, str):
                content["text_summary"] = summary + block
                return
        return


def _build_seed_messages(
    messages: Sequence[Dict[str, Any]],
    tool_calls: Sequence[Any],
) -> List[Dict[str, Any]]:
    user_text = _last_user_text(messages) or "(no user message)"
    digest = _recent_tool_digest(messages, tool_calls)
    user = (
        "The main agent is gathering information for this user request:\n\n"
        f"{user_text}\n\n"
        "It already ran these tools:\n\n"
        f"{digest or '(no tool results yet)'}\n\n"
        "If more of the provided read-only tools would help, call them. "
        "Otherwise reply briefly with what you found and stop."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def maybe_run_aux_simple_tool_followup(
    agent: Any,
    messages: List[Dict[str, Any]],
    assistant_message: Any,
    effective_task_id: str,
    api_call_count: int = 0,
    conversation_history: Optional[list] = None,
) -> int:
    """Run zero or more aux simple-tool iterations. Returns iterations executed."""
    tool_calls = getattr(assistant_message, "tool_calls", None) or []
    cfg = _task_config()
    if not should_run_aux_simple_followup(agent, tool_calls, cfg=cfg):
        return 0

    allowlist = resolve_allowlist(cfg)
    schemas = _simple_tool_schemas(agent, allowlist)
    if not schemas:
        return 0

    from agent.auxiliary_client import call_llm

    compact = _build_seed_messages(messages, tool_calls)
    iterations = 0
    agent._in_aux_simple_tools = True
    try:
        _vprint = getattr(agent, "_vprint", None)
        prefix = getattr(agent, "log_prefix", "") or ""
        if callable(_vprint) and not getattr(agent, "quiet_mode", False):
            _vprint(f"{prefix}⚡ Auxiliary model continuing with simple tools...")

        for _ in range(_max_iterations(cfg)):
            if getattr(agent, "_interrupt_requested", False):
                break
            try:
                response = call_llm(
                    task=TASK_KEY,
                    messages=compact,
                    tools=schemas,
                    temperature=0,
                    max_tokens=_AUX_OUTPUT_TOKENS,
                )
            except Exception as exc:
                logger.info("auxiliary simple-tools call failed: %s", exc)
                break

            raw_calls = _parse_tool_calls(response)
            adapted = _adapt_tool_calls(raw_calls, allowlist)
            if not adapted:
                _fold_aux_note(messages, _response_text(response))
                break

            assistant_dict = _assistant_dict(adapted)
            messages.append(assistant_dict)
            compact.append({
                "role": "assistant",
                "content": assistant_dict.get("content") or "",
                "tool_calls": assistant_dict.get("tool_calls"),
            })
            before = len(messages)
            exec_msg = SimpleNamespace(tool_calls=adapted, content="")
            try:
                agent._execute_tool_calls(
                    exec_msg, messages, effective_task_id, api_call_count,
                )
            except Exception:
                logger.warning(
                    "auxiliary simple-tools execution failed", exc_info=True,
                )
                break
            for msg in messages[before:]:
                if isinstance(msg, dict) and msg.get("role") == "tool":
                    compact.append(_compact_copy_tool_message(msg))
            iterations += 1
            if getattr(agent, "_tool_guardrail_halt_decision", None) is not None:
                break
            try:
                agent._flush_messages_to_session_db(
                    messages, conversation_history or [],
                )
            except Exception:
                pass
        return iterations
    finally:
        agent._in_aux_simple_tools = False
