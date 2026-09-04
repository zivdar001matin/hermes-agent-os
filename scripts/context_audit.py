#!/usr/bin/env python3
"""Audit turn-1 context payload for a Hermes profile without a live session.

Builds the same system prompt, tool schemas, and memory blocks that ship on the
first provider request, then prints a token breakdown by category (matching
``agent.context_breakdown``) plus a per-tool ranking under ``tool_definitions``.

Usage:
    python scripts/context_audit.py
    python scripts/context_audit.py --profile coder
    python scripts/context_audit.py --json
    python scripts/context_audit.py --real-tokens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _parse_early_args(argv: List[str]) -> argparse.Namespace:
    """Parse only --profile before HERMES_HOME-dependent imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", "-p", default="default")
    return parser.parse_known_args(argv)[0]


def _apply_profile(profile_name: str) -> None:
    from hermes_cli.profiles import resolve_profile_env

    os.environ["HERMES_HOME"] = resolve_profile_env(profile_name)


def _build_inspection_agent(platform: str = "cli") -> Any:
    """Construct an offline AIAgent whose turn-1 payload matches a real session."""
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools

    cfg = load_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    model = model_cfg.get("default") or model_cfg.get("model") or ""

    enabled_toolsets = sorted(_get_platform_tools(cfg, platform))
    agent_cfg = cfg.get("agent") or {}
    disabled_toolsets = agent_cfg.get("disabled_toolsets") or None

    return AIAgent(
        model=model,
        api_key="inspect-only",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        save_trajectories=False,
        platform=platform,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
    )


def _rough_tokens(text: str) -> int:
    from agent.model_metadata import estimate_tokens_rough

    return estimate_tokens_rough(text)


def _rough_json_tokens(value: Any) -> int:
    if not value:
        return 0
    return _rough_tokens(json.dumps(value, ensure_ascii=False))


def _per_tool_rough_tokens(tool: dict) -> int:
    return _rough_json_tokens(tool)


def _get_tiktoken_encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _real_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def _real_json_tokens(value: Any, enc) -> int:
    if not value:
        return 0
    return _real_tokens(json.dumps(value, ensure_ascii=False), enc)


def _per_tool_real_tokens(tool: dict, enc) -> int:
    return _real_json_tokens(tool, enc)


def _effective_input_budget(context_length: int, max_tokens: Optional[int]) -> int:
    reserved = max_tokens or 0
    budget = context_length - reserved
    return budget if budget > 0 else context_length


def compute_context_audit(
    agent: Any,
    *,
    messages: Optional[List[dict]] = None,
    use_real_tokens: bool = False,
) -> Dict[str, Any]:
    """Return a full turn-1 context audit for *agent*."""
    from agent.context_breakdown import (
        _SKILLS_BLOCK_RE,
        _memory_blocks,
        _split_tools,
        _strip_blocks,
        _tool_name,
    )
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.prompt_builder import drain_truncation_warnings
    from agent.system_prompt import build_system_prompt_parts
    from model_tools import get_tool_definitions
    from toolsets import resolve_toolset  # noqa: F401 — assembly path for get_tool_definitions

    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    context = parts.get("context", "") or ""
    volatile = parts.get("volatile", "") or ""

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(
        part for part in (system_core, system_tail) if part
    ).strip()

    # Tool schemas come from model_tools.get_tool_definitions (toolsets.resolve_toolset).
    tools = list(getattr(agent, "tools", None) or get_tool_definitions(quiet_mode=True))
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)

    enc = None
    if use_real_tokens:
        try:
            enc = _get_tiktoken_encoder()
        except Exception:
            enc = None

    if enc is not None:
        count_text = lambda text: _real_tokens(text, enc)
        count_json = lambda value: _real_json_tokens(value, enc)
        per_tool_count = lambda tool: _per_tool_real_tokens(tool, enc)
        conversation_tokens = sum(
            _real_json_tokens(msg, enc) for msg in (messages or [])
        )
        estimator = "tiktoken:cl100k_base"
    else:
        count_text = _rough_tokens
        count_json = _rough_json_tokens
        per_tool_count = _per_tool_rough_tokens
        conversation_tokens = estimate_messages_tokens_rough(messages or [])
        estimator = "rough:chars/4"

    category_specs: List[Tuple[str, str, int]] = [
        ("system_prompt", "System prompt", count_text(system_prompt_text)),
        ("tool_definitions", "Tool definitions", count_json(builtin_tools)),
        ("rules", "Rules", count_text(context)),
        ("skills", "Skills", count_text(skills_index)),
        ("mcp", "MCP", count_json(mcp_tools)),
        ("subagent_definitions", "Subagent definitions", count_json(subagent_tools)),
        ("memory", "Memory", count_text(memory_text)),
        ("conversation", "Conversation", conversation_tokens),
    ]

    tool_rows: List[Dict[str, Any]] = []
    for tool in builtin_tools:
        name = _tool_name(tool) or "<unnamed>"
        tool_rows.append({"name": name, "tokens": per_tool_count(tool)})
    tool_rows.sort(key=lambda row: row["tokens"], reverse=True)

    total_tokens = sum(tokens for _, _, tokens in category_specs)

    comp = getattr(agent, "context_compressor", None)
    context_length = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    max_tokens = getattr(agent, "max_tokens", None)
    if max_tokens is None and comp is not None:
        max_tokens = getattr(comp, "max_tokens", None)
    if isinstance(max_tokens, bool):
        max_tokens = None
    elif max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                max_tokens = None
        except (TypeError, ValueError):
            max_tokens = None

    effective_budget = _effective_input_budget(context_length, max_tokens)
    budget_percent = (
        round(total_tokens / effective_budget * 100, 1)
        if effective_budget > 0
        else 0.0
    )

    truncation_warnings = drain_truncation_warnings()

    return {
        "profile": "",
        "hermes_home": os.environ.get("HERMES_HOME", ""),
        "model": getattr(agent, "model", "") or "",
        "platform": getattr(agent, "platform", "") or "cli",
        "estimator": estimator,
        "context_length": context_length,
        "max_tokens": max_tokens,
        "effective_input_budget": effective_budget,
        "total_tokens": total_tokens,
        "budget_percent": budget_percent,
        "categories": [
            {"id": category_id, "label": label, "tokens": tokens}
            for category_id, label, tokens in category_specs
        ],
        "tool_definitions": {
            "tokens": next(
                tokens for cid, _, tokens in category_specs if cid == "tool_definitions"
            ),
            "tools": tool_rows,
        },
        "tool_count": len(tools),
        "truncation_warnings": truncation_warnings,
    }


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def render_audit(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    profile_label = data.get("profile") or "default"
    lines.append(
        f"Context audit (profile={profile_label}, model={data.get('model') or 'unset'}, "
        f"platform={data.get('platform') or 'cli'})"
    )
    lines.append(f"Token estimator: {data.get('estimator', 'rough:chars/4')}")
    if data.get("hermes_home"):
        lines.append(f"HERMES_HOME: {data['hermes_home']}")
    lines.append("")

    lines.append("Categories:")
    total = int(data.get("total_tokens") or 0)
    for category in data.get("categories") or []:
        tokens = int(category.get("tokens") or 0)
        pct = (tokens / total * 100) if total else 0.0
        lines.append(
            f"  {category['id']:<22} {_fmt_int(tokens):>10} tokens  ({pct:5.1f}% of total)"
        )

    tool_block = data.get("tool_definitions") or {}
    tool_rows = tool_block.get("tools") or []
    if tool_rows:
        lines.append("")
        lines.append("Tool definitions (per tool, descending):")
        for row in tool_rows:
            lines.append(f"  {row['name']:<28} {_fmt_int(int(row['tokens'])):>10} tokens")

    lines.append("")
    lines.append(f"Total                 : {_fmt_int(total):>10} tokens")
    context_length = int(data.get("context_length") or 0)
    max_tokens = data.get("max_tokens")
    effective_budget = int(data.get("effective_input_budget") or 0)
    if context_length:
        if max_tokens:
            lines.append(
                f"Context window        : {_fmt_int(context_length)} tokens "
                f"(max_tokens={_fmt_int(int(max_tokens))})"
            )
        else:
            lines.append(f"Context window        : {_fmt_int(context_length)} tokens")
        lines.append(f"Effective input budget: {_fmt_int(effective_budget)} tokens")
        lines.append(
            f"Budget usage          : {float(data.get('budget_percent') or 0):.1f}% "
            f"of effective input budget"
        )

    warnings = data.get("truncation_warnings") or []
    if warnings:
        lines.append("")
        lines.append("Truncation warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    early = _parse_early_args(argv or sys.argv[1:])
    _apply_profile(early.profile)

    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(project_env=os.path.join(_PROJECT_ROOT, ".env"))

    parser = argparse.ArgumentParser(
        description="Audit turn-1 context payload and token breakdown for a profile.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=early.profile,
        help='Hermes profile name (default: "default")',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    parser.add_argument(
        "--real-tokens",
        action="store_true",
        help="Use tiktoken (cl100k_base) when available instead of chars/4 heuristic",
    )
    args = parser.parse_args(argv)

    if args.profile != early.profile:
        _apply_profile(args.profile)

    try:
        agent = _build_inspection_agent(platform="cli")
        data = compute_context_audit(agent, messages=[], use_real_tokens=args.real_tokens)
        data["profile"] = args.profile
    except Exception as exc:
        print(f"Could not compute context audit: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_audit(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
