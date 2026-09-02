"""Tests for auxiliary simple-tool follow-up."""

from __future__ import annotations

from types import SimpleNamespace

from agent.aux_simple_tools import (
    DEFAULT_SIMPLE_TOOLS,
    is_simple_tool,
    maybe_run_aux_simple_tool_followup,
    should_run_aux_simple_followup,
)


def _tc(name: str, call_id: str = "call_1", arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeAgent:
    def __init__(self):
        self.quiet_mode = True
        self.log_prefix = ""
        self._interrupt_requested = False
        self._tool_guardrail_halt_decision = None
        self.model = "ggml-org/Qwen3.8-27B-GGUF:Q8_0"
        self.provider = "custom"
        self.base_url = "http://host.docker.internal:9931/v1"
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search files",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        self.valid_tool_names = {"read_file", "search_files", "write_file"}
        self.executed: list[str] = []

    def _vprint(self, *args, **kwargs):
        pass

    def _execute_tool_calls(self, assistant_message, messages, task_id, api_call_count=0):
        for tc in assistant_message.tool_calls:
            self.executed.append(tc.function.name)
            messages.append({
                "role": "tool",
                "name": tc.function.name,
                "tool_call_id": tc.id,
                "content": f"result of {tc.function.name}",
            })

    def _flush_messages_to_session_db(self, messages, conversation_history):
        pass


_ENABLED_DISTINCT = {
    "enabled": True,
    "provider": "custom",
    "model": "unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
    "base_url": "http://host.docker.internal:9932/v1",
    "max_iterations": 4,
    "tools": [],
}


def test_is_simple_tool_matches_native_and_mcp_suffix():
    allow = DEFAULT_SIMPLE_TOOLS
    assert is_simple_tool("read_file", allow)
    assert is_simple_tool("mcp__hermes_os_files__read_file", allow)
    assert is_simple_tool("mcp__hermes_os_files__list_files", allow)
    assert is_simple_tool("mcp__hermes_os_files__find_files", allow)
    assert not is_simple_tool("write_file", allow)
    assert not is_simple_tool("mcp__hermes_os_files__write_file", allow)
    assert not is_simple_tool("terminal", allow)
    assert not is_simple_tool("mcp__hermes_os_actions__send_telegram", allow)


def test_should_run_requires_enabled_distinct_and_all_simple():
    agent = _FakeAgent()
    reads = [_tc("read_file")]
    assert should_run_aux_simple_followup(agent, reads, cfg=_ENABLED_DISTINCT)
    assert not should_run_aux_simple_followup(
        agent, reads, cfg={**_ENABLED_DISTINCT, "enabled": False},
    )
    assert not should_run_aux_simple_followup(
        agent, [_tc("write_file")], cfg=_ENABLED_DISTINCT,
    )
    assert not should_run_aux_simple_followup(
        agent, [_tc("read_file"), _tc("write_file", call_id="c2")],
        cfg=_ENABLED_DISTINCT,
    )
    same_server = {
        **_ENABLED_DISTINCT,
        "base_url": agent.base_url,
        "model": agent.model,
        "provider": "custom",
    }
    assert not should_run_aux_simple_followup(agent, reads, cfg=same_server)
    auto_main = {
        "enabled": True,
        "provider": "auto",
        "model": "",
        "base_url": "",
    }
    assert not should_run_aux_simple_followup(agent, reads, cfg=auto_main)


def test_followup_executes_aux_tool_then_stops_on_text(monkeypatch):
    agent = _FakeAgent()
    monkeypatch.setattr(
        "agent.aux_simple_tools._task_config",
        lambda: {**_ENABLED_DISTINCT, "max_iterations": 3},
    )

    calls = {"n": 0}

    def fake_call_llm(**kwargs):
        calls["n"] += 1
        assert kwargs["task"] == "simple_tools"
        assert kwargs["tools"]
        names = {t["function"]["name"] for t in kwargs["tools"]}
        assert "read_file" in names
        assert "search_files" in names
        assert "write_file" not in names
        if calls["n"] == 1:
            tc = SimpleNamespace(
                id="aux_search",
                function=SimpleNamespace(
                    name="search_files", arguments='{"pattern": "foo"}',
                ),
            )
            msg = SimpleNamespace(content="", tool_calls=[tc])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        msg = SimpleNamespace(content="Found foo in bar.py", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)

    messages = [
        {"role": "user", "content": "Where is foo defined?"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {
            "role": "tool",
            "name": "read_file",
            "tool_call_id": "call_1",
            "content": "file contents",
        },
    ]
    n = maybe_run_aux_simple_tool_followup(
        agent, messages, SimpleNamespace(tool_calls=[_tc("read_file")]), "task-1",
    )
    assert n == 1
    assert agent.executed == ["search_files"]
    assert any(m.get("role") == "assistant" and m.get("_aux_simple_tools") for m in messages)
    last_tool = next(m for m in reversed(messages) if m.get("role") == "tool")
    assert "auxiliary follow-up" in last_tool["content"]
    assert "Found foo in bar.py" in last_tool["content"]
    assert calls["n"] == 2


def test_followup_skips_when_disabled(monkeypatch):
    agent = _FakeAgent()
    monkeypatch.setattr(
        "agent.aux_simple_tools._task_config",
        lambda: {**_ENABLED_DISTINCT, "enabled": False},
    )
    called = {"n": 0}

    def fake_call_llm(**kwargs):
        called["n"] += 1
        raise AssertionError("aux should not be called")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    n = maybe_run_aux_simple_tool_followup(
        agent,
        [{"role": "user", "content": "hi"}],
        SimpleNamespace(tool_calls=[_tc("read_file")]),
        "task-1",
    )
    assert n == 0
    assert called["n"] == 0
    assert agent.executed == []


def test_followup_drops_non_simple_aux_calls(monkeypatch):
    agent = _FakeAgent()
    monkeypatch.setattr(
        "agent.aux_simple_tools._task_config",
        lambda: _ENABLED_DISTINCT,
    )

    def fake_call_llm(**kwargs):
        tc = SimpleNamespace(
            id="aux_write",
            function=SimpleNamespace(name="write_file", arguments="{}"),
        )
        msg = SimpleNamespace(content="should not write", tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    messages = [
        {"role": "user", "content": "edit the file"},
        {"role": "tool", "name": "read_file", "tool_call_id": "call_1", "content": "x"},
    ]
    n = maybe_run_aux_simple_tool_followup(
        agent, messages, SimpleNamespace(tool_calls=[_tc("read_file")]), "task-1",
    )
    assert n == 0
    assert agent.executed == []
    assert "auxiliary follow-up" in messages[-1]["content"]


def test_followup_call_llm_error_does_not_raise(monkeypatch):
    agent = _FakeAgent()
    monkeypatch.setattr(
        "agent.aux_simple_tools._task_config",
        lambda: _ENABLED_DISTINCT,
    )
    def boom(**kwargs):
        raise RuntimeError("aux down")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", boom)
    n = maybe_run_aux_simple_tool_followup(
        agent,
        [{"role": "user", "content": "hi"}],
        SimpleNamespace(tool_calls=[_tc("read_file")]),
        "task-1",
    )
    assert n == 0
