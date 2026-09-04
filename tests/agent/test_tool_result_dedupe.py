"""Session-level identical tool-result dedupe (tools.dedupe_results)."""

from types import SimpleNamespace

from agent.conversation_loop import maybe_dedupe_tool_results


def _assistant(*call_ids_and_names):
    calls = []
    for call_id, name in call_ids_and_names:
        calls.append(SimpleNamespace(id=call_id, function=SimpleNamespace(name=name)))
    return SimpleNamespace(tool_calls=calls)


def _tool(name, content, tool_call_id):
    return {
        "role": "tool",
        "name": name,
        "tool_name": name,
        "content": content,
        "tool_call_id": tool_call_id,
    }


def test_second_identical_result_is_replaced_with_pointer(monkeypatch):
    monkeypatch.setattr(
        "agent.conversation_loop._dedupe_tool_results_enabled", lambda: True
    )
    payload = "سلام" * 2000
    agent = SimpleNamespace()
    first = _tool("read_file", payload, "call-1")
    second = _tool("read_file", payload, "call-2")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        first,
        {"role": "assistant", "tool_calls": [{"id": "call-2"}]},
        second,
    ]

    n = maybe_dedupe_tool_results(agent, messages, _assistant(("call-2", "read_file")))
    assert n == 1
    assert first["content"] == payload
    assert second["content"].startswith("[duplicate tool result")
    assert "call-1" in second["content"]
    assert "sha256=" in second["content"]


def test_first_result_is_kept(monkeypatch):
    monkeypatch.setattr(
        "agent.conversation_loop._dedupe_tool_results_enabled", lambda: True
    )
    payload = "unique-once"
    agent = SimpleNamespace()
    msg = _tool("read_file", payload, "call-1")
    messages = [msg]
    n = maybe_dedupe_tool_results(agent, messages, _assistant(("call-1", "read_file")))
    assert n == 0
    assert msg["content"] == payload


def test_distinct_results_are_not_collapsed(monkeypatch):
    monkeypatch.setattr(
        "agent.conversation_loop._dedupe_tool_results_enabled", lambda: True
    )
    agent = SimpleNamespace()
    a = _tool("read_file", "alpha-content", "call-1")
    b = _tool("read_file", "beta-content", "call-2")
    messages = [a, b]
    n = maybe_dedupe_tool_results(
        agent, messages, _assistant(("call-1", "read_file"), ("call-2", "read_file"))
    )
    assert n == 0
    assert a["content"] == "alpha-content"
    assert b["content"] == "beta-content"


def test_dedupe_can_be_disabled(monkeypatch):
    monkeypatch.setattr(
        "agent.conversation_loop._dedupe_tool_results_enabled", lambda: False
    )
    payload = "same" * 50
    agent = SimpleNamespace()
    first = _tool("read_file", payload, "call-1")
    second = _tool("read_file", payload, "call-2")
    messages = [first, second]
    n = maybe_dedupe_tool_results(
        agent, messages, _assistant(("call-2", "read_file"))
    )
    assert n == 0
    assert second["content"] == payload
