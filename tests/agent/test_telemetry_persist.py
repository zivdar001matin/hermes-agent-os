"""Unit tests for per-call telemetry helpers in conversation_loop."""

from types import SimpleNamespace

from agent.conversation_loop import (
    persist_completed_api_call,
    persist_pending_tool_results,
)
from hermes_state import SessionDB


def test_persist_completed_api_call_writes_row(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        model="qwen",
        base_url="http://host:9931/v1",
    )
    persist_completed_api_call(
        agent,
        started_at=10.0,
        finished_at=12.5,
        prompt_tokens=1000,
        completion_tokens=40,
        cached_tokens=200,
        estimated_prompt_tokens=1100,
        finish_reason="stop",
        tool_names=["read_file"],
    )
    row = db._conn.execute("SELECT * FROM api_calls WHERE session_id = 's1'").fetchone()
    assert row["prompt_tokens"] == 1000
    assert row["estimated_prompt_tokens"] == 1100
    assert row["completion_tokens"] == 40
    assert abs(row["finished_at"] - row["started_at"] - 2.5) < 0.001
    db.close()


def test_persist_completed_api_call_stores_request_id(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        model="qwen",
        base_url="http://host:9931/v1",
        _telemetry_request_id="mhndsi:s1:1",
        _telemetry_call_index=1,
    )
    persist_completed_api_call(
        agent,
        started_at=10.0,
        finished_at=11.0,
        prompt_tokens=10,
        completion_tokens=2,
    )
    row = db._conn.execute("SELECT request_id, call_index FROM api_calls").fetchone()
    assert row["request_id"] == "mhndsi:s1:1"
    assert row["call_index"] == 1
    db.close()


def test_stamp_agent_request_header_sets_extra_headers(tmp_path, monkeypatch):
    from agent.telemetry_headers import HERMES_REQUEST_HEADER, stamp_agent_request_header

    monkeypatch.setenv("HERMES_PROFILE_NAME", "mhndsi")
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-1", source="api_server")
    agent = SimpleNamespace(_session_db=db, session_id="sess-1")
    kwargs = {"model": "qwen", "messages": []}
    rid = stamp_agent_request_header(agent, kwargs)
    assert rid == "mhndsi:sess-1:1"
    assert kwargs["extra_headers"][HERMES_REQUEST_HEADER] == rid
    assert agent._telemetry_call_index == 1
    db.close()


def test_persist_pending_tool_results_flags(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", source="api_server")
    db.record_api_call("s1", model="qwen", started_at=1, finished_at=2)
    agent = SimpleNamespace(_session_db=db, session_id="s1")
    messages = [
        {
            "role": "tool",
            "name": "read_file",
            "content": "hello world",
            "_duration_ms": 12,
        },
        {
            "role": "tool",
            "name": "read_file",
            "content": "[duplicate tool result — identical to an earlier copy]",
            "_deduped_tool_result": True,
        },
        {
            "role": "tool",
            "name": "read_file",
            "content": "[Full tool result was ~9000 estimated tokens and was summarized]",
            "_aux_large_result_summary": True,
        },
    ]
    persist_pending_tool_results(agent, messages, routed_to="main")
    rows = db._conn.execute(
        "SELECT tool_name, truncated, deduped, duration_ms FROM tool_results "
        "WHERE session_id = 's1' ORDER BY id"
    ).fetchall()
    assert len(rows) == 3
    assert rows[0]["truncated"] == 0 and rows[0]["deduped"] == 0
    assert rows[0]["duration_ms"] == 12
    assert rows[1]["deduped"] == 1
    assert rows[2]["truncated"] == 1
    persist_pending_tool_results(agent, messages, routed_to="main")
    n = db._conn.execute("SELECT COUNT(*) FROM tool_results WHERE session_id = 's1'").fetchone()[0]
    assert n == 3
    db.close()
