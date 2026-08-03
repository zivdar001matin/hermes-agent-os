"""Unit tests for Hermes OS Telegram inbound persist helper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from plugins.platforms.telegram.hermesos_persist import (
    hermesos_persist_configured,
    schedule_hermesos_inbound_persist,
    serialize_update_for_hermesos,
)


def test_serialize_group_text_message():
    update = SimpleNamespace(
        update_id=42,
        message=SimpleNamespace(
            message_id=7,
            text="hi group",
            caption=None,
            date=None,
            message_thread_id=None,
            photo=None,
            document=None,
            voice=None,
            video=None,
            chat=SimpleNamespace(id=-1001, type="supergroup", title="Ops", full_name=None),
            from_user=SimpleNamespace(id=99, full_name="Alice", username="alice"),
        ),
        edited_message=None,
        channel_post=None,
        edited_channel_post=None,
        callback_query=None,
    )
    payload = serialize_update_for_hermesos(update)
    assert payload["update_id"] == 42
    assert payload["chat_id"] == "-1001"
    assert payload["chat_type"] == "group"
    assert payload["kind"] == "text"
    assert payload["text"] == "hi group"
    assert payload["user_id"] == "99"


def test_serialize_callback():
    update = SimpleNamespace(
        update_id=5,
        message=None,
        edited_message=None,
        channel_post=None,
        edited_channel_post=None,
        callback_query=SimpleNamespace(
            data="approve:1",
            from_user=SimpleNamespace(id=1, full_name="Bob"),
            message=SimpleNamespace(
                message_id=3,
                chat=SimpleNamespace(id=10, type="private", title=None),
            ),
        ),
    )
    payload = serialize_update_for_hermesos(update)
    assert payload["kind"] == "callback"
    assert payload["text"] == "approve:1"
    assert payload["chat_type"] == "dm"


@patch.dict(
    "os.environ",
    {
        "HERMES_OS_BACKEND_URL": "http://django:8000",
        "HERMES_MCP_SECRET": "sec",
        "HERMES_OS_PROFILE_ID": "prof-1",
    },
    clear=False,
)
def test_configured_when_env_present():
    assert hermesos_persist_configured() is True


@patch.dict("os.environ", {"HERMES_OS_BACKEND_URL": "", "HERMES_MCP_SECRET": "", "HERMES_OS_PROFILE_ID": ""}, clear=False)
def test_schedule_noop_when_unconfigured():
    with patch(
        "plugins.platforms.telegram.hermesos_persist._post_payload"
    ) as post:
        schedule_hermesos_inbound_persist(SimpleNamespace(update_id=1, message=None))
        post.assert_not_called()
