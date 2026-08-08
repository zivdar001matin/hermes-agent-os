"""Unit tests for Hermes OS email SMTP timeout logout helper."""

from __future__ import annotations

from unittest.mock import patch

from plugins.platforms.email.hermesos_logout import (
    hermesos_logout_configured,
    schedule_hermesos_email_smtp_logout,
)


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
    assert hermesos_logout_configured() is True


@patch.dict(
    "os.environ",
    {
        "HERMES_OS_BACKEND_URL": "",
        "HERMES_MCP_SECRET": "",
        "HERMES_OS_PROFILE_ID": "",
    },
    clear=False,
)
def test_schedule_noop_when_unconfigured():
    with patch("plugins.platforms.email.hermesos_logout._post_logout") as post:
        schedule_hermesos_email_smtp_logout("timed out")
        post.assert_not_called()
