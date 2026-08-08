"""Ask Hermes OS to disconnect email when SMTP becomes unreachable.

Profile containers set:
  HERMES_OS_BACKEND_URL, HERMES_MCP_SECRET, HERMES_OS_PROFILE_ID

POST /api/integrations/email/smtp-timeout-logout/ with MCP Bearer auth.
Failures are logged and never raise into the Email adapter path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request

logger = logging.getLogger("hermes.email.hermesos_logout")

_LOGOUT_PATH = "/api/integrations/email/smtp-timeout-logout/"


def _env_backend() -> tuple[str, str, str]:
    backend = (os.getenv("HERMES_OS_BACKEND_URL") or "").rstrip("/")
    secret = os.getenv("HERMES_MCP_SECRET") or ""
    profile_id = os.getenv("HERMES_OS_PROFILE_ID") or ""
    return backend, secret, profile_id


def hermesos_logout_configured() -> bool:
    backend, secret, profile_id = _env_backend()
    return bool(backend and secret and profile_id)


def _post_logout(detail: str) -> None:
    backend, secret, profile_id = _env_backend()
    if not (backend and secret and profile_id):
        return
    url = f"{backend}{_LOGOUT_PATH}"
    body = json.dumps(
        {
            "reason": "smtp_timeout",
            "detail": (detail or "")[:2000],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
            "X-Profile-ID": profile_id,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if getattr(resp, "status", 200) >= 400:
                logger.warning(
                    "Hermes OS email SMTP logout HTTP %s",
                    getattr(resp, "status", "?"),
                )
    except urllib.error.HTTPError as exc:
        logger.warning("Hermes OS email SMTP logout failed: HTTP %s", exc.code)
    except Exception as exc:
        logger.warning("Hermes OS email SMTP logout failed: %s", exc)


def schedule_hermesos_email_smtp_logout(detail: str) -> None:
    """Fire-and-forget disconnect request; safe to call from connect()."""
    if not hermesos_logout_configured():
        return
    threading.Thread(
        target=_post_logout,
        args=(detail,),
        name="hermesos-email-smtp-logout",
        daemon=True,
    ).start()
