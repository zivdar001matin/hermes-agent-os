"""Fan-out inbound Telegram updates to Hermes OS Postgres history.

Profile containers set:
  HERMES_OS_BACKEND_URL, HERMES_MCP_SECRET, HERMES_OS_PROFILE_ID

POST /api/integrations/telegram/inbound/ with MCP Bearer auth.
Failures are logged and never raise into the Telegram handler path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger("hermes.telegram.hermesos_persist")

_INGEST_PATH = "/api/integrations/telegram/inbound/"
_TEXT_LIMIT = 16_000


def _env_backend() -> tuple[str, str, str]:
    backend = (os.getenv("HERMES_OS_BACKEND_URL") or "").rstrip("/")
    secret = os.getenv("HERMES_MCP_SECRET") or ""
    profile_id = os.getenv("HERMES_OS_PROFILE_ID") or ""
    return backend, secret, profile_id


def hermesos_persist_configured() -> bool:
    backend, secret, profile_id = _env_backend()
    return bool(backend and secret and profile_id)


def _chat_type_label(chat: Any) -> str:
    raw = str(getattr(chat, "type", "") or "").split(".")[-1].lower()
    if raw in {"group", "supergroup"}:
        return "group"
    if raw == "channel":
        return "channel"
    if raw == "private":
        return "dm"
    return raw or ""


def _message_kind(message: Any, *, edited: bool = False, channel: bool = False) -> str:
    if edited:
        return "edited"
    if channel:
        return "channel_post"
    text = getattr(message, "text", None) or ""
    if text.startswith("/"):
        return "command"
    if text:
        return "text"
    for attr in ("photo", "video", "audio", "voice", "document", "sticker", "animation"):
        if getattr(message, attr, None):
            return "media"
    if getattr(message, "caption", None):
        return "media"
    return "other"


def _message_payload(message: Any, update_id: int, *, kind: str) -> dict[str, Any]:
    chat = getattr(message, "chat", None)
    user = getattr(message, "from_user", None)
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    if isinstance(text, str) and len(text) > _TEXT_LIMIT:
        text = text[: _TEXT_LIMIT - 1] + "…"
    date = getattr(message, "date", None)
    telegram_date = None
    if date is not None:
        try:
            telegram_date = date.isoformat()
        except Exception:
            telegram_date = str(date)
    thread_id = getattr(message, "message_thread_id", None)
    return {
        "update_id": int(update_id),
        "chat_id": str(getattr(chat, "id", "") or ""),
        "chat_type": _chat_type_label(chat) if chat is not None else "",
        "chat_title": (
            getattr(chat, "title", None)
            or (getattr(chat, "full_name", None) if chat is not None else None)
            or ""
        ),
        "message_id": str(getattr(message, "message_id", "") or ""),
        "thread_id": str(thread_id) if thread_id is not None else "",
        "user_id": str(getattr(user, "id", "") or "") if user is not None else "",
        "user_name": (getattr(user, "full_name", None) or getattr(user, "username", None) or "")
        if user is not None
        else "",
        "text": text,
        "kind": kind,
        "telegram_date": telegram_date,
        "raw": {
            "has_photo": bool(getattr(message, "photo", None)),
            "has_document": bool(getattr(message, "document", None)),
            "has_voice": bool(getattr(message, "voice", None)),
            "has_video": bool(getattr(message, "video", None)),
        },
    }


def serialize_update_for_hermesos(update: Any) -> Optional[dict[str, Any]]:
    """Build ingest payload from a python-telegram-bot Update (or duck-typed)."""
    update_id = getattr(update, "update_id", None)
    if update_id is None:
        return None

    message = getattr(update, "message", None)
    if message is not None:
        return _message_payload(message, update_id, kind=_message_kind(message))

    edited = getattr(update, "edited_message", None)
    if edited is not None:
        return _message_payload(edited, update_id, kind=_message_kind(edited, edited=True))

    channel = getattr(update, "channel_post", None)
    if channel is not None:
        return _message_payload(
            channel, update_id, kind=_message_kind(channel, channel=True)
        )

    edited_channel = getattr(update, "edited_channel_post", None)
    if edited_channel is not None:
        return _message_payload(
            edited_channel, update_id, kind=_message_kind(edited_channel, edited=True, channel=True)
        )

    callback = getattr(update, "callback_query", None)
    if callback is not None:
        msg = getattr(callback, "message", None)
        user = getattr(callback, "from_user", None)
        chat = getattr(msg, "chat", None) if msg is not None else None
        data = getattr(callback, "data", None) or ""
        return {
            "update_id": int(update_id),
            "chat_id": str(getattr(chat, "id", "") or "") if chat is not None else "",
            "chat_type": _chat_type_label(chat) if chat is not None else "",
            "chat_title": (getattr(chat, "title", None) or "") if chat is not None else "",
            "message_id": str(getattr(msg, "message_id", "") or "") if msg is not None else "",
            "thread_id": "",
            "user_id": str(getattr(user, "id", "") or "") if user is not None else "",
            "user_name": (getattr(user, "full_name", None) or "") if user is not None else "",
            "text": str(data)[:_TEXT_LIMIT],
            "kind": "callback",
            "telegram_date": None,
            "raw": {"callback": True},
        }

    return {
        "update_id": int(update_id),
        "chat_id": "",
        "chat_type": "",
        "chat_title": "",
        "message_id": "",
        "thread_id": "",
        "user_id": "",
        "user_name": "",
        "text": "",
        "kind": "other",
        "telegram_date": None,
        "raw": {"unknown_update": True},
    }


def _post_payload(payload: dict[str, Any]) -> None:
    backend, secret, profile_id = _env_backend()
    if not (backend and secret and profile_id):
        return
    url = f"{backend}{_INGEST_PATH}"
    body = json.dumps(payload).encode("utf-8")
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
                    "Hermes OS Telegram inbound ingest HTTP %s",
                    getattr(resp, "status", "?"),
                )
    except urllib.error.HTTPError as exc:
        logger.warning("Hermes OS Telegram inbound ingest failed: HTTP %s", exc.code)
    except Exception as exc:
        logger.warning("Hermes OS Telegram inbound ingest failed: %s", exc)


def schedule_hermesos_inbound_persist(update: Any) -> None:
    """Fire-and-forget persist; safe to call from PTB handlers."""
    if not hermesos_persist_configured():
        return
    try:
        payload = serialize_update_for_hermesos(update)
    except Exception:
        logger.debug("Failed to serialize Telegram update for Hermes OS", exc_info=True)
        return
    if not payload:
        return
    threading.Thread(
        target=_post_payload,
        args=(payload,),
        name="hermesos-tg-inbound",
        daemon=True,
    ).start()
