"""Proprioception plugin — a body sense for Hermes.

Three tiers:

1. **Delta heartbeat** (``pre_llm_call`` hook): when the machine's state
   materially changes between turns — a model server goes down, VRAM
   pressure trips the dashboard, context fill crosses a bucket — a short
   status line rides into the current turn's user message. Ephemeral:
   injected at API-call time via ``plugin_user_context``, never persisted,
   so exactly one heartbeat is ever visible in context.
2. **``body_state`` tool**: on-demand full reading, grouped by category.
3. **Event language**: sensor loss ("dashboard unreachable") is reported
   as a state like any other, so the agent knows when it's flying blind.

Fail-closed: without ``proprioception.enabled: true`` in config.yaml the
hook returns ``None`` and the tool's check_fn hides it. Bundled backend
kind → auto-loads like the spotify plugin; the config flag is the single
switch (soft-launch friendly, ConfigSentinel-visible).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _pre_llm_call(**kwargs: Any) -> Optional[dict]:
    """Turn-prologue hook: contribute an ephemeral heartbeat, or nothing.

    ``invoke_hook`` already isolates exceptions, but we catch everything
    anyway: a broken body sense must never cost a turn.
    """
    try:
        from .settings import get_settings

        settings = get_settings()
        if not settings["enabled"]:
            return None

        from .heartbeat import build_heartbeat

        # A falsy session id must not collapse distinct conversations into
        # one shared delta state; fall back to a per-thread key.
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            import threading

            session_id = f"no-session-{threading.get_ident()}"

        text = build_heartbeat(
            session_id=session_id,
            conversation_history=kwargs.get("conversation_history"),
            settings=settings,
            last_turn=kwargs.get("last_turn"),
            model=str(kwargs.get("model") or ""),
            platform_name=str(kwargs.get("platform") or ""),
            sender_id=str(kwargs.get("sender_id") or ""),
            user_message=str(kwargs.get("user_message") or ""),
        )
        if text:
            # Tell the output firewall a self-signal rode into THIS turn, so it
            # scans the reply for machine-state leakage before it reaches a user.
            from .firewall import mark_signal_fired

            mark_signal_fired(session_id)
            return {"context": text}
        return None
    except Exception:
        logger.debug("proprioception heartbeat failed; staying silent", exc_info=True)
        return None


def register(ctx) -> None:
    """Called once by the plugin loader."""
    from .tools import (
        BODY_STATE_SCHEMA,
        check_body_state_available,
        handle_body_state,
    )

    ctx.register_tool(
        name="body_state",
        toolset="proprioception",
        schema=BODY_STATE_SCHEMA,
        handler=handle_body_state,
        check_fn=check_body_state_available,
        emoji="🫀",
    )
    ctx.register_hook("pre_llm_call", _pre_llm_call)

    from .firewall import transform_llm_output

    ctx.register_hook("transform_llm_output", transform_llm_output)
    ctx.register_hook("transform_tool_result", _transform_tool_result)


def _transform_tool_result(**kwargs: Any) -> Optional[str]:
    """In-loop senses (pain, self-guard): append one line to THIS tool result.

    Tool results are the live tail of the conversation, so the note costs no
    cache re-prefill and arrives while the loop can still use it. Returns None
    (pass-through) unless a sense has something to say.
    """
    try:
        from .settings import get_settings

        settings = get_settings()
        ss = settings.get("senses") or {}
        if not settings["enabled"] or not (ss.get("pain") or ss.get("self_guard")):
            return None
        result = kwargs.get("result")
        if not isinstance(result, str):
            return None
        from .senses import tool_result_note

        note = tool_result_note(
            session_id=str(kwargs.get("session_id") or ""),
            tool_name=str(kwargs.get("tool_name") or "?"),
            args=kwargs.get("args"),
            status=kwargs.get("status"),
            error_message=kwargs.get("error_message"),
            ss=ss,
        )
        return f"{result}\n{note}" if note else None
    except Exception:
        logger.debug("proprioception tool-result sense failed", exc_info=True)
        return None
