"""LLM stage: the OpenAI-compatible proxy (e.g. claude-api.io).

The proxy speaks the standard ``/v1/chat/completions`` protocol, so Pipecat's
:class:`OpenAILLMService` drops in unchanged -- we only repoint ``base_url`` at
the proxy and select a fast model. Streaming is on by default, which is what
lets TTS start on the first sentence while the model is still writing.

Vision (M4)
-----------
The fast ``model`` (claude-3-5-haiku) is **text-only** and rejects image
content. When a turn carries a client JPEG (injected into the shared
``LLMContext`` as an ``image_url`` content part by :class:`vision.VisionCoordinator`),
:class:`ProxyLLMService` transparently upgrades that single request to
``vision_model``; plain text turns keep the cheap/fast model. If the vision
model is unavailable or errors, the request is retried text-only with the fast
model so the user still gets a spoken reply instead of silence.
"""

from __future__ import annotations

import os
import time

from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

from config import Config, MOOD_STEER, moment_line


def _messages_have_image(messages) -> bool:
    """True if any message carries an OpenAI-style ``image_url`` content part."""
    if not messages:
        return False
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _last_user_text(messages) -> str:
    """Plain text of the last user message in an outgoing request (concatenated text parts)."""
    for m in reversed(list(messages or [])):
        if not (isinstance(m, dict) and m.get("role") == "user"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""
    return ""


def _prefix_last_user(params: dict, steer: str) -> None:
    """Fold a one-shot steer into a PER-REQUEST COPY of the last user message.

    A NEW list and a NEW dict every time: the shared LLMContext feeds memory, so a steer that
    landed there would be replayed as something the user actually said, on every later turn.
    """
    msgs = list(params.get("messages") or [])
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                msgs[i] = {**m, "content": f"{steer} {c}"}
            elif isinstance(c, list):
                msgs[i] = {**m, "content": [{"type": "text", "text": steer}, *c]}
            params["messages"] = msgs
            break


def _strip_images(context: LLMContext) -> None:
    """Rewrite image-bearing messages down to their text parts, in place.

    Used only for the graceful fallback: if the vision model fails, we drop the
    image content so the same context can be re-sent to the text-only model and
    the user still hears an answer (about the words they said, minus the frame).
    """
    messages = context.get_messages()
    rewritten = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in content
        ):
            texts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            joined = " ".join(t for t in texts if t).strip()
            m = {
                **m,
                "content": joined or "Опиши, о чём шла речь, без картинки.",
            }
        rewritten.append(m)
    context.set_messages(rewritten)


class ProxyLLMService(OpenAILLMService):
    """OpenAI-compatible proxy LLM with per-turn text/vision model selection.

    ``text_model`` handles normal turns; a turn whose outgoing request contains
    an image is upgraded to ``vision_model``. Model choice is derived from the
    actual request payload (does it carry an image?), so it stays correct no
    matter how/when the image was injected into the context.
    """

    def __init__(self, *, text_model: str, vision_model: str, persona=None, **kwargs) -> None:
        from pipecat.services.settings import LLMSettings

        # settings= replaces the deprecated model= kwarg (it warned on every request).
        super().__init__(settings=LLMSettings(model=text_model), **kwargs)
        self._text_model = text_model
        self._vision_model = vision_model
        # Session persona: carries the one-shot typed-turn flag + chat model/effort picks.
        self._persona = persona

    def build_chat_completion_params(self, params_from_context) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        if _messages_have_image(params.get("messages")):
            params["model"] = self._vision_model
            logger.debug("vision turn -> model={}", self._vision_model)
        else:
            params["model"] = self._text_model
        # Typed-chat flag rides to the shim as extra JSON body fields (one-shot per turn).
        extra = {"edit_mode": "voice"}
        p = self._persona
        # The typed flag belongs to ONE turn. That turn can die before it ever reaches here (barge-in,
        # dropped turn), and a leaked flag then made the next SPOKEN turn answer in markdown, on the
        # typed model — and stay silent (TypedTurnTTSGate swallows its text). Honour it only while the
        # request still carries the typed text; otherwise this is a different turn, so drop it.
        if p is not None and getattr(p, "typed_turn", False):
            typed_text = (getattr(p, "typed_text", "") or "").strip()
            if typed_text and typed_text not in _last_user_text(params.get("messages")):
                logger.info("stale typed-turn flag dropped — this turn is spoken")
                p.typed_turn = False
                p.typed_text = ""
        # Capture typed-ness BEFORE the branch below resets p.typed_turn — the brevity/warmth
        # knobs must apply only to spoken (voice) turns, never to long typed markdown replies.
        is_typed = bool(p is not None and getattr(p, "typed_turn", False))
        mood = ""
        if p is not None:
            p.current_response_typed = bool(getattr(p, "typed_turn", False))
            mood = getattr(p, "user_mood", "") or ""
            if mood and not getattr(p, "typed_turn", False):
                extra["edit_mood"] = mood
                p.user_mood = ""                     # one-shot: applies to THIS turn only
        if p is not None and getattr(p, "typed_turn", False):
            extra = {
                "edit_mode": "typed",
                "edit_model": getattr(p, "chat_model", "") or "",
                "edit_effort": getattr(p, "chat_effort", "") or "",
            }
            p.typed_turn = False
            p.typed_text = ""
            logger.info("typed turn -> model={} effort={}", extra["edit_model"], extra["edit_effort"])
        # Fast-brain engine choice (auto/fast/smart) rides on every turn; the shim routes
        # conversational turns to the local Молния brain when it's "fast" or "auto" + not tool-y.
        if p is not None:
            extra["edit_engine"] = getattr(p, "chat_engine", "") or "auto"
        params["extra_body"] = {**(params.get("extra_body") or {}), **extra}
        # Voice brevity + warmth knobs (audit item #20), env-gated and OFF unless explicitly set,
        # so the default deploy is byte-for-byte unchanged. Applied to spoken turns ONLY — typed
        # markdown replies keep the model's own length. Set EDIT_VOICE_MAX_TOKENS generously (e.g.
        # 200) so action blocks like [AGENT:…]/[TG:…] are never truncated mid-bracket.
        if not is_typed:
            mt = os.environ.get("EDIT_VOICE_MAX_TOKENS")
            if mt:
                try:
                    params["max_tokens"] = int(mt)
                except ValueError:
                    logger.warning("bad EDIT_VOICE_MAX_TOKENS={!r} — ignored", mt)
            tmp = os.environ.get("EDIT_VOICE_TEMP")
            if tmp:
                try:
                    params["temperature"] = float(tmp)
                except ValueError:
                    logger.warning("bad EDIT_VOICE_TEMP={!r} — ignored", tmp)
        # SHE HAS A CLOCK. The time reached her only inside the self-initiated loops, so an ordinary
        # «сколько времени?» was answered by a model whose only clue was its training data — and
        # «уже поздно, ложись» could arrive at noon. It rides on the request COPY, per turn, so it is
        # always current and never accumulates in the stored conversation.
        if True:
            try:
                from events import _tz_offset_hours, _WEEKDAYS
                t = time.time() + _tz_offset_hours(p) * 3600
                # Дата ЗДЕСЬ, а не только в системном промпте: метка едет в копии реплики и
                # потому может меняться хоть каждую минуту, не трогая кешируемый префикс.
                import datetime as _dt
                _d = _dt.datetime.utcfromtimestamp(t)
                _m = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
                      "августа", "сентября", "октября", "ноября", "декабря")
                stamp = (f"(Сейчас {int((t // 3600) % 24):02d}:{int((t // 60) % 60):02d}, "
                         f"{_WEEKDAYS[int((t // 86400 + 3) % 7)]}, "
                         f"{_d.day} {_m[_d.month - 1]} {_d.year}.) ")
                msgs = list(params.get("messages") or [])
                for i in range(len(msgs) - 1, -1, -1):
                    m = msgs[i]
                    if isinstance(m, dict) and m.get("role") == "user":
                        c = m.get("content")
                        if isinstance(c, str):
                            msgs[i] = {**m, "content": stamp + c}
                        elif isinstance(c, list):
                            msgs[i] = {**m, "content": [{"type": "text", "text": stamp}, *c]}
                        params["messages"] = msgs
                        break
            except Exception:  # noqa: BLE001 — a missing clock must never cost him the answer
                pass

        # Emotional attunement steer (item #18), gated EDIT_MOOD_STEER. The OpenAI-compat proxy drops
        # the edit_mood field, so fold a short one-shot steer into a PER-REQUEST COPY of the last user
        # message — never the shared LLMContext/memory (a NEW list + NEW dict, so nothing persists).
        if not is_typed and mood and os.environ.get("EDIT_MOOD_STEER", "0") == "1":
            steer = MOOD_STEER.get(mood)
            if steer:
                _prefix_last_user(params, steer)
        # «Момент», gated EDIT_MOMENT: local time + how long the pause was, in the SAME
        # per-request copy. Deliberately NOT in the system prompt — that must stay byte-identical
        # across turns (one system message for the Anthropic-backed proxy, and a stable prefix is
        # the precondition for caching it). Spoken turns only; typed turns keep their own shape.
        if not is_typed and os.environ.get("EDIT_MOMENT", "0") == "1":
            moment = moment_line(p)
            if moment:
                _prefix_last_user(params, moment)
            if p is not None:
                # Stamped AFTER the read, so the NEXT turn's «прошлая реплика» is this one.
                p.last_exchange_ts = time.time()
        return params

    async def get_chat_completions(self, context: LLMContext):
        had_image = _messages_have_image(context.get_messages())
        try:
            return await super().get_chat_completions(context)
        except Exception:
            if not had_image:
                raise  # ordinary text failure -> base handles/pushes the error
            # Vision path failed (model missing, image rejected, timeout...).
            # Degrade to a spoken-safe text reply instead of dying silently.
            logger.exception(
                "vision model '{}' failed; retrying text-only with '{}'",
                self._vision_model,
                self._text_model,
            )
            _strip_images(context)
            return await super().get_chat_completions(context)


def create_llm(config: Config, persona=None) -> ProxyLLMService:
    """Build the proxy-backed LLM service with text + vision model selection."""
    return ProxyLLMService(
        api_key=config.proxy_api_key,
        base_url=config.proxy_url,
        text_model=config.model,
        vision_model=config.vision_model,
        persona=persona,
    )
