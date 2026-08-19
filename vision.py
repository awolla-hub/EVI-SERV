"""Server-side vision: turn-aligned image injection + endpointing hold.

Wire additions (the rest of the protocol is unchanged):

CLIENT -> SERVER
  * {"type":"vision_pending","turn":"<id>"}  — heads-up: the client just decided
        to capture; sent *before* the JPEG is ready so the server can briefly
        hold the finished turn while the image is in flight.
  * {"type":"vision","image_b64":"<jpeg b64>","turn":"<id>","for":"<query>"} —
        the captured frame + the query text it answers.

SERVER -> CLIENT
  * {"type":"need_photo","turn":"<id>"} — fallback: the server heard a visual
        query but the client offered no photo; the client should capture and
        reply with {"type":"vision"}.

Flow
----
``transport.deserialize`` turns the two client messages into
:class:`VisionPendingFrame` / :class:`VisionImageFrame`, which travel downstream
(passed through by VAD/STT/aggregator) to the :class:`VisionCoordinator` placed
between the user context aggregator and the LLM. The aggregator fires a turn by
pushing an ``LLMContextFrame``; the coordinator intercepts it and:

* injects the JPEG into the shared ``LLMContext`` as an ``image_url`` user
  message (via ``LLMContext.create_image_url_message``) so the OpenAI-compatible
  request carries ``content=[{type:text},{type:image_url,...}]``;
* holds the turn up to ``hold_secs`` when a capture is pending but the image
  hasn't landed yet, then releases it (with or without the image);
* if no client photo was offered but the text reads as a visual query, answers
  from the AMBIENT frame she is already looking at when one is fresh enough,
  and only otherwise emits ``need_photo`` and holds for the client to send
  ``{vision}``.

The model swap (text vs. vision) is handled downstream by
:class:`services.proxy_llm.ProxyLLMService`, which keys off whether the request
actually carries an image — the coordinator only has to get the image into the
context before releasing the turn.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    OutputTransportMessageFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# -- control frames (created in transport.deserialize) -----------------------
@dataclass
class VisionPendingFrame(Frame):
    """Client is about to capture a frame for ``turn``; image still in flight."""

    turn: str = ""


@dataclass
class VisionImageFrame(Frame):
    """A captured JPEG (already a ``data:`` URL) + the query it answers."""

    turn: str = ""
    data_url: str = ""
    query: str = ""


# «Что это?» about a scene she is ALREADY watching: an ambient frame this fresh IS the answer, and
# a capture round trip cannot make it any more true — only later. Kept deliberately tight: past this
# window he may have turned away, and answering about the wrong scene is worse than the pause.
_REUSE_MAXAGE = float(os.environ.get("EDIT_VISION_REUSE_MAXAGE", "8"))
# A hold that already timed out is about to answer BLIND, so a looser frame still wins there.
_REUSE_TIMEOUT_MAXAGE = float(os.environ.get("EDIT_VISION_REUSE_TIMEOUT_MAXAGE", "20"))
# A turn-bound capture stays reusable for the short window above, but must go invisible to the
# 30-s ambient read long before it — otherwise AmbientVisionEngine volunteers an unprompted remark
# about the very frame they just finished discussing, and ships a 1280px/q0.6 JPEG to do it.
_TURN_FRAME_MAXAGE = float(os.environ.get("EDIT_VISION_TURN_MAXAGE", "8"))


class VisualMemory:
    """The latest AMBIENT camera frame from the glasses (data URL + monotonic timestamp).

    Survives per-turn resets and is read by :class:`events.AmbientVisionEngine` so she can remark on
    what the user is looking at WITHOUT a query — the «живое зрение» lever (audit items #8/#17). It is
    refreshed both by the app's periodic ``vision_ambient`` push (transport) and by any turn-bound
    capture (VisionCoordinator), so a «что это» moment can be answered from what she already sees.
    """

    def __init__(self) -> None:
        self.data_url: str = ""
        self.ts: float = 0.0
        self._max_age_cap: float = float("inf")  # per-frame ceiling; turn images expire early

    def update(self, data_url: str, *, ambient: bool = True) -> None:
        """Store a frame. ``ambient=False`` marks a turn-bound capture, which no reader may see
        for longer than ``_TURN_FRAME_MAXAGE`` however generous its own ``max_age``."""
        if data_url:
            self.data_url = data_url
            self.ts = time.monotonic()
            self._max_age_cap = float("inf") if ambient else _TURN_FRAME_MAXAGE

    def recent(self, max_age: float) -> Optional[str]:
        """The frame's data URL if it's fresher than ``max_age`` seconds, else None."""
        if self.data_url and (time.monotonic() - self.ts) <= min(max_age, self._max_age_cap):
            return self.data_url
        return None


# Conservative Russian visual-intent heuristic for the need_photo fallback.
# Deliberately narrow: bare "это" is far too common, so we only match phrases
# that clearly point at something the assistant would need to *see*.
# WORD-ANCHORED on both sides — without it «посмотрим», «поглядим», «гляньте» matched their
# imperative prefixes and every such sentence paid a need_photo round trip plus a multi-second
# turn hold. Alternatives that are legitimately followed by an inflection end in \w*.
_VISUAL_INTENT = re.compile(
    r"(?<!\w)("
    r"что\s+(это|тут|здесь|за|на\s+(фото|картинке|экране|изображени\w*))|"
    r"посмотри|погляди|глянь|взгляни|"
    r"прочитай|прочти|распознай|определи\s+что|"
    r"как(ой|ого)\s+цвет\w*|перед\s+тобой|передо\s+мной|что\s+видишь|видишь\s+ли|"
    r"сфотографируй|сфоткай|сфотай|фоткай|фотай|"
    r"сними\s+(фото|кадр)|сделай\s+(фото|фотку|снимок)|щ[её]лкни|"
    r"что\s+(у\s+меня\s+)?в\s+рук\w*|что\s+я\s+держу|держу\s+кое.?что|"
    r"что\s+на\s+мне|оцени\s+(это|вид)"
    r")(?!\w)",
    re.IGNORECASE,
)

# Text-class subset of the above: these need a CRISP frame. The ambient push is 512px / quality 0.4
# (VisionImage.ambientJPEG in the app) — enough to say «это кофемашина», far too coarse to read a
# price tag or a menu. Narrow on purpose: when in doubt we fall through and pay for the capture.
_OCR_INTENT = re.compile(
    r"(?<!\w)("
    r"прочитай|прочти|распознай|что\s+(там\s+)?написано|переведи|мелк\w*\s+шрифт|"
    r"меню|этикетк\w*|состав|срок\s+годности|цен[аыу]|ценник\w*|надпис\w*|"
    r"инструкци\w*|документ\w*|номер\w*"
    r")(?!\w)",
    re.IGNORECASE,
)


def _last_user_text(context: LLMContext) -> str:
    """Plain text of the most recent user message (concatenated text parts)."""
    for m in reversed(context.get_messages()):
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


class VisionCoordinator(FrameProcessor):
    """Aligns client images with the current turn and holds endpointing for them.

    One instance per connection. Placed between ``context_aggregator.user()`` and
    the LLM so it sees both the inbound vision control frames and the outbound
    ``LLMContextFrame`` that triggers inference.
    """

    def __init__(
        self,
        *,
        context: LLMContext,
        hold_secs: float = 2.5,
        need_photo: bool = True,
        visual: "VisualMemory | None" = None,
    ) -> None:
        super().__init__()
        self._context = context
        self._hold_secs = hold_secs
        self._need_photo_enabled = need_photo
        self._visual = visual        # ambient VisualMemory — refreshed by any turn-bound capture
        self._expecting = False  # client sent vision_pending, image not here yet
        self._have_image = False  # image for the current turn already injected
        self._pending_turn: Optional[str] = None
        self._held: Optional[LLMContextFrame] = None
        self._image_event = asyncio.Event()
        self._waiter: Optional[asyncio.Task] = None

    # -- lifecycle -----------------------------------------------------------
    def _reset_turn(self) -> None:
        self._expecting = False
        self._have_image = False
        self._pending_turn = None
        self._held = None
        self._image_event = asyncio.Event()
        self._waiter = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # -- inbound control frames (consumed, never forwarded) --------------
        if isinstance(frame, VisionPendingFrame):
            self._expecting = True
            self._have_image = False
            self._pending_turn = frame.turn
            self._image_event = asyncio.Event()
            logger.debug("vision_pending (turn={})", frame.turn)
            return

        if isinstance(frame, VisionImageFrame):
            self._inject_image(frame.data_url, frame.query)
            self._have_image = True
            self._pending_turn = frame.turn
            self._image_event.set()  # wake a held turn, if any
            logger.info("vision image injected (turn={})", frame.turn)
            return

        # -- turn trigger ----------------------------------------------------
        if isinstance(frame, LLMContextFrame):
            if self._have_image:
                # Image landed before/with the turn: run it straight away.
                await self.push_frame(frame, direction)
                self._reset_turn()
                return
            if self._expecting:
                # Client promised a photo; wait a beat for it.
                await self._begin_hold(frame, direction, "client capture")
                return
            query_text = _last_user_text(frame.context) if self._need_photo_enabled else ""
            if query_text and _VISUAL_INTENT.search(query_text):
                # She is already looking at it: answer from that frame instead of paying a capture
                # round trip plus the full hold (VISION_HOLD_SECS is 20 s in prod). Text-class
                # questions never take this path — they need the crisp capture.
                seen = (
                    None
                    if _OCR_INTENT.search(query_text)
                    else self._recent_frame(_REUSE_MAXAGE)
                )
                if seen is not None:
                    self._inject_image(seen, query_text, remember=False)
                    logger.info("visual query answered from ambient frame (no capture)")
                    await self.push_frame(frame, direction)
                    self._reset_turn()
                    return
                turn = uuid.uuid4().hex[:8]
                self._pending_turn = turn
                await self.push_frame(
                    OutputTransportMessageFrame(
                        message={"type": "need_photo", "turn": turn}
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                logger.info("visual query detected -> need_photo (turn={})", turn)
                await self._begin_hold(frame, direction, "need_photo")
                return
            # Ordinary text turn.
            await self.push_frame(frame, direction)
            return

        # -- interruption / teardown: drop any held turn ---------------------
        if isinstance(frame, (InterruptionFrame, EndFrame, CancelFrame)):
            if self._waiter is not None and not self._waiter.done():
                self._waiter.cancel()
            self._reset_turn()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    # -- helpers -------------------------------------------------------------
    async def _begin_hold(
        self, frame: LLMContextFrame, direction: FrameDirection, reason: str
    ) -> None:
        self._held = frame
        logger.debug("holding turn up to {}s ({})", self._hold_secs, reason)
        self._waiter = self.create_task(self._wait_and_release(frame, direction))

    async def _wait_and_release(
        self, frame: LLMContextFrame, direction: FrameDirection
    ) -> None:
        try:
            await asyncio.wait_for(self._image_event.wait(), timeout=self._hold_secs)
        except asyncio.TimeoutError:
            # The capture never landed. Answering about the last thing she actually saw beats
            # answering blind, so the window here is looser than the short-circuit one.
            seen = self._recent_frame(_REUSE_TIMEOUT_MAXAGE)
            if seen is not None:
                self._inject_image(seen, _last_user_text(frame.context), remember=False)
                logger.info("vision hold timed out; falling back to last ambient frame")
            else:
                logger.info("vision hold timed out; proceeding text-only")
        except asyncio.CancelledError:
            return
        # The image (if it arrived) is already injected into the shared context.
        if self._held is frame:
            await self.push_frame(frame, direction)
        self._reset_turn()

    def _recent_frame(self, max_age: float) -> Optional[str]:
        """The ambient frame she is currently looking at, if it is fresher than ``max_age``."""
        return self._visual.recent(max_age) if self._visual is not None else None

    def _inject_image(self, data_url: str, query: str, *, remember: bool = True) -> None:
        if not data_url:
            logger.warning("vision frame with empty image; skipping injection")
            return
        try:
            msg = LLMContext.create_image_url_message(
                role="user", url=data_url, text=(query or None)
            )
            self._context.add_message(msg)
            # remember=False on the reuse paths: the frame is ALREADY in VisualMemory, and
            # re-storing it would restamp its timestamp — a stale frame could then be reused
            # forever, and would resurface to the ambient reader long after it went cold.
            if self._visual is not None and remember:
                self._visual.update(data_url, ambient=False)
        except Exception:  # noqa: BLE001 - a bad image must never wedge the turn
            logger.exception("failed to inject vision image into context")
