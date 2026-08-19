"""Wire protocol <-> Pipecat frame bridge.

This module implements the OpenGlasses "Пятница Realtime" wire contract on top
of Pipecat's :class:`FastAPIWebsocketTransport`. A single persistent TLS
WebSocket carries:

CLIENT -> SERVER
  * binary : 16 kHz mono PCM16 LE audio, ~20 ms chunks, streamed continuously.
  * text   : {"type":"hello","session":..,"resume_seq":..,"codec":"pcm16"}
             {"type":"barge_in"}
             {"type":"ping","seq":N}
             {"type":"set_voice","voice":"xenia"}  (live TTS speaker switch)
             {"type":"set_persona","name":"Джарвис","character":"jarvis"}
                 (live persona: freeform name and/or character preset; either field optional)
             {"type":"vision_pending","turn":".."}  (capture heads-up; hold turn)
             {"type":"vision","image_b64":..,"turn":..,"for":".."}  (captured frame)
             {"type":"bye"}

SERVER -> CLIENT
  * binary : 24 kHz mono PCM16 LE TTS audio (straight into the app's player).
  * text   : {"type":"partial","text":..,"seq":N}
             {"type":"user_final","text":..}
             {"type":"assistant_text","text":..}
             {"type":"speaking_start"} / {"type":"speaking_end"}
             {"type":"interrupted"}
             {"type":"need_photo","turn":".."}  (visual query, no client photo)
             {"type":"pong","seq":N}
             {"type":"voices","list":[..],"current":"..",
              "characters":[{"id":..,"name":..,"voice":..}],"name":..,"character":..}  (sent on hello)

Design notes
------------
Pipecat's FastAPI output transport only routes a *fixed* set of frame types to
``serializer.serialize`` (audio, transport-message, interruption, end/cancel).
The ``partial`` / ``user_final`` / ``assistant_text`` / ``speaking_*`` events
are therefore emitted by lightweight tap processors (see ``events.py``) that
wrap them in ``TransportMessageFrame`` objects, which *do* reach the serializer.

``ping`` needs a reply that must be sent from the single output task to avoid
concurrent writes on the same WebSocket, so ``deserialize`` returns a
``TransportMessageFrame`` for the ``pong`` and lets it flow down the pipeline to
the output transport, where ``serialize`` turns it back into JSON.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Optional

from loguru import logger

from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

from config import CHARACTERS, character_voice
from services.silero_tts import ALLOWED_SPEAKERS
from services import fish_voices
from services.fish_tts import _KEY as _FISH_KEY, _VOICE as _FISH_DEFAULT_VOICE


def fish_key() -> str:
    return _FISH_KEY


def fish_default_voice() -> str:
    return _FISH_DEFAULT_VOICE
from vision import VisionImageFrame, VisionPendingFrame

if TYPE_CHECKING:
    from config import SessionPersona
    from pipecat.processors.aggregators.llm_context import LLMContext
    from services.silero_tts import SileroTTSService


class PyatnitsaSerializer(FrameSerializer):
    """Custom :class:`FrameSerializer` implementing the wire contract above.

    One instance lives per WebSocket connection. It is intentionally stateless
    apart from the negotiated session metadata captured on ``hello``.

    ``tts`` is the per-session :class:`SileroTTSService`; the serializer holds a
    reference so a ``set_voice`` control message can flip the speaker for this
    session directly (see ``deserialize``).
    """

    def __init__(
        self,
        tts: "SileroTTSService | None" = None,
        context: "LLMContext | None" = None,
        persona: "SessionPersona | None" = None,
        stt=None,
        memory=None,
        visual=None,
    ) -> None:
        super().__init__()
        self.session_id: Optional[str] = None
        self.resume_seq: int = 0
        self.codec: str = "pcm16"
        self._tts = tts
        # Ambient VisualMemory — the app's periodic vision_ambient push lands here (item #8/#17).
        self._visual = visual
        # The per-session STT service — enroll_voice / voice_gate control messages reach the
        # owner-voice gate through it (duck-typed: only the T-one service implements the methods).
        self._stt = stt
        # Shared live LLM context + the mutable persona. set_voice / set_persona edit the persona
        # and rewrite the system message in place so the change lands on the next turn.
        self._context = context
        self._persona = persona
        # Cross-session conversation memory (MemoryStore) — recent turns ride along in the
        # hello reply so a fresh app install can hydrate its chat history.
        self._memory = memory

    def _rebuild_prompt(self) -> None:
        """Recompute the system prompt from the current persona and write it into the live context.

        Edits the existing role=='system' message IN PLACE (rather than replacing the list) so it
        can't race a mid-turn vision-image injection appending to the same messages list.
        """
        if self._context is None or self._persona is None:
            return
        content = self._persona.prompt()
        try:
            messages = self._context.messages
        except Exception:  # noqa: BLE001
            logger.warning("Could not read context messages for persona rebuild")
            return
        for m in messages:
            # Editing the system dict IN PLACE updates the live context even if `.messages` returned
            # a shallow-copied list (the dict objects are shared). If the system message isn't a
            # plain dict (a different LLMContext impl), skip rather than risk corrupting the context —
            # the initial connect-time prompt is already correct; only live edits would be missed.
            if isinstance(m, dict) and m.get("role") == "system":
                m["content"] = content
                return
        logger.warning("No dict system message found; persona live-update skipped")

    # -- lifecycle -----------------------------------------------------------
    async def setup(self, frame: StartFrame) -> None:
        """Capture the negotiated audio rates when the pipeline starts."""
        self._in_rate = frame.audio_in_sample_rate
        self._out_rate = frame.audio_out_sample_rate

    # -- outbound: Pipecat frame -> wire -------------------------------------
    async def serialize(self, frame: Frame) -> str | bytes | None:
        # TTS audio -> raw binary PCM16. TTSAudioRawFrame subclasses
        # OutputAudioRawFrame, so this one check covers both.
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio or None

        # Control messages injected by the tap processors (partial, user_final,
        # assistant_text, speaking_start/end) or by deserialize (pong).
        if isinstance(
            frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)
        ):
            return json.dumps(frame.message, ensure_ascii=False)

        # Barge-in / server-side VAD interruption -> ack so the client flushes
        # its playback buffer.
        if isinstance(frame, InterruptionFrame):
            return json.dumps({"type": "interrupted"})

        # EndFrame / CancelFrame and anything else: nothing to put on the wire.
        return None

    # -- inbound: wire -> Pipecat frame --------------------------------------
    async def deserialize(self, data: str | bytes) -> Frame | None:
        # Binary payloads are raw 16 kHz mono PCM16 audio chunks.
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=self.session_in_rate(),
                num_channels=1,
            )

        # Everything else is a JSON control message.
        try:
            msg = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Dropping non-JSON text frame: {!r}", data[:80])
            return None

        # An exception escaping here kills the transport's RECEIVE task: the socket stays open with
        # the mic streaming into nothing (a half-dead session). One malformed control message must
        # only cost that message, so the whole dispatch is contained.
        try:
            return await self._handle_control(msg)
        except Exception:  # noqa: BLE001
            logger.exception("Dropping control message that raised: {!r}", str(data)[:80])
            return None

    async def _handle_control(self, msg) -> Frame | None:
        """Dispatch one decoded JSON control message (see the wire contract above)."""
        mtype = msg.get("type")

        if mtype == "hello":
            self.session_id = msg.get("session")
            self.resume_seq = int(msg.get("resume_seq") or 0)
            self.codec = msg.get("codec", "pcm16")
            # A reconnect (resume_seq>0) is NOT a fresh session — used by ProactiveInjector to avoid
            # re-greeting on a mid-conversation LTE blip (audit item #15). hello_seen lets the delayed
            # greeting wait for this instead of racing a fixed timer.
            if self._persona is not None:
                self._persona.is_resume = self.resume_seq > 0
                self._persona.hello_seen = True
            logger.info(
                "hello: session={} resume_seq={} codec={}",
                self.session_id,
                self.resume_seq,
                self.codec,
            )
            # Advertise the available voices + the active one, PLUS the character catalog and the
            # session's current name/character, so the client can populate/confirm its pickers.
            # Routed through the pipeline (like pong) so the write happens on the single output task.
            current = self._tts.speaker if self._tts is not None else None
            catalog = [
                {"id": cid, "name": spec["name"], "voice": spec["voice"]}
                for cid, spec in CHARACTERS.items()
            ]
            history: list[dict] = []
            if self._memory is not None:
                try:
                    # SQLite read off the loop: a concurrent writer (embed service, recap job) can
                    # hold the lock for seconds and would otherwise stall the whole audio pipeline.
                    history = await asyncio.to_thread(self._memory.recent_history)
                except Exception:  # noqa: BLE001
                    logger.warning("Could not read memory for hello history")
            # Fish catalogue too, so the voice picker is populated the moment settings opens rather
            # than after a round trip. Cached process-wide (15 min), so this is normally free; on a
            # cold cache it is one thread-bound call and an empty list is a survivable answer.
            fish_list: list[dict] = []
            try:
                # На hello по-прежнему одна страница самых используемых: экран настроек откроется
                # мгновенно, а листать и фильтровать он будет отдельными запросами list_voices.
                fish_list = (await asyncio.to_thread(
                    fish_voices.list_voices, fish_key(), "ru", 40)).get("items", [])
            except Exception:  # noqa: BLE001
                logger.warning("Fish: каталог голосов недоступен на hello")
            return OutputTransportMessageFrame(
                message={
                    "type": "voices",
                    "list": list(ALLOWED_SPEAKERS),
                    "current": current,
                    "fish_list": fish_list,
                    "fish_current": (getattr(self._persona, "fish_voice", "")
                                     or fish_default_voice()),
                    "characters": catalog,
                    "name": self._persona.display_name if self._persona is not None else None,
                    "character": self._persona.character if self._persona is not None else None,
                    "history": history,
                }
            )

        if mtype == "set_voice":
            # Live voice switch for THIS session: flip the Silero speaker so the next TTS uses it,
            # AND reset the character to that voice's default + rebuild the prompt so the persona's
            # gender/character follow the new voice (the custom name, if any, is kept).
            voice = msg.get("voice")
            if self._tts is None:
                logger.warning("set_voice received but no TTS service bound")
            elif self._tts.set_speaker(voice):
                logger.info("set_voice -> {}", voice)
                if self._persona is not None:
                    self._persona.voice = voice
                    self._persona.character = None  # revert to this voice's default character
                    self._rebuild_prompt()
            return None

        if mtype == "set_persona":
            # Live persona edit: a freeform NAME and/or a CHARACTER preset (e.g. "jarvis"). Choosing a
            # character also switches the TTS voice to the one that character is spoken with, so voice
            # and character never mismatch. Rebuilds the system prompt for the next turn.
            if self._persona is None:
                logger.warning("set_persona received but no persona bound")
                return None
            name = msg.get("name")
            character = msg.get("character")
            changed = False
            if name is not None:
                self._persona.name = (str(name).strip() or None)
                changed = True
            if character is not None:
                cid = str(character).lower()
                if cid in CHARACTERS:
                    self._persona.character = cid
                    cvoice = character_voice(cid)
                    if cvoice and cvoice != self._persona.voice:
                        self._persona.voice = cvoice
                        if self._tts is not None:
                            self._tts.set_speaker(cvoice)
                    changed = True
                else:
                    logger.warning("set_persona: unknown character {!r}, ignoring", character)
            if changed:
                logger.info(
                    "set_persona -> name={!r} character={!r} voice={!r}",
                    self._persona.name,
                    self._persona.character,
                    self._persona.voice,
                )
                self._rebuild_prompt()
            return None

        if mtype == "text_input":
            # TYPED message from the app's chat screen — becomes a normal user turn (opens the turn
            # via the aggregator exactly like speech; the answer streams back as text + TTS).
            text = str(msg.get("text") or "").strip()
            if not text:
                return None
            logger.info("text_input: {}", text[:60])
            if self._persona is not None:
                # Typed turn: the LLM answers in full markdown with the picked model/effort. The
                # text is recorded too, so the flag can only be consumed by THIS turn — if it never
                # reaches the LLM (barge-in, dropped turn) the next spoken turn clears it instead of
                # inheriting markdown/model/mute behaviour (see ProxyLLMService).
                self._persona.typed_turn = True
                self._persona.typed_text = text
                self._persona.chat_model = str(msg.get("model") or "")
                self._persona.chat_effort = str(msg.get("effort") or "")
            from pipecat.frames.frames import TranscriptionFrame
            from pipecat.utils.time import time_now_iso8601

            return TranscriptionFrame(text, "", time_now_iso8601())

        if mtype == "enroll_voice":
            # Owner-voice enrollment: the next ~8 s of speech become the voice profile. The spoken
            # instruction goes through TTS so the user knows to start talking.
            if self._stt is not None and hasattr(self._stt, "begin_enrollment"):
                self._stt.begin_enrollment()
                from pipecat.frames.frames import TTSSpeakFrame

                return TTSSpeakFrame(
                    "Говори со мной обычным голосом секунд десять — я запоминаю твой голос."
                )
            logger.warning("enroll_voice received but STT has no gate")
            return None

        if mtype == "voice_gate":
            if self._stt is not None and hasattr(self._stt, "set_gate"):
                self._stt.set_gate(bool(msg.get("on", True)))
            return None

        if mtype == "set_pref":
            if self._persona is not None:
                key = str(msg.get("key") or "")
                on = bool(msg.get("on", True))
                if key == "proactive":
                    self._persona.proactive_enabled = on
                elif key == "backchannel":
                    self._persona.backchannel_enabled = on
                logger.info("set_pref {} -> {}", key, on)
            return None

        if mtype == "set_engine":
            if self._persona is not None:
                eng = str(msg.get("engine") or "auto").lower()
                if eng in ("auto", "fast", "smart"):
                    self._persona.chat_engine = eng
                    logger.info("set_engine -> {}", eng)
            return None

        if mtype == "set_tts_engine":
            # Live TTS engine switch: "fish" (Fish Audio S2, cloud) or "silero" (local neural
            # voice). An older build of the app may still have "xtts" persisted from the removed
            # clone engine; it is not in the tuple, so it is IGNORED and the session keeps whatever
            # it had — never silently reinterpreted as a voice he did not pick.
            if self._persona is not None:
                eng = str(msg.get("engine") or "silero").lower()
                if eng in ("fish", "silero"):
                    self._persona.tts_engine = eng
                    logger.info("set_tts_engine -> {}", eng)
                else:
                    logger.info("set_tts_engine: неизвестный движок {!r} — игнорирую", eng)
            return None

        if mtype == "set_fish_voice":
            # A reference_id from the Fish catalogue, or one of his own trained clones.
            if self._persona is not None:
                vid = str(msg.get("voice") or "").strip()
                self._persona.fish_voice = vid
                logger.info("set_fish_voice -> {}", vid or "(сброшено на умолчание)")
            return None

        if mtype == "list_voices":
            # Explicit refresh for the settings screen. Served from the shared cache when warm, so
            # this is usually free; the network call is on a worker thread either way.
            #
            # ФИЛЬТР ЖИВЁТ ЗДЕСЬ, а не на телефоне: в библиотеке 1002 русских голоса, и тянуть их
            # все, чтобы отобрать десяток, — это мегабайты по сокету ради того, что Fish умеет
            # сделать сам. Все поля необязательны: запрос без них ведёт себя как прежний.
            res = await asyncio.to_thread(
                fish_voices.list_voices,
                fish_key(),
                "ru",
                int(msg.get("limit") or 40),
                int(msg.get("page") or 1),
                str(msg.get("tag") or "").strip()[:32],
                str(msg.get("query") or "").strip()[:64],
                str(msg.get("sort") or "task_count"),
            )
            return OutputTransportMessageFrame(
                message={
                    "type": "fish_voices",
                    "list": res.get("items", []),
                    "page": res.get("page", 1),
                    "has_more": res.get("has_more", False),
                    "current": getattr(self._persona, "fish_voice", "") or fish_default_voice(),
                }
            )

        if mtype == "enroll_tts_voice":
            # ~10 s of his voice -> a PRIVATE Fish model, selected for this session the moment it
            # exists. Training is a network call measured in seconds, so it goes to a thread: on the
            # event loop it would stall every session's audio, which is what the old XTTS enroll did.
            audio_b64 = str(msg.get("audio_b64") or "")
            if not audio_b64 or self._persona is None:
                return None
            try:
                wav = base64.b64decode(audio_b64)
            except Exception:  # noqa: BLE001
                logger.warning("enroll_tts_voice: не декодируется base64")
                return None
            vid = await asyncio.to_thread(fish_voices.create_clone, fish_key(), wav)
            if vid:
                self._persona.fish_voice = vid
                # Selecting it is the point: a clone he has to go and pick afterwards is a clone he
                # will assume failed.
                logger.info("enroll_tts_voice -> клон {} создан и выбран", vid)
            return OutputTransportMessageFrame(
                message={"type": "clone_result", "ok": bool(vid), "voice": vid or ""}
            )

        if mtype == "set_location":
            if self._persona is not None:
                self._persona.user_city = str(msg.get("city") or "")
                try:
                    self._persona.user_lat = float(msg.get("lat") or 0.0)
                    self._persona.user_lon = float(msg.get("lon") or 0.0)
                except (TypeError, ValueError):
                    pass
                # Часовой пояс ТЕЛЕФОНА, а не сервера: сервер живёт в UTC, и без сдвига
                # «сегодня» в промпте расходится с настоящим на девять часов каждый вечер.
                try:
                    if msg.get("tz_offset_hours") is not None:
                        self._persona.user_tz_offset_hours = int(msg["tz_offset_hours"])
                except (TypeError, ValueError):
                    pass
                logger.info("set_location -> {}", self._persona.user_city)
                self._rebuild_prompt()
            return None

        if mtype == "ping":
            # URGENT: a plain OutputTransportMessageFrame queues BEHIND real-time-paced TTS audio, so
            # mid-answer a pong could lag seconds — the client's liveness watchdog would false-kill
            # healthy connections. Urgent frames bypass the paced media queue.
            return OutputTransportMessageUrgentFrame(
                message={"type": "pong", "seq": msg.get("seq", 0)}
            )

        if mtype == "barge_in":
            # Push a system interruption downstream: cancels the in-flight LLM
            # + TTS and (via serialize) acks the client with {"type":"interrupted"}.
            logger.debug("client barge_in")
            return InterruptionFrame()

        if mtype == "vision_ambient":
            # Periodic AMBIENT frame from the glasses (not a turn): store it in VisualMemory so the
            # AmbientVisionEngine can remark on what he's looking at unprompted («живое зрение»).
            image_b64 = msg.get("image_b64") or ""
            if image_b64 and self._visual is not None:
                self._visual.update(f"data:image/jpeg;base64,{image_b64}")
                logger.debug("vision_ambient stored ({} b64 chars)", len(image_b64))
            return None

        if mtype == "vision_pending":
            # Heads-up that the client just decided to capture: the JPEG is on
            # its way. Emit a control frame so the VisionCoordinator briefly
            # holds this turn's endpointing for the image (see vision.py).
            turn = str(msg.get("turn") or "")
            logger.debug("vision_pending (turn={})", turn)
            return VisionPendingFrame(turn=turn)

        if mtype == "vision":
            # A captured JPEG (base64). Wrap it as a data: URL and hand it to the
            # VisionCoordinator, which injects it into the current turn's context
            # and (if the turn is being held) releases the LLM immediately.
            image_b64 = msg.get("image_b64") or ""
            if not image_b64:
                logger.warning("vision frame with no image_b64; ignoring")
                return None
            return VisionImageFrame(
                turn=str(msg.get("turn") or ""),
                data_url=f"data:image/jpeg;base64,{image_b64}",
                query=str(msg.get("for") or ""),
            )

        if mtype == "bye":
            logger.info("client bye")
            return EndFrame()

        logger.warning("Unknown control message type: {}", mtype)
        return None

    # -- helpers -------------------------------------------------------------
    def session_in_rate(self) -> int:
        """Input sample rate; falls back to 16 kHz before ``setup`` runs."""
        return getattr(self, "_in_rate", 16000)
