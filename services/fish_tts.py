"""Fish Audio S2 — the «Живой» voice, beside the local Silero one.

WHY IT IS HERE. Silero is clean and lifeless; Fish S2 wins blind comparisons against ElevenLabs and
is the only strong option with open weights. But the open weights need 12-24 GB of VRAM and this
server has no GPU at all, so what is wired up here is their CLOUD API — which is the one place in
this project where audio leaves the box. That is a deliberate, stated trade, not an oversight.

WHY IT SITS ON TOP OF SileroTTSService. The two engines form one chain — fish -> silero — so the
pipeline keeps constructing exactly one TTS object and the live switch from the phone keeps working
unchanged. Anything that is not "fish" falls straight through to the local voice.

FAILURE POSTURE. The fallback is the LOCAL Silero voice: a cloud outage or an expired key is not a
reason for the assistant to go mute mid-sentence, and that fallback is always available — it is the
same process, with no network between it and the speaker. A circuit breaker keeps a dead or unpaid
endpoint from costing every clause its full timeout.

RUSSIAN IS UNPROVEN. Fish's tier-1 languages are Japanese, English and Chinese; Russian is in the
80+ list without published evidence. Nothing here assumes it sounds good — that is what the A/B
harness in scratchpad/ab_voice.py is for, and it should be listened to before this becomes default.
"""
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, TTSAudioRawFrame

from services.silero_tts import SileroTTSService, _sanitize_for_tts


# EXPRESSION MARKUP SURVIVES THE SANITISER — and until now it did not.
#
# Fish reads inline tags: `[singing]`, `[whispering]`, `[very excited]`, `[warm and gentle]`. They
# were verified against the API by hand and written up as working, but they never once reached it
# from here: `_sanitize_for_tts` keeps only characters Silero can pronounce, and square brackets are
# not among them. Every tag she has ever written was flattened into a bare word before the request
# left the building — so at best the markup did nothing, and at worst she SAID «singing».
#
# The local voice must keep the old behaviour: Silero has no notion of a tag and would read it out.
# So the tags are protected only on the path that can use them, which is this one.
#
# WHAT COUNTS AS A TAG is deliberately narrow: ASCII lowercase words in brackets, at most forty
# characters. Fish's vocabulary is English, so this passes everything it understands and nothing
# else — a Russian aside like «[неразборчиво]» is still stripped exactly as before, rather than
# being handed to the synthesiser as an instruction.
_FISH_TAG_RE = re.compile(r"\[[a-z][a-z ,\-]{0,38}\]")


def _sanitize_for_fish(text: str) -> str:
    """`_sanitize_for_tts` applied to everything BETWEEN the tags, with the tags left intact."""
    parts: list[str] = []
    last = 0
    for m in _FISH_TAG_RE.finditer(text):
        parts.append(_sanitize_for_tts(text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(_sanitize_for_tts(text[last:]))
    return " ".join(p for p in parts if p).strip()

_URL = os.environ.get("FISH_URL", "https://api.fish.audio/v1/tts")
# The key is read from the environment, never committed and never logged. Absent key = engine off,
# and `available()` says so, so nothing has to fail at synthesis time to discover it.
_KEY = os.environ.get("FISH_KEY", "")
# s2.1-pro is the paid production model; s2.1-pro-free is the free tier and the safe default for a
# first listen, so a mistyped env var cannot quietly start spending.
_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free")
# The FALLBACK voice id, used when the session has not picked one.
#
# It must not be empty. Fish has no "default voice" to fall back on: given no reference_id it
# invents a voice for every request, and this pipeline synthesizes clause by clause — so an unset
# id did not mean "their standard voice", it meant a different speaker every few words, mid-sentence.
_VOICE = os.environ.get("FISH_VOICE", "")
_TIMEOUT = float(os.environ.get("FISH_TIMEOUT", "9.0"))
_COOLDOWN = float(os.environ.get("FISH_COOLDOWN", "45.0"))
_TRIP_N = max(1, int(os.environ.get("FISH_TRIP_N", "2")))

# STREAMING. Their WebSocket endpoint returns audio while it is still generating, so the wait no
# longer scales with how much she has to say. Measured on this server, same text, same model:
#
#     clause      HTTP 1.69 s -> WS 1.15 s
#     sentence    HTTP 1.51 s -> WS 1.14 s
#     long reply  HTTP 2.63 s -> WS 1.29 s
#
# The shape matters more than the size: HTTP grows with the reply, the socket does not, because it
# is answering with the FIRST chunk rather than the last. Their docs list only `s1` and `s2-pro` for
# this endpoint — that is wrong, `s2.1-pro-free` was verified working here, while `s1`/`s2-pro`
# reject the handshake with 402 for want of API credit exactly as the HTTP path does.
# FISH_STREAM=0 returns to the plain HTTP request below, which is kept intact for that reason.
_STREAM = os.environ.get("FISH_STREAM", "1") == "1"
_WS_URL = os.environ.get("FISH_WS_URL", "wss://api.fish.audio/v1/tts/live")

_FRAME_SAMPLES = 480             # 20 ms @ 24 kHz
_RATE = 24000                    # ask for the pipeline's rate directly — never resample
_TAIL_PAD = np.zeros(2880, dtype="<i2")     # ~120 ms, so a final syllable is not clipped

# Both are already installed (websockets ships with pipecat; msgpack is in requirements), but the
# streaming path must not be the reason the whole service fails to import on a box where one is
# missing — it degrades to HTTP instead, which needs neither.
try:
    import inspect

    import msgpack
    import websockets

    # `extra_headers` was renamed `additional_headers` in websockets 14 (this server runs 16.1).
    # Read off the actual signature once, at import — not guessed from a version string, and not
    # discovered by catching a TypeError on every clause.
    _WS_HEADER_KW = (
        "additional_headers"
        if "additional_headers" in inspect.signature(websockets.connect).parameters
        else "extra_headers"
    )
except Exception:  # noqa: BLE001
    msgpack = None
    websockets = None
    _WS_HEADER_KW = "additional_headers"


def _streaming_ready() -> bool:
    return _STREAM and msgpack is not None and websockets is not None


def available() -> bool:
    """True when a key is present. Checked at wire-up so the engine simply never appears without
    one, rather than failing on his first sentence."""
    return bool(_KEY)


class FishTTSService(SileroTTSService):
    """Fish Audio S2 over their HTTP API, with a local-voice fallback and a breaker."""

    _cooldown_until: float = 0.0     # class-level: shared across per-connection instances
    _fails: int = 0
    _persona = None                  # set by the pipeline; run_tts reads persona.tts_engine

    @classmethod
    def _in_cooldown(cls) -> bool:
        return time.monotonic() < cls._cooldown_until

    @classmethod
    def _note_failure(cls, exc: BaseException, consequence: str) -> None:
        """One place where a Fish failure is counted, so the two paths can never drift on when the
        breaker trips. `consequence` names what he actually gets, which is the part worth logging."""
        cls._fails += 1
        logger.warning("Fish не ответил ({}); подряд {}/{} — {}",
                       exc, cls._fails, _TRIP_N, consequence)
        if cls._fails >= _TRIP_N:
            cls._cooldown_until = time.monotonic() + _COOLDOWN
            logger.warning("Fish отключён на {:.0f} с", _COOLDOWN)

    @classmethod
    def _note_success(cls) -> None:
        cls._fails = 0
        cls._cooldown_until = 0.0

    def _announce(self, what: str) -> None:
        """Say ONCE, per session and per change, which voice is actually coming out.

        The success path used to log nothing at all, so «Fish не работает» and «Fish работает» left
        exactly the same trace — none — and the only way to tell them apart was to guess. Logged on
        change rather than per clause: a switch mid-conversation is the interesting event, and one
        line per clause would bury it.
        """
        if getattr(self, "_said_engine", "") != what:
            self._said_engine = what
            logger.info("голос этой сессии -> {}", what)

    def _fish_pcm(self, text: str) -> bytes:
        """One clause of raw PCM16. Asks for 24 kHz PCM directly, so nothing is resampled on a
        4-core box, and for low latency, since this is a live conversation and not a render job."""
        payload = {
            "text": text,
            "format": "pcm",
            "sample_rate": _RATE,
            "latency": "low",
            "normalize": True,
        }
        voice = self._voice_id()
        if voice:
            payload["reference_id"] = voice
        req = urllib.request.Request(
            _URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_KEY}",
                "Content-Type": "application/json",
                "model": _MODEL,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            # 401 and 402 are the two that will actually happen — a wrong key and an empty balance —
            # and both are worth naming out loud, because "no sound" is otherwise indistinguishable
            # from a bug in our own pipeline.
            if e.code == 401:
                raise RuntimeError("Fish: ключ отвергнут (401)") from e
            if e.code == 402:
                raise RuntimeError("Fish: закончился баланс (402)") from e
            raise RuntimeError(f"Fish: HTTP {e.code}") from e

    async def _fetch(self, text: str) -> bytes:
        return await asyncio.wait_for(
            asyncio.to_thread(self._fish_pcm, text), timeout=_TIMEOUT + 1.5
        )

    def _voice_id(self) -> str:
        """The voice this session speaks in, most specific first.

        1. what he PICKED in the app — an explicit choice outranks everything;
        2. the CHARACTER's own Fish voice — so Джарвис sounds like Джарвис without him having to
           pick a voice every time he changes character;
        3. the environment default.

        Step 2 is why this is not a one-liner. The character catalogue's `voice` field is a SILERO
        speaker id, which Fish cannot use at all — so before this existed every character, male or
        female, came out in whichever single voice FISH_VOICE named.
        """
        picked = getattr(self._persona, "fish_voice", "")
        if picked:
            return picked
        char = (getattr(self._persona, "character", "") or "").lower()
        if char:
            try:
                from config import CHARACTERS

                by_character = (CHARACTERS.get(char) or {}).get("fish", "")
            except Exception:  # noqa: BLE001 — a catalogue problem must not mute her
                by_character = ""
            if by_character:
                return by_character
        return _VOICE

    def _ws_request(self, text: str) -> dict:
        """The `start` payload. `text` is sent empty here and carried by the `text` event, which is
        what makes the socket a stream rather than a slower POST."""
        req = {
            "text": "",
            "format": "pcm",
            "sample_rate": _RATE,
            "latency": "low",
            "normalize": True,
        }
        voice = self._voice_id()
        if voice:
            req["reference_id"] = voice
        return req

    async def _stream_pcm(self, text: str):
        """Raw PCM16 chunks, yielded as they arrive.

        One socket per clause. A persistent connection would save the handshake, but it would also
        have to survive barge-in, a dropped turn and a model switch — and every number quoted above
        was measured WITH the handshake included, so the win is real without that complexity.
        """
        headers = {"Authorization": f"Bearer {_KEY}", "model": _MODEL}
        connect = websockets.connect(_WS_URL, max_size=None, **{_WS_HEADER_KW: headers})
        async with await asyncio.wait_for(connect, timeout=_TIMEOUT) as ws:
            await ws.send(msgpack.packb({"event": "start", "request": self._ws_request(text)}))
            await ws.send(msgpack.packb({"event": "text", "text": text}))
            await ws.send(msgpack.packb({"event": "flush"}))
            await ws.send(msgpack.packb({"event": "stop"}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=_TIMEOUT)
                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception:  # noqa: BLE001 — a frame we cannot read is not a reason to stop
                    continue
                if not isinstance(msg, dict):
                    continue
                event = msg.get("event")
                if event == "audio":
                    chunk = msg.get("audio")
                    if chunk:
                        yield chunk
                elif event in ("finish", "stop"):
                    return

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        engine = getattr(self._persona, "tts_engine", "fish")
        # Either another engine is selected, or Fish is selected but unusable (no key, or the
        # endpoint is known-bad). Both answers are the same one: speak locally. There is no third
        # place to fall to any more, which is the point — the local voice cannot itself go missing.
        if engine != "fish" or not _KEY or self._in_cooldown():
            why = ("выбран движок " + repr(engine)) if engine != "fish" else (
                "нет ключа" if not _KEY else "Fish в остывании после отказов")
            self._announce(f"Silero локально ({why})")
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        clean = _sanitize_for_fish(text)
        if not clean:
            return

        if _streaming_ready():
            async for frame in self._run_streamed(clean, text, context_id):
                yield frame
            return

        await self.start_ttfb_metrics()
        try:
            data = await self._fetch(clean)
            self._note_success()
        except Exception as exc:  # noqa: BLE001
            self._note_failure(exc, "говорю локально")
            await self.stop_ttfb_metrics()
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        await self.stop_ttfb_metrics()
        if not data or len(data) < 2:
            async for frame in super().run_tts(text, context_id):
                yield frame
            return

        pcm = np.frombuffer(data[: len(data) - (len(data) % 2)], dtype="<i2")
        pcm = np.concatenate([pcm, _TAIL_PAD])
        for start in range(0, len(pcm), _FRAME_SAMPLES):
            yield TTSAudioRawFrame(
                audio=pcm[start : start + _FRAME_SAMPLES].tobytes(),
                sample_rate=_RATE,
                num_channels=1,
                context_id=context_id,
            )

    async def _run_streamed(self, clean: str, raw_text: str, context_id: str):
        """The socket path: frames leave as chunks land, so she starts speaking before the clause
        has finished generating.

        FAILURE SPLITS IN TWO, on purpose. A socket that dies BEFORE any audio is a clause that was
        never spoken — that falls back to the local voice like any other Fish outage. A socket that
        dies AFTER audio is already playing must NOT: re-running the clause locally would make her
        say the first half twice, in a different voice. That one is cut short instead. The breaker
        counts both, so a flapping endpoint stops being asked either way.
        """
        await self.start_ttfb_metrics()
        carry = b""
        spoken = False
        try:
            async for chunk in self._stream_pcm(clean):
                if not spoken:
                    await self.stop_ttfb_metrics()
                    spoken = True
                    self._announce(f"Fish {_MODEL}, поток")
                buf = carry + chunk
                # A chunk boundary can fall inside a 16-bit sample; the odd byte waits for the next.
                usable = len(buf) - (len(buf) % 2)
                carry = buf[usable:]
                if not usable:
                    continue
                pcm = np.frombuffer(buf[:usable], dtype="<i2")
                for start in range(0, len(pcm), _FRAME_SAMPLES):
                    yield TTSAudioRawFrame(
                        audio=pcm[start : start + _FRAME_SAMPLES].tobytes(),
                        sample_rate=_RATE,
                        num_channels=1,
                        context_id=context_id,
                    )
        except asyncio.CancelledError:
            # Barge-in. The pipeline is tearing this turn down — that is not Fish failing, and
            # counting it would trip the breaker on a working endpoint every time he interrupts.
            raise
        except Exception as exc:  # noqa: BLE001
            if spoken:
                self._note_failure(exc, "фраза оборвана на полуслове")
                return
            self._note_failure(exc, "говорю локально")
            await self.stop_ttfb_metrics()
            async for frame in super().run_tts(raw_text, context_id):
                yield frame
            return

        if not spoken:
            # A clean close that said nothing. Silence is not an answer, so it counts as a failure.
            self._note_failure(RuntimeError("пустой поток"), "говорю локально")
            await self.stop_ttfb_metrics()
            async for frame in super().run_tts(raw_text, context_id):
                yield frame
            return

        self._note_success()
        # ~120 ms so the socket flush cannot clip her last syllable, same as the HTTP path.
        yield TTSAudioRawFrame(
            audio=_TAIL_PAD.tobytes(),
            sample_rate=_RATE,
            num_channels=1,
            context_id=context_id,
        )
