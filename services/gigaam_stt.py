"""Russian speech-to-text services.

Two interchangeable :class:`SegmentedSTTService` implementations:

* :class:`GigaAMSTTService`   -- Sber's GigaAM v3 RNNT (best Russian WER), via
  the ``gigaam`` package. Preferred once the model is set up on the box.
* :class:`FasterWhisperSTTService` -- faster-whisper CPU fallback so the whole
  loop runs out of the box before GigaAM is installed.

Both subclass :class:`SegmentedSTTService`: Pipecat's Silero VAD segments the
user's utterance and hands us the whole segment, which we transcribe in one shot
and return as a single (finalized) :class:`TranscriptionFrame`. The heavy,
blocking model call is pushed to a worker thread so the event loop keeps
streaming audio.

The factory :func:`create_stt` picks the backend from configuration.
"""

from __future__ import annotations

import asyncio
import io
import os
import wave
from typing import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from config import Config

TARGET_SAMPLE_RATE = 16000

# Consecutive GigaAM failures before the faster-whisper fallback becomes run-long. Below this a
# failed utterance (typically one longer than GigaAM's ~25 s RNNT limit) only falls back for itself.
_MAX_GIGAAM_FAILS = 3

# Emotional attunement (audit item #18), gated EDIT_MOOD_STEER (default OFF → inert). When on, a
# coarse acoustic read (RMS + speech rate) of each utterance tags persona.user_mood for the reply to
# adapt. Wide dead-bands: only clear extremes tag, everything else clears the one-shot mood.
_MOOD_STEER = os.environ.get("EDIT_MOOD_STEER", "0") == "1"


def _pcm_to_float32(data: bytes) -> np.ndarray:
    """Decode a VAD audio segment to mono float32 in [-1, 1] at 16 kHz.

    ``SegmentedSTTService`` may hand us either a WAV container or raw PCM16
    depending on the ``wants_wav_segments`` default, so we sniff the RIFF
    header and handle both. Non-16 kHz input is linearly resampled.
    """
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    else:
        # Raw PCM16 mono at the transport input rate.
        channels, width, rate, raw = 1, 2, TARGET_SAMPLE_RATE, data

    if width != 2:
        raise ValueError(f"Unsupported sample width: {width} bytes")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if rate != TARGET_SAMPLE_RATE and samples.size:
        # Simple linear resample; good enough for ASR feature extraction.
        n_out = int(round(samples.size * TARGET_SAMPLE_RATE / rate))
        samples = np.interp(
            np.linspace(0.0, samples.size, num=n_out, endpoint=False),
            np.arange(samples.size),
            samples,
        ).astype(np.float32)

    return samples


def _float32_to_wav_bytes(samples: np.ndarray, rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Encode float32 mono audio to an in-memory 16-bit PCM WAV."""
    pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class GigaAMSTTService(SegmentedSTTService):
    """GigaAM RNNT Russian STT (Sber, MIT) with a faster-whisper safety net.

    ``model_name`` is passed straight to :func:`gigaam.load_model`. ``"rnnt"``
    resolves to the newest RNNT weights the installed gigaam ships (``v2_rnnt``
    in the current build); ``gigaam.transcribe`` takes a 16 kHz mono WAV file
    path and returns the decoded string.

    Robustness: a failing utterance falls back to faster-whisper for THAT utterance only —
    GigaAM's RNNT refuses audio longer than ~25 s, and one long question must not condemn the
    whole connection to whisper. Only a model that won't load, or ``_MAX_GIGAAM_FAILS``
    consecutive failures, flips the run-long fallback. A failure never crashes the connection.
    """

    def __init__(
        self,
        model_name: str = "rnnt",
        *,
        fallback_model_size: str = "small",
        fallback_compute_type: str = "int8",
        **kwargs,
    ) -> None:
        # Declared settings silence pipecat's per-session validator, which logs an ERROR for
        # every NOT_GIVEN field on connect. Cosmetic today; pipecat has a habit of promoting
        # these to hard failures, and an ERROR line per session buries the ones that matter.
        super().__init__(
            sample_rate=TARGET_SAMPLE_RATE,
            settings=STTSettings(model=f"gigaam-{model_name}", language="ru"),
            **kwargs,
        )
        self._model_name = model_name
        self._model = None
        self._load_lock = asyncio.Lock()
        # Attached by pipeline.py (`if hasattr(stt, "persona")`) so run_stt can tag the user's mood
        # (item #18). None until then; a None persona simply disables the mood read.
        self.persona = None
        # Lazily-built faster-whisper fallback. Once GigaAM is deemed unusable
        # (`_gigaam_failed`), every subsequent utterance goes through it.
        self._fallback_size = fallback_model_size
        self._fallback_compute = fallback_compute_type
        self._fallback: FasterWhisperSTTService | None = None
        self._gigaam_failed = False
        # Consecutive per-utterance GigaAM failures; reset by the next success.
        self._gigaam_fails = 0

    # Loaded models shared ACROSS sessions: a fresh pipeline (and service instance) is built per
    # WebSocket connection, and without this cache every connection re-paid the 5-15 s model load.
    _model_cache: dict = {}
    _model_cache_lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with type(self)._model_cache_lock:
            if self._model is not None:
                return
            cached = type(self)._model_cache.get(self._model_name)
            if cached is not None:
                self._model = cached
                return
            logger.info("Loading GigaAM model '{}' (first use)...", self._model_name)
            import gigaam  # imported lazily so the fallback works without it

            self._model = await asyncio.to_thread(gigaam.load_model, self._model_name)
            type(self)._model_cache[self._model_name] = self._model
            logger.info("GigaAM model ready.")

    def _transcribe(self, samples: np.ndarray) -> str:
        # gigaam.transcribe reads audio from a file path; write a temp WAV.
        import tempfile
        import os

        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(_float32_to_wav_bytes(samples))
            result = self._model.transcribe(path)
        finally:
            os.unlink(path)
        # Some gigaam versions return a dict; normalise to text.
        if isinstance(result, dict):
            return str(result.get("transcription") or result.get("text") or "")
        return str(result)

    async def _fallback_transcribe(self, samples: np.ndarray) -> str:
        """Transcribe via the faster-whisper safety net (built on first use)."""
        if self._fallback is None:
            logger.warning(
                "Building the faster-whisper '{}' fallback (first use).",
                self._fallback_size,
            )
            self._fallback = FasterWhisperSTTService(
                model_size=self._fallback_size,
                compute_type=self._fallback_compute,
            )
        # These two calls don't need the fallback to be wired into a pipeline.
        await self._fallback._ensure_model()
        return await asyncio.to_thread(self._fallback._transcribe, samples)

    def _set_mood(self, samples: np.ndarray, text: str) -> None:
        """Coarse acoustic mood from the utterance (RMS loudness + speech rate). Wide dead-bands:
        only clear extremes tag; neutral clears the one-shot mood. Sets persona.user_mood."""
        dur = samples.size / TARGET_SAMPLE_RATE if samples.size else 0.0
        if dur < 0.5:
            return                                  # too short to judge
        rms = float(np.sqrt(np.mean(np.square(samples))))
        rate = len(text) / dur                      # characters per second
        mood = ""
        if rms < 0.010 and rate < 8.0:              # quiet AND slow → subdued
            mood = "устал"
        elif rms > 0.080 or rate > 20.0:            # loud OR fast → animated
            mood = "оживлён"
        self.persona.user_mood = mood               # "" clears (one-shot, neutral)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        samples = _pcm_to_float32(audio)
        if samples.size == 0:
            return
        await self.start_ttfb_metrics()
        text = ""
        # Per-utterance decision: a single bad segment (GigaAM's RNNT rejects audio longer than
        # ~25 s) must not degrade the whole connection — only a model that won't load, or
        # _MAX_GIGAAM_FAILS in a row, makes the whisper fallback stick.
        use_fallback = self._gigaam_failed

        if not use_fallback:
            try:
                await self._ensure_model()
                text = (await asyncio.to_thread(self._transcribe, samples)).strip()
                self._gigaam_fails = 0
            except Exception:  # noqa: BLE001 - degrade to fallback, keep alive
                use_fallback = True
                self._gigaam_fails += 1
                if self._model is None or self._gigaam_fails >= _MAX_GIGAAM_FAILS:
                    self._gigaam_failed = True
                    self._model = None
                    logger.exception(
                        "GigaAM failed {} time(s); falling back to faster-whisper for the "
                        "rest of this run",
                        self._gigaam_fails,
                    )
                else:
                    logger.exception(
                        "GigaAM failed on this utterance ({:.1f}s); using faster-whisper for it "
                        "only",
                        samples.size / TARGET_SAMPLE_RATE,
                    )

        if use_fallback:
            try:
                text = (await self._fallback_transcribe(samples)).strip()
            except Exception as exc:  # noqa: BLE001
                await self.stop_ttfb_metrics()
                logger.exception("faster-whisper fallback also failed")
                yield ErrorFrame(f"STT error (GigaAM + fallback): {exc}")
                return

        # GigaAM succeeded but returned EMPTY — common on short/quiet segments («да», a soft onset).
        # The segment is real audio (VAD gated it through), so don't silently drop it: retry once with
        # faster-whisper. Gate on energy so we never retry (and risk a whisper hallucination) on true
        # silence. This was a real "иногда не слышит" cause: an empty GigaAM result = a lost turn.
        if not text and not use_fallback:
            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            if rms >= 0.006:
                try:
                    alt = (await self._fallback_transcribe(samples)).strip()
                    if alt:
                        logger.info("GigaAM empty (rms={:.3f}) → whisper recovered: {!r}", rms, alt)
                        text = alt
                except Exception:  # noqa: BLE001 - a failed retry just leaves text empty
                    logger.warning("empty-STT whisper retry failed", exc_info=True)

        # Emotional attunement (item #18): tag HOW he sounded so the reply can adapt. Gated + fully
        # wrapped so a bad read can never break transcription.
        if _MOOD_STEER and self.persona is not None and text:
            try:
                self._set_mood(samples, text)
            except Exception:  # noqa: BLE001
                pass

        await self.stop_ttfb_metrics()
        if text:
            logger.debug("STT: {}", text)
            yield TranscriptionFrame(text, "", time_now_iso8601())


class FasterWhisperSTTService(SegmentedSTTService):
    """faster-whisper CPU fallback STT."""

    def __init__(
        self,
        model_size: str = "small",
        compute_type: str = "int8",
        language: str = "ru",
        **kwargs,
    ) -> None:
        super().__init__(
            sample_rate=TARGET_SAMPLE_RATE,
            settings=STTSettings(model=f"faster-whisper-{model_size}", language=language),
            **kwargs,
        )
        self._model_size = model_size
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._load_lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            logger.info(
                "Loading faster-whisper '{}' ({}) on CPU...",
                self._model_size,
                self._compute_type,
            )
            from faster_whisper import WhisperModel

            self._model = await asyncio.to_thread(
                WhisperModel,
                self._model_size,
                device="cpu",
                compute_type=self._compute_type,
            )
            logger.info("faster-whisper model ready.")

    def _transcribe(self, samples: np.ndarray) -> str:
        segments, _info = self._model.transcribe(
            samples,
            language=self._language,
            beam_size=1,  # greedy: lowest latency on CPU
            vad_filter=False,  # VAD already segmented the audio upstream
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            await self._ensure_model()
            samples = _pcm_to_float32(audio)
            if samples.size == 0:
                return
            await self.start_ttfb_metrics()
            text = (await asyncio.to_thread(self._transcribe, samples)).strip()
            await self.stop_ttfb_metrics()
            if text:
                logger.debug("whisper: {}", text)
                yield TranscriptionFrame(text, "", time_now_iso8601())
        except Exception as exc:  # noqa: BLE001
            logger.exception("faster-whisper transcription failed")
            yield ErrorFrame(f"faster-whisper STT error: {exc}")


def create_stt(config: Config):
    """Instantiate the STT service selected by ``STT_BACKEND``.

    ``tone`` (streaming, lowest latency) falls back to GigaAM if the ``tone`` package or its
    model can't be loaded, so a broken install degrades to the proven batch path instead of
    killing the server.
    """
    backend = config.stt_backend.lower()
    if backend == "tone":
        try:
            from services.tone_stt import TOneSTTService

            return TOneSTTService(decoder=config.tone_decoder)
        except Exception:  # noqa: BLE001 - degrade to batch GigaAM
            logger.exception("T-one unavailable — falling back to GigaAM")
            return GigaAMSTTService(model_name=config.gigaam_model)
    if backend == "gigaam":
        return GigaAMSTTService(model_name=config.gigaam_model)
    if backend in ("faster_whisper", "whisper"):
        return FasterWhisperSTTService(
            model_size=config.whisper_model,
            compute_type=config.whisper_compute_type,
        )
    raise ValueError(f"Unknown STT_BACKEND: {config.stt_backend!r}")
