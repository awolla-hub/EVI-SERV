"""Silero v5_ru text-to-speech service.

Wraps the Silero TTS model (torch.hub, ``snakers4/silero-models``) as a Pipecat
:class:`TTSService`. The base class aggregates streamed LLM tokens up to
sentence boundaries (``TextAggregationMode.SENTENCE``) and calls
:meth:`run_tts` once per sentence, so synthesis overlaps generation. Output is
24 kHz mono PCM16, chunked into ~20 ms frames that drop straight into the app's
existing 24 kHz player.
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import AsyncGenerator, Optional

import numpy as np
import torch
from loguru import logger

# A web-search answer can arrive as a markdown report with URLs; read aloud verbatim that's
# "h-t-t-p-s colon slash slash www dot..." — unspeakable. Strip URLs and markdown link/source
# scaffolding BEFORE the char filter so only human words reach Silero.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")  # [label](url) -> label
_MD_MARK_RE = re.compile(r"[*#`>]")  # markdown emphasis/heading/quote markers — never spoken
# Trailing "Источники: ..." / "Sources: ..." block, tolerating leading markdown markers/whitespace.
_SOURCES_RE = re.compile(r"(?is)\n[\s*#>\-]*(?:источник|source)[a-zа-я]*[\s*]*:.*$")

# Silero v5_ru raises ValueError on emoji / unsupported symbols. Keep only
# Cyrillic, Latin, digits, whitespace and speakable punctuation; drop the rest
# (emoji, pictographs, currency/tech symbols) so one stray glyph can't kill TTS.
_TTS_DROP = re.compile(r"[^0-9A-Za-zЀ-ӿ\s.,!?;:%№()\"'«»\-—–…]")


def _sanitize_for_tts(text: str) -> str:
    text = _MD_LINK_RE.sub(r"\1", text)      # keep the link label, drop the URL
    text = _SOURCES_RE.sub("", text)         # drop any "Источники: ..." tail (before markers stripped)
    text = re.sub(r"(?i)\b(sources?|сорс\w*)\b:?", "", text)   # the word itself must never be spoken
    text = _URL_RE.sub(" ", text)            # nuke bare URLs
    text = _MD_MARK_RE.sub(" ", text)        # strip *, #, `, > markers
    text = _TTS_DROP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

from services import ru_accent


class ClauseFirstAggregator(SimpleTextAggregator):
    """Sentence aggregation, except the FIRST chunk of each response flushes at the first CLAUSE
    boundary (comma/semicolon/dash after a minimum length) instead of waiting for the full sentence.

    Why: time-to-first-audio = LLM finishing the whole first sentence + synthesis. Cutting at the
    first clause starts speech as soon as «Сейчас плюс восемнадцать,» exists — typically
    0.3-0.6 s earlier. Later chunks keep normal sentence granularity (better prosody), and synthesis
    overlaps generation as before.
    """

    _CLAUSE_MARKS = {",", ";", ":", "—"}
    _MIN_CLAUSE_CHARS = 12

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._first_flushed = False

    async def aggregate(self, text: str):
        if self.aggregation_type != AggregationType.SENTENCE:
            async for a in super().aggregate(text):
                yield a
            return
        for char in text:
            self._text += char
            if (
                not self._first_flushed
                and char in self._CLAUSE_MARKS
                and len(self._text.strip()) >= self._MIN_CLAUSE_CHARS
            ):
                result = self._text
                self._text = ""
                self._needs_lookahead = False
                self._first_flushed = True
                yield Aggregation(text=result.strip(" "), type=AggregationType.SENTENCE)
                continue
            result = await self._check_sentence_with_lookahead(char)
            if result:
                self._first_flushed = True
                yield result

    async def flush(self):
        # End of one response → the NEXT response gets a fresh early-clause cut.
        self._first_flushed = False
        return await super().flush()

    async def handle_interruption(self):
        self._first_flushed = False
        await super().handle_interruption()

# ~20 ms of 24 kHz mono audio per output frame.
_FRAME_SAMPLES = 480

# Speakers shipped in the Silero v5_ru pack. Used to validate live voice
# switches (any other name is rejected so a bad client message can never wedge
# apply_tts with an unknown speaker id).
ALLOWED_SPEAKERS = ["xenia", "baya", "kseniya", "aidar", "eugene"]


class SileroTTSService(TTSService):
    """Silero v5_ru TTS, male voice ``aidar`` by default, 24 kHz."""

    def __init__(
        self,
        *,
        repo: str = "snakers4/silero-models",
        speaker_pack: str = "v5_ru",
        speaker: str = "aidar",
        sample_rate: int = 24000,
        **kwargs,
    ) -> None:
        # Let the base class own the TTS audio context lifecycle: it creates the
        # audio context and emits TTSStartedFrame (push_start_frame) before
        # run_tts and TTSStoppedFrame (push_stop_frames) after it. Our run_tts
        # then only yields TTSAudioRawFrames. This is required in pipecat 1.5.0 —
        # frames yielded by run_tts are appended to that base-managed context.
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            # Declared settings silence the per-session pipecat validator errors.
            settings=TTSSettings(model="silero-v5_ru", voice=speaker, language="ru"),
            **kwargs,
        )
        # Swap the base sentence aggregator for the clause-first one (pipecat 1.5 constructs it
        # internally with no injection point, so we replace the private attr right after init).
        self._text_aggregator = ClauseFirstAggregator(
            aggregation_type=self._text_aggregator.aggregation_type
        )
        self._repo = repo
        self._speaker_pack = speaker_pack
        # ``_speaker`` is mutated live by ``set_speaker`` (from the event-loop
        # thread) and read inside ``_synthesize`` (on a worker thread via
        # asyncio.to_thread), so guard it with a plain threading.Lock.
        self._speaker = speaker
        self._speaker_lock = threading.Lock()
        self._model = None
        self._load_lock = asyncio.Lock()

    @property
    def speaker(self) -> str:
        """The speaker id used for the next synthesis."""
        with self._speaker_lock:
            return self._speaker

    def set_speaker(self, name: str) -> bool:
        """Switch the Silero speaker for this session (thread-safe).

        Validates ``name`` against :data:`ALLOWED_SPEAKERS`; an unknown name is
        ignored and ``False`` is returned. The v5_ru model bundles every speaker
        in one pack, so the switch only changes the ``apply_tts`` argument — no
        model reload — and takes effect on the next ``run_tts`` call.
        """
        if name not in ALLOWED_SPEAKERS:
            logger.warning("Ignoring unknown TTS speaker: {!r}", name)
            return False
        with self._speaker_lock:
            if name == self._speaker:
                return True
            self._speaker = name
        logger.info("TTS speaker switched to '{}'.", name)
        return True

    # Loaded models shared ACROSS sessions (a fresh service instance is built per connection;
    # without the cache every reconnect re-paid the torch.hub load).
    _model_cache: dict = {}
    _model_cache_lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with type(self)._model_cache_lock:
            if self._model is not None:
                return
            key = (self._repo, self._speaker_pack)
            cached = type(self)._model_cache.get(key)
            if cached is not None:
                self._model = cached
                return
            logger.info("Loading Silero TTS '{}' (first use)...", self._speaker_pack)
            self._model = await asyncio.to_thread(self._load_model)
            type(self)._model_cache[key] = self._model
            logger.info("Silero TTS ready (speaker={}).", self._speaker)

    def _load_model(self):
        # Use all 4 cores: this is GLOBAL, and capping it at 2 (an earlier attempt) HALVED GigaAM
        # transcription (2.35 s → 1.2 s for a 3 s utterance = the «долго слушает» delay). STT and TTS
        # run sequentially per turn, so the short synth burst doesn't starve the lightweight VAD.
        torch.set_num_threads(4)
        model, _example = torch.hub.load(
            repo_or_dir=self._repo,
            model="silero_tts",
            language="ru",
            speaker=self._speaker_pack,
            trust_repo=True,
        )
        model.to(torch.device("cpu"))
        return model

    def _output_rate(self) -> int:
        """The rate synthesis actually runs at — and therefore the rate the frames must carry.

        Silero's apply_tts REJECTS any rate outside [8000, 24000, 48000]. The pipeline sets
        self.sample_rate on StartFrame, but guard against 0/unset so synthesis never silently
        no-ops (this path is reached whenever Fish is off or unavailable, and on every turn at all
        once FISH_KEY is unset, so it is the normal path rather than a fallback nobody exercises).
        """
        return self.sample_rate if self.sample_rate in (8000, 24000, 48000) else 24000

    def _synthesize(self, text: str) -> Optional[np.ndarray]:
        """Blocking synthesis -> float32 mono waveform at :meth:`_output_rate`."""
        with self._speaker_lock:
            speaker = self._speaker
        sr = self._output_rate()
        # Context-aware stress BEFORE Silero's own. `put_accent` stays on: it fills in
        # whatever RUAccent did not mark, and it is the whole placement when RUAccent is
        # absent or still loading (see services/ru_accent.py — this call is the identity
        # function until then, so it can never delay or break a reply).
        text = ru_accent.stress(text)
        try:
            audio = self._model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=sr,
                put_accent=True,  # fills any word RUAccent left unmarked
                put_yo=True,
                # v5_ru ships its OWN context-aware homograph resolver, and it was switched off
                # here for no reason: `put_accent`/`put_yo` alone are flat dictionary lookups, so
                # «за́мок» and «замо́к» came out the same. These two read the sentence around the
                # word. Measured on this box: 0.46 s vs 0.49 s for 6.4 s of speech — free.
                #
                # This matters more than it looks, because RUAccent — which was supposed to be
                # doing this job — is NOT INSTALLED on the server, so `ru_accent.stress` above has
                # been the identity function in production all along. It degraded silently, exactly
                # as designed, which is why nobody noticed for weeks.
                put_stress_homo=True,
                put_yo_homo=True,
            )
        except Exception as exc:  # noqa: BLE001 — one bad sentence must not kill the reply
            logger.warning("Silero could not synthesize {!r}: {} — skipping.", text[:40], exc)
            return None
        if audio is None or len(audio) == 0:
            return None
        return audio.detach().cpu().numpy().astype(np.float32)

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        text = _sanitize_for_tts(text)
        if not text:
            return
        try:
            await self._ensure_model()
            await self.start_ttfb_metrics()

            waveform = await asyncio.to_thread(self._synthesize, text)
            if waveform is None:
                return

            pcm16 = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2")
            # ~120 ms tail silence so a player/socket flush can't clip the last syllable.
            pcm16 = np.concatenate([pcm16, np.zeros(2880, dtype="<i2")])
            await self.stop_ttfb_metrics()
            # Stamp the rate the audio was SYNTHESIZED at, not the (possibly unset) self.sample_rate —
            # a mismatch plays the reply back at the wrong pitch and speed.
            rate = self._output_rate()

            # Slice into ~20 ms frames so the client can start playing (and
            # flush on barge-in) at fine granularity. TTSStarted/Stopped are
            # emitted by the base class (push_start_frame / push_stop_frames).
            for start in range(0, len(pcm16), _FRAME_SAMPLES):
                chunk = pcm16[start : start + _FRAME_SAMPLES]
                yield TTSAudioRawFrame(
                    audio=chunk.tobytes(),
                    sample_rate=rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Silero TTS failed")
            yield ErrorFrame(f"Silero TTS error: {exc}")
