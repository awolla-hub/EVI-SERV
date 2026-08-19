"""T-one streaming Russian STT service (T-Bank, Apache 2.0) + owner-voice gate.

Streaming recognition (see the original docstring below) PLUS speaker verification: every decoded
phrase's audio is sliced from a rolling 16 kHz buffer (T-one reports phrase start/end times) and
scored against the enrolled owner fingerprint (services/speaker_gate.py). Phrases from OTHER voices
are dropped BEFORE they reach the turn logic — strangers can neither open a turn nor interrupt the
assistant, and ambient chatter/noise dies here too.

Streaming details: pipecat's continuous ``STTService`` calls :meth:`run_stt` for every ~20 ms input
frame; we downsample 16 kHz → 8 kHz (pairwise mean), accumulate exactly 2400 samples = 300 ms and
feed ``pipeline.forward(chunk, state)``. Phrases finalize at natural micro-pauses; on
``VADUserStoppedSpeakingFrame`` we flush with ``is_last=True`` so the tail phrase lands BEFORE the
stop signal reaches the aggregator. Enrollment: the client sends ``{"type":"enroll_voice"}``; the
next ~8 s of speech become the owner profile, confirmed aloud.
"""

from __future__ import annotations

import asyncio
import time
import os

_BACKCHANNEL_ON = os.environ.get("EDIT_BACKCHANNEL", "1") == "1"
import os
from typing import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

from config import MOOD_STEER
from services import asr_fix
from services.speaker_gate import SpeakerVerifier

# Emotional attunement, gated EDIT_MOOD_STEER (default OFF → inert). It MUST be the same flag
# gigaam_stt.py reads: tagging the mood and consuming the tag cannot be enabled independently, or
# the tag rides to the shim on every turn while nothing acts on it.
_MOOD_STEER_ON = os.environ.get("EDIT_MOOD_STEER", "0") == "1"
# The tags this backend may write into persona.user_mood. They MUST be MOOD_STEER keys — the steer
# is looked up by exact key, so a tag with no entry is a silent no-op at request time. Checked once
# at import so the two vocabularies cannot diverge quietly.
_MOOD_TAGS = ("взволнован", "устал", "оживлён", "громко")
_unmapped = [t for t in _MOOD_TAGS if t not in MOOD_STEER]
if _unmapped:
    logger.error("tone_stt mood tags missing from config.MOOD_STEER: {} — steer would no-op", _unmapped)

INPUT_SAMPLE_RATE = 16_000
TONE_SAMPLE_RATE = 8_000
TONE_CHUNK = 2400  # 300 ms at 8 kHz — forward() requires EXACTLY this many samples

GATE_THRESHOLD = float(os.environ.get("VOICE_GATE_THRESHOLD", "0.38"))

# Разбор слуха: запись входного звука рядом с расшифровкой (см. _dump_phrase). Выключено.
_DUMP_AUDIO = os.environ.get("EDIT_DUMP_AUDIO", "0") == "1"
_DUMP_DIR = os.environ.get("EDIT_DUMP_DIR", "/opt/pyatnitsa/dumps")
_DUMP_KEEP = int(os.environ.get("EDIT_DUMP_KEEP", "300"))
_EPOCH_CAP_16K = 60 * INPUT_SAMPLE_RATE   # keep at most 60 s of raw audio per stream epoch
_ENROLL_TARGET_16K = 8 * INPUT_SAMPLE_RATE  # ~8 s of speech makes a profile

_verifier_singleton: SpeakerVerifier | None = None


def _verifier() -> SpeakerVerifier:
    global _verifier_singleton
    if _verifier_singleton is None:
        _verifier_singleton = SpeakerVerifier()
    return _verifier_singleton


class TOneSTTService(STTService):
    """Streaming Russian STT over the T-one pipeline, with the owner-voice gate."""

    _pipeline_cache = None
    _pipeline_lock = asyncio.Lock()

    def __init__(self, *, decoder: str = "beam", **kwargs) -> None:
        super().__init__(
            sample_rate=INPUT_SAMPLE_RATE,
            # Declared settings + honest TTFS silence the per-session pipecat validator errors.
            settings=STTSettings(model="t-one", language="ru"),
            ttfs_p99_latency=0.5,
            **kwargs,
        )
        self._decoder = decoder
        self._pipeline = None
        self._state = None
        self._buf = np.zeros(0, dtype=np.int32)
        self._state_lock = asyncio.Lock()
        self._speaking = False
        # Rolling raw-audio history for the CURRENT stream epoch (reset with T-one state) — phrase
        # times index into it for speaker verification.
        self._epoch16 = np.zeros(0, dtype=np.float32)
        self._epoch16_offset = 0            # samples trimmed off the front
        self._fed16 = 0                     # total 16 k samples fed this epoch
        self._total_fed16 = 0               # global sample clock (never reset — gate window)
        self._last_verified_t = -1e9        # global-seconds of the last owner-verified phrase
        # -- prosody mood (per-utterance) + backchannel («ага» during long speech) --
        self._utt_rms: list[float] = []     # per-phrase RMS values of the current utterance
        self._utt_chars = 0                 # emitted chars this utterance
        self._utt_start_t = 0.0             # global-seconds when the utterance started
        self._last_backchannel_t = -1e9     # global-seconds of the last spoken «ага»
        self._bot_speaking = False          # she is talking right now (echo-guard input)
        self._bot_stopped_wall = 0.0        # wall time her TTS last stopped
        # The session persona (set by the pipeline) — mood tags land here for the LLM prompt.
        self.persona = None
        # Owner-voice gate + enrollment.
        # Owner-voice gating defaults ON (that is the point of the enrolled profile), but it is
        # env-overridable so the STT backend can be switched WITHOUT also switching who she can
        # hear. A fingerprint recorded on one microphone may not clear the bar on another, and
        # "she went deaf" is a far worse failure than "a stranger could talk to her".
        self.gate_enabled = os.environ.get("EDIT_VOICE_GATE", "1") == "1"
        self.enroll_mode = False
        # Meeting mode: log EVERY speaker (labelled Я/Гость by voice fingerprint), answer nothing
        # until «стоп встреча» — then a debrief turn is synthesized.
        self.meeting_mode = False
        self.meeting_log: list[tuple[str, str]] = []
        self._enroll_snips: list[np.ndarray] = []
        self._enroll_len = 0
        self._announce: list[Frame] = []

    # -- model ---------------------------------------------------------------

    async def _ensure_model(self) -> None:
        if self._pipeline is not None:
            return
        async with type(self)._pipeline_lock:
            if type(self)._pipeline_cache is not None:
                self._pipeline = type(self)._pipeline_cache
                return
            logger.info("Loading T-one streaming STT (decoder={})...", self._decoder)
            self._pipeline = await asyncio.to_thread(self._load)
            type(self)._pipeline_cache = self._pipeline
            logger.info("T-one ready.")

    def _load(self):
        from tone import StreamingCTCPipeline

        if self._decoder == "greedy":
            try:
                from tone.decoder import DecoderType

                return StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
            except Exception:  # noqa: BLE001
                logger.warning("T-one greedy decoder unavailable; using default")
        return StreamingCTCPipeline.from_hugging_face()

    # -- enrollment control (called by the transport on client messages) ------

    def begin_enrollment(self) -> None:
        self.enroll_mode = True
        self._enroll_snips = []
        self._enroll_len = 0
        self._enroll_started = time.monotonic()
        logger.info("Voice enrollment STARTED")

    def set_gate(self, on: bool) -> None:
        self.gate_enabled = on
        logger.info("Voice gate {}", "ON" if on else "OFF")

    def gate_status(self) -> dict:
        v = _verifier()
        return {"available": v.available, "enrolled": v.enrolled, "on": self.gate_enabled}

    # -- audio helpers --------------------------------------------------------

    @staticmethod
    def _downsample(pcm16: np.ndarray) -> np.ndarray:
        if len(pcm16) % 2:
            pcm16 = pcm16[:-1]
        return (pcm16[0::2].astype(np.int32) + pcm16[1::2].astype(np.int32)) // 2

    def _epoch_reset(self) -> None:
        self._epoch16 = np.zeros(0, dtype=np.float32)
        self._epoch16_offset = 0
        self._fed16 = 0
        # NOTE: _last_verified_t / _total_fed16 deliberately survive the reset — the
        # continuation window must span utterances, or short follow-ups are always dropped.

    def _phrase_audio(self, ph) -> np.ndarray:
        """Slice the phrase's raw 16 kHz audio out of the epoch history."""
        try:
            s = int(float(ph.start_time) * INPUT_SAMPLE_RATE) - self._epoch16_offset
            e = int(float(ph.end_time) * INPUT_SAMPLE_RATE) - self._epoch16_offset
            s = max(0, s)
            e = min(len(self._epoch16), max(s, e))
            return self._epoch16[s:e]
        except Exception:  # noqa: BLE001
            return np.zeros(0, dtype=np.float32)

    def _phrase_allowed(
        self, snippet: np.ndarray, text: str, precomputed: float | None = None
    ) -> bool:
        """The owner-voice gate. Also feeds enrollment when it is active."""
        v = _verifier()
        if self.enroll_mode:
            if time.monotonic() - getattr(self, "_enroll_started", 0.0) > 60.0:
                # Enrollment never completed (too little speech) — bail out instead of
                # swallowing every phrase forever.
                self.enroll_mode = False
                self._enroll_snips = []
                self._enroll_len = 0
                self._announce.append(OutputTransportMessageUrgentFrame(
                    message={"type": "enrolled", "ok": False}
                ))
                self._announce.append(TTSSpeakFrame(
                    "Я так и не расслышала достаточно голоса — отменяю запоминание, скажи ещё раз «запомни мой голос»."
                ))
                return False
            if snippet.size >= INPUT_SAMPLE_RATE:          # ≥1 s pieces only
                self._enroll_snips.append(snippet.copy())
                self._enroll_len += snippet.size
            if self._enroll_len >= _ENROLL_TARGET_16K:
                ok = v.enroll(self._enroll_snips) if v.available else False
                self.enroll_mode = False
                self._enroll_snips = []
                self._enroll_len = 0
                self._announce.append(
                    OutputTransportMessageUrgentFrame(message={"type": "enrolled", "ok": ok})
                )
                self._announce.append(TTSSpeakFrame(
                    "Готово, я запомнила твой голос. Теперь слушаю только тебя."
                    if ok else
                    "Не получилось запомнить голос, попробуй ещё раз."
                ))
            return False                                    # enrollment speech is not a query
        if not (self.gate_enabled and v.enrolled):
            return True
        now = self._total_fed16 / INPUT_SAMPLE_RATE
        score = precomputed if precomputed is not None else v.score(snippet)
        if score is None:
            # Too short to verify — allow only as continuation of a recently verified voice.
            return (now - self._last_verified_t) < 8.0
        # Echo guard: while SHE is speaking (and ~1.2s after), her own TTS bleeds into the mic —
        # keep the FULL bar so echo fragments can't open phantom turns («Момент» из ниоткуда).
        echo_risk = self._bot_speaking or (time.monotonic() - self._bot_stopped_wall) < 1.2
        # Он только что говорил и был опознан — разговор ИДЁТ. Это меняет не порог доверия к
        # голосу, а цену ошибки: посреди своей же реплики отброшенная фраза — это её молчание и
        # его «она опять не слышит», тогда как чужой голос именно в этот момент маловероятен.
        in_conversation = (now - self._last_verified_t) < 8.0
        if echo_risk:
            threshold = GATE_THRESHOLD
        elif snippet.size >= 2 * INPUT_SAMPLE_RATE:
            # ПЛАНКА ДЛЯ ДЛИННЫХ ФРАЗ БЫЛА СЛИШКОМ ВЫСОКОЙ — это видно по журналу, а не по
            # рассуждению: его собственные длинные фразы набирают 0.32-0.35 и улетали в DROP
            # («а все все все хватит», «плоть до того говорить на камеру»), пока короткие с 0.27
            # проходили. Замысел «длиннее фраза → больше доказательств → выше балл» на реальном
            # звуке не выполняется: в длинный кусок попадают шум, музыка и чужие реплики рядом.
            # Внутри разговора планка опускается до 0.30 — всё ещё заметно выше гостевой полосы
            # (гости держатся ниже 0.20), но уже не режет хозяина.
            threshold = 0.30 if in_conversation else GATE_THRESHOLD
        else:
            # Short phrases in the clear: «что это»/«рассказывай» from the owner score 0.25-0.32
            # (little voice evidence), guests stay well under 0.20 — 0.24 hears him, blocks them.
            threshold = 0.24
        # Планка в строке журнала: без неё по PASS/DROP нельзя понять, КАКОЕ правило сработало,
        # и следующая настройка снова будет гаданием.
        if score >= threshold:
            self._last_verified_t = now
            logger.info("voice-gate PASS {:.2f}/{:.2f} {!r}", score, threshold, text[:40])
            return True
        logger.info("voice-gate DROP {:.2f}/{:.2f} {!r}", score, threshold, text[:40])
        return False

    _ENROLL_TRIGGERS = ("запомни мой голос", "запомни голос", "выучи мой голос")
    _MEET_START = ("режим встречи", "начни встречу", "запиши встречу")
    _MEET_STOP = ("стоп встреча", "закончи встречу", "конец встречи", "стоп запись")

    def _dump_phrase(self, snippet: np.ndarray, text: str) -> None:
        """Сохранить ТОТ САМЫЙ звук, который получил распознаватель, рядом с тем, что он услышал.

        Зачем: «слышит криво» может значить две совершенно разные вещи — микрофон отдаёт кашу, или
        модель не справляется с нормальным звуком. Лечатся они в разных местах и стоят разных денег
        (замена наушников против платного распознавателя), а различить их можно ТОЛЬКО послушав
        вход. Без этих файлов любой выбор — ставка.

        Пара «wav + txt» на фразу и есть готовый набор для сравнения распознавателей: один и тот же
        звук прогоняется через T-one, GigaAM, SpeechKit — и спор о том, кто лучше, решается
        числом на ЕГО голосе и ЕГО микрофоне, а не бенчмарком на чужом.

        ВЫКЛЮЧЕНО ПО УМОЛЧАНИЮ: это запись голоса на диск. Включается `EDIT_DUMP_AUDIO=1` на время
        разбора и гасится обратно; старые файлы подчищаются сами, чтобы забытый флаг не превратился
        в бесконечный архив разговоров.
        """
        try:
            import wave

            os.makedirs(_DUMP_DIR, exist_ok=True)
            files = sorted(f for f in os.listdir(_DUMP_DIR) if f.endswith(".wav"))
            for stale in files[: max(0, len(files) - _DUMP_KEEP + 1)]:
                for ext in (".wav", ".txt"):
                    try:
                        os.remove(os.path.join(_DUMP_DIR, stale[:-4] + ext))
                    except OSError:
                        pass
            stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.monotonic() * 1000) % 1000:03d}"
            path = os.path.join(_DUMP_DIR, stamp)
            with wave.open(path + ".wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(INPUT_SAMPLE_RATE)
                w.writeframes(snippet.astype("<i2").tobytes())
            with open(path + ".txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:  # noqa: BLE001 — диагностика не имеет права ломать разговор
            logger.warning("не смогла сохранить фразу для разбора", exc_info=True)

    def _process_phrase(self, snippet: np.ndarray, text: str) -> str | None:
        """Meeting mode + owner gate router. Returns the text to emit as a user turn, or None."""
        if _DUMP_AUDIO and snippet.size:
            self._dump_phrase(snippet, text)     # в файл — СЫРОЕ, до починки: это наша правда о слухе
        # Починка по звучанию ДО всего остального: от неё зависят и триггеры («запомни мой голос»),
        # и то, что увидит мозг. Словарь маленький и злой — см. services/asr_fix.py.
        text = asr_fix.repair(text)
        tl = text.lower()
        v = _verifier()
        owner = None
        owner_score: float | None = None
        if v.enrolled and snippet.size >= 8000:
            owner_score = v.score(snippet)
            owner = owner_score is not None and owner_score >= GATE_THRESHOLD

        # Voice-triggered enrollment — bulletproof (no WS race like the app button had).
        if not self.enroll_mode and any(k in tl for k in self._ENROLL_TRIGGERS):
            if owner is not False:
                self.begin_enrollment()
                self._announce.append(TTSSpeakFrame(
                    "Говори со мной обычным голосом секунд десять — я запоминаю твой голос."
                ))
            return None

        if not self.meeting_mode and any(k in tl for k in self._MEET_START):
            if owner is not False:
                self.meeting_mode = True
                self.meeting_log = []
                logger.info("MEETING mode ON")
                self._announce.append(TTSSpeakFrame(
                    "Записываю встречу и помечаю, кто говорит. Скажи «стоп встреча», когда закончите."
                ))
                self._announce.append(OutputTransportMessageUrgentFrame(
                    message={"type": "assistant_text", "text": "🎙 запись встречи…"}
                ))
            return None

        if self.meeting_mode:
            if owner is not False and any(k in tl for k in self._MEET_STOP):
                self.meeting_mode = False
                log = "\n".join(f"{who}: {t}" for who, t in self.meeting_log)[-6000:]
                n = len(self.meeting_log)
                self.meeting_log = []
                logger.info("MEETING mode OFF ({} phrases)", n)
                self._announce.append(TTSSpeakFrame("Секунду, готовлю итоги встречи."))
                return (
                    "Встреча закончена. Составь короткий дебриф по записи ниже: главные темы, "
                    "решения, задачи и договорённости, и отправь его в телеграм блоком [TG: ...]. "
                    "Голосом скажи только самое главное, в одно-два предложения. Запись встречи:\n"
                    + (log or "запись пуста")
                )
            who = "Я" if owner else "Гость"
            self.meeting_log.append((who, text))
            self._announce.append(OutputTransportMessageUrgentFrame(
                message={"type": "partial", "text": f"{who}: {text}", "seq": 0}
            ))
            return None

        return text if self._phrase_allowed(snippet, text, precomputed=owner_score) else None

    # -- prosody mood + backchannel -------------------------------------------

    _BACKCHANNELS = ("Ага.", "Угу.", "Так.", "М-м.")

    def _note_prosody(self, snippet: np.ndarray, text: str) -> None:
        """Accumulate loudness + tempo evidence for the current utterance."""
        now = self._total_fed16 / INPUT_SAMPLE_RATE
        if not self._utt_rms:
            self._utt_start_t = now
        if snippet.size:
            self._utt_rms.append(float(np.sqrt(np.mean(snippet.astype(np.float64) ** 2))))
        self._utt_chars += len(text)

    def _finish_mood(self) -> None:
        """Classify the finished utterance's prosody into one of the _MOOD_TAGS for the prompt.

        Deliberately conservative: tag only clear signals, else empty (no tag beats a wrong one).
        The result must be a MOOD_STEER key, not a free-text description — the steer is looked up
        by exact key, so a phrase here means the read is computed and then silently discarded.
        """
        try:
            rms_vals, chars = self._utt_rms, self._utt_chars
            start = self._utt_start_t
            self._utt_rms, self._utt_chars = [], 0
            if not rms_vals or chars < 6:
                return
            now = self._total_fed16 / INPUT_SAMPLE_RATE
            dur = max(0.5, now - start)
            rms = sorted(rms_vals)[len(rms_vals) // 2]     # median phrase loudness
            rate = chars / dur                              # chars/sec ≈ tempo
            loud, quiet = rms > 0.11, rms < 0.028
            fast, slow = rate > 15.0, rate < 6.5
            if loud and fast:
                mood = "взволнован"
            elif quiet and slow:
                mood = "устал"
            elif fast:
                mood = "оживлён"
            elif loud:
                mood = "громко"
            else:
                mood = ""
            if self.persona is not None:
                self.persona.user_mood = mood
        except Exception:  # noqa: BLE001 - mood must never break the audio path
            self._utt_rms, self._utt_chars = [], 0

    def _maybe_backchannel(self) -> None:
        """Fake half-duplex: a short «ага» while the user is mid-monologue, like a human listener.

        Only during ACTIVE speech (VAD open), only after ~4s of continuous talking, at most one
        per 7s, and never during enrollment/meeting mode.
        """
        if (not _BACKCHANNEL_ON or self.enroll_mode or self.meeting_mode or not self._speaking
                or not getattr(self.persona, "backchannel_enabled", True)):
            return
        now = self._total_fed16 / INPUT_SAMPLE_RATE
        if not self._utt_rms or now - self._utt_start_t < 4.0:
            return
        if now - self._last_backchannel_t < 7.0:
            return
        self._last_backchannel_t = now
        import random
        self._announce.append(TTSSpeakFrame(random.choice(self._BACKCHANNELS)))

    def _drain_announcements(self) -> list[Frame]:
        out, self._announce = self._announce, []
        return out

    # -- audio path ----------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        try:
            await self._ensure_model()
            pcm = np.frombuffer(audio, dtype=np.int16)
            if pcm.size == 0:
                yield None
                return
            phrases = []
            async with self._state_lock:
                f32 = pcm.astype(np.float32) / 32768.0
                self._epoch16 = np.concatenate([self._epoch16, f32])
                self._fed16 += len(f32)
                self._total_fed16 += len(f32)
                if len(self._epoch16) > _EPOCH_CAP_16K:
                    drop = len(self._epoch16) - _EPOCH_CAP_16K
                    self._epoch16 = self._epoch16[drop:]
                    self._epoch16_offset += drop
                self._buf = np.concatenate([self._buf, self._downsample(pcm)])
                while len(self._buf) >= TONE_CHUNK:
                    chunk, self._buf = self._buf[:TONE_CHUNK], self._buf[TONE_CHUNK:]
                    new_phrases, self._state = await asyncio.to_thread(
                        self._pipeline.forward, chunk, self._state
                    )
                    phrases.extend(new_phrases)
                emit = []
                for ph in phrases:
                    raw = getattr(ph, "text", ph)
                    text = (raw if isinstance(raw, str) else "").strip()
                    if text:
                        snippet = self._phrase_audio(ph)
                        res = self._process_phrase(snippet, text)
                        if res:
                            emit.append(res)
                            self._note_prosody(snippet, res)
                            self._maybe_backchannel()
            for fr in self._drain_announcements():
                yield fr
            for text in emit:
                logger.debug("T-one phrase: {}", text)
                yield TranscriptionFrame(text, "", time_now_iso8601())
        except Exception as exc:  # noqa: BLE001
            logger.exception("T-one run_stt failed")
            self._state = None
            self._buf = np.zeros(0, dtype=np.int32)
            self._epoch_reset()
            yield ErrorFrame(f"T-one STT error: {exc}")

    # -- VAD hooks: flush the tail the moment the user stops ------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Track HER speech (frames travel upstream through the STT): while she talks — and for a
        # short tail after — the gate goes strict so her own echo can't open phantom turns.
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_wall = time.monotonic()
        await super().process_frame(frame, direction)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        await super()._handle_vad_user_started_speaking(frame)
        self._speaking = True

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        await super()._handle_vad_user_stopped_speaking(frame)
        if not self._speaking:
            return
        self._speaking = False
        if self._pipeline is None:
            return
        try:
            emit = []
            async with self._state_lock:
                if self._state is None and len(self._buf) == 0:
                    return
                chunk = np.zeros(TONE_CHUNK, dtype=np.int32)
                take = min(len(self._buf), TONE_CHUNK)
                if take:
                    chunk[:take] = self._buf[:take]
                self._buf = np.zeros(0, dtype=np.int32)
                phrases, _ = await asyncio.to_thread(
                    self._pipeline.forward, chunk, self._state, is_last=True
                )
                self._state = None
                for ph in phrases:
                    raw = getattr(ph, "text", ph)
                    text = (raw if isinstance(raw, str) else "").strip()
                    if text:
                        snippet = self._phrase_audio(ph)
                        res = self._process_phrase(snippet, text)
                        if res:
                            emit.append(res)
                            self._note_prosody(snippet, res)
                if _MOOD_STEER_ON:
                    self._finish_mood()
                else:
                    self._utt_rms, self._utt_chars = [], 0   # keep the accumulator bounded
                self._epoch_reset()
            for fr in self._drain_announcements():
                await self.push_frame(fr, FrameDirection.DOWNSTREAM)
            for text in emit:
                logger.debug("T-one tail phrase: {}", text)
                await self.push_frame(
                    TranscriptionFrame(text, "", time_now_iso8601()),
                    FrameDirection.DOWNSTREAM,
                )
        except Exception:  # noqa: BLE001
            logger.exception("T-one finalize failed")
            self._state = None
            self._buf = np.zeros(0, dtype=np.int32)
            self._epoch_reset()
