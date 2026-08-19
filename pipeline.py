"""Assemble the per-connection Pipecat pipeline.

Pipeline order (per plan §8):

    transport.input()
      -> STT                (VAD-segmented GigaAM / faster-whisper)
      -> [tap: partial / user_final]
      -> user context aggregator
      -> LLM                (proxy, streaming)
      -> [tap: assistant_text]
      -> TTS                (Silero v5_ru, 24 kHz, sentence-chunked)
      -> [tap: speaking_start / speaking_end]
      -> transport.output()
      -> assistant context aggregator

Interruptions are enabled so both client ``barge_in`` and server-side VAD can
cut off the assistant mid-sentence.
"""

from __future__ import annotations

import asyncio
import os

from fastapi import WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from config import Config, SessionPersona
from config import USER_NAME, USER_NAME_GEN  # noqa: E402
from events import (AmbientVisionEngine, AppCommandFilter, CueVoiceWarmer, ClientEventTap, FillerCue, FillerVoice, PresenceEngine, ProactiveInjector, ResponseWatch, SearchFillerInjector, SemanticRecall, TypedTurnTTSGate)
from services.memory_store import get_store
from services.gigaam_stt import create_stt
from services.proxy_llm import create_llm
from services import ru_accent
from services.silero_tts import SileroTTSService
from services.fish_tts import FishTTSService, available as fish_available
from transport import PyatnitsaSerializer
from vision import VisionCoordinator, VisualMemory


def _build_stop_strategy(config: Config):
    """User-turn stop strategy: Smart Turn v3 primary, VAD-timeout fallback.

    Constructing :class:`LocalSmartTurnAnalyzerV3` loads its bundled ONNX model
    (downloaded on first run). If that import/load fails — missing deps, missing
    model, unsupported runtime — we log and fall back to the timeout endpointer
    so a Smart-Turn failure degrades gracefully instead of breaking the server.
    """
    # Smart Turn v3 is opt-in (SMART_TURN=1). It can stall turn completion on some
    # audio, so the reliable VAD-timeout endpointer is the default.
    if not config.use_smart_turn:
        # NB: the strategy's own user_speech_timeout (default 0.6 s) STACKS on the VAD stop_secs —
        # effective pause tolerance = sum. Field consensus (LiveKit/Pipecat guides) calls a long
        # VAD-only wait the #1 self-inflicted latency; 0.35+0.15=0.5 s total is the sweet spot for
        # conversational RU without clipping slow speakers (was 0.4+0.3=0.7 s).
        logger.info("VAD-timeout endpointing (stop_secs=%s + 0.12s strategy).", config.vad_stop_secs)
        return SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.12)
    try:
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
            LocalSmartTurnAnalyzerV3,
        )

        turn_analyzer = LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(stop_secs=config.vad_stop_secs),
        )
        logger.info("Smart Turn v3 semantic endpointing enabled.")
        return TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer)
    except Exception:  # noqa: BLE001 - degrade to the timeout endpointer
        logger.exception(
            "Smart Turn v3 unavailable; falling back to VAD-timeout endpointing."
        )
        return SpeechTimeoutUserTurnStopStrategy()


async def build_pipeline_task(websocket: WebSocket, config: Config) -> PipelineTask:
    """Wire up all stages for one accepted WebSocket connection.

    Async because the connect path reads SQLite four times (store open, facts, recall, seed) and a
    concurrent writer holding the lock can stall for up to the 15 s busy_timeout — that stall would
    land on the realtime event loop, i.e. on his audio.
    """
    # Build the TTS service first so the serializer can hold a reference to it
    # and route live {"type":"set_voice"} control messages to its speaker setter.
    # Loads a small context model on a worker thread; nothing awaits it, and until it is
    # ready the stress step is a no-op (see services/ru_accent.py).
    ru_accent.warm()

    # LOOP HEARTBEAT (flagged): sleeps 200 ms in a loop and reports how late it actually woke.
    # If timers in this process are being starved by blocking CPU work, the drift shows here — and
    # a starved timer is exactly what the delayed filler is.
    if os.environ.get("EDIT_LOOP_HEARTBEAT", "0") == "1":
        async def _heartbeat() -> None:
            import time as _t
            while True:
                t0 = _t.monotonic()
                await asyncio.sleep(0.2)
                drift = (_t.monotonic() - t0 - 0.2) * 1000
                if drift > 250:
                    logger.warning("[loop] цикл стоял {:.0f} мс", drift)
        asyncio.get_event_loop().create_task(_heartbeat())

    # Fish Audio S2 is only wired in when a key is present, so an unset environment is plain local
    # Silero and no outage can come from a feature nobody configured. The chain is fish -> silero,
    # one object either way.
    _TTS = FishTTSService if fish_available() else SileroTTSService
    tts = _TTS(
        repo=config.tts_silero_repo,
        speaker_pack=config.tts_silero_speaker_pack,
        speaker=config.tts_speaker,
        sample_rate=config.tts_sample_rate,
    )

    # Live-editable persona (name / character / voice). The serializer mutates it and rebuilds the
    # system message in place when the client sends set_voice / set_persona, so a voice or name
    # change takes effect on the very next turn without reconnecting.
    persona = SessionPersona(voice=config.tts_speaker)
    tts._persona = persona   # so FishTTSService.run_tts can honour the live fish/silero engine switch
    # Cross-session memory: the last few exchanges are seeded into the fresh context so she
    # remembers earlier conversations («помнил разговоры между сессиями»).
    # First call opens the DB and runs the CREATE/ALTER migrations — off the loop with the reads.
    memory = await asyncio.to_thread(get_store)
    # Shared ambient VisualMemory (item #8/#17): the app's periodic vision_ambient push lands here and
    # the AmbientVisionEngine reads it. Empty/harmless until the app streams frames.
    visual = VisualMemory()
    # Durable «запомни X» facts (audit item #4), gated EDIT_FACTS. The always-on prompt PROMISES
    # «запомни …» works, so this must stay in step with the same flag in events.py — off, she
    # confirms and stores nothing. Folded into persona.extra_system so it (a) stays in the SINGLE
    # system message — never a 2nd system message, which the Anthropic-backed proxy would reject —
    # and (b) survives set_voice/set_persona rebuilds (transport._rebuild_prompt recomputes from
    # persona.prompt()). facts_block() is already dated and char-capped: an uncapped always-on block
    # grows with the table on EVERY turn.
    if os.environ.get("EDIT_FACTS", "1") == "1":
        _facts = await asyncio.to_thread(memory.facts_block)
        persona.extra_system = (
            "ПАМЯТЬ-ФАКТЫ: если " + USER_NAME + " просит запомнить («запомни…», «запиши, что…», «на будущее…»), "
            "добавь в ответ блок [MEMO: короткий факт от первого лица " + USER_NAME_GEN + "] — он НЕ озвучивается, "
            "а сохраняется навсегда; голосом коротко подтверди: «готово». "
            + _facts
        ).strip()
    # Cross-session recall (audit item #16), gated EDIT_RECALL. A relative-time digest of recent
    # non-trivial exchanges, folded into the SAME single system message via extra_system so
    # «помнишь, вчера…» lands and it survives set_voice/set_persona rebuilds.
    #
    # ДАЙДЖЕСТ УБРАН ИЗ СИСТЕМНОГО ПРОМПТА, и это не про экономию текста. Тёплый клиент в шиме
    # ключуется хешем системного промпта, а `recall_block()` — сводка последних реплик, то есть
    # строка, МЕНЯЮЩАЯСЯ КАЖДУЮ СЕССИЮ. Ключ каждый раз новый, тёплый клиент каждый раз мимо, и
    # первая реплика любого разговора платила холодный запуск: 4-6 с против 1,7 с — ровно то, что
    # человек называет «она долго думает, когда только начинаешь говорить».
    #
    # Содержимое не потеряно: `recent_messages` ниже отдаёт те же реплики НАСТОЯЩИМИ сообщениями
    # (24 против прежних 10), где им и место — модель видит диалог, а не пересказ диалога.
    _seed = await asyncio.to_thread(memory.recent_messages)
    context = LLMContext(
        messages=[{"role": "system", "content": persona.prompt()}] + _seed
    )

    # STT is built early so the serializer can route enroll_voice / voice_gate control messages to
    # the owner-voice gate inside it.
    stt = create_stt(config)
    # The STT's prosody analyser writes mood tags onto the persona (tone_stt only).
    if hasattr(stt, "persona"):
        stt.persona = persona

    serializer = PyatnitsaSerializer(
        tts=tts, context=context, persona=persona, stt=stt, memory=memory, visual=visual
    )

    # Silero VAD. In pipecat 1.5.0 the transport no longer owns VAD; a
    # VADProcessor placed right after transport.input() emits the
    # VADUserStarted/StoppedSpeakingFrames that the SegmentedSTTService uses to
    # slice utterances and that the user-turn aggregator uses to detect turns.
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            stop_secs=config.vad_stop_secs,
            confidence=config.vad_confidence,
            min_volume=config.vad_min_volume,
        )
    )
    vad_processor = VADProcessor(vad_analyzer=vad_analyzer)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=config.audio_in_sample_rate,
            audio_out_sample_rate=config.audio_out_sample_rate,
            serializer=serializer,
            session_timeout=None,
        ),
    )

    llm = create_llm(config, persona=persona)

    # The LLMContext (built above with the persona's system prompt) is shared by the aggregator
    # pair — it turns finalized transcripts into user turns and captures the assistant's streamed
    # reply back into that same context.
    # User-turn endpointing. Primary is Smart Turn v3 semantic detection
    # (LocalSmartTurnAnalyzerV3 -> TurnAnalyzerUserTurnStopStrategy): it feeds
    # audio/VAD/transcription frames to an on-device ONNX model that predicts
    # whether the user has actually finished their thought, instead of firing on
    # a fixed silence timeout. The analyzer downloads/loads a bundled ONNX model
    # in its constructor; if that fails for any reason we degrade gracefully to
    # the plain VAD-timeout endpointer so the loop still runs. Start detection
    # stays on the defaults (VAD + transcription), which consume the upstream
    # VADProcessor frames.
    stop_strategy = _build_stop_strategy(config)
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # Turn-start gate. The word count only applies WHILE she's speaking (idle needs 1 word);
            # it was 2 to resist her own TTS echo opening a turn, but that dropped short real replies
            # spoken over her TTS tail («иногда не слышит»). vad_min_words=1 hears them; intentional
            # interrupts still use the client's explicit {"type":"barge_in"}.
            user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(min_words=config.vad_min_words)],
                stop=[stop_strategy],
            ),
            # A 1-2 s LTE blip used to finalize a half-spoken question (default 1 s audio idle).
            audio_idle_timeout=4.0,
        ),
    )

    # Sits between the user aggregator and the LLM: intercepts the turn's
    # LLMContextFrame to inject a client JPEG into this turn's context and to
    # briefly hold endpointing while the image is in flight (see vision.py). The
    # actual text/vision model swap happens in ProxyLLMService based on whether
    # the outgoing request carries an image.
    vision_coordinator = VisionCoordinator(
        context=context,
        hold_secs=config.vision_hold_secs,
        need_photo=config.vision_need_photo,
        visual=visual,
    )

    # Shared signal: assistant-text tap sets it on the first LLM token; the delayed filler awaits it
    # and stays SILENT when the answer is fast (filler only on slow search/photo/cold turns).
    watch = ResponseWatch()
    # The filler DECIDES upstream of the LLM and SPEAKS downstream of it; this carries the line
    # between the two halves (see FillerCue).
    filler_cue = FillerCue()

    pipeline = Pipeline(
        [
            transport.input(),
            vad_processor,
            stt,
            ClientEventTap(emit_transcripts=True, memory=memory),
            context_aggregator.user(),
            vision_coordinator,
            # Per-turn semantic recall: prepend meaning-similar past messages to the user turn (item #5).
            # DECIDES here (only upstream does a context frame announce a dispatched turn — the
            # LLM consumes it) and SPEAKS from FillerVoice below the gate: a frame pushed from
            # here would sit in the LLM's queue until the answer finished, which is exactly the
            # «секунду после ответа» he complained about.
            SemanticRecall(memory, context),
            SearchFillerInjector(watch=watch, persona=persona, cue=filler_cue),
            llm,
            # Strips [OPEN: имя] from the spoken stream → urgent open_app message to the client.
            AppCommandFilter(memory, persona),
            ClientEventTap(emit_assistant_text=True, watch=watch, memory=memory),
            TypedTurnTTSGate(persona),
            FillerVoice(filler_cue),
            ProactiveInjector(persona, memory, context),
            # The living loop: she may speak on her own initiative during a lull (see PresenceEngine).
            # Sits right before TTS so its TTSSpeakFrame is synthesized like any other line.
            PresenceEngine(persona, memory, context),
            # «Живое зрение»: she may remark on what the glasses SEE, unprompted (items #8/#17).
            AmbientVisionEngine(persona, memory, visual, context),
            tts,
            # Pre-synthesizes the app's system cues in her voice (see CueVoiceWarmer).
            CueVoiceWarmer(tts, persona),
            ClientEventTap(emit_speaking=True),
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Interruptions are always enabled in pipecat 1.5.0 (the allow_interruptions
    # flag was removed); client barge_in -> InterruptionFrame and server-side VAD
    # both cut the assistant off.
    return PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=config.audio_in_sample_rate,
            audio_out_sample_rate=config.audio_out_sample_rate,
            enable_metrics=True,
            enable_usage_metrics=False,
        ),
        # A quiet-but-alive session (user idle between turns; only pings flow) must NOT be killed:
        # pipecat's default idle timeout cancelled live sessions («отключается» reports) and spammed
        # "Idle pipeline detected" warnings.
        idle_timeout_secs=None,
    )
