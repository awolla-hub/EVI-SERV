"""FastAPI entrypoint for the Пятница Realtime server.

One persistent WebSocket per session at ``/ws``; a fresh Pipecat pipeline runs
for the lifetime of each connection. ``/health`` is a cheap liveness probe for
Docker / nginx.

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8080
or simply:
    python server.py
"""

from __future__ import annotations

import asyncio
import hmac
import os
import sys
from contextlib import suppress

from dotenv import load_dotenv

# Load /opt/pyatnitsa/.env into os.environ BEFORE importing config/pipeline/events. The EDIT_*/FISH_*
# feature flags are read via os.environ.get() at those modules' IMPORT time, so the process
# environment must carry the .env values first — otherwise a flag set in .env is silently ignored and
# the feature stays at its default. (pydantic Config reads the .env file separately for its own typed
# fields; this is what makes the os.environ.get() feature flags honour .env too.)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import uvicorn  # noqa: E402 — must follow load_dotenv so imported modules see the .env
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from loguru import logger  # noqa: E402
from starlette.websockets import WebSocketState  # noqa: E402

from config import load_config  # noqa: E402
from pipeline import build_pipeline_task  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402

config = load_config()

# Route pipecat's loguru logs at the configured level.
logger.remove()
logger.add(sys.stderr, level=config.log_level)

# An unauthenticated /ws hands anyone who learns the URL a free LLM *and* the owner's conversation
# history (it rides in the hello reply), so say so once, loudly, at boot.
if not config.auth_token:
    logger.warning(
        "AUTH_TOKEN is NOT set — /ws accepts ANY client (free LLM + conversation history). "
        "Set AUTH_TOKEN in .env and send it from the app as 'Authorization: Bearer <token>'."
    )

app = FastAPI(title="Пятница Realtime", version="0.1.0")


def _authorized(websocket: WebSocket) -> bool:
    """Constant-time ``Authorization: Bearer <token>`` check. True when no token is configured."""
    expected = config.auth_token
    if not expected:
        return True
    scheme, _, presented = (websocket.headers.get("authorization") or "").partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


@app.on_event("startup")
async def prewarm_models() -> None:
    """Load STT + TTS models once at boot (they're cached across sessions), so the first
    utterance of the first session doesn't pay the 5-15 s GigaAM + Silero cold load."""
    import asyncio

    async def _warm() -> None:
        # STT and TTS warm INDEPENDENTLY: the configured STT backend may not be installed
        # (gigaam is optional in requirements.txt), and one ImportError used to skip the TTS
        # prewarm too — every first reply then paid the Silero cold load.
        try:
            import torch

            torch.set_num_threads(4)  # all cores for STT/TTS — 2 halved GigaAM speed
            from services.gigaam_stt import create_stt

            stt = create_stt(config)
            if hasattr(stt, "_ensure_model"):
                await stt._ensure_model()  # noqa: SLF001 - deliberate prewarm
            logger.info("STT prewarm complete.")
        except Exception:  # noqa: BLE001 - prewarm must never block serving
            logger.exception("STT prewarm failed (sessions will lazy-load)")

        try:
            from services.silero_tts import SileroTTSService

            tts = SileroTTSService(
                repo=config.tts_silero_repo,
                speaker_pack=config.tts_silero_speaker_pack,
                speaker=config.tts_speaker,
                sample_rate=config.tts_sample_rate,
            )
            await tts._ensure_model()  # noqa: SLF001
            logger.info("TTS prewarm complete.")
        except Exception:  # noqa: BLE001 - prewarm must never block serving
            logger.exception("TTS prewarm failed (sessions will lazy-load)")

    asyncio.get_event_loop().create_task(_warm())


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "stt": config.stt_backend, "model": config.model}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Handle one realtime voice session."""
    if not _authorized(websocket):
        # Rejected BEFORE accept: no pipeline is built and no hello reply (which carries the
        # owner's recent history) can ever be sent. The token itself is never logged.
        logger.warning("WebSocket rejected — bad or missing token: {}", websocket.client)
        await websocket.close(code=1008)
        return
    await websocket.accept()
    logger.info("WebSocket connected: {}", websocket.client)

    run_task: "asyncio.Task | None" = None
    watchdog: "asyncio.Task | None" = None
    try:
        # Awaited: building the pipeline reads SQLite (memory seed, facts, recall digest) and those
        # reads inherit PRAGMA busy_timeout=15000, so they run in a worker thread. Doing them inline
        # would let a concurrent writer stall the realtime audio loop for up to 15 s at connect.
        task = await build_pipeline_task(websocket, config)
        runner = PipelineRunner(handle_sigint=False)

        # The runner alone is not a reliable end-of-session signal: when a client vanishes
        # without a `bye` (backgrounded app, dropped LTE), runner.run() has been observed to
        # never return, leaving the whole pipeline — including the always-on presence loop —
        # running forever against a dead socket. Racing it against an explicit disconnect
        # watchdog guarantees teardown, so a session can never outlive its WebSocket.
        run_task = asyncio.create_task(runner.run(task))
        watchdog = asyncio.create_task(_await_disconnect(websocket))
        done, _ = await asyncio.wait(
            {run_task, watchdog}, return_when=asyncio.FIRST_COMPLETED
        )
        if run_task not in done:
            logger.info("Client disconnected — cancelling pipeline: {}", websocket.client)
            await task.cancel()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: {}", websocket.client)
    except Exception:  # noqa: BLE001 - never let one session crash the server
        logger.exception("Session error")
    finally:
        for pending in (watchdog, run_task):
            if pending is not None and not pending.done():
                pending.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pending
        logger.info("Session closed: {}", websocket.client)


async def _await_disconnect(websocket: WebSocket, poll_seconds: float = 1.0) -> None:
    """Return as soon as the socket is no longer connected."""
    while websocket.client_state == WebSocketState.CONNECTED:
        await asyncio.sleep(poll_seconds)


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        # Protocol-level WS pings: detect half-dead LTE sockets server-side so a stale session
        # is torn down instead of silently eating mic audio forever.
        ws_ping_interval=15,
        ws_ping_timeout=10,
    )
