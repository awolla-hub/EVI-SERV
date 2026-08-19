"""The drawing channel: she says what she is drawing, and the phone watches it appear.

SHAPE OF THE THING. She emits `[DRAW: подсолнух]` mid-sentence. The marker is stripped before TTS,
so she never reads it aloud, and it costs nothing to produce — which is why the stage can be claimed
on the phone BEFORE any artist has drawn a single line. A second, separate call to the model then
streams a scene of SOLIDS back — an ellipsoid, a torus, a segment repeated eight times about an axis
— and every time a few more complete bodies have arrived they are validated and pushed, and the phone
scatters points over their surfaces.

WHY A SECOND CALL AND NOT A TOOL. The voice turn must not wait. A drawing takes 8-20 s to generate;
her sentence takes two. Running the artist as a detached task means the drawing lands WHILE she is
talking, and if it fails she has still already answered. The two never share a client, a pool entry
or a context.

WHAT THE PHONE IS TRUSTED WITH: nothing. Every frame leaving here has been through
`scene_compile.compile_scene`, so it is a fixed set of primitive kinds with clamped finite numbers.
Nothing she typed crosses the wire, and nothing is ever executed.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

import aiohttp
from loguru import logger

import scene_compile

# Push a frame at most this often. Faster looks no better — the assembly animation on the phone runs
# ~1.5 s per batch — and each frame costs a JSON encode of the whole cumulative scene.
_TICK = float(os.environ.get("EDIT_DRAW_TICK", "1.1"))
# A drawing that has not finished by now is not going to. Measured: a full scene is 8-20 s.
_DEADLINE = float(os.environ.get("EDIT_DRAW_TIMEOUT", "70"))
# The receive side has its own limit; an oversized frame throws there rather than truncating, so the
# scene is closed as `capped` instead.
_MAX_FRAME = 96 * 1024
_MAX_SOLIDS = 96

# THE SECOND ARTIST — see mesh3d.py. Off unless a python for it exists, so a server without the
# extra venv behaves exactly as before.
_MESH_PY = os.environ.get("EDIT_MESH_PY", "/opt/pyatnitsa/.venv-mesh/bin/python")
_MESH_SCRIPT = os.environ.get("EDIT_MESH_SCRIPT", "/opt/pyatnitsa/mesh3d.py")
# Generous, because it borrows a free shared GPU and may sit in a queue. Nothing waits on it: the
# solid drawing is already on screen the whole time.
_MESH_DEADLINE = float(os.environ.get("EDIT_MESH_TIMEOUT", "180"))
_MESH_ON = os.environ.get("EDIT_MESH", "1") not in ("0", "false", "")


class DrawSession:
    """One drawing, from marker to settled picture.

    Cancellable: a barge-in kills it, because a picture for a sentence he interrupted is worse than
    no picture — it would still be assembling itself while she answers something else.
    """

    def __init__(self, subject: str, push, proxy_url: str, api_key: str = "") -> None:
        self.subject = subject.strip()[:120]
        self._push = push                        # async (dict) -> None
        self._url = proxy_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self.id = uuid.uuid4().hex[:6]
        self.seed = int.from_bytes(self.id.encode()[:4], "little") & 0x7FFFFFFF
        self._task: asyncio.Task | None = None
        self._mesh_task: asyncio.Task | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        # A SECOND, INDEPENDENT ARTIST. It is not awaited and its failure is not reported: the
        # solid scene is the drawing, and this one is an improvement that may or may not arrive.
        if _MESH_ON and os.path.exists(_MESH_PY) and os.path.exists(_MESH_SCRIPT):
            self._mesh_task = asyncio.create_task(self._mesh())

    def cancel(self) -> None:
        for t in (self._task, self._mesh_task):
            if t and not t.done():
                t.cancel()

    # -- the run -----------------------------------------------------------

    async def _open(self) -> None:
        """Claim the stage. Synchronous with the marker, no model involved — this is the frame that
        makes the drawing feel instant even though the artist takes fifteen seconds."""
        await self._push({
            "type": "visual", "phase": "open", "id": self.id,
            "title": self.subject, "kit": "scene", "seed": self.seed,
        })

    async def _done(self, status: str, scene: dict | None) -> None:
        msg = {"type": "visual", "phase": "done", "id": self.id, "status": status}
        if scene:
            msg["solids"] = scene["solids"]
            msg["spin"] = scene["spin"]
            msg["dropped"] = scene["dropped"]
        await self._push(msg)

    async def _run(self) -> None:
        t0 = time.monotonic()
        raw = ""
        sent_solids = -1
        last_tick = 0.0
        scene: dict | None = None
        try:
            await self._open()
            headers = {"Content-Type": "application/json"}
            if self._key:
                headers["Authorization"] = f"Bearer {self._key}"
            body = {
                "model": "draw", "stream": True, "edit_mode": "draw",
                "messages": [{"role": "user", "content": self.subject}],
            }
            timeout = aiohttp.ClientTimeout(total=_DEADLINE)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(self._url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning("[draw] upstream {} for «{}»", resp.status, self.subject)
                        await self._done("upstream", None)
                        return
                    async for line in resp.content:
                        chunk = _sse_text(line)
                        if chunk is None:
                            continue
                        raw += chunk
                        now = time.monotonic()
                        if now - last_tick < _TICK:
                            continue
                        last_tick = now
                        scene = await asyncio.to_thread(
                            scene_compile.compile_scene, raw, _MAX_SOLIDS)
                        n = len(scene["solids"])
                        # Only speak when there is something new to say: a frame that repeats the
                        # previous list makes the phone resample its whole point cloud for nothing.
                        if n > sent_solids and n > 0:
                            if await self._emit(scene):
                                sent_solids = n
            scene = await asyncio.to_thread(scene_compile.compile_scene, raw, _MAX_SOLIDS)
            logger.info("[draw] «{}» {} тел, {} отброшено, {:.1f}s, {} B",
                        self.subject, len(scene["solids"]), scene["dropped"],
                        time.monotonic() - t0, len(raw))
            await self._done(scene["status"], scene)
        except asyncio.CancelledError:
            await self._done("cancelled", None)
            raise
        except asyncio.TimeoutError:
            logger.warning("[draw] «{}» timed out after {:.0f}s", self.subject, _DEADLINE)
            await self._done("upstream", scene if scene and scene["solids"] else None)
        except Exception as e:  # noqa: BLE001
            logger.warning("[draw] «{}» failed: {}", self.subject, e)
            await self._done("upstream", None)

    async def _mesh(self) -> None:
        """Text to a surface, in a subprocess, on its own clock.

        A SUBPROCESS AND NOT AN IMPORT, deliberately: the mesh path pulls in numpy, trimesh and a
        gradio client, and none of that belongs in the address space of the process that carries
        her voice. It also gives the one thing an unreliable free service needs — a hard kill.

        The frame it sends is an ordinary cumulative `ops` on the SAME id, so the phone treats it
        as a new beat of the same drawing and morphs the cloud from the bodies into the surface.
        """
        t0 = time.monotonic()
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                _MESH_PY, _MESH_SCRIPT, "--emit", self.subject,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=_MESH_DEADLINE)
            except asyncio.TimeoutError:
                proc.kill()
                logger.info("[mesh] «{}» gave up after {:.0f}s — the solids stand",
                            self.subject, _MESH_DEADLINE)
                return
            b64 = out.decode("ascii", "ignore").strip()
            if not b64:
                tail = err.decode("utf-8", "ignore").strip().splitlines()[-1:] or [""]
                logger.info("[mesh] «{}» no cloud — {}", self.subject, tail[0][:160])
                return
            msg = {"type": "visual", "phase": "ops", "id": self.id, "pts": b64}
            size = len(json.dumps(msg).encode())
            if size > _MAX_FRAME:
                logger.warning("[mesh] cloud {} B over the {} B cap — dropped", size, _MAX_FRAME)
                return
            await self._push(msg)
            logger.info("[mesh] «{}» landed in {:.1f}s, {} B", self.subject,
                        time.monotonic() - t0, size)
        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.kill()
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("[mesh] «{}» failed: {}: {}", self.subject, type(e).__name__, e)

    async def _emit(self, scene: dict) -> bool:
        """One cumulative frame of bodies. Returns False if it was too big to send, which closes the
        scene rather than truncating it — a half-sent list would decode into a broken object."""
        msg = {"type": "visual", "phase": "ops", "id": self.id,
               "solids": scene["solids"], "spin": scene["spin"]}
        size = len(json.dumps(msg, ensure_ascii=False).encode())
        if size > _MAX_FRAME:
            logger.warning("[draw] frame {} B over cap — closing as capped", size)
            await self._done("capped", None)
            return False
        await self._push(msg)
        return True


def _sse_text(line: bytes) -> str | None:
    """The text delta out of one SSE line, or None for keepalives and the terminator."""
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        d = json.loads(payload)
        return (d.get("choices") or [{}])[0].get("delta", {}).get("content") or None
    except Exception:  # noqa: BLE001
        return None
