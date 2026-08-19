"""THE SECOND ARTIST: a drawing that comes back as a surface instead of as a list of bodies.

WHY. `draw.py` asks a language model for solids — an ellipsoid here, a cone there. That vocabulary
is right for machines (the arc reactor is eleven bodies and reads as an arc reactor) and wrong for
anything alive: a cat assembled from ellipsoids is a snowman, and no amount of prompt work fixes it,
because the language has no word for the shape. So a second path exists — text to image, image to
mesh, mesh to points — and the phone draws the result with exactly the renderer it already has.

IT NEVER REPLACES THE FIRST ONE, IT ARRIVES AFTER IT. The solid scene lands in eight to twenty
seconds; this takes thirty to sixty. Both run, the fast one shows first, and when the mesh lands the
phone MORPHS from one into the other. If this path fails, times out, or the free GPU it borrows is
busy, nothing is lost: the drawing is already on screen. That is what makes a free, unreliable
source safe to depend on.

WHAT IT COSTS: nothing. Both models are public Hugging Face Spaces on shared GPUs. The price is the
queue — at a busy moment a call waits, and sometimes a Space is asleep. Fine for a few drawings a
day; not a thing to hammer.

THE WIRE. Points are sent as one base64 blob, not as JSON numbers: 7600 points as `[x, y, z]`
literals is about half a megabyte and would be refused by the frame cap. Quantised, a point is nine
bytes — int16 position, int8 normal — and the whole cloud fits in the same budget as a photograph.
"""
from __future__ import annotations

import base64
import os
import struct
import tempfile

import numpy as np
from loguru import logger

# A FREE HUGGING FACE TOKEN, if there is one. Anonymous callers share a very small ZeroGPU
# allowance and hit "you have exceeded your quota" after a handful of images — the service says so
# itself and points at the tokens page. An account costs nothing and no card; the token simply
# raises the allowance. Without it everything still works, just more often not.
# It must be a Hugging Face token — `hf_...` from huggingface.co/settings/tokens, read scope. A
# key for zerogpu.ai (`zgpu-api-...`) is a DIFFERENT service and does nothing here: tried through
# every header the client offers, the Spaces still answer "quota exceeded".
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

# The two Spaces. Both are public.
SPACE_IMAGE = os.environ.get("EDIT_MESH_IMAGE_SPACE", "black-forest-labs/FLUX.1-schnell")
SPACE_MESH = os.environ.get("EDIT_MESH_SHAPE_SPACE", "tencent/Hunyuan3D-2")

# Points in the cloud. 7600 matches the phone's own budget for a solid scene, and at nine bytes a
# point it lands just inside the frame cap.
POINTS = int(os.environ.get("EDIT_MESH_POINTS", "7600"))
# How much a crease outweighs a flank when the points are handed out. Area alone spends the budget
# uniformly, so an eye socket gets as many points as a plain side; this is the knob the solid path
# always had per body (`d`) and the mesh path was missing.
DETAIL = float(os.environ.get("EDIT_MESH_DETAIL", "4.0"))

# Written for the model that will be photographed, not for a person: a plain background and one
# centred subject are what a single-image reconstruction can actually lift into three dimensions.
# THE VIEW COMES FIRST, and that is not stylistic. A single-image reconstruction can only lift
# into three dimensions what the photograph shows, and head-on shows no depth at all: the model
# invents the whole length of the animal. The same fact is already written into the renderer — she
# authors facing the camera, and the stage turns the object a quarter before showing it.
#
# Asking for the view at the END of the prompt did not work: with a Russian subject leading, the
# English view clause got diluted and the cat came out face-on every time. Leading with it, and
# naming an angle in degrees, produced a proper three-quarter on the first try.
#
# CONSTRAINT: "turned away from the camera" is the wording that was actually tested. "toward" reads
# better on paper — the face would be visible — but it is untested, and the failure mode of getting
# this wrong is silently back to head-on.
_PROMPT = ("three-quarter side view photograph of {subject}, the subject is turned 40 degrees "
           "away from the camera, full body in frame, plain white background, even studio light, "
           "sharp focus, no text, no watermark")


def _client(space: str):
    from gradio_client import Client
    # CONSTRAINT: the parameter is `token`, not `hf_token`. The older name is what everyone
    # reaches for and it raises a TypeError on gradio_client 2.x — silently turning an
    # authenticated call into no call at all if the error is swallowed.
    return Client(space, verbose=False, token=HF_TOKEN) if HF_TOKEN \
        else Client(space, verbose=False)


def _first_file(result, suffixes: tuple[str, ...]) -> str | None:
    """Gradio returns tuples of dicts of paths, and the shape differs per Space. Walk it."""
    stack = [result]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if x.lower().endswith(suffixes) and os.path.exists(x):
                return x
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return None


def render_image(subject: str) -> str | None:
    """Text to a reference photograph. Four steps — this is the turbo model, and more steps buy
    nothing a reconstruction can see."""
    r = _client(SPACE_IMAGE).predict(
        prompt=_PROMPT.format(subject=subject), seed=0, randomize_seed=True,
        width=1024, height=1024, num_inference_steps=4, api_name="/infer")
    return _first_file(r, (".png", ".jpg", ".jpeg", ".webp"))


def render_mesh(image_path: str) -> str | None:
    """Photograph to a white mesh. No texture is requested: the phone draws her two pigments, so a
    textured mesh would cost three times as much for something never looked at."""
    from gradio_client import handle_file
    r = _client(SPACE_MESH).predict(
        caption="", image=handle_file(image_path),
        mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
        steps=30, guidance_scale=5.0, api_name="/shape_generation")
    return _first_file(r, (".glb", ".obj", ".ply"))


def sample_points(mesh_path: str, n: int = POINTS, detail: float = DETAIL) -> bytes:
    """A mesh to a quantised point cloud, weighted by curvature.

    CURVATURE, CHEAPLY: the angle between a face and each of its neighbours. A crease — an eye
    socket, the edge of an ear, the line where a leg meets the body — has large dihedral angles; a
    flank has almost none. One pass over the adjacency list, no ball queries, no per-vertex solve,
    which is what lets this run on a server with no GPU and four cores.
    """
    import trimesh

    m = trimesh.load(mesh_path, force="mesh")
    m.fix_normals()
    areas = m.area_faces.astype(np.float64)

    curv = np.zeros(len(m.faces))
    cnt = np.zeros(len(m.faces))
    pairs = m.face_adjacency
    ang = np.abs(m.face_adjacency_angles)
    np.add.at(curv, pairs[:, 0], ang); np.add.at(cnt, pairs[:, 0], 1)
    np.add.at(curv, pairs[:, 1], ang); np.add.at(cnt, pairs[:, 1], 1)
    curv = np.where(cnt > 0, curv / np.maximum(cnt, 1), 0.0)
    # Normalised against a high percentile rather than the maximum: one malformed triangle would
    # otherwise own the whole budget and starve everything else.
    scale = float(np.percentile(curv, 97)) or 1.0
    curv = np.clip(curv / scale, 0, 1)

    w = areas * (1.0 + detail * curv)      # area keeps the silhouette continuous,
    w = w / w.sum()                        # curvature buys the detail its extra points

    rng = np.random.default_rng(7)
    fid = rng.choice(len(m.faces), size=n, p=w)
    tri = m.triangles[fid]
    u, v = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])
    nrm = m.face_normals[fid]

    lo, hi = pts.min(0), pts.max(0)
    pts = (pts - (lo + hi) / 2) * (300.0 / max(float((hi - lo).max()), 1e-6))
    nrm = nrm / np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6, None)

    # int16 position at 0.01 of a unit, int8 normal at about half a degree — far finer than either
    # is looked at, and nine bytes a point instead of the forty JSON would spend.
    out = struct.pack("<I", n)
    out += np.clip(pts * 100, -32000, 32000).astype("<i2").tobytes()
    out += np.clip(nrm * 127, -127, 127).astype("<i1").tobytes()
    return out


def build(subject: str) -> str | None:
    """The whole path, text to a base64 cloud. Returns None on any failure — the caller already has
    a drawing on screen and must not be made to care why this one did not arrive."""
    img = mesh = None
    try:
        img = render_image(subject)
        if not img:
            logger.warning("[mesh] «{}»: no image came back", subject)
            return None
        mesh = render_mesh(img)
        if not mesh:
            logger.warning("[mesh] «{}»: no mesh came back", subject)
            return None
        blob = sample_points(mesh)
        logger.info("[mesh] «{}» ok — {} points, {} B on the wire",
                    subject, POINTS, len(blob) * 4 // 3)
        return base64.b64encode(blob).decode()
    except Exception as e:  # noqa: BLE001
        logger.warning("[mesh] «{}» failed: {}: {}", subject, type(e).__name__, e)
        return None


if __name__ == "__main__":
    import sys, time

    # `--emit <subject>` prints the base64 cloud and NOTHING else on stdout — this is the mode
    # `draw.py` runs as a subprocess. Loguru writes to stderr, so the two never mix, and a failure
    # is simply empty output rather than a parse error at the other end.
    if len(sys.argv) > 2 and sys.argv[1] == "--emit":
        blob = build(" ".join(sys.argv[2:]))
        if blob:
            sys.stdout.write(blob)
        raise SystemExit(0 if blob else 1)

    subject = sys.argv[1] if len(sys.argv) > 1 else "сидящий кот"
    t0 = time.monotonic()
    b64 = build(subject)
    print(f"{'OK' if b64 else 'FAILED'} in {time.monotonic() - t0:.1f}s"
          + (f", {len(b64)} b64 chars" if b64 else ""))
    if b64 and len(sys.argv) > 2:
        with open(sys.argv[2], "wb") as f:
            f.write(base64.b64decode(b64))
        print("wrote", sys.argv[2])
