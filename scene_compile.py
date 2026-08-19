"""A streamed scene of solids in, a validated list out.

She now describes BODIES rather than markup — an ellipsoid, a torus, a segment repeated eight times
about an axis — because a flat vector format has no notion of a thing with a far side, and it showed:
a cat came back as an ellipse, a circle and two triangles.

This module does the same job `draw_compile` did for SVG, and for the same reason: the phone must
never see a byte she typed. It pulls COMPLETE objects out of a half-arrived JSON array by brace
matching (a partial stream is never valid JSON, so `json.loads` on the whole thing is useless until
the very end), then clamps every field to a range where nothing downstream can be surprised.

CONSTRAINT: standard library only, and it must never raise. Every entry point returns a scene — a
thin one, or an empty one, but a scene.
"""
from __future__ import annotations

import json
import math

MAX_SOLIDS = 96
LIMIT = 2000.0          # coordinates beyond this are nonsense at any scale
MAX_PATH = 24

KINDS = {"sphere", "ellipsoid", "torus", "ring", "disc", "cone", "cyl", "cylinder",
         "tube", "box", "helix"}


def _f(v, d=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    if not math.isfinite(x):
        return d
    return max(-LIMIT, min(LIMIT, x))


def _vec(v, d=(0.0, 0.0, 0.0)):
    if not isinstance(v, (list, tuple)) or len(v) < 3:
        return list(d)
    return [_f(v[0]), _f(v[1]), _f(v[2])]


def _pos(v, d):
    """A radius-like number: never zero or negative. A zero radius is an invisible body that would
    still consume part of the phone's point budget."""
    x = abs(_f(v, d))
    return d if x < 0.01 else min(x, LIMIT)


def _solid(raw):
    """One validated body, or None."""
    if not isinstance(raw, dict):
        return None
    k = str(raw.get("k", "")).lower().strip()
    if k not in KINDS:
        return None
    out = {"k": k, "c": _vec(raw.get("c"))}
    axis = _vec(raw.get("axis"), (0, 0, 1))
    if axis == [0, 0, 0]:
        axis = [0, 0, 1]

    if k == "sphere":
        out["r"] = _pos(raw.get("r"), 10)
    elif k == "ellipsoid":
        r = raw.get("r")
        # `r` may be one number or three; she writes both, and rejecting either would be a rule she
        # has to remember rather than a shape we can simply accept.
        if isinstance(r, (list, tuple)) and len(r) >= 3:
            out["r"] = [_pos(r[0], 10), _pos(r[1], 10), _pos(r[2], 10)]
        else:
            out["r"] = _pos(r, 10)
    elif k == "torus":
        out["R"] = _pos(raw.get("R", raw.get("bigR")), 60)
        out["r"] = _pos(raw.get("r"), 8)
        out["axis"] = axis
    elif k in ("ring", "disc"):
        r1 = _pos(raw.get("r1", raw.get("r")), 60)
        out["r1"] = r1
        out["r0"] = min(abs(_f(raw.get("r0"))), r1 * 0.98)
        out["axis"] = axis
    elif k == "cone":
        out["r"] = _pos(raw.get("r"), 20)
        out["h"] = _pos(raw.get("h"), 40)
        out["axis"] = axis
    elif k in ("cyl", "cylinder"):
        r0 = _pos(raw.get("r0", raw.get("r")), 12)
        out["r0"] = r0
        out["r1"] = _pos(raw.get("r1", r0), r0)
        out["h"] = _pos(raw.get("h"), 40)
        out["axis"] = axis
    elif k == "tube":
        pts = raw.get("p", raw.get("path"))
        if not isinstance(pts, (list, tuple)):
            return None
        path = [_vec(p) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 3]
        if len(path) < 2:
            return None
        out["p"] = path[:MAX_PATH]
        r0 = _pos(raw.get("r0", raw.get("r")), 8)
        out["r0"] = r0
        out["r1"] = _pos(raw.get("r1", raw.get("r")), r0)
    elif k == "box":
        s = _vec(raw.get("s", raw.get("size")), (20, 20, 20))
        if min(abs(v) for v in s) < 0.01:
            return None
        out["s"] = [abs(v) for v in s]
    elif k == "helix":
        out["R"] = _pos(raw.get("R"), 50)
        out["r"] = _pos(raw.get("r"), 4)
        out["turns"] = min(24.0, _pos(raw.get("turns"), 3))
        out["h"] = _f(raw.get("h"), 40)
        out["axis"] = axis

    rep = int(_f(raw.get("rep"), 1))
    if rep > 1:
        out["rep"] = max(1, min(64, rep))
        out["repAxis"] = _vec(raw.get("repAxis"), tuple(axis))
    d = _f(raw.get("d"), 1)
    if abs(d - 1) > 0.01:
        out["d"] = max(0.15, min(4.0, d))
    return out


def _complete_objects(text):
    """Every COMPLETE `{...}` inside the first `"solids": [` array of a partial stream.

    A live stream is never valid JSON, so this is brace matching rather than parsing: track depth,
    ignore braces inside strings, honour backslash escapes, and hand back each object as soon as its
    closing brace arrives. Anything still open is simply not yielded yet.
    """
    i = text.find('"solids"')
    if i < 0:
        return []
    i = text.find("[", i)
    if i < 0:
        return []
    out = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for j in range(i + 1, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = j
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:j + 1])
                start = -1
                if len(out) >= MAX_SOLIDS:
                    break
        elif ch == "]" and depth == 0:
            break
    return out


def compile_scene(text, max_solids=MAX_SOLIDS):
    """Partial or complete model output -> {"solids": [...], "spin": float, "status": str,
    "dropped": int}. Never raises."""
    out = {"solids": [], "spin": 2.2, "status": "ok", "dropped": 0}
    if not text:
        out["status"] = "invalid"
        return out
    # She occasionally fences the block or prefixes a sentence despite being told not to.
    i = text.find("{")
    if i < 0:
        out["status"] = "invalid"
        return out
    body = text[i:]

    m = body.find('"spin"')
    if m >= 0:
        seg = body[m + 6:m + 24].lstrip(": ")
        num = ""
        for ch in seg:
            if ch in "-0123456789.":
                num += ch
            elif num:
                break
        if num:
            out["spin"] = max(0.0, min(8.0, _f(num, 2.2)))

    dropped = 0
    for chunk in _complete_objects(body):
        try:
            raw = json.loads(chunk)
        except Exception:  # noqa: BLE001
            dropped += 1
            continue
        s = _solid(raw)
        if s is None:
            dropped += 1
            continue
        out["solids"].append(s)
        if len(out["solids"]) >= max_solids:
            break

    out["dropped"] = dropped
    if not out["solids"]:
        out["status"] = "invalid"
    elif len(out["solids"]) < 4:
        out["status"] = "thin"
    return out


if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as fh:
            scene = compile_scene(fh.read())
        sys.stderr.write("%-24s solids=%-3d dropped=%-3d %s\n"
                         % (path.split("/")[-1], len(scene["solids"]), scene["dropped"],
                            scene["status"]))
        print(json.dumps(scene, separators=(",", ":"), ensure_ascii=False))
