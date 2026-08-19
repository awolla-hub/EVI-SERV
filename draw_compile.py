"""SVG in, draw-ops out. The server is a COMPILER, not a pipe.

She draws in SVG because that is the dialect she is provably fluent in — the two artefacts that
started this feature (an Iron Man arc reactor and a cat) were both first-try, and no invented
vocabulary was going to beat that. But SVG is a large, sharp format, and the phone must never see one
byte she typed. So everything hostile is resolved HERE, in Python, where a bad number is a caught
exception rather than a crash on his phone:

  * elliptical arcs become cubics — the nastiest piece of SVG maths never enters Swift
  * transform chains compose into one 6-float matrix
  * <defs> gradients are resolved against each element's own bounding box and inlined
  * relative path commands become absolute; S/T become C/Q
  * colours normalise to #RRGGBB; every number is clamped and checked finite
  * <text>, <image>, <script>, <foreignObject>, <style>, <use>, href and <filter> are DROPPED

The app therefore decodes JSON with three op kinds and nothing else. Nothing is executed, so no
markup she emits can reach a browser engine, and a poisoned web page she read cannot do better than
change which of three primitives appear.

CONSTRAINT worth stating: this file must never import anything outside the standard library, and must
never raise. Every entry point returns a scene — a thin one, or an empty one, but a scene.
"""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET

SVG_NS = "{http://www.w3.org/2000/svg}"

# Anything not here is dropped and counted. An allowlist, never a denylist: a denylist is a promise
# that nobody will ever add a dangerous element to SVG again.
SHAPES = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}
CONTAINERS = {"svg", "g", "defs", "linearGradient", "radialGradient", "stop", "clipPath", "mask",
              "title", "desc", "metadata"}

LIMIT = 1e5          # coordinates beyond this are nonsense at any viewBox
MAX_OPS = 64
MAX_PATH_SEGS = 600


# ---------------------------------------------------------------- numbers

def _num(v, default=0.0):
    """A float, or the default. Never raises, never returns NaN or infinity."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return max(-LIMIT, min(LIMIT, f))


def _pct(v, basis, default=0.0):
    """A length that may be written as a percentage of some basis."""
    if v is None:
        return default
    s = str(v).strip()
    if s.endswith("%"):
        return _num(s[:-1], default * 100 / basis if basis else 0) / 100 * basis
    return _num(s, default)


_NUM_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _nums(s):
    return [_num(m.group()) for m in _NUM_RE.finditer(s or "")]


# ---------------------------------------------------------------- colour

_NAMED = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#008000",
    "blue": "#0000ff", "gray": "#808080", "grey": "#808080", "orange": "#ffa500",
    "yellow": "#ffff00", "brown": "#a52a2a", "pink": "#ffc0cb", "purple": "#800080",
    "cyan": "#00ffff", "magenta": "#ff00ff", "silver": "#c0c0c0", "gold": "#ffd700",
}


def _color(v):
    """#RRGGBB, or None for 'none'/unparseable. `url(#id)` is handled by the caller."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s or s in ("none", "transparent", "currentcolor"):
        return None
    if s in _NAMED:
        return _NAMED[s]
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            return "#" + "".join(c * 2 for c in h)
        if len(h) == 6:
            return "#" + h if all(c in "0123456789abcdef" for c in h) else None
        return None
    m = re.match(r"rgba?\(([^)]*)\)", s)
    if m:
        parts = _nums(m.group(1))
        if len(parts) >= 3:
            r, g, b = (max(0, min(255, int(round(p)))) for p in parts[:3])
            return "#%02x%02x%02x" % (r, g, b)
    return None


# ---------------------------------------------------------------- matrices

IDENT = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _mul(m, n):
    """m then n, in the SVG/CoreGraphics [a b c d tx ty] convention."""
    a1, b1, c1, d1, e1, f1 = m
    a2, b2, c2, d2, e2, f2 = n
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _apply(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _parse_transform(s):
    """A transform attribute into one matrix. Unknown functions are skipped, not fatal."""
    if not s:
        return IDENT
    out = IDENT
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s):
        v = _nums(args)
        if name == "translate":
            out = _mul(out, (1, 0, 0, 1, v[0] if v else 0, v[1] if len(v) > 1 else 0))
        elif name == "scale":
            sx = v[0] if v else 1
            out = _mul(out, (sx, 0, 0, v[1] if len(v) > 1 else sx, 0, 0))
        elif name == "rotate":
            a = math.radians(v[0] if v else 0)
            cos, sin = math.cos(a), math.sin(a)
            r = (cos, sin, -sin, cos, 0, 0)
            if len(v) >= 3:                      # rotate about a point
                r = _mul(_mul((1, 0, 0, 1, v[1], v[2]), r), (1, 0, 0, 1, -v[1], -v[2]))
            out = _mul(out, r)
        elif name == "matrix" and len(v) >= 6:
            out = _mul(out, tuple(v[:6]))
        elif name == "skewX":
            out = _mul(out, (1, 0, math.tan(math.radians(v[0] if v else 0)), 1, 0, 0))
        elif name == "skewY":
            out = _mul(out, (1, math.tan(math.radians(v[0] if v else 0)), 0, 1, 0, 0))
    return out


# ---------------------------------------------------------------- path data

def _arc_to_cubics(x0, y0, rx, ry, phi, large, sweep, x, y):
    """One SVG elliptical arc as a list of cubic segments.

    This is the reason the compiler exists. The endpoint-to-centre conversion is fiddly and its
    degenerate cases (zero radius, coincident endpoints, radii too small for the chord) are exactly
    the inputs that produce NaN — and a NaN that reaches a GPU path is a rendering corruption, not an
    exception. Resolving it here means the phone only ever sees cubics.
    """
    if rx == 0 or ry == 0 or (x0 == x and y0 == y):
        return [("L", x, y)]
    rx, ry = abs(rx), abs(ry)
    rad = math.radians(phi)
    cosp, sinp = math.cos(rad), math.sin(rad)
    dx2, dy2 = (x0 - x) / 2.0, (y0 - y) / 2.0
    x1 = cosp * dx2 + sinp * dy2
    y1 = -sinp * dx2 + cosp * dy2
    # Radii too small to span the chord are scaled up, per the spec's correction step.
    lam = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    denom = rx * rx * y1 * y1 + ry * ry * x1 * x1
    if denom <= 0:
        return [("L", x, y)]
    num = max(0.0, rx * rx * ry * ry - denom)
    coef = math.sqrt(num / denom) * (-1 if large == sweep else 1)
    cx1 = coef * rx * y1 / ry
    cy1 = -coef * ry * x1 / rx
    cx = cosp * cx1 - sinp * cy1 + (x0 + x) / 2.0
    cy = sinp * cx1 + cosp * cy1 + (y0 + y) / 2.0

    def angle(ux, uy, vx, vy):
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        if n == 0:
            return 0.0
        c = max(-1.0, min(1.0, (ux * vx + uy * vy) / n))
        a = math.acos(c)
        return -a if ux * vy - uy * vx < 0 else a

    th0 = angle(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dth = angle((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi

    segs = []
    n = max(1, int(math.ceil(abs(dth) / (math.pi / 2))))
    step = dth / n
    k = 4.0 / 3.0 * math.tan(step / 4.0)
    th = th0
    px, py = x0, y0
    for _ in range(n):
        th2 = th + step
        # Ellipse point and derivative, rotated back into user space.
        def pt(t):
            ex, ey = rx * math.cos(t), ry * math.sin(t)
            return (cosp * ex - sinp * ey + cx, sinp * ex + cosp * ey + cy)

        def dv(t):
            ex, ey = -rx * math.sin(t), ry * math.cos(t)
            return (cosp * ex - sinp * ey, sinp * ex + cosp * ey)

        ex, ey = pt(th2)
        d1x, d1y = dv(th)
        d2x, d2y = dv(th2)
        segs.append(("C", px + k * d1x, py + k * d1y, ex - k * d2x, ey - k * d2y, ex, ey))
        px, py, th = ex, ey, th2
    return segs


_CMD_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])([^MmLlHhVvCcSsQqTtAaZz]*)")


def _parse_path(d):
    """Path data into absolute M/L/C/Q/Z segments. Malformed tails are truncated, never fatal."""
    segs = []
    cx = cy = sx = sy = 0.0
    prev_c = prev_q = None
    for cmd, argstr in _CMD_RE.findall(d or ""):
        a = _nums(argstr)
        rel = cmd.islower()
        c = cmd.upper()
        i = 0
        first = True
        while True:
            if len(segs) > MAX_PATH_SEGS:
                return segs
            if c == "Z":
                segs.append(("Z",))
                cx, cy = sx, sy
                break
            need = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}[c]
            if i + need > len(a):
                break
            v = a[i:i + need]
            i += need
            if c == "M":
                x, y = (cx + v[0], cy + v[1]) if rel else (v[0], v[1])
                if first:
                    segs.append(("M", x, y)); sx, sy = x, y
                else:
                    segs.append(("L", x, y))       # extra pairs after M are implicit lineto
                cx, cy = x, y
                prev_c = prev_q = None
            elif c in ("L", "H", "V"):
                if c == "L":
                    x, y = (cx + v[0], cy + v[1]) if rel else (v[0], v[1])
                elif c == "H":
                    x, y = (cx + v[0]) if rel else v[0], cy
                else:
                    x, y = cx, (cy + v[0]) if rel else v[0]
                segs.append(("L", x, y)); cx, cy = x, y
                prev_c = prev_q = None
            elif c in ("C", "S"):
                if c == "C":
                    p = [(cx + v[k] if rel and k % 2 == 0 else cy + v[k] if rel else v[k])
                         for k in range(6)]
                    x1, y1, x2, y2, x, y = p
                else:
                    # Smooth cubic: the first control point mirrors the previous one.
                    x1, y1 = (2 * cx - prev_c[0], 2 * cy - prev_c[1]) if prev_c else (cx, cy)
                    x2, y2 = (cx + v[0], cy + v[1]) if rel else (v[0], v[1])
                    x, y = (cx + v[2], cy + v[3]) if rel else (v[2], v[3])
                segs.append(("C", x1, y1, x2, y2, x, y))
                prev_c = (x2, y2); prev_q = None
                cx, cy = x, y
            elif c in ("Q", "T"):
                if c == "Q":
                    x1, y1 = (cx + v[0], cy + v[1]) if rel else (v[0], v[1])
                    x, y = (cx + v[2], cy + v[3]) if rel else (v[2], v[3])
                else:
                    x1, y1 = (2 * cx - prev_q[0], 2 * cy - prev_q[1]) if prev_q else (cx, cy)
                    x, y = (cx + v[0], cy + v[1]) if rel else (v[0], v[1])
                segs.append(("Q", x1, y1, x, y))
                prev_q = (x1, y1); prev_c = None
                cx, cy = x, y
            elif c == "A":
                x, y = (cx + v[5], cy + v[6]) if rel else (v[5], v[6])
                segs.extend(_arc_to_cubics(cx, cy, v[0], v[1], v[2], v[3] != 0, v[4] != 0, x, y))
                cx, cy = x, y
                prev_c = prev_q = None
            first = False
            if i >= len(a):
                break
    return segs


def _rel(cmd_rel, base, v):                     # kept for readability at the call site
    return base + v if cmd_rel else v


# ---------------------------------------------------------------- geometry helpers

def _seg_points(segs):
    pts = []
    for s in segs:
        if s[0] == "M" or s[0] == "L":
            pts.append((s[1], s[2]))
        elif s[0] == "C":
            pts += [(s[1], s[2]), (s[3], s[4]), (s[5], s[6])]
        elif s[0] == "Q":
            pts += [(s[1], s[2]), (s[3], s[4])]
    return pts


def _bbox(pts):
    if not pts:
        return (0.0, 0.0, 1.0, 1.0)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(1e-6, max(xs) - min(xs)), max(1e-6, max(ys) - min(ys)))


# ---------------------------------------------------------------- gradients

def _collect_gradients(root):
    grads = {}
    for el in root.iter():
        tag = el.tag.replace(SVG_NS, "")
        if tag not in ("linearGradient", "radialGradient"):
            continue
        gid = el.get("id")
        if not gid:
            continue
        stops = []
        for st in el:
            if st.tag.replace(SVG_NS, "") != "stop":
                continue
            off = _pct(st.get("offset", "0"), 1.0, 0.0)
            style = st.get("style") or ""
            sc = st.get("stop-color") or _style_prop(style, "stop-color")
            so = st.get("stop-opacity") or _style_prop(style, "stop-opacity")
            stops.append([max(0.0, min(1.0, off)), _color(sc) or "#ffffff",
                          max(0.0, min(1.0, _num(so, 1.0)))])
        if stops:
            grads[gid] = {"tag": tag, "el": el, "stops": stops}
    return grads


def _style_prop(style, name):
    m = re.search(r"(?:^|;)\s*%s\s*:\s*([^;]+)" % re.escape(name), style or "")
    return m.group(1).strip() if m else None


def _resolve_gradient(grads, ref, bbox):
    """A paint reference into an inlined, user-space gradient. Percentages in the default
    objectBoundingBox space are resolved against THIS element's box, which is why gradients cannot be
    hoisted or shared on the wire."""
    m = re.match(r"url\(#([^)]+)\)", (ref or "").strip())
    if not m:
        return None
    g = grads.get(m.group(1))
    if not g:
        return None
    el = g["el"]
    bx, by, bw, bh = bbox
    obb = (el.get("gradientUnits", "objectBoundingBox") != "userSpaceOnUse")

    def coord(v, default, basis, origin):
        if obb:
            return origin + _pct(v, 1.0, default) * basis
        return _pct(v, basis, default)

    if g["tag"] == "linearGradient":
        return {"t": "linear",
                "x1": coord(el.get("x1"), 0.0, bw, bx), "y1": coord(el.get("y1"), 0.0, bh, by),
                "x2": coord(el.get("x2"), 1.0, bw, bx), "y2": coord(el.get("y2"), 0.0, bh, by),
                "stops": g["stops"]}
    r_basis = math.hypot(bw, bh) / math.sqrt(2)
    return {"t": "radial",
            "cx": coord(el.get("cx"), 0.5, bw, bx), "cy": coord(el.get("cy"), 0.5, bh, by),
            "r": (_pct(el.get("r"), 1.0, 0.5) * r_basis) if obb else _pct(el.get("r"), r_basis, 0.5),
            "stops": g["stops"]}


# ---------------------------------------------------------------- the compiler

def compile_svg(text, max_ops=MAX_OPS):
    """SVG text -> {"ops": [...], "status": str, "dropped": int, "w": int, "h": int}.

    Never raises. A truncated or wholly unparseable document yields an empty scene, which the app
    renders as the ignition armature — failure reads as patience rather than as breakage.
    """
    out = {"ops": [], "status": "ok", "dropped": 0, "w": 400, "h": 400}
    if not text:
        out["status"] = "invalid"
        return out
    # She sometimes prefixes a sentence or fences the block despite being told not to.
    i = text.find("<svg")
    if i < 0:
        out["status"] = "invalid"
        return out
    j = text.rfind("</svg>")
    frag = text[i:j + 6] if j > i else text[i:]
    root = None
    try:
        root = ET.fromstring(frag)
    except ET.ParseError:
        # A LIVE stream is almost always cut mid-element, so this is the normal case, not the odd
        # one: walk back to the last complete tag and close the document there. Without the trim, a
        # half-written attribute makes the whole partial scene unparseable and the picture would
        # only ever appear at the very end.
        tail = frag
        for _ in range(24):
            k = tail.rfind(">")
            if k < 0:
                break
            tail = tail[:k + 1]
            try:
                root = ET.fromstring(tail + "</svg>")
                break
            except ET.ParseError:
                tail = tail[:k]
    if root is None:
        out["status"] = "invalid"
        return out

    vb = _nums(root.get("viewBox") or "")
    if len(vb) == 4 and vb[2] > 0 and vb[3] > 0:
        vx, vy, vw, vh = vb
    else:
        vx, vy = 0.0, 0.0
        vw = _num(root.get("width"), 400) or 400
        vh = _num(root.get("height"), 400) or 400
    out["w"], out["h"] = int(round(vw)), int(round(vh))
    # Normalise the viewBox origin into the matrix so the app always works in 0,0-w,h.
    base = (1, 0, 0, 1, -vx, -vy)

    grads = _collect_gradients(root)
    ops = []
    dropped = [0]

    def walk(el, mat, inherit):
        tag = el.tag.replace(SVG_NS, "") if isinstance(el.tag, str) else ""
        if tag in ("defs", "clipPath", "mask", "title", "desc", "metadata",
                   "linearGradient", "radialGradient", "stop"):
            return
        if tag not in SHAPES and tag not in CONTAINERS:
            dropped[0] += 1            # text, image, script, use, foreignObject, filter, style…
            return
        m = _mul(mat, _parse_transform(el.get("transform")))
        style = el.get("style") or ""
        attrs = dict(inherit)
        for k in ("fill", "stroke", "stroke-width", "opacity", "fill-opacity", "stroke-opacity",
                  "stroke-linecap"):
            v = el.get(k) or _style_prop(style, k)
            if v is not None:
                attrs[k] = v
        if tag in ("svg", "g"):
            for child in el:
                walk(child, m, attrs)
            return
        if len(ops) >= max_ops:
            dropped[0] += 1
            return
        op = _shape(tag, el, m, attrs, grads)
        if op is not None:
            ops.append(op)
        else:
            dropped[0] += 1

    walk(root, base, {})

    # THE ROOM IS HERS, NOT THE MODEL'S. A near-full-bleed rect is the model painting a background,
    # which is what makes generated art read as a picture pasted onto the screen instead of something
    # drawn on it. Dropping it is the single most load-bearing art-direction rule here.
    area = vw * vh
    kept = []
    for op in ops:
        if op["k"] == "rect" and op.get("w", 0) * op.get("h", 0) >= 0.92 * area:
            dropped[0] += 1
            continue
        if op["k"] == "ellipse" and math.pi * op.get("rx", 0) * op.get("ry", 0) >= 0.92 * area:
            dropped[0] += 1
            continue
        kept.append(op)
    out["ops"] = kept
    out["dropped"] = dropped[0]
    if not kept:
        out["status"] = "invalid"
    elif len(kept) < 6:
        out["status"] = "thin"
    return out


def _shape(tag, el, m, attrs, grads):
    """One shape element into one op, or None if it carries no paint."""
    fill_ref = attrs.get("fill", "#000000")
    stroke_ref = attrs.get("stroke")
    sw = _num(attrs.get("stroke-width"), 1.0)
    alpha = max(0.0, min(1.0, _num(attrs.get("opacity"), 1.0)))
    cap = (attrs.get("stroke-linecap") or "butt").strip()

    op = {"k": None}
    pts = []

    if tag == "path":
        segs = _parse_path(el.get("d"))
        if not segs:
            return None
        op["k"] = "path"
        op["d"] = [list(s) for s in segs]
        pts = _seg_points(segs)
    elif tag in ("circle", "ellipse"):
        cx, cy = _num(el.get("cx")), _num(el.get("cy"))
        if tag == "circle":
            rx = ry = _num(el.get("r"))
        else:
            rx, ry = _num(el.get("rx")), _num(el.get("ry"))
        if rx <= 0 or ry <= 0:
            return None
        op.update(k="ellipse", cx=cx, cy=cy, rx=rx, ry=ry)
        pts = [(cx - rx, cy - ry), (cx + rx, cy + ry)]
    elif tag == "rect":
        x, y = _num(el.get("x")), _num(el.get("y"))
        w, h = _num(el.get("width")), _num(el.get("height"))
        if w <= 0 or h <= 0:
            return None
        rx = _num(el.get("rx"), _num(el.get("ry"), 0.0))
        op.update(k="rect", x=x, y=y, w=w, h=h, rx=max(0.0, min(min(w, h) / 2, rx)))
        pts = [(x, y), (x + w, y + h)]
    elif tag == "line":
        x1, y1 = _num(el.get("x1")), _num(el.get("y1"))
        x2, y2 = _num(el.get("x2")), _num(el.get("y2"))
        op["k"] = "path"
        op["d"] = [["M", x1, y1], ["L", x2, y2]]
        pts = [(x1, y1), (x2, y2)]
        fill_ref = None                      # a line has no interior
    elif tag in ("polyline", "polygon"):
        v = _nums(el.get("points"))
        if len(v) < 4:
            return None
        d = [["M", v[0], v[1]]] + [["L", v[k], v[k + 1]] for k in range(2, len(v) - 1, 2)]
        if tag == "polygon":
            d.append(["Z"])
        op["k"] = "path"
        op["d"] = d
        pts = [(v[k], v[k + 1]) for k in range(0, len(v) - 1, 2)]
        if tag == "polyline":
            fill_ref = None
    else:
        return None

    box = _bbox(pts)
    fill_grad = _resolve_gradient(grads, fill_ref, box)
    stroke_grad = _resolve_gradient(grads, stroke_ref, box)
    fill = None if fill_grad else _color(fill_ref)
    stroke = None if stroke_grad else _color(stroke_ref)
    if fill is None and stroke is None and not fill_grad and not stroke_grad:
        return None                          # invisible; not worth a frame

    if fill_grad:
        op["grad"] = fill_grad
    elif fill:
        op["fill"] = fill
    if stroke_grad:
        op["sgrad"] = stroke_grad
        op["sw"] = max(0.05, sw)
    elif stroke:
        op["stroke"] = stroke
        op["sw"] = max(0.05, sw)
    if cap in ("round", "square"):
        op["cap"] = cap
    op["alpha"] = alpha
    if m != IDENT:
        op["m"] = [round(v, 5) for v in m]
    return op


# ---------------------------------------------------------------- CLI

if __name__ == "__main__":
    import sys
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8", errors="replace") as f:
            scene = compile_svg(f.read())
        sys.stderr.write("%-28s ops=%-3d dropped=%-3d %s\n"
                         % (path.split("/")[-1], len(scene["ops"]), scene["dropped"],
                            scene["status"]))
        print(json.dumps(scene, separators=(",", ":"), ensure_ascii=False))
