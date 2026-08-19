"""Russian stress placement for the TTS text, one step before synthesis.

WHY THIS EXISTS
---------------
Silero already accepts stress marks and already places them itself
(``put_accent=True``), but its placement is a flat dictionary lookup: it cannot
see the sentence. So every homograph is a coin flip, and Russian is full of them
in exactly the sentences an assistant says out loud — «за+мок» / «зам+ок»,
«сто+ит» / «стои+т», «бо+льшая» / «больша+я», «до+ма» / «дома+», and every
numeral that inflects. One wrong stress per reply is the difference between a
voice that sounds foreign and one that sounds Russian, and it is the single
cheapest fix available: no model swap, no latency budget, no new hardware.

``RUAccent`` places the same ``+`` marks with a context model, so the sentence
decides. The marked text goes to Silero unchanged otherwise.

CONSTRAINT — this must never be able to break a reply. The accentizer loads a
transformer on first use, which means a download on a cold machine and a few
hundred milliseconds of CPU. So:

* loading happens ONCE, in a worker thread, kicked off at startup by
  :func:`warm`; the pipeline never awaits it;
* until it is ready — and forever, if the package is missing or the load fails —
  :func:`stress` is the identity function, and Silero falls back to the built-in
  placement it uses today;
* a failure is logged once, not per sentence.

That makes this module additive in the strictest sense: with the dependency
absent the server behaves exactly as it did before.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from loguru import logger

# ``turbo3.1`` is the small context model — it is the homograph resolver, which
# is the whole point; the big variants buy accuracy this use does not need and
# cost startup time on a CPU-only VPS.
_OMOGRAPH_MODEL = "turbo3.1"

_accentizer = None
_lock = threading.Lock()
_state = "cold"          # cold -> loading -> ready | failed
_warm_task: Optional[asyncio.Task] = None


def _load_blocking() -> None:
    """Import and load the accentizer. Runs on a worker thread, never the loop."""
    global _accentizer, _state
    try:
        from ruaccent import RUAccent  # imported here so the dep stays optional

        acc = RUAccent()
        acc.load(omograph_model_size=_OMOGRAPH_MODEL, use_dictionary=True, tiny_mode=False)
        with _lock:
            _accentizer = acc
            _state = "ready"
        logger.info("RUAccent ready ({}) — stress marks now come from context.", _OMOGRAPH_MODEL)
    except Exception as exc:  # noqa: BLE001 — a missing dep is a normal deployment
        with _lock:
            _state = "failed"
        logger.info(
            "RUAccent unavailable ({}) — falling back to Silero's own stress placement.", exc
        )


def warm() -> None:
    """Start loading in the background. Safe to call more than once."""
    global _warm_task, _state
    with _lock:
        if _state != "cold":
            return
        _state = "loading"
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop (a script, a test): load inline rather than not at all.
        _load_blocking()
        return
    _warm_task = loop.create_task(asyncio.to_thread(_load_blocking))


def ready() -> bool:
    with _lock:
        return _state == "ready"


def stress(text: str) -> str:
    """Return ``text`` with ``+`` before each stressed vowel, or unchanged.

    Identity while the model is still loading, and identity forever if it never
    loads. Any per-sentence failure also returns the input untouched — a stress
    mark is a nicety, and no reply is worth losing over one.
    """
    if not text:
        return text
    with _lock:
        acc = _accentizer if _state == "ready" else None
    if acc is None:
        return text
    try:
        return acc.process_all(text)
    except Exception as exc:  # noqa: BLE001
        logger.debug("RUAccent skipped {!r}: {}", text[:40], exc)
        return text
