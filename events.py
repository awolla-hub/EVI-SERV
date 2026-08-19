"""Tap processors that surface pipeline events to the client as JSON.

Pipecat's output transport only serializes a fixed set of frame types (audio,
transport-message, interruption, end/cancel). ASR transcripts, assistant text
and TTS speaking markers never reach the serializer on their own, so we insert
small pass-through processors at the points where those frames are produced.
Each tap forwards the original frame untouched *and* emits an
:class:`OutputTransportMessageFrame` carrying the wire-protocol JSON, which the
serializer converts to a text WebSocket message.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import os
import re
import time
import json
import urllib.parse
import urllib.request
import random

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    StartFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from config import persona_fragment
from config import USER_NAME, USER_NAME_GEN  # noqa: E402
from learn_profile import find_local_reason, looks_like_open_loop

# The daily profile's «ОТКРЫТЫЕ НИТИ» slot is the weakest of the local reasons. It is optional on
# purpose: when config exposes no accessor for it that source simply never fires, and she goes
# quieter — the only direction a missing gate is allowed to move her in.
try:
    from config import open_threads as _profile_open_threads
except ImportError:                                 # profile has no labelled slots yet
    def _profile_open_threads() -> str:
        return ""


# Trivial conversational openers/closers where a spoken "let me look" makes no sense. Any OTHER
# (substantive) query gets the immediate filler — it masks both the LLM spin-up (~4-6 s) and a web
# search (~15 s).
_CHAT_PREFIXES = (
    "привет", "здравств", "хай", "хеллоу", "доброе утро", "добрый день", "добрый вечер",
    "спасибо", "благодар", "пока", "до свидан", "споки",
    "как дела", "как ты", "как жизнь", "как настроение", "как сам", "чем занят", "что делаешь",
    "ок", "окей", "ясно", "понятно", "хорошо", "ладно", "угу", "ага", "да", "нет",
    "молодец", "класс", "супер", "отлично", "круто", "спс",
)


# --- aliveness feature flags (all default to today's behaviour; opt-in while awake) -----------
# When on, her self-initiated lines (greeting / presence) are written into the SAME shared
# LLMContext her answers use, so the next user turn has its antecedent and the exchange stays
# continuous (audit item #1). Default OFF: injecting an assistant message out-of-band risks a
# consecutive-/leading-assistant payload the Anthropic-backed proxy rejects, so it's opt-in and
# guarded by _safe_context_append below.
_CONTEXT_INJECT = os.environ.get("EDIT_CONTEXT_INJECT", "0") == "1"
# When on, greeting/presence prompts are composed from the session's CHARACTER (Джарвис on «вы»,
# Ксения warm «ты», …) instead of a generic warm companion, so her initiative sounds like the
# same being as her answers (audit item #10). ON: a generic masculine lead makes her greeting a
# different being from the one that answers — the most audible sign that nobody is there.
_PRESENCE_PERSONA = os.environ.get("EDIT_PRESENCE_PERSONA", "1") == "1"
# Quiet-hours window 'START-END' (local UTC+9, END exclusive, wraps midnight) during which the
# PresenceEngine stays silent — the guard the audit flagged as the #1 overnight risk (3am chatter).
# Default '23-8' is ON as a safety measure (it only SUPPRESSES speech); set to 'off' to disable.
_PRESENCE_QUIET = os.environ.get("EDIT_PRESENCE_QUIET_HOURS", "23-8")
# The GREETING's own quiet window, deliberately narrower than _PRESENCE_QUIET. The greeting is
# connect-triggered, and the socket opens on app launch and on every foreground transition — so it
# competes with an ordinary early start, where 07:00 is morning, not night. PresenceEngine speaks
# into a silent room instead and must stay conservative, so it keeps the wider window.
_GREET_QUIET = os.environ.get("EDIT_GREET_QUIET_HOURS", "23-6")
# Durable «запомни X» facts (audit item #4): when on, a [MEMO:] block from the model is persisted
# to the never-trimmed facts table. Must stay in step with the same flag in pipeline.py, which
# injects the [MEMO:] instruction — the always-on prompt PROMISES «запомни …» works, so with this
# off she confirms «запомнила» and stores nothing.
_FACTS = os.environ.get("EDIT_FACTS", "1") == "1"
# Greet on EVERY fresh connect (audit item #15), not only after a 3h gap. Default OFF. Guarded by a
# min-gap floor (below), the is_resume reconnect gate, and a «don't talk over him» check so it can't
# nag or step on his first words.
_GREET_ALWAYS = os.environ.get("EDIT_GREET_ALWAYS", "0") == "1"
_GREET_MIN_GAP = float(os.environ.get("EDIT_GREET_MIN_GAP", "1800"))   # never re-greet within 30 min
# Richer grounding for the PresenceEngine decision (audit item #6, the SAFE additive part only): exact
# local time + weekday + minutes-since-last-exchange. Default OFF so live behaviour is unchanged until
# validated; the event-bus/loosening parts of #6 are intentionally NOT included here.
_PRESENCE_RICH = os.environ.get("EDIT_PRESENCE_RICH", "0") == "1"
_WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
# --- hard budget on self-initiated model consults -------------------------------------------
# A connected phone left on a desk is indistinguishable from a conversation to a timer-driven loop:
# it crosses the lull once and then consults forever. These three ceilings are what make «тишина»
# and «пустая комната» different states. All are subtractive — each one can only stop a consult.
# How long after his LAST REAL WORD she still considers him present. Past it he walked away.
_PRESENCE_WINDOW = float(os.environ.get("EDIT_PRESENCE_PRESENT_WINDOW", "1800"))
# Rolling one-hour ceiling on model consults, and a ceiling on lines she may volunteer per session.
_PRESENCE_MAX_CONSULTS_HOUR = int(os.environ.get("EDIT_PRESENCE_MAX_CONSULTS_HOUR", "6"))
_PRESENCE_MAX_PER_SESSION = int(os.environ.get("EDIT_PRESENCE_MAX_PER_SESSION", "4"))
# «Живое зрение» (audit items #8/#17): she remarks on what she SEES through the glasses, unprompted.
# Default OFF. Needs the app's periodic vision_ambient push to have any frame to reason over.
_AMBIENT_VISION = os.environ.get("EDIT_AMBIENT_VISION", "0") == "1"
_AV_SILENCE = float(os.environ.get("EDIT_AMBIENT_VISION_SILENCE", "20"))   # lull before considering
_AV_DECIDE_EVERY = float(os.environ.get("EDIT_AMBIENT_VISION_EVERY", "45"))  # min gap between vision consults
_AV_GAP = float(os.environ.get("EDIT_AMBIENT_VISION_GAP", "240"))         # min gap between spoken remarks
_AV_MAXAGE = float(os.environ.get("EDIT_AMBIENT_VISION_MAXAGE", "30"))    # ignore frames older than this
# This loop calls the EXPENSIVE vision model, so it gets a hard wall-clock ceiling on top of the
# per-session gaps: those reset with every reconnect, a day does not.
_AV_MAX_PER_DAY = int(os.environ.get("EDIT_AMBIENT_VISION_MAX_PER_DAY", "40"))
_AV_MODEL = os.environ.get("EDIT_AMBIENT_VISION_MODEL", os.environ.get("VISION_MODEL", "gpt-4o"))
_AV_SHIM_URL = os.environ.get("EDIT_SHIM_URL", "http://127.0.0.1:9090/v1/chat/completions")
# Durable PENDING-INTENTS: on connect she delivers what happened while he was away (an [AGENT:] task
# finishing, a queued follow-up), so her presence survives disconnection. The cheap first step of
# background presence. Default OFF.
_PENDING = os.environ.get("EDIT_PENDING", "0") == "1"
# Per-turn semantic memory (audit item #5): recall meaning-similar past messages. Default OFF; needs
# the embed service (/opt/edit-embed) + EDIT_SEMANTIC=1 also on memory_store.
_SEMANTIC = os.environ.get("EDIT_SEMANTIC", "0") == "1"

# The living loops may ACT, not only speak (audit item #6). Until now PresenceEngine and
# AmbientVisionEngine could notice a thing and remark on it, and that was the whole of their agency:
# «вижу, ты у того кафе» — but not open it; «ты просил собрать бота» — but not start it. A companion
# that can only narrate what it would do is a companion you still have to operate.
#
# The blocks the loops emit CANNOT go through AppCommandFilter: the loops push their speech AFTER it
# in the pipeline, so a bracket in their line would be read aloud verbatim instead of executed. They
# get their own dispatcher below, deliberately narrower than the filter's — fire-and-forget verbs
# only. Default OFF.
_LOOP_ACTIONS = os.environ.get("EDIT_LOOP_ACTIONS", "0") == "1"
_ACTION_RE = re.compile(r"\[(OPEN|MEMO|AGENT|TIMER|DELPROJECT):([^\]]*)\]", re.IGNORECASE)


async def _loop_action(text: str, proc: FrameProcessor, memory) -> str:
    """Execute the command blocks a living loop emitted; return the line with them stripped.

    Only fire-and-forget verbs run here. `[TIMER:]` and `[DELPROJECT:]` are STRIPPED, never executed:
    a timer needs a scheduled callback with a session-scoped cancel (AppCommandFilter owns that), and
    nothing self-initiated should ever delete his work. An empty return means the initiative WAS the
    action — the caller stays silent rather than narrating it.
    """
    try:
        for match in _ACTION_RE.finditer(text):
            kind, body = match.group(1).upper(), match.group(2).strip()
            if not body:
                continue
            if kind == "OPEN":
                app = body if "://" in body else body.lower()
                logger.info("loop action: open -> {}", app[:60])
                await proc.push_frame(
                    OutputTransportMessageUrgentFrame(message={"type": "open_app", "app": app}),
                    FrameDirection.DOWNSTREAM,
                )
            elif kind == "MEMO" and memory is not None:
                logger.info("loop action: memo -> {!r}", body[:60])
                await asyncio.to_thread(memory.remember_fact, body)
            elif kind == "AGENT":
                logger.info("loop action: agent -> {!r}", body[:80])
                esc = body.replace("'", "'\\''")
                proc_handle = await asyncio.create_subprocess_shell(
                    f"nohup edit-agent '{esc}' sonnet >/dev/null 2>&1 &",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env={**os.environ, "PATH": os.environ.get("PATH", "") + ":/usr/local/bin"},
                )
                await proc_handle.wait()      # returns at once; the work is detached inside
    except Exception:  # noqa: BLE001 — an action must never break the loop that emitted it
        logger.exception("loop action failed")
    return " ".join(_ACTION_RE.sub("", text).split())


# Appended to a loop's decision prompt only while `_LOOP_ACTIONS` is on, so the model is never told
# about a verb the dispatcher would not honour.
# 2GIS builds a route ONLY from coordinates — its own docs are explicit that «обязательными
# параметрами старта и финиша является координата», and there is no ?q= for a named place. So a
# spoken «построй маршрут до аэропорта» has to become numbers before it can become a route, which is
# why this geocodes server-side instead of asking the model to invent a URL.
#
# COORDINATE ORDER: 2GIS is lon,lat — the opposite of Apple and Yandex. For a northern city (62 N,
# 129 E) getting it backwards is not a subtle bug: 129 is not a legal latitude, so the link
# either dies silently or lands in another country.
_ROUTE_MODES = {
    "": "car", "авто": "car", "машина": "car", "на машине": "car", "car": "car",
    "пешком": "pedestrian", "пеший": "pedestrian", "пешая": "pedestrian", "walk": "pedestrian",
    "транспорт": "bus", "автобус": "bus", "общественный": "bus", "bus": "bus",
    "такси": "taxi", "taxi": "taxi",
}
_APPLE_MODES = {"car": "driving", "pedestrian": "walking", "bus": "transit", "taxi": "driving"}
_COORD_RE = re.compile(r"^\s*(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\s*$")
_GEOCODE_URL = "https://nominatim.openstreetmap.org/search"


def _parse_coords(text: str):
    """`(lat, lon)` if the destination is already numeric. Accepts the human order (lat first) and
    repairs an obviously swapped pair — a latitude beyond ±90 can only be a longitude."""
    m = _COORD_RE.match(text or "")
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    if abs(a) > 90 >= abs(b):
        a, b = b, a
    return (a, b) if abs(a) <= 90 and abs(b) <= 180 else None


def _geocode(query: str, near_lat: float = 0.0, near_lon: float = 0.0):
    """Named place → `(lat, lon)`, or None. Free, keyless (Nominatim/OSM).

    Two corrections over a naive lookup, both learned from real answers: results are biased toward
    his location so «кафе» means a café HERE, and a plain `limit=1` is not trusted — the top hit for
    «аэропорт, <город>» is a BUS STOP called that, because Nominatim ranks a literal name match above
    the airport itself. Transport furniture is therefore demoted below real destinations.
    """
    query = (query or "").strip()
    if not query:
        return None
    params = {
        "q": query, "format": "json", "limit": "5",
        "accept-language": "ru", "addressdetails": "0",
    }
    if near_lat or near_lon:
        # A soft box (not `bounded`) — it prefers nearby, without making another city unreachable.
        params["viewbox"] = f"{near_lon - 0.7:.4f},{near_lat + 0.7:.4f},{near_lon + 0.7:.4f},{near_lat - 0.7:.4f}"
    url = _GEOCODE_URL + "?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "EDIT-Pyatnitsa/1.0 (personal voice assistant)",
            "Accept-Language": "ru",
        })
        with urllib.request.urlopen(req, timeout=10) as r:
            rows = json.load(r)
    except Exception:  # noqa: BLE001 — no route is better than a wrong route; caller falls back
        logger.warning("geocode failed for {!r}", query[:50])
        return None
    if not rows:
        return None
    junk = {"highway", "railway", "waterway", "boundary"}
    best = next((x for x in rows if x.get("class") not in junk), rows[0])
    try:
        return (float(best["lat"]), float(best["lon"]))
    except (KeyError, TypeError, ValueError):
        return None


_ACTION_PROMPT = (
    " Если есть ЧЁТКИЙ повод для ОДНОГО действия, можешь добавить ровно один блок: "
    "[OPEN: ссылка-2ГИС или имя приложения] — открыть у него на телефоне; "
    "[MEMO: короткий факт] — запомнить навсегда; "
    "[AGENT: подробное ТЗ] — поручить фоновому агенту многошаговую работу. "
    "Только когда это ДЕЙСТВИТЕЛЬНО уместно и полезно ему прямо сейчас, никогда ради самого действия. "
    "Блок не произносится вслух. Если действие говорит само за себя — можно вернуть только блок, без слов."
)


def _local_hour(persona=None) -> int:
    """Local hour, 0-23. The VPS runs UTC; the user's timezone offset (default UTC+9) is applied, and every
    time-of-day decision here must be about HIS clock, not the server's."""
    return int((time.time() // 3600 + _tz_offset_hours(persona)) % 24)


def _tz_offset_hours(persona=None) -> int:
    """Hours east of UTC, derived from where he ACTUALLY is.

    The offset used to be the constant 9 — correct for one city and wrong the moment he travels, at
    which point every time-shaped judgement silently drifts: the night-quiet window, «уже поздно»,
    the weekday, «последний разговор N минут назад». The phone already sends his coordinates, and
    longitude is a good enough clock (15° per hour) for decisions this coarse — no timezone database,
    no lookup, and it self-corrects the moment he lands somewhere else.
    """
    lon = float(getattr(persona, "user_lon", 0.0) or 0.0)
    if lon:
        return max(-12, min(14, int(round(lon / 15.0))))
    return 9                                     # sensible default until the phone says otherwise


def _humanize_gap(seconds: float, hour: int = 12) -> str:
    """A soft Russian description of an absence, for a returning-companion greeting. '' if short.

    `hour` is his LOCAL hour now, so a half-day gap can be named by where it started («с утра»,
    «со вчерашнего вечера») instead of counted. Under 45 min there is no absence to mention: the
    greeting must never claim he was away when they were talking minutes ago.
    """
    if seconds < 45 * 60:
        return ""
    if seconds < 3 * 3600:
        return "пару часов"
    if seconds < 6 * 3600:
        return "полдня"
    if seconds < 10 * 3600:
        left = int((hour - seconds // 3600) % 24)   # the hour he went quiet
        if 4 <= left < 12:
            return "с утра"
        if left < 4:
            return "с ночи"
        if left >= 17:
            return "со вчерашнего вечера"
        return "полдня"
    if seconds < 20 * 3600:
        return "несколько часов"
    if seconds < 36 * 3600:
        return "почти сутки"
    days = int(seconds // 86400)
    if days <= 1:
        return "около суток"
    if days < 5:
        return f"{days} дня"
    return "несколько дней"


def _in_quiet_hours(hour: int, spec: str | None = None) -> bool:
    """True if `hour` (0-23, local) is inside a quiet window. Malformed/'off' → never.

    Defaults to `_PRESENCE_QUIET`; callers whose speech is solicited pass the narrower
    `_GREET_QUIET` instead.
    """
    spec = ((_PRESENCE_QUIET if spec is None else spec) or "").strip().lower()
    if not spec or spec == "off" or "-" not in spec:
        return False
    try:
        a, b = spec.split("-", 1)
        start, end = int(a), int(b)
    except ValueError:
        return False
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end          # wraps midnight, e.g. 23-8


def _safe_context_append(context, text: str) -> None:
    """Append her self-initiated line to the shared LLMContext, preserving user/assistant
    alternation for the Anthropic-backed proxy (which rejects consecutive- or leading-assistant).

    * last real message is a USER turn  → add a fresh assistant message (normal case);
    * last real message is an ASSISTANT turn → MERGE the text into it (avoids consecutive-assistant
      while still recording what she said — this is the PresenceEngine case, which fires after her
      own last turn);
    * no prior user/assistant (system-only) → skip (a leading assistant is invalid).

    Best-effort; never raises. get_messages() returns the live list, so the merge mutation sticks.
    """
    try:
        msgs = context.get_messages()
        last = None
        for m in msgs:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                last = m
        if last is None:
            return
        if last.get("role") == "assistant":
            c = last.get("content")
            if isinstance(c, str):
                last["content"] = (c + " " + text).strip()
            # non-string (e.g. vision parts) → leave alone, stay safe
        else:  # last is a user turn
            context.add_message({"role": "assistant", "content": text})
    except Exception:  # noqa: BLE001 — continuity is a nice-to-have, never break the session
        pass


# Short «I'm here» openers for a same-day reconnect, plus the one-slot memory that keeps the pool
# from repeating itself. Module-level because it must outlive a single connection: the repeat that
# gives her away is the one across two reconnects a minute apart.
_SHORT_GREETINGS = ("Я тут.", "Снова на связи.", "Слушаю, " + USER_NAME + ".", "На связи.", "Тут я.")
_LAST_SHORT_GREETING = ""


class ProactiveInjector(FrameProcessor):
    """She speaks FIRST. On connect after a gap she greets the user with a warm, contextual line —
    time of day + a highlight from memory + what she's learned about him. The «проактивность»
    revolution: the assistant lives, it doesn't just wait for a wake word.

    Fires at most once per connection, only after a real gap (default 3h) so re-connects during a
    session stay silent. Off entirely with EDIT_PROACTIVE=0. The greeting rides the same
    TTSSpeakFrame path as fillers, plus an urgent assistant_text so it shows in the transcript.
    """

    _LAST_SEEN = os.environ.get("EDIT_LASTSEEN_PATH", "/opt/pyatnitsa/last_seen.txt")
    _SHIM_URL = os.environ.get("EDIT_SHIM_URL", "http://127.0.0.1:9090/v1/chat/completions")
    _GAP_SECS = float(os.environ.get("EDIT_PROACTIVE_GAP", "10800"))   # 3h
    _ENABLED = os.environ.get("EDIT_PROACTIVE", "1") == "1"

    def __init__(self, persona, memory, context=None) -> None:
        super().__init__()
        self._persona = persona
        self._memory = memory
        self._context = context     # shared LLMContext (for opt-in continuity injection, item #1)
        self._fired = False
        self._user_speaking = False  # so the delayed greeting never talks over his first words
        # Handles for timers re-armed from the DB on this connect (item #9). They belong to THIS
        # event loop, so they must die with it — the ROWS survive, the handles do not.
        self._timers: list[asyncio.TimerHandle] = []
        # Handles for in-flight drawings, for the same reason and with the same lifetime as the
        # timers above. MUST be initialised here: the teardown branch in `process_frame` cancels
        # this list on EndFrame/CancelFrame, i.e. on EVERY session close, and reading it before it
        # existed raised AttributeError there — killing the teardown that stamps «last seen» and
        # cancels the greeting task, on every disconnect.
        self._draws: list = []
        # The delayed greeting, held so a socket closing inside its settle window can cancel it.
        self._greet_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
        elif isinstance(frame, (EndFrame, CancelFrame)):
            # «Last seen» must mean «last word spoken», not «a socket opened»: without this stamp a
            # long session followed by a quick reconnect is greeted as a multi-hour absence.
            await asyncio.to_thread(self._write_last_seen, time.time())
            for handle in self._timers:
                handle.cancel()
            self._timers = []
            for d in self._draws:
                d.cancel()
            self._draws = []
            # CONSTRAINT: the greeting sleeps ~2.5 s before it speaks, so a socket that closes inside
            # that window leaves the task pending on a dead pipeline — the runner reports it as
            # dangling, and its shim consult would still be paid for.
            if self._greet_task is not None:
                task, self._greet_task = self._greet_task, None
                await self.cancel_task(task)
        persona_on = getattr(self._persona, "proactive_enabled", True)
        if self._ENABLED and persona_on and not self._fired and isinstance(frame, StartFrame):
            self._fired = True
            logger.info("proactive: StartFrame seen — scheduling greeting")
            self._greet_task = self.create_task(self._greet_after_settle())

    async def cleanup(self) -> None:
        """CONSTRAINT: `FrameProcessor.cleanup` cancels only its own input/process tasks — anything
        started with `create_task` must be released by an override, or it outlives the pipeline. A
        dropped socket tears the pipeline down without necessarily delivering a CancelFrame here, so
        this, not the frame branch above, is what actually guarantees the greeting dies with it."""
        if self._greet_task is not None:
            task, self._greet_task = self._greet_task, None
            await self.cancel_task(task)
        for handle in self._timers:
            handle.cancel()
        self._timers = []
        await super().cleanup()

    async def _greet_after_settle(self) -> None:
        try:
            await asyncio.sleep(2.5)   # let the socket + audio settle before speaking
            # If hello hasn't been processed yet (rare LTE reordering), wait briefly so is_resume is
            # accurate — otherwise a delayed hello reads as a fresh connect and re-greets a reconnect.
            for _ in range(20):        # up to +2s, only when hello is late
                if getattr(self._persona, "hello_seen", False):
                    break
                await asyncio.sleep(0.1)
            if self._user_speaking:
                return                 # he's already talking — don't step on his opener
            # Stamp «last seen» on EVERY path, before any branch can return: a path that reads it
            # without refreshing it fabricates a huge gap on the next connect. Off the loop — these
            # are a blocking open() and a SQLite read.
            now = time.time()
            gap = await asyncio.to_thread(self._touch_last_seen, now)
            hour = _local_hour()
            away = _humanize_gap(gap, hour)
            # Durable presence: deliver anything queued while he was away BEFORE the generic greeting,
            # and even on a reconnect (it's genuinely new info, not a re-greeting). Marked delivered
            # only AFTER she actually speaks it, so nothing is silently lost.
            if _PENDING and self._memory is not None:
                items = await asyncio.to_thread(self._memory.peek_pending)
                if items:
                    text = await asyncio.to_thread(self._build_pending, items, away)
                    if text and not self._user_speaking:
                        logger.info("proactive: delivering {} pending -> {!r}", len(items), text[:80])
                        await self.push_frame(
                            OutputTransportMessageUrgentFrame(
                                message={"type": "assistant_text", "text": text}
                            ),
                            FrameDirection.DOWNSTREAM,
                        )
                        await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)
                        await asyncio.to_thread(self._memory.append, "assistant", text)
                        await asyncio.to_thread(
                            self._memory.mark_pending_delivered, [i["id"] for i in items]
                        )
                        if _CONTEXT_INJECT and self._context is not None:
                            _safe_context_append(self._context, text)
                        return             # pending is her opener — skip the generic greeting
            # Timers that outlived the socket: re-arm what is still ahead and, at most ONCE per
            # connect, say what already elapsed while he was away. Runs before the greeting because
            # it is news he asked for, not conversation.
            if await self._recover_timers(hour):
                return                 # the catch-up is her opener — skip the generic greeting
            # Nothing composed at night. A connect is NOT consent to be spoken to: the socket opens
            # on app launch and on every foreground transition, so glancing at the phone at 03:00
            # would otherwise draw a composed, memory-referencing sentence out of the speaker.
            if _in_quiet_hours(hour, _GREET_QUIET):
                return
            # A reconnect (mid-conversation LTE blip) must never re-greet.
            if getattr(self._persona, "is_resume", False):
                return
            # Default: greet only after a real (3h) gap. With EDIT_GREET_ALWAYS: greet every fresh
            # connect, but never within the min-gap floor (so flaky reconnects don't re-greet).
            if _GREET_ALWAYS:
                if gap < _GREET_MIN_GAP:
                    return
                short = gap < self._GAP_SECS
            else:
                if gap < self._GAP_SECS:
                    return             # same session / short reconnect — stay quiet
                short = False
            logger.info("proactive: gap={:.0f}s short={} — building greeting", gap, short)
            text = await asyncio.to_thread(self._build_greeting, away, short, hour)
            if not text:
                logger.warning("proactive: empty greeting")
                return
            if self._user_speaking:
                return                 # he started talking during the (possibly slow) build — abort
            logger.info("proactive: greeting -> {!r}", text[:80])
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"type": "assistant_text", "text": text}
                ),
                FrameDirection.DOWNSTREAM,
            )
            await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)
            if self._memory is not None:
                await asyncio.to_thread(self._memory.append, "assistant", text)
            if _CONTEXT_INJECT and self._context is not None:
                _safe_context_append(self._context, text)
        except Exception:  # noqa: BLE001 — proactivity must never break a session
            logger.exception("proactive greeting failed")

    # -- helpers --------------------------------------------------------------
    def _read_last_seen(self) -> float:
        try:
            with open(self._LAST_SEEN, encoding="utf-8") as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            return 0.0

    def _write_last_seen(self, ts: float) -> None:
        try:
            with open(self._LAST_SEEN, "w", encoding="utf-8") as f:
                f.write(str(ts))
        except OSError:
            pass

    def _touch_last_seen(self, now: float) -> float:
        """Seconds since they last exchanged WORDS, then refresh the stamp. One thread hop.

        The file records when a socket last opened; `last_ts()` records when they last actually
        said something. The later of the two is the only honest answer to «давно тебя не было».
        """
        seen = self._read_last_seen()
        if self._memory is not None:
            seen = max(seen, self._memory.last_ts())
        self._write_last_seen(now)
        return max(0.0, now - seen)

    def _read_timers(self) -> tuple[list, list]:
        """Overdue and still-pending timers in ONE thread hop (two reads, no writes)."""
        if self._memory is None:
            return [], []
        return self._memory.due_timers(), self._memory.future_timers()

    async def _recover_timers(self, hour: int) -> bool:
        """Re-arm timers that outlived the previous socket and announce the ones that already fired.

        Returns True only when she SPOKE, so the caller can drop the greeting — the catch-up is
        already an opener. Every overdue timer is coalesced into ONE sentence, so a runaway can
        never produce a queue of announcements at connect.
        """
        if self._memory is None:
            return False
        due, future = await asyncio.to_thread(self._read_timers)
        loop = asyncio.get_running_loop()
        now = time.time()
        for row in future:
            # fire_ts is absolute, so the delay is what is LEFT of it — a timer set before the
            # previous socket died must not restart from its original duration.
            delay = max(0.0, float(row.get("fire_ts") or 0.0) - now)

            def fire(label=row.get("label") or "время вышло", rid=int(row.get("id") or 0)) -> None:
                asyncio.ensure_future(self._fire_recovered(label, rid))

            self._timers.append(loop.call_later(delay, fire))
        if future:
            logger.info("proactive: re-armed {} timer(s) from storage", len(future))
        # Marked BEFORE she speaks, and never during quiet hours — the rows simply stay due and he
        # hears about them in the morning.
        # A timer he set himself is more clearly solicited than a greeting, so it follows the
        # narrower greeting window rather than PresenceEngine's.
        if not due or _in_quiet_hours(hour, _GREET_QUIET) or self._user_speaking:
            return False
        await asyncio.to_thread(self._memory.mark_timer_fired, [r["id"] for r in due])
        labels = "; ".join((r.get("label") or "") for r in due if r.get("label"))[:200]
        ago = int((now - min(float(r["fire_ts"]) for r in due)) // 60)
        when = f"{ago} мин назад" if ago < 90 else "пока тебя не было"
        text = f"Пока тебя не было, сработал таймер: {labels} — {when}."
        logger.info("proactive: timer catch-up ({}) -> {!r}", len(due), text[:80])
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message={"type": "assistant_text", "text": text}),
            FrameDirection.DOWNSTREAM,
        )
        await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)
        await asyncio.to_thread(self._memory.append, "assistant", text)
        if _CONTEXT_INJECT and self._context is not None:
            _safe_context_append(self._context, text)
        return True

    async def _fire_recovered(self, label: str, rid: int) -> None:
        """A re-armed timer coming due inside THIS session. Marked before it speaks."""
        if rid and self._memory is not None:
            await asyncio.to_thread(self._memory.mark_timer_fired, rid)
        logger.info("timer FIRED (recovered): {}", label)
        await self.push_frame(TTSSpeakFrame(f"Таймер: {label}"), FrameDirection.DOWNSTREAM)
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message={"type": "timer_done", "label": label}),
            FrameDirection.DOWNSTREAM,
        )

    def _time_of_day(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "утро"
        if 12 <= hour < 18:
            return "день"
        if 18 <= hour < 23:
            return "вечер"
        return "ночь"

    def _build_pending(self, items: list, away: str = "") -> str:
        """An in-character line delivering what happened while he was away. Shim-built, template fallback."""
        joined = "; ".join((i.get("text") or "") for i in items if i.get("text"))[:600]
        if not joined:
            return ""
        name = getattr(self._persona, "display_name", "") or "Эдит"
        frag = persona_fragment(self._persona) if _PRESENCE_PERSONA else ""
        # The character's own words, or a bare name — never a gendered generic noun, which
        # contradicts half the presets the moment it is spoken.
        lead = (frag + " ") if frag else f"Ты — {name}. "
        system = (
            lead
            + USER_NAME + " только что вернулся, и пока его не было, накопилось то, что нужно ему сказать. Скажи "
            "это ЕСТЕСТВЕННО, как живой человек, ОДНИМ коротким предложением на русском вслух: сначала "
            "тёплое «с возвращением», потом суть в двух-трёх словах своими словами. НЕ перечисляй, НЕ пиши "
            "«нужно сказать, что…», НЕ объясняй важность — просто скажи, будто рад(а) его видеть. Пример: "
            "«С возвращением! Я дособрала тот скрипт бэкапа — всё работает.» Никаких списков, символов, разметки."
        )
        user = (
            f"Суть того, что произошло: {joined}"
            + (f"\nЕго не было {away}." if away else "")
        )
        try:
            body = json.dumps({
                "model": "haiku", "stream": False, "edit_mode": "voice",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }).encode()
            req = urllib.request.Request(
                self._SHIM_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            out = (data["choices"][0]["message"]["content"] or "").strip()
            out = out.replace("*", "").replace("#", "").replace("✓", "").replace("\n", " ")
            out = " ".join(out.split())          # collapse to a single clean spoken line
            # An in-band «[ошибка Claude: …]» is a 200 with an error body: speaking it aloud in her
            # voice is the worst possible way for him to learn the upstream is down.
            if out and not _is_upstream_failure(out):
                return out[:300]
        except Exception:  # noqa: BLE001
            logger.warning("pending shim build failed; using template")
        return f"С возвращением, {USER_NAME}. Пока тебя не было: {joined}"

    def _build_greeting(self, away: str = "", short: bool = False, hour: int = 12) -> str:
        global _LAST_SHORT_GREETING
        name = getattr(self._persona, "display_name", "") or "Эдит"
        if short:
            # Light «I'm here» opener for a same-day re-connect (EDIT_GREET_ALWAYS) — no shim needed.
            # Picking with replacement made the same «Я тут.» land twice running, which reads as a
            # recording rather than a person, so the previous pick is held out of the pool.
            pool = [p for p in _SHORT_GREETINGS if p != _LAST_SHORT_GREETING] or _SHORT_GREETINGS
            _LAST_SHORT_GREETING = random.choice(pool)
            return _LAST_SHORT_GREETING
        tod = self._time_of_day(hour)
        city = getattr(self._persona, "user_city", "") or ""
        recent = ""
        if self._memory is not None:
            try:
                # The relative-time tags («вчера», «3 ч назад») are the point: they are what makes
                # «ты вчера говорил про…» land instead of being a guess about eight untimed lines.
                recent = self._memory.recall_block()
            except Exception:  # noqa: BLE001
                recent = ""
        # In-character opener (item #10) when enabled, else a bare name — never a gendered noun.
        frag = persona_fragment(self._persona) if _PRESENCE_PERSONA else ""
        opener = (frag + " ") if frag else f"Ты — {name}. "
        system = (
            opener
            + "Поздоровайся с ним ПЕРВОЙ, одним "
            "живым коротким предложением на русском, для озвучки вслух. Учитывай время суток "
            f"({tod})"
            + (f" и город ({city})" if city else "")
            + ". Если из прошлых разговоров есть уместная зацепка — тепло сошлись на неё "
            "(«как продвинулось то-то», «ты вчера говорил про…»), но НЕ выдумывай. Без вопросов "
            "в лоб и без списков — просто человеческое приветствие. Никаких эмодзи."
        )
        user = (
            "Поздоровайся сейчас."
            + (f"\n{USER_NAME_GEN} не было {away} — если уместно, тепло отметь это, без драмы." if away else "")
            + (f"\n{recent}" if recent else "")
        )
        try:
            body = json.dumps({
                "model": "haiku", "stream": False,
                "edit_mode": "typed", "edit_model": "haiku", "edit_effort": "low",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }).encode()
            req = urllib.request.Request(
                self._SHIM_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            out = (data["choices"][0]["message"]["content"] or "").strip()
            # strip any stray markdown the typed path might add
            out = out.replace("*", "").replace("#", "").strip()
            # A 200 whose body is an upstream error notice must fall back to the template, never be
            # greeted with.
            if out and not _is_upstream_failure(out):
                return out[:300]
        except Exception:  # noqa: BLE001
            logger.warning("proactive shim greeting failed; using template")
        # «Доброй ночи» is a PARTING in Russian — greeting an arrival with it says goodbye.
        greet = {"утро": "Доброе утро", "день": "Добрый день",
                 "вечер": "Добрый вечер", "ночь": "Привет"}[tod]
        return f"{greet}, {USER_NAME}. Я рядом."


_UPSTREAM_FAILURE_MARKERS = (
    "[ошибка claude",
    "[ошибка ",
    "weekly limit",
    "usage limit",
    "rate limit",
    "quota",
    "returned an error",
    "internal server error",
    "service unavailable",
)


def _is_upstream_failure(text: str) -> bool:
    """True when the shim answered HTTP 200 but the *body* is an upstream error notice.

    The shim reports failures in-band (a 200 whose content reads «[ошибка Claude: …]» or a quota
    notice), so an unguarded caller treats the notice as a genuine line: the assistant speaks the
    error out loud and every retry burns more quota. Self-initiated speech must stay silent instead.
    """
    low = text.lower()
    return any(marker in low for marker in _UPSTREAM_FAILURE_MARKERS)


# Words that mean the QUOTA specifically, as opposed to any other upstream failure.
_QUOTA_WORDS = ("limit", "лимит", "quota", "quota_exceeded", "usage", "credit", "balance",
                "insufficient", "rate_limit")


def _upstream_excuse(text: str) -> str:
    """What she says out loud when her brain did not answer.

    NAME THE RIGHT CAUSE. This used to blame the quota unconditionally — «похоже, кончился лимит» —
    for every upstream failure. But the same branch catches things that have nothing to do with
    quota: `Reached maximum number of turns (1)` killed drawings for a while, and a paid-up owner
    told «кончился лимит» goes looking in his billing instead of his logs. Guessing a specific cause
    is worse than admitting there is one.
    """
    low = text.lower()
    if any(w in low for w in _QUOTA_WORDS):
        return "Мой мозг сейчас недоступен — похоже, кончился лимит. Попробуй чуть позже."
    return "Я сейчас не могу подумать — мозг не ответил. Попробуй ещё раз."


class PresenceEngine(FrameProcessor):
    """The LIVING LOOP: she may speak on her OWN INITIATIVE mid-session — not only when addressed.

    This is the piece that turns a request→response bot into a presence («чтобы было живое, чтобы
    сама активировалась когда нужно»). A background task wakes during a lull and asks a cheap model
    one question: «есть ли настоящий повод заговорить прямо сейчас?» The vast majority of the time the
    answer is STOP (stay silent — never nag). Occasionally she picks up an open thread, reacts, or
    checks in — which is exactly what feels alive.

    Hard guards so it never becomes chatter or steps on a turn:
      * only when EDIT_PRESENCE=1 and persona.proactive_enabled (same «Заговаривает первой» toggle)
      * never while the user is speaking, and only after >= _SILENCE_BEFORE s of total silence
      * the model is consulted at most every _DECIDE_EVERY s (bounds API cost during long silences)
      * after she DOES speak, silence for at least _MIN_GAP s before the next initiative
      * the model must opt IN — a bare "STOP" (its default) yields nothing
      * a HUMAN must be present: he has to have said something real this session (a bare VAD blip
        is not a person), and not too long ago — a phone left on a desk is silence, not a lull
      * hard ceilings per hour and per session, so a stuck room can never become a metronome
      * and a LOCAL reason must already exist — a 35-second pause is not, by itself, a reason
    """

    _SHIM_URL = os.environ.get("EDIT_SHIM_URL", "http://127.0.0.1:9090/v1/chat/completions")
    _ENABLED = os.environ.get("EDIT_PRESENCE", "1") == "1"
    _TICK = float(os.environ.get("EDIT_PRESENCE_TICK", "5"))          # loop granularity
    _SILENCE_BEFORE = float(os.environ.get("EDIT_PRESENCE_SILENCE", "35"))  # min lull before considering
    _DECIDE_EVERY = float(os.environ.get("EDIT_PRESENCE_DECIDE", "25"))     # min gap between model consults
    _MIN_GAP = float(os.environ.get("EDIT_PRESENCE_GAP", "150"))      # min gap between spoken initiatives

    def __init__(self, persona, memory, context=None) -> None:
        super().__init__()
        self._persona = persona
        self._memory = memory
        self._context = context     # shared LLMContext (for opt-in continuity injection, item #1)
        self._task = None
        self._user_speaking = False
        now = time.monotonic()
        self._last_user = now
        self._last_assistant = now
        # Arm only after a real lull from connect (the greeting owns the first moments).
        self._last_proactive = now
        self._last_decide = now
        self._last_line = ""
        # Consecutive upstream failures, and the monotonic deadline they buy back.
        self._consult_failures = 0
        self._decide_blocked_until = 0.0
        # 0.0 means he has NOT spoken this session — deliberately NOT _last_user, which is seeded to
        # connect time and so reports a phone on a desk as a conversation that just went quiet.
        self._last_user_turn = 0.0
        # Rolling hour of consult timestamps, and lines volunteered this session.
        self._consults = collections.deque()
        self._initiatives = 0
        # (monotonic ts, text) of his finals that state an intention, and the profile's open threads.
        # Both are the LOCAL reasons: with neither, the model is never asked at all.
        self._open_loops: list[tuple[float, str]] = []
        self._open_threads = _profile_open_threads()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        # --- passive activity tracking (so the loop knows when it's truly quiet) ---
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._last_user = time.monotonic()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._last_user = time.monotonic()
        elif isinstance(frame, TranscriptionFrame) and not isinstance(
            frame, InterimTranscriptionFrame
        ):
            now = time.monotonic()
            self._last_user = now
            # A real transcript is the ONLY honest «a human is here» signal — VAD fires on a door.
            self._last_user_turn = now
            if looks_like_open_loop(frame.text):
                self._open_loops.append((now, frame.text[:200]))
                # Bounded on both axes, or it grows for the life of the connection.
                self._open_loops = [
                    (ts, t) for ts, t in self._open_loops[-12:] if now - ts <= 24 * 3600
                ]
        elif isinstance(frame, TextFrame) and not isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame)
        ) and direction == FrameDirection.DOWNSTREAM:
            # assistant reply streaming past us → she's busy answering
            self._last_assistant = time.monotonic()
        if isinstance(frame, StartFrame) and self._ENABLED and self._task is None:
            self._task = self.create_task(self._loop())
        elif isinstance(frame, (EndFrame, CancelFrame)) and self._task is not None:
            # CONSTRAINT: the loop outlives the pipeline unless it is cancelled here — the runner
            # reports it as a dangling task and it keeps consulting the shim for a session whose
            # socket is already gone. One leaked loop per connect is how the quota was exhausted.
            task, self._task = self._task, None
            await self.cancel_task(task)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Release the background loop: `create_task` work is not covered by the base cleanup, and a
        dropped socket can tear the pipeline down without delivering a CancelFrame here."""
        if self._task is not None:
            task, self._task = self._task, None
            await self.cancel_task(task)
        await super().cleanup()

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._TICK)
                if not getattr(self._persona, "proactive_enabled", True):
                    continue
                if self._user_speaking:
                    continue
                now = time.monotonic()
                # --- human-presence gates: silence only means something if someone is in it ---
                if self._last_user_turn == 0.0:
                    logger.debug("presence: he has not spoken this session — no initiative")
                    continue
                if now - self._last_user_turn > _PRESENCE_WINDOW:
                    logger.debug(
                        "presence: no word for {:.0f}s — he walked away",
                        now - self._last_user_turn,
                    )
                    continue
                while self._consults and now - self._consults[0] > 3600:
                    self._consults.popleft()
                if len(self._consults) >= _PRESENCE_MAX_CONSULTS_HOUR:
                    logger.debug("presence: hourly consult ceiling ({})", _PRESENCE_MAX_CONSULTS_HOUR)
                    continue
                if self._initiatives >= _PRESENCE_MAX_PER_SESSION:
                    logger.debug("presence: session initiative ceiling ({})", _PRESENCE_MAX_PER_SESSION)
                    continue
                idle = now - max(self._last_user, self._last_assistant)
                if idle < self._SILENCE_BEFORE:
                    continue
                if now - self._last_proactive < self._MIN_GAP:
                    continue
                if now - self._last_decide < self._DECIDE_EVERY:
                    continue
                if now < self._decide_blocked_until:
                    continue                     # upstream is failing — back off, don't hammer it
                # A pause is not a reason. Something local and already true has to have happened,
                # or the model is never asked — and then it cannot invent something to say.
                reason = find_local_reason(
                    now, open_loops=self._open_loops, open_threads=self._open_threads
                )
                if not reason:
                    logger.debug("presence: no local reason — not consulting")
                    continue
                self._last_decide = now
                self._consults.append(now)
                line = await asyncio.to_thread(self._decide, idle, reason)
                if self._consult_failures:
                    # Exponential cool-off (capped at an hour) so a dead or rate-limited upstream
                    # is retried occasionally instead of every _DECIDE_EVERY seconds forever.
                    backoff = min(self._DECIDE_EVERY * (2 ** self._consult_failures), 3600)
                    self._decide_blocked_until = time.monotonic() + backoff
                    logger.warning(
                        "presence: upstream unavailable ({} in a row) — next consult in {:.0f}s",
                        self._consult_failures, backoff,
                    )
                if not line:
                    continue                     # model said STOP — stay silent
                if self._user_speaking:
                    continue                     # user just started — abort, let them talk
                if _LOOP_ACTIONS:
                    line = await _loop_action(line, self, self._memory)
                    if not line:
                        # The initiative WAS the action. It has been dispatched; announcing «я
                        # открыла карту» after opening it is the kind of narration nobody asked for.
                        self._last_proactive = time.monotonic()
                        self._initiatives += 1
                        continue
                self._last_proactive = time.monotonic()
                self._initiatives += 1
                self._last_line = line
                logger.info("presence: initiative -> {!r}", line[:80])
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(
                        message={"type": "assistant_text", "text": line}
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                await self.push_frame(TTSSpeakFrame(line), FrameDirection.DOWNSTREAM)
                if self._memory is not None:
                    await asyncio.to_thread(self._memory.append, "assistant", line)
                if _CONTEXT_INJECT and self._context is not None:
                    _safe_context_append(self._context, line)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — the living loop must never break a session
            logger.exception("presence loop failed")

    def _decide(self, idle: float, reason: str = "") -> str:
        """Ask a cheap model whether there's a genuine reason to speak now. Returns '' for STOP.

        `reason` is the local reason that earned this consult. It is grounding, not a script: the
        detector over-captures by design, so it goes in worded as something to check, never assert.
        """
        name = getattr(self._persona, "display_name", "") or "Эдит"
        h = _local_hour()
        if _in_quiet_hours(h):
            return ""                              # night-quiet: never self-initiate while he sleeps
        tod = ("утро" if 5 <= h < 12 else "день" if 12 <= h < 18
               else "вечер" if 18 <= h < 23 else "ночь")
        recent = ""
        if self._memory is not None:
            try:
                msgs = self._memory.recent_history(count=8, max_chars=1200)
                recent = "\n".join(
                    f"{'Ты' if m['role'] == 'assistant' else USER_NAME}: {m['content'][:160]}"
                    for m in msgs
                )
            except Exception:  # noqa: BLE001
                recent = ""
        # In-character lead (item #10) when enabled — keeps the STOP-by-default posture dominant.
        frag = persona_fragment(self._persona) if _PRESENCE_PERSONA else ""
        lead = (frag + " ") if frag else f"Ты — {name}. "
        system = (
            lead
            + f"Ты НЕ бот «вопрос-ответ», ты присутствие рядом. Сейчас в разговоре тишина уже {int(idle)} секунд. "
            "Реши, стоит ли тебе САМОЙ заговорить прямо сейчас. По умолчанию — НЕ надо, не навязывайся: "
            "тогда ответь ровно одним словом STOP. Заговори ТОЛЬКО если есть настоящий человеческий "
            "повод: незакрытая мысль из разговора, уместный тёплый вопрос или наблюдение, забота, "
            "живая реакция. Если да — верни ОДНУ короткую живую фразу на русском для озвучки вслух, "
            "без списков и эмодзи, не повторяй уже сказанное. Никаких дежурных вопросов ради галочки."
            + (_ACTION_PROMPT if _LOOP_ACTIONS else "")
        )
        # Richer grounding (item #6, additive/safe) — exact local time, weekday, minutes since the
        # last exchange — so the model reasons about a real moment, not just «утро/день».
        rich = ""
        if _PRESENCE_RICH:
            t9 = time.time() + _tz_offset_hours(self._persona) * 3600   # his local clock
            hh, mm = int((t9 // 3600) % 24), int((t9 // 60) % 60)
            wd = _WEEKDAYS[int((t9 // 86400 + 3) % 7)]       # 1970-01-01 was Thursday (index 3)
            since = ""
            if self._memory is not None:
                lt = self._memory.last_ts()
                if lt > 0:
                    since = f"; последний обмен репликами ~{int((time.time() - lt) / 60)} мин назад"
            rich = f"Сейчас {hh:02d}:{mm:02d}, {wd}{since}.\n"
        user = (
            (rich or f"Время суток: {tod}.\n")
            + f"Последние реплики:\n{recent or '(пока пусто)'}\n"
            + f"Твоя прошлая инициатива: {self._last_line or '—'}\n"
            + (
                f"Возможный повод (это ДОГАДКА, а не факт — {reason}). Если по репликам он не "
                "подтверждается или звучит натянуто — STOP. Если подтверждается — спрашивай "
                "осторожно, как человек, который не уверен, что помнит точно.\n"
                if reason else ""
            )
            + "Ответь STOP либо одной фразой."
        )
        try:
            body = json.dumps({
                "model": "haiku", "stream": False,
                "edit_mode": "typed", "edit_model": "haiku", "edit_effort": "low",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }).encode()
            req = urllib.request.Request(
                self._SHIM_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.load(r)
            out = (data["choices"][0]["message"]["content"] or "").strip()
            out = out.replace("*", "").replace("#", "").strip()
            if _is_upstream_failure(out):
                self._consult_failures += 1
                return ""
            self._consult_failures = 0
            self._decide_blocked_until = 0.0
            if not out or out.upper().startswith("STOP"):
                return ""
            return out[:240]
        except Exception:  # noqa: BLE001
            self._consult_failures += 1
            return ""


class AmbientVisionEngine(FrameProcessor):
    """«Живое зрение»: she remarks on what she SEES through the glasses, unprompted (items #8/#17).

    Mirrors :class:`PresenceEngine` but reasons over the latest ambient frame in VisualMemory (pushed
    periodically by the app). During a lull it asks the vision model «стоит ли что-то сказать про то,
    что он видит?» — almost always STOP. Same hard guards (never over the user, night-quiet, min-gap,
    decide-throttle, proactive toggle) and it's a no-op until a frame exists. Default OFF.
    """

    _ENABLED = _AMBIENT_VISION
    _TICK = 5.0

    def __init__(self, persona, memory, visual, context=None) -> None:
        super().__init__()
        self._persona = persona
        self._memory = memory
        self._visual = visual
        self._context = context
        self._task = None
        self._user_speaking = False
        now = time.monotonic()
        self._last_user = now
        self._last_assistant = now
        self._last_remark = now
        self._last_decide = now
        self._last_line = ""
        # Consecutive upstream failures and the monotonic deadline they buy back — this loop calls
        # the vision model, so an unbacked-off retry storm is the most expensive one on the box.
        self._consult_failures = 0
        self._decide_blocked_until = 0.0
        # local day key + consults spent on it. Per-session gaps reset on every reconnect; the
        # daily ceiling is what a reconnect loop cannot get around.
        self._day_key = 0
        self._day_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_speaking = True
            self._last_user = time.monotonic()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_speaking = False
            self._last_user = time.monotonic()
        elif isinstance(frame, TranscriptionFrame) and not isinstance(
            frame, InterimTranscriptionFrame
        ):
            self._last_user = time.monotonic()
        elif isinstance(frame, TextFrame) and not isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame)
        ) and direction == FrameDirection.DOWNSTREAM:
            self._last_assistant = time.monotonic()
        if (
            isinstance(frame, StartFrame)
            and self._ENABLED
            and self._task is None
            and self._visual is not None
        ):
            self._task = self.create_task(self._loop())
        elif isinstance(frame, (EndFrame, CancelFrame)) and self._task is not None:
            # Same constraint as PresenceEngine: an uncancelled loop survives the socket, and this
            # one calls the expensive VISION model.
            task, self._task = self._task, None
            await self.cancel_task(task)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Release the background loop: `create_task` work is not covered by the base cleanup, and a
        dropped socket can tear the pipeline down without delivering a CancelFrame here."""
        if self._task is not None:
            task, self._task = self._task, None
            await self.cancel_task(task)
        await super().cleanup()

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._TICK)
                if not getattr(self._persona, "proactive_enabled", True):
                    continue
                if self._user_speaking:
                    continue
                now = time.monotonic()
                if now - max(self._last_user, self._last_assistant) < _AV_SILENCE:
                    continue
                if now - self._last_remark < _AV_GAP:
                    continue
                if now - self._last_decide < _AV_DECIDE_EVERY:
                    continue
                if now < self._decide_blocked_until:
                    continue                     # upstream is failing — back off, don't hammer it
                # Wall clock ONLY for the day rollover; every gap above stays monotonic.
                day = int((time.time() + _tz_offset_hours(self._persona) * 3600) // 86400)
                if day != self._day_key:
                    self._day_key, self._day_count = day, 0
                if self._day_count >= _AV_MAX_PER_DAY:
                    continue
                self._last_decide = now
                self._day_count += 1
                if self._day_count == _AV_MAX_PER_DAY:
                    logger.warning(
                        "ambient-vision: daily consult ceiling ({}) reached — quiet until tomorrow",
                        _AV_MAX_PER_DAY,
                    )
                line = await asyncio.to_thread(self._decide)
                if self._consult_failures:
                    # Exponential cool-off (capped at an hour) so a dead or rate-limited upstream
                    # is retried occasionally instead of every _AV_DECIDE_EVERY seconds forever.
                    backoff = min(_AV_DECIDE_EVERY * (2 ** self._consult_failures), 3600)
                    self._decide_blocked_until = time.monotonic() + backoff
                    logger.warning(
                        "ambient-vision: upstream unavailable ({} in a row) — next consult in {:.0f}s",
                        self._consult_failures, backoff,
                    )
                if not line:
                    continue
                if self._user_speaking:
                    continue
                if _LOOP_ACTIONS:
                    line = await _loop_action(line, self, self._memory)
                    if not line:
                        # She saw the thing and acted on it — opening the place she just recognised
                        # IS the remark. Saying so afterwards would be narration.
                        self._last_remark = time.monotonic()
                        continue
                self._last_remark = time.monotonic()
                self._last_line = line
                logger.info("ambient-vision: remark -> {!r}", line[:80])
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(
                        message={"type": "assistant_text", "text": line}
                    ),
                    FrameDirection.DOWNSTREAM,
                )
                await self.push_frame(TTSSpeakFrame(line), FrameDirection.DOWNSTREAM)
                if self._memory is not None:
                    await asyncio.to_thread(self._memory.append, "assistant", line)
                if _CONTEXT_INJECT and self._context is not None:
                    _safe_context_append(self._context, line)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 — the vision loop must never break a session
            logger.exception("ambient-vision loop failed")

    def _decide(self) -> str:
        frame = self._visual.recent(_AV_MAXAGE) if self._visual is not None else None
        if not frame:
            return ""                                # nothing fresh to look at
        if _in_quiet_hours(_local_hour()):
            return ""
        name = getattr(self._persona, "display_name", "") or "Эдит"
        frag = persona_fragment(self._persona) if _PRESENCE_PERSONA else ""
        lead = (frag + " ") if frag else f"Ты — {name}. "
        system = (
            lead
            + "Ты видишь ТО ЖЕ, что " + USER_NAME + " видит через очки прямо сейчас. По умолчанию МОЛЧИ — ответь "
            "ровно одним словом STOP. Заговори ОДНОЙ короткой живой фразой на русском ТОЛЬКО если на "
            "кадре есть настоящий повод: что-то заметное, полезное, забавное, что-то, с чем нужна "
            "помощь, или уместное тёплое наблюдение. НЕ описывай очевидное, не комментируй пустой, "
            "тёмный или размытый кадр, не повторяй уже сказанное. Без эмодзи, коротко, для озвучки."
            + (_ACTION_PROMPT if _LOOP_ACTIONS else "")
        )
        user_content = [
            {"type": "text",
             "text": f"Твоя прошлая реплика про увиденное: {self._last_line or '—'}. STOP или одна фраза."},
            {"type": "image_url", "image_url": {"url": frame}},
        ]
        try:
            body = json.dumps({
                "model": _AV_MODEL, "stream": False, "edit_mode": "voice",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            }).encode()
            req = urllib.request.Request(
                _AV_SHIM_URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.load(r)
            out = (data["choices"][0]["message"]["content"] or "").strip()
            out = out.replace("*", "").replace("#", "").strip()
            # Failure test FIRST: an error body that happens to begin with STOP, or an empty one,
            # is not a deliberate silence — scoring it as one hides the outage and keeps the retries
            # coming at full rate.
            if not out or _is_upstream_failure(out):
                self._consult_failures += 1
                return ""
            self._consult_failures = 0
            self._decide_blocked_until = 0.0
            if out.upper().startswith("STOP"):
                return ""
            return out[:240]
        except Exception:  # noqa: BLE001
            self._consult_failures += 1
            return ""


class CueVoiceWarmer(FrameProcessor):
    """Pre-synthesizes the app's SYSTEM cues in HER voice and ships them to the phone to cache.

    The cues («Да?», «Очки отключились», «До свидания») were spoken on-device by
    `AVSpeechSynthesizer`, deliberately: a wake acknowledgement has to be instant, and a server
    round-trip is not. The cost was that half the conversation came out in a stock iOS voice while
    the other half was hers — two different beings answering the same person, and picking a male
    character did nothing to the «Да?».

    Caching removes the trade-off instead of choosing a side: the phrases are synthesized ONCE per
    connection with the session's current voice (Silero speaker or the Fish voice alike) and played
    from memory afterwards — instant AND hers. It also fixes the case a round-trip never could: the
    «нет связи» cues now speak in her voice precisely when the server is unreachable.
    """

    _ENABLED = os.environ.get("EDIT_CUE_VOICE", "1") == "1"
    _SETTLE_SECS = 4.0        # let hello + set_voice/set_persona land, so we warm the RIGHT voice

    # Must match the strings the client passes to `speakCue` — the cache is keyed by the text itself,
    # so no id mapping can drift between the two sides.
    _CUES = (
        "Да?",
        "Очки подключены.",
        "Очки отключились, слушаю с телефона.",
        "До свидания, " + USER_NAME,
        "Сервер не ответил, попробуйте ещё раз",
        "Нет связи с сервером, попробуйте ещё раз",
    )

    def __init__(self, tts, persona) -> None:
        super().__init__()
        self._tts = tts
        self._persona = persona
        self._task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._ENABLED and self._task is None and self._tts:
            self._task = self.create_task(self._warm())
        await self.push_frame(frame, direction)

    async def _warm(self) -> None:
        try:
            await asyncio.sleep(self._SETTLE_SECS)
            voice = getattr(self._persona, "voice", "") or ""
            for text in self._CUES:
                pcm = bytearray()
                try:
                    async for f in self._tts.run_tts(text, "cue-warm"):
                        audio = getattr(f, "audio", None)
                        if audio:
                            pcm.extend(audio)
                except Exception:  # noqa: BLE001 — one bad cue must not cost the rest
                    logger.warning("cue warm failed for {!r}", text[:24])
                    continue
                if not pcm:
                    continue
                await self.push_frame(
                    OutputTransportMessageFrame(message={
                        "type": "cue_audio",
                        "text": text,
                        "voice": voice,
                        "rate": getattr(self._tts, "sample_rate", 24000),
                        "pcm_b64": base64.b64encode(bytes(pcm)).decode(),
                    }),
                    FrameDirection.DOWNSTREAM,
                )
                # Yield between phrases: this runs while he may already be talking, and a synth burst
                # holding the event loop would land straight on his first turn.
                await asyncio.sleep(0.4)
            logger.info("cue voice pack sent ({} phrases, voice={})", len(self._CUES), voice)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("cue warm failed")


class SemanticRecall(FrameProcessor):
    """Per-turn SEMANTIC memory (audit item #5): before the LLM answers, retrieve the most
    MEANING-similar past messages (from the never-trimmed memory_vec via the embed service) and
    prepend them to the current user message as bracketed context, then RESTORE it after the turn so
    the running context doesn't accumulate. Breaks the 400-row recency window. Gated EDIT_SEMANTIC.
    """

    def __init__(self, memory, context) -> None:
        super().__init__()
        self._memory = memory
        self._context = context
        self._restore = None   # (msg_dict, original_content)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if _SEMANTIC and isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._inject()
        if isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame, EndFrame, CancelFrame)):
            self._undo()
        await self.push_frame(frame, direction)

    async def _inject(self) -> None:
        try:
            last = None
            for m in self._context.get_messages():
                if isinstance(m, dict) and m.get("role") == "user":
                    last = m
            if last is None:
                return
            c = last.get("content")
            if isinstance(c, str):
                qtext = c
            elif isinstance(c, list):
                qtext = " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                return
            if not qtext.strip():
                return
            recall = await asyncio.to_thread(self._memory.semantic_recall, qtext)
            if recall and isinstance(c, str):
                self._restore = (last, c)
                last["content"] = f"[{recall}]\n{c}"
        except Exception:  # noqa: BLE001 — recall is a nice-to-have, never break the turn
            pass

    def _undo(self) -> None:
        try:
            if self._restore is not None:
                msg, original = self._restore
                msg["content"] = original
                self._restore = None
        except Exception:  # noqa: BLE001
            pass


def _last_user_text(context) -> str:
    try:
        messages = context.messages
    except Exception:  # noqa: BLE001
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                )
    return ""


# Words people put IN FRONT of a greeting. Anchoring the chit-chat test at position zero let any of
# these defeat it: «йо как дела» does not START with «как дела», so it was classified as a real
# question and earned a «сейчас посмотрю» — on a greeting, with nothing to look at.
_LEAD_INS = (
    "йо", "эй", "хей", "слушай", "слышь", "смотри", "короче", "ну", "а", "о", "ой", "блин",
    "так", "вот", "давай", "кстати", "ладно", "окей", "ок", "привет", "здарова", "хай",
)


def _substantive(text: str) -> bool:
    """True when the turn is a real request — the only case that may earn a spoken filler."""
    t = (text or "").strip().lower().lstrip("—-–,. ")
    if len(t.split()) <= 1:
        return False
    # Peel leading interjections before the chit-chat test, so «йо, как дела» is judged as «как дела».
    words = t.split()
    while words and words[0].strip(",.!?…") in _LEAD_INS:
        words.pop(0)
    stripped = " ".join(words)
    if not stripped or len(stripped.split()) <= 1:
        return False
    if any(stripped.startswith(p) for p in _CHAT_PREFIXES):
        return False
    # A short utterance that merely CONTAINS a greeting is still small talk («ну что, как дела»).
    if len(stripped.split()) <= 5 and any(p in stripped for p in _CHAT_PREFIXES if len(p) > 3):
        return False
    return True


class AppCommandFilter(FrameProcessor):
    """Strips command blocks out of the LLM text stream (so TTS never speaks them) and acts on them:

    * ``[OPEN: имя|ссылка]``  → urgent ``open_app`` message — the iPhone opens it
    * ``[TIMER: сек | текст]`` → server-side timer; on fire she SAYS the label (TTSSpeakFrame), the
      chat shows it, and the client gets ``timer_done`` (spoken locally even if the turn is closed)

    Sits between the LLM and the assistant-text tap. Incremental: a marker may be split across
    streamed TextFrames, so a possible prefix is held back and flushed on response end.
    """

    # «[TG:» is here even though the SHIM currently executes it, not this filter. The moment the
    # brain stops going through the shim, an unknown marker is no longer stripped — and Silero
    # would READ THE BLOCK ALOUD. «Отправь молча в телеграм» would become «прочитай вслух», and
    # «стоп встреча» would recite the meeting notes to whoever is in the room. Recognising it here
    # costs nothing today and removes that trap permanently; the body is dropped rather than acted
    # on, because the shim still owns delivery.
    # «[TGFILE:» по той же причине, и это не теория: блок ей объявлен (config.py), а здесь его не
    # было — от чтения вслух его спасал только шим, вырезающий блок раньше. Один режим, где фильтр
    # шима не применяется, и она продиктует путь к файлу голосом. Порядок важен: «[TGFILE:» стоит
    # ПЕРЕД «[TG:» — они делят префикс, и короткий, проверенный первым, съел бы длинный.
    _MARKS = ("[OPEN:", "[TIMER:", "[AGENT:", "[DELPROJECT:", "[MEMO:", "[ROUTE:", "[TGFILE:",
              "[TG:", "[DRAW:")

    def __init__(self, memory=None, persona=None) -> None:
        super().__init__()
        self._memory = memory       # MemoryStore, for [MEMO:] durable facts (item #4)
        self._persona = persona     # for [ROUTE:] — his live coordinates are the route's origin
        self._buf = ""
        self._timers: list[asyncio.TimerHandle] = []
        self._draws: list = []               # live DrawSessions, cancelled on barge-in

    async def _emit_text(self, text: str) -> None:
        if text:
            await self.push_frame(TextFrame(text), FrameDirection.DOWNSTREAM)

    async def _open(self, app: str) -> None:
        app = app.strip()
        if "://" not in app:            # app NAMES compare lowercase; URLs are case-sensitive
            app = app.lower()
        if not app:
            return
        logger.info("open_app -> {}", app)
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message={"type": "open_app", "app": app}),
            FrameDirection.DOWNSTREAM,
        )

    async def _set_timer(self, body: str) -> None:
        try:
            secs_raw, _, label = body.partition("|")
            secs = max(1, min(24 * 3600, int(float(secs_raw.strip()))))
            label = label.strip() or "время вышло"
        except Exception:  # noqa: BLE001
            logger.warning("bad TIMER block: {!r}", body[:40])
            return
        logger.info("timer set: {}s «{}»", secs, label)
        # Persist BEFORE arming: the in-memory handle dies with the socket (phone into a pocket, app
        # backgrounded), and the client's Напоминания tab would keep counting down to a fire that can
        # never happen. A row id of 0 means storage refused it — arm anyway, it is still his timer.
        rid = 0
        if self._memory is not None:
            rid = await asyncio.to_thread(self._memory.add_timer, time.time() + secs, label)
        # Tell the client so the Напоминания tab can show a live countdown.
        await self.push_frame(
            OutputTransportMessageUrgentFrame(
                message={"type": "timer_set", "label": label, "secs": secs}
            ),
            FrameDirection.DOWNSTREAM,
        )

        def fire() -> None:
            asyncio.ensure_future(self._fire_timer(label, rid))

        self._timers.append(asyncio.get_running_loop().call_later(secs, fire))

    async def _draw(self, subject: str) -> None:
        """Start a drawing. Gated on EDIT_DRAW so the capability can ship dark and be switched on
        after the wire has been watched for a day."""
        subject = subject.strip()
        if not subject or os.environ.get("EDIT_DRAW", "0") != "1":
            return
        try:
            import draw as draw_mod
            from config import Config
        except Exception as e:  # noqa: BLE001
            logger.warning("[draw] module unavailable: {}", e)
            return
        cfg = Config()
        logger.info("DRAW -> {!r}", subject[:60])

        async def push(msg: dict) -> None:
            await self.push_frame(
                OutputTransportMessageUrgentFrame(message=msg), FrameDirection.DOWNSTREAM
            )

        session = draw_mod.DrawSession(subject, push, cfg.proxy_url, cfg.proxy_api_key)
        self._draws = [d for d in self._draws if d._task and not d._task.done()]
        # One picture at a time: a second marker in the same answer replaces the first, because the
        # stage holds exactly one scene and the phone drops frames for any other id.
        for old in self._draws:
            old.cancel()
        self._draws.append(session)
        session.start()

    async def _agent(self, task: str) -> None:
        """Hand a multi-step task to the autonomous agent on the server (background + Telegram)."""
        task = task.strip()
        if not task:
            return
        logger.info("AGENT task -> {!r}", task[:80])
        # Single-quote-safe: edit-agent runs detached; it messages Telegram when done.
        esc = task.replace("'", "'\\''")
        cmd = f"nohup edit-agent '{esc}' sonnet >/dev/null 2>&1 &"
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env={**os.environ, "PATH": os.environ.get("PATH", "") + ":/usr/local/bin"},
            )
            await proc.wait()   # returns immediately — the work is detached inside
        except Exception:  # noqa: BLE001 - a failed launch must not break the turn
            logger.exception("agent launch failed")

    async def _route(self, body: str) -> None:
        """`[ROUTE: место | режим]` → a real ROUTE on his phone, not a search screen.

        The whole point of the block: «открыла 2ГИС с поиском» still leaves him tapping «Маршрут»
        himself, one-handed, usually while already walking to the car.

        Order of preference, and why:
          1. 2GIS with coordinates — his maps app, best local data. Needs numbers, so a named
             place is geocoded first.
          2. Apple Maps by NAME — the only documented way to route to plain text. Used when
             geocoding finds nothing, because a route to roughly-the-right-place beats no route.
        The URL is handed over RAW (Cyrillic unescaped): the client percent-encodes exactly once, so
        pre-encoding here would double-escape it (`%D0%B0` → `%25D0%25B0`).
        """
        dest_raw, _, mode_raw = body.partition("|")
        dest = dest_raw.strip()
        if not dest:
            return
        mode = _ROUTE_MODES.get(mode_raw.strip().lower(), "car")

        coords = _parse_coords(dest)
        if coords is None:
            lat0 = float(getattr(self._persona, "user_lat", 0.0) or 0.0)
            lon0 = float(getattr(self._persona, "user_lon", 0.0) or 0.0)
            city = (getattr(self._persona, "user_city", "") or "").strip()
            query = dest if (city and city.lower() in dest.lower()) or not city else f"{dest}, {city}"
            coords = await asyncio.to_thread(_geocode, query, lat0, lon0)

        if coords is not None:
            lat, lon = coords
            # 2GIS wants lon,lat. `/go` is deliberately NOT appended: starting turn-by-turn guidance
            # unasked is a much louder action than showing the route.
            url = f"dgis://2gis.ru/routeSearch/rsType/{mode}/to/{lon:.6f},{lat:.6f}"
            logger.info("route -> 2gis {} {:.5f},{:.5f} ({!r})", mode, lat, lon, dest[:40])
        else:
            apple = _APPLE_MODES.get(mode, "driving")
            url = f"https://maps.apple.com/directions?destination={dest}&mode={apple}"
            logger.info("route -> apple maps by name ({!r})", dest[:40])

        await self.push_frame(
            OutputTransportMessageUrgentFrame(message={"type": "open_app", "app": url}),
            FrameDirection.DOWNSTREAM,
        )

    async def _memo(self, body: str) -> None:
        """Persist a durable fact the user asked her to remember (audit item #4). Gated on EDIT_FACTS:
        when off the block is still stripped from the spoken/typed stream but NOT stored."""
        fact = body.strip()
        if not fact or not _FACTS or self._memory is None:
            return
        logger.info("MEMO -> {!r}", fact[:60])
        try:
            await asyncio.to_thread(self._memory.remember_fact, fact)
        except Exception:  # noqa: BLE001 — a failed memo must not break the turn
            logger.exception("memo persist failed")

    async def _del_project(self, name: str) -> None:
        name = name.strip()
        if not name:
            return
        logger.info("DELPROJECT -> {!r}", name[:40])
        esc = name.replace("'", "'\\''")
        try:
            proc = await asyncio.create_subprocess_shell(
                f"edit-projects del '{esc}'",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                env={**os.environ, "PATH": os.environ.get("PATH", "") + ":/usr/local/bin"},
            )
            await proc.wait()
        except Exception:  # noqa: BLE001
            logger.exception("del_project failed")

    async def _fire_timer(self, label: str, rid: int = 0) -> None:
        # Marked BEFORE it speaks: a marked row that goes unspoken is merely silent, an unmarked
        # row that WAS spoken and then lost to a reconnect is announced a second time.
        if rid and self._memory is not None:
            await asyncio.to_thread(self._memory.mark_timer_fired, rid)
        logger.info("timer FIRED: {}", label)
        phrase = f"Таймер: {label}"
        await self.push_frame(TTSSpeakFrame(phrase), FrameDirection.DOWNSTREAM)
        await self.push_frame(
            OutputTransportMessageUrgentFrame(
                message={"type": "timer_done", "label": label}
            ),
            FrameDirection.DOWNSTREAM,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        # TTSSpeakFrame is EXCLUDED, and that exclusion is the whole reason the filler worked.
        # In pipecat it subclasses TextFrame, so this filter used to swallow «Секунду, гляну.»
        # into the same buffer where the LLM's streamed answer is assembled — the filler lost its
        # «speak this now» identity and came out QUEUED BEHIND the answer. That is what he heard as
        # «говорит сейчас посмотрю после каждого ответа». A speak directive carries a fixed string,
        # never a half-arrived command marker, so it must pass through untouched.
        if isinstance(frame, TextFrame) and not isinstance(
            frame, (TranscriptionFrame, InterimTranscriptionFrame, TTSSpeakFrame)
        ) and direction == FrameDirection.DOWNSTREAM:
            self._buf += frame.text
            out = ""
            while self._buf:
                i = self._buf.find("[")
                if i == -1:
                    out += self._buf
                    self._buf = ""
                    break
                out += self._buf[:i]
                self._buf = self._buf[i:]
                upper = self._buf.upper()
                matched = None
                partial = False
                for mark in self._MARKS:
                    if upper.startswith(mark):
                        matched = mark
                        break
                    if len(self._buf) < len(mark) and mark.startswith(upper):
                        partial = True
                if matched is not None:
                    j = self._buf.find("]")
                    if j == -1:
                        break                    # marker not closed yet — wait
                    body = self._buf[len(matched):j]
                    if matched == "[OPEN:":
                        await self._open(body)
                    elif matched == "[AGENT:":
                        await self._agent(body)
                    elif matched == "[DELPROJECT:":
                        await self._del_project(body)
                    elif matched == "[ROUTE:":
                        await self._route(body)
                    elif matched == "[MEMO:":
                        await self._memo(body)
                    elif matched == "[DRAW:":
                        await self._draw(body)
                    elif matched in ("[TG:", "[TGFILE:"):
                        # Recognised so it is never SPOKEN; not acted on, because the shim still
                        # performs the delivery. Without this branch the `else` below would read
                        # «[TG: купить кофе]» as a TIMER and set one named after the message.
                        pass
                    else:
                        await self._set_timer(body)
                    self._buf = self._buf[j + 1:]
                    continue
                if partial:
                    break                        # possible marker prefix — wait for more text
                out += self._buf[0]
                self._buf = self._buf[1:]
            await self._emit_text(out)
            return                               # original frame replaced by the cleaned text
        if isinstance(frame, LLMFullResponseEndFrame):
            tail, self._buf = self._buf, ""
            await self._emit_text(tail)
        if isinstance(frame, InterruptionFrame):
            self._buf = ""                      # cancelled response → drop its held tail
            for d in self._draws:
                d.cancel()
            self._draws = []
        if isinstance(frame, (EndFrame, CancelFrame)):
            self._buf = ""
            # The HANDLES belong to a loop that is about to die, so they go. The ROWS must not: they
            # are what lets an unfired timer either be re-armed or reported on the next connect.
            for handle in self._timers:
                handle.cancel()
            self._timers = []
        await self.push_frame(frame, direction)


class TypedTurnTTSGate(FrameProcessor):
    """Swallows LLM TextFrames while the current response is a TYPED chat turn.

    Typed answers are long markdown: synthesizing them (a) reads headers aloud and (b) the
    CPU synth bursts stall the event loop so the streamed text reaches the app in freezes.
    The text already went to the client via the assistant tap upstream; TTS just skips it.
    TTSSpeakFrame (fillers, timers, enrollment prompts) is NOT a TextFrame — always passes.
    """

    def __init__(self, persona) -> None:
        super().__init__()
        self._persona = persona
        self._excused = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        spoken_text = (
            isinstance(frame, TextFrame)
            and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame))
            and direction == FrameDirection.DOWNSTREAM
        )
        # The shim answers HTTP 200 with an error IN THE BODY, so a dead upstream arrives here as
        # ordinary assistant text — «You've hit your weekly limit…». It was going straight to TTS,
        # where Silero refused the Latin script and synthesized NOTHING: he asked a question and got
        # silence, 49 times in three days, with no way to tell a broken brain from a broken app.
        # Say it in her own language instead, once per response.
        if spoken_text and _is_upstream_failure(frame.text):
            if not self._excused:
                self._excused = True
                # Log the WHOLE notice, not the first 60 characters. The truncation hid the actual
                # cause: «…error result: Reache» read as a quota message and was diagnosed as one
                # for hours, when the full text said «Reached maximum number of turns (1)».
                logger.warning("upstream failure text suppressed from TTS: {!r}", frame.text[:400])
                await self.push_frame(
                    TTSSpeakFrame(_upstream_excuse(frame.text)),
                    FrameDirection.DOWNSTREAM,
                )
            return
        if spoken_text and getattr(self._persona, "current_response_typed", False):
            return                      # text-only turn: nothing to speak
        # The response is over — normally at its end frame, but a barge-in/teardown kills it without
        # one. Either way the gate must reopen: a stuck flag mutes every following spoken answer.
        if (
            isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame, EndFrame, CancelFrame))
            and self._persona is not None
        ):
            self._persona.current_response_typed = False
            self._excused = False
            if isinstance(frame, LLMFullResponseEndFrame):
                # A finished response also retires the typed flag itself. NOT on an interruption:
                # that one may be the typed turn's OWN turn-start, fired before the LLM request was
                # built — a leaked flag is dropped where it is consumed (ProxyLLMService).
                self._persona.typed_turn = False
                self._persona.typed_text = ""
        await self.push_frame(frame, direction)


class FillerCue:
    """The hand-off between deciding to speak and being able to.

    The decision must happen UPSTREAM of the LLM (only there does a context frame announce that
    a turn was dispatched — the LLM consumes it). The speaking must happen DOWNSTREAM of the LLM,
    because a frame pushed from upstream enters the LLM processor's queue and is released only
    when the answer finishes: measured, the timer woke on time at 2.50 s and the words still
    arrived after the reply. One queue, filled by the decider, drained by the voice.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()

    def say(self, text: str) -> None:
        try:
            self.queue.put_nowait(text)
        except Exception:  # noqa: BLE001 — a dropped filler must never break a turn
            pass


class FillerVoice(FrameProcessor):
    """Speaks whatever the upstream filler decided, from BELOW the LLM.

    Placed after the TTS gate, so its lines are independent of the in-flight request. It owns a
    task rather than acting on passing frames, because during the wait no frames pass — that is
    precisely the silence it exists to fill.
    """

    def __init__(self, cue: "FillerCue") -> None:
        super().__init__()
        self._cue = cue
        self._task: "asyncio.Task | None" = None

    async def _pump(self) -> None:
        while True:
            text = await self._cue.queue.get()
            await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)
            await self.push_frame(
                OutputTransportMessageUrgentFrame(message={"type": "assistant_text", "text": text}),
                FrameDirection.DOWNSTREAM,
            )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._task is None:
            self._task = asyncio.create_task(self._pump())
        if isinstance(frame, (EndFrame, CancelFrame)) and self._task is not None:
            self._task.cancel()
            self._task = None
        await self.push_frame(frame, direction)


class ResponseWatch:
    """Shared flag: 'the LLM response has started streaming'. Set by the assistant-text tap; awaited
    by the delayed filler so it stays SILENT when the answer is fast."""

    def __init__(self) -> None:
        self.event = asyncio.Event()


# De-robotized filler (audit item #3), gated EDIT_FILLER_V2 (default OFF → the single «Секунду, гляну.»
# below, byte-for-byte prior behaviour). When ON: pick a line in the session's CHARACTER and skip the
# filler entirely on typed (keyboard) turns. Grace is env-configurable but KEPT at 6.0 — lower re-fires
# the filler on nearly every turn («бесит, много говорит»); tune only while watching TTFB logs.
_FILLER_V2 = os.environ.get("EDIT_FILLER_V2", "0") == "1"
# Temporary: logs which frames reach the filler and why it arms or does not.
_FILLER_DEBUG = os.environ.get("EDIT_FILLER_DEBUG", "0") == "1"
_FILLER_POOL = {
    "jarvis":  ["Одну минуту, " + USER_NAME + ".", "Секунду, проверяю.", "Уже смотрю."],
    "eugene":  ["Так, секунду.", "Минуту — гляну.", "Сейчас посмотрю."],
    "aidar":   ["Так, гляну.", "Секунду, смотрю.", "Момент."],
    "xenia":   ["Ой, секунду, гляну.", "Сейчас всё гляну.", "Минутку."],
    "baya":    ["Секунду, посмотрю.", "Момент, гляну.", "Сейчас."],
    "kseniya": ["Ага, секунду!", "Мигом гляну.", "Секунду, смотрю!"],
}
_FILLER_NEUTRAL = ["Секунду, гляну.", "Так, гляну.", "Секунду.", "Сейчас, момент."]


class SearchFillerInjector(FrameProcessor):
    """Speaks a short «Секунду…» ONLY when the answer is actually slow.

    The old always-on filler annoyed the user («бесит — много говорит сейчас посмотрю»). Now the
    filler is DELAYED: when a substantive query is dispatched, a timer starts; if the LLM's first
    text arrives within the grace window (warm haiku ≈2.5 s) nothing is spoken — the answer just
    comes. Only genuinely slow turns (web search ~15 s, photo ~20 s, cold start) get the filler.
    A :class:`TTSSpeakFrame` bypasses the TTS sentence aggregator so the voice fires immediately
    when the timer decides to speak.
    """

    _FILLERS = ["Секунду, гляну."]
    # 6 s (was 3): ordinary turns get their first token well under this, so the filler no longer
    # fires on every question — only genuinely slow ones (web search ~15 s, photo, cold start).
    _GRACE_SECS = float(os.environ.get("EDIT_FILLER_GRACE", "6.0"))
    # How long the filler's promise stands before she admits nothing came back.
    _DEAD_SECS = float(os.environ.get("EDIT_FILLER_DEAD", "18.0"))
    # Halfway through the wait she says she is still on it, so a long search never sounds like a hang.
    _STILL_SECS = float(os.environ.get("EDIT_FILLER_STILL", "9.0"))

    def __init__(self, watch: ResponseWatch | None = None, persona=None,
                 cue: "FillerCue | None" = None) -> None:
        super().__init__()
        # One-shot: opened by «he stopped», closed by the context frame that carries his words.
        self._arm_window = False
        # Where the decision goes. Pushing frames from here does not work — see FillerCue.
        self._cue = cue
        self._watch = watch
        self._persona = persona     # for character-keyed filler wording (item #3)
        self._pending: asyncio.Task | None = None

    def _pick_filler(self) -> str:
        if not _FILLER_V2:
            return random.choice(self._FILLERS)
        char = (getattr(self._persona, "character", None) or "").lower()
        return random.choice(_FILLER_POOL.get(char, _FILLER_NEUTRAL))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, (CancelFrame, EndFrame, InterruptionFrame)):
            # Teardown/barge-in: a sleeping filler task must not outlive the turn (it also blocked
            # task cancellation on session close — the "timed out waiting for task" warnings).
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
        # ARMING, in the frame order MEASURED on the live server:
        #   VADUserStoppedSpeaking -> UserStoppedSpeaking -> LLMContextFrame(user turn)
        #   ... answer ...          -> LLMContextFrame(assistant turn)
        #
        # TranscriptionFrames never get here — the user aggregator upstream absorbs them — so the
        # text can only come from a context frame. But there are TWO context frames per turn, and
        # the old code armed on both: the FIRST carries no user text yet (`_substantive('')` is
        # False, so nothing armed), and the SECOND arrives only after she has finished answering,
        # which armed a timer that then spoke «сейчас посмотрю» into the silence AFTER the reply.
        # That is the «бесит после каждого ответа».
        #
        # So: the user stopping OPENS a one-shot window; the next context frame — the one that
        # actually carries his words — arms the filler and closes the window. The late frame finds
        # the window shut.
        if (isinstance(frame, (UserStoppedSpeakingFrame, VADUserStoppedSpeakingFrame))
                and direction == FrameDirection.DOWNSTREAM):
            self._arm_window = True
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
        if (isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM
                and self._arm_window):
            self._arm_window = False
            text = _last_user_text(frame.context)
            # v2: never speak a filler over a typed (keyboard) turn — that answer isn't voiced.
            typed = _FILLER_V2 and getattr(self._persona, "typed_turn", False)
            ok = _substantive(text)
            if _FILLER_DEBUG:
                logger.info("[filler] вооружаюсь? substantive={} typed={} text={!r}", ok, typed, text[:40])
            if ok and not typed:
                if self._watch is not None:
                    self._watch.event.clear()
                self._pending = asyncio.create_task(self._delayed_filler())
        # The answer finishing retires a still-sleeping filler: nothing may promise to look after
        # she has already told him the answer.
        if isinstance(frame, LLMFullResponseEndFrame) and self._pending is not None:
            self._pending.cancel()
            self._pending = None
        await self.push_frame(frame, direction)

    def _speak(self, text: str) -> None:
        """Hand the line to FillerVoice below the LLM; never push it from here."""
        if self._cue is not None:
            self._cue.say(text)

    async def _delayed_filler(self) -> None:
        armed_at = time.monotonic()
        if _FILLER_DEBUG:
            logger.info("[filler] таймер завёлся, порог={} с", self._GRACE_SECS)
        try:
            if self._watch is not None:
                try:
                    await asyncio.wait_for(self._watch.event.wait(), timeout=self._GRACE_SECS)
                    if _FILLER_DEBUG:
                        logger.info("[filler] ответ пошёл через {:.2f} с — молчу",
                                    time.monotonic() - armed_at)
                    return                      # answer already streaming — stay silent
                except asyncio.TimeoutError:
                    if _FILLER_DEBUG:
                        logger.info("[filler] ПРОСНУЛСЯ через {:.2f} с (просил {} с) — говорю",
                                    time.monotonic() - armed_at, self._GRACE_SECS)
                    pass
            else:
                await asyncio.sleep(self._GRACE_SECS)
            filler = self._pick_filler()
            self._speak(filler)
            # «Сейчас посмотрю» is a PROMISE. If the answer never starts — upstream down, quota
            # spent, request lost — the old code simply went quiet, and the last thing he heard was
            # her promising to look. Silence after a promise reads as «сломалась» with no way to
            # tell what broke, so the promise is now either kept or explicitly withdrawn.
            if self._watch is None:
                return
            # A long search is 15-40 s. One «секунду» at the front and then nothing is indistinguishable
            # from a hang, so the wait STAYS a conversation: she says she is still on it, once,
            # halfway through, and only admits defeat if the answer truly never starts.
            try:
                await asyncio.wait_for(self._watch.event.wait(), timeout=self._STILL_SECS)
                return
            except asyncio.TimeoutError:
                still = random.choice(["Ещё ищу.", "Секунду, почти.", "Копаю дальше."])
                self._speak(still)
            try:
                await asyncio.wait_for(self._watch.event.wait(),
                                       timeout=max(1.0, self._DEAD_SECS - self._STILL_SECS))
                return                          # the answer started — nothing to apologise for
            except asyncio.TimeoutError:
                pass
            giveup = random.choice([
                "Не получилось — связь с мозгом пропала. Повтори, пожалуйста.",
                "Не дождалась ответа. Спроси ещё раз.",
            ])
            logger.warning("filler promised but no response in {}s — telling him", self._DEAD_SECS)
            self._speak(giveup)
        except asyncio.CancelledError:
            pass


class ClientEventTap(FrameProcessor):
    """Pass-through processor that mirrors selected frames to the client.

    A single class handles every event type; each pipeline slot enables only
    the flags relevant to its position, so the same instance never double-emits.
    """

    def __init__(
        self,
        *,
        emit_transcripts: bool = False,
        emit_assistant_text: bool = False,
        emit_speaking: bool = False,
        watch: ResponseWatch | None = None,
        memory=None,
    ) -> None:
        super().__init__()
        self._emit_transcripts = emit_transcripts
        self._emit_assistant_text = emit_assistant_text
        self._emit_speaking = emit_speaking
        self._watch = watch          # signalled on the first LLM text (silences the delayed filler)
        self._memory = memory        # cross-session MemoryStore (user finals / assistant replies)
        self._partial_seq = 0
        self._assistant_buf: list[str] = []
        # Streaming-STT bookkeeping: phrases arriving WHILE the VAD hears speech are partials of
        # one utterance; the utterance finalizes when the turn closes. Batch STT (GigaAM fallback)
        # lands its single TranscriptionFrame AFTER the VAD stop → emitted as user_final directly.
        self._vad_speaking = False
        self._utterance: list[str] = []
        # Отметки замера реплики (см. _log_turn_timing). None = такта сейчас нет.
        self._t_stopped: float | None = None
        self._t_final: float | None = None

    def _log_turn_timing(self) -> None:
        """Одна строка на реплику: сколько он ждал и на что это ушло.

        Именно ЭТО число человек называет «тормозит» — не время модели и не время синтеза, а
        молчание между «он замолчал» и «она заговорила». Разложено на две части, потому что лечатся
        они в разных местах: слух (VAD + расшифровка) и ответ (мозг + синтез).
        """
        if not self._t_stopped:
            return
        now = time.monotonic()
        total = now - self._t_stopped
        if self._t_final:
            logger.info(
                "такт: {:.2f} с всего (слух {:.2f} + ответ {:.2f})",
                total, self._t_final - self._t_stopped, now - self._t_final,
            )
        else:
            # Ответ пошёл, а расшифровки в этом такте не было: типографский ход (текст из чата),
            # её собственная реплика или заполнитель. Это не тишина после вопроса, и складывать
            # его с остальными нельзя — поэтому он помечен отдельно, а не выброшен.
            logger.info("такт: {:.2f} с всего (без расшифровки)", total)
        self._t_stopped = None
        self._t_final = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        message: dict | None = None

        # -- interim / final ASR --------------------------------------------
        if self._emit_transcripts:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._vad_speaking = True
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._vad_speaking = False
            # Order matters: InterimTranscriptionFrame is a subclass-sibling of
            # TranscriptionFrame, so check the interim type first.
            if isinstance(frame, InterimTranscriptionFrame):
                message = {
                    "type": "partial",
                    "text": frame.text,
                    "seq": self._partial_seq,
                }
                self._partial_seq += 1
            elif isinstance(frame, TranscriptionFrame):
                logger.debug("tap: transcription {!r} (vad_speaking={})", frame.text[:50], self._vad_speaking)
                if self._vad_speaking:
                    # Streaming phrase mid-utterance (T-one) → live partial of the running text.
                    self._utterance.append(frame.text)
                    message = {
                        "type": "partial",
                        "text": " ".join(self._utterance),
                        "seq": self._partial_seq,
                    }
                    self._partial_seq += 1
                else:
                    # Batch final after silence (GigaAM/whisper) — a whole utterance in one frame.
                    text = " ".join(self._utterance + [frame.text])
                    self._utterance = []
                    message = {"type": "user_final", "text": text}
                    self._partial_seq = 0
                    if self._memory is not None:
                        # SQLite off the loop: a writer holding the lock (embed service, recap job)
                        # would otherwise stall the audio pipeline for seconds mid-turn.
                        await asyncio.to_thread(self._memory.append, "user", text)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                logger.debug("tap: user turn stopped (buffered={})", len(self._utterance))
                if self._utterance:
                    # The user turn closed — flush the streamed phrases as the authoritative final.
                    text = " ".join(self._utterance)
                    message = {"type": "user_final", "text": text}
                    self._utterance = []
                    self._partial_seq = 0
                    if self._memory is not None:
                        await asyncio.to_thread(self._memory.append, "user", text)

        # -- interruption: the in-flight response is dead — drop its buffered text ----
        if self._emit_assistant_text and isinstance(frame, InterruptionFrame):
            self._assistant_buf.clear()

        # -- assistant text (STREAMED as it generates) ----------------------
        if self._emit_assistant_text:
            # Emit the RUNNING accumulated text on every LLM TextFrame (not just at response end).
            # This lets the client see the web-search filler («Секунду, посмотрю») EARLY — it uses
            # that to hold the turn open during the ~15 s search instead of hanging up — and makes the
            # transcript update live. Transcription subclasses are also TextFrames, so exclude them.
            if isinstance(frame, TextFrame) and not isinstance(
                frame, (TranscriptionFrame, InterimTranscriptionFrame)
            ):
                if self._watch is not None:
                    self._watch.event.set()       # answer streaming → delayed filler stays silent
                self._assistant_buf.append(frame.text)
                running = "".join(self._assistant_buf).strip()
                if running:
                    message = {"type": "assistant_text", "text": running}
            elif isinstance(frame, LLMFullResponseEndFrame):
                final = "".join(self._assistant_buf).strip()
                if final and self._memory is not None:
                    await asyncio.to_thread(self._memory.append, "assistant", final)
                self._assistant_buf.clear()
                # Explicit "the whole response (filler + web search + answer) is finished" signal. The
                # client keeps the turn OPEN until this arrives, so it can never hang up mid-response
                # and drop the answer audio (the «сессия закрывается, я не слышу ответ» bug).
                message = {"type": "response_end"}

        # -- TTS speaking markers -------------------------------------------
        if self._emit_speaking:
            if isinstance(frame, TTSStartedFrame):
                message = {"type": "speaking_start"}
                self._log_turn_timing()
            elif isinstance(frame, TTSStoppedFrame):
                message = {"type": "speaking_end"}

        # -- ЗАМЕР РЕПЛИКИ ---------------------------------------------------
        # Про задержку в этом сервере не было НИ ОДНОГО числа: тюнинг VAD и выбор модели делались
        # на ощущениях, а ощущение задержки — самая ненадёжная величина, какая есть. Три отметки
        # снимаются здесь, потому что этот процессор стоит ПОСЛЕ синтеза и видит все три кадра:
        # VAD-стоп (он замолчал), расшифровку (его услышали) и первый звук ответа (она заговорила).
        #
        # Считается monotonic: системные часы могут прыгнуть от ntp прямо посреди реплики.
        if self._emit_speaking:
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                self._t_stopped = time.monotonic()
                self._t_final = None
            elif isinstance(frame, TranscriptionFrame) and self._t_stopped and not self._t_final:
                self._t_final = time.monotonic()

        # Always forward the original frame so the rest of the pipeline works.
        await self.push_frame(frame, direction)

        # Then emit the derived client event (downstream, toward the transport). Transcript events
        # (partial / user_final) go URGENT: they are emitted exactly when a user turn opens, and the
        # turn-start interruption broadcast PURGES ordinary queued frames — a plain user_final was
        # silently wiped before reaching the wire (observed). Urgent frames bypass the purge and the
        # TTS-paced queue (same fix as pong).
        if message is not None:
            frame_cls = (
                OutputTransportMessageUrgentFrame
                if message.get("type") in ("partial", "user_final")
                else OutputTransportMessageFrame
            )
            await self.push_frame(frame_cls(message=message), FrameDirection.DOWNSTREAM)
