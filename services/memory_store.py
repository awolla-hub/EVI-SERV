"""Cross-session conversation memory — «чтобы помнил разговоры между сессиями».

A tiny SQLite log of every exchange. On each new WebSocket session the last few exchanges are
injected into the LLM context right after the system prompt, so E.D.I.T. remembers yesterday's
conversation. Writes happen from the event tap (user_final / response end); the table is trimmed
so it can never grow unbounded.
"""

from __future__ import annotations

import json
import os
_OWNER = __import__("os").environ.get("PYATNITSA_OWNER", "хозяин")
import re
import sqlite3
import threading
import time
import urllib.request

from loguru import logger

DB_PATH = os.environ.get("MEMORY_DB", "/opt/pyatnitsa/memory.db")

# Мусор, который НЕЛЬЗЯ подавать как «наш прошлый разговор». В журнале лежат сбойные ответы модели —
# отказы, рассуждения про инструкции, «это первое сообщение в чате, история сконструирована». Они
# отлично совпадают по смыслу с вопросами о прошлом (в них ведь про прошлое и говорится) и всплывали
# бы в выдаче первыми — то есть на «помнишь, мы решали?» она получала бы свой же старый сбой в
# качестве воспоминания. Ловится списком, потому что это узкий и стабильный класс фраз.
_META_JUNK = (
    "конфликт инструкц", "сконструирован", "первое сообщение в нашем", "я не могу выполнить",
    "как языковая модель", "как ии", "system prompt", "системный промпт", "мои инструкции",
    "не имею доступа к прошл", "истории нет",
)


def _is_meta_junk(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _META_JUNK)


def _ago_mark(age: float) -> str:
    """«[вчера] », «[3 ч назад] » — или пустая строка для свежего.

    Грубо и словами, а не датой: человек и сам помнит прошлое как «вчера» и «на той неделе», а
    «2026-08-04 09:33» в устной реплике пришлось бы ещё и произносить.
    """
    if age < 7200:                       # два часа — это всё ещё «сейчас», один разговор
        return ""
    if age < 86400:
        return f"[{int(age // 3600)} ч назад] "
    if age < 172800:
        return "[вчера] "
    if age < 604800:
        return f"[{int(age // 86400)} дн назад] "
    return "[давно] "
# Скользящий журнал разговоров. Было 400 — и таблица СТОЯЛА НА ЭТОМ ПОТОЛКЕ: всё, что старше двух
# суток плотного общения, стиралось. «Ты же вчера говорила» упиралось в физически удалённую строку.
# Четыре тысячи строк — это месяцы: 3,5 МБ базы при нынешнем размере записи, то есть ничего.
# В контекст по-прежнему уходит горстка последних (recent_messages), так что цена — только диск.
_MAX_ROWS = int(os.environ.get("EDIT_MEMORY_ROWS", "4000"))
# Semantic memory (audit item #5): copy every message into the never-trimmed memory_vec store and
# recall by meaning. Default OFF; needs the embed service (/opt/edit-embed) running.
_SEMANTIC = os.environ.get("EDIT_SEMANTIC", "0") == "1"
_EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8181")

# Trivial closers/acks that shouldn't be surfaced as "what we talked about" (recall, item #16).
_TRIVIAL = (
    "спасибо", "пока", "ок", "окей", "угу", "ага", "да", "нет", "хорошо", "ладно",
    "ясно", "понятно", "споки", "до свидания", "спс", "класс", "супер", "отлично",
)
# Matched against the WHOLE message, never as a prefix: most of these words also OPEN real
# sentences («Покажи, что там на кухне», «Ладно, тогда завтра в девять», «Нет времени, перенеси
# встречу»), so a prefix test silently deletes exactly the lines recall exists to surface.
_TRIVIAL_SET = frozenset(_TRIVIAL)

# «Запомни X» facts whose truth expires: a parking spot or a «сегодня»-plan is current for days,
# not forever. Matched rows keep a TTL and stop being INJECTED once it passes — they are never
# deleted, so «а где машина стояла в прошлый вторник» can still be answered.
_VOLATILE_RE = re.compile(
    r"парков|машина стоит|я сейчас|сегодня|на этой неделе|занял место", re.IGNORECASE
)
_VOLATILE_TTL = 3 * 86400.0

# Timers: hard bounds so a runaway can never queue a pile of announcements at connect.
_TIMER_MAX_LIVE = 10
_TIMER_MAX_AGE = 24 * 3600.0


def _ago(ts: float) -> str:
    """Compact Russian relative-time tag for a stored message (recall grounding)."""
    d = max(0.0, time.time() - ts)
    if d < 3600:
        return f"{int(d // 60)} мин назад"
    if d < 86400:
        return f"{int(d // 3600)} ч назад"
    days = int(d // 86400)
    if days == 1:
        return "вчера"
    if days < 5:
        return f"{days} дн назад"
    return "давно"


def _fact_key(text: str) -> str:
    """Comparison key for near-identical facts — case, punctuation and spacing carry no meaning
    here, so «Машина — у Ленина 15.» and «машина у ленина 15» are the SAME fact and must
    not both sit in the always-on system message."""
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def _normalize_seed(rows: list[dict]) -> list[dict]:
    """Make a seeded history legal before it reaches the model: it must open on a `user` turn and
    must never carry two same-role rows in a row.

    Not cosmetic — four paths append assistant rows with no user turn between them (greeting,
    pending delivery, presence initiative, ambient remark), so leading- and consecutive-assistant
    seeds are the normal case, and nothing downstream repairs them."""
    start = 0
    while start < len(rows) and rows[start].get("role") == "assistant":
        start += 1
    out: list[dict] = []
    for row in rows[start:]:
        role, content = row.get("role"), (row.get("content") or "")
        if out and out[-1]["role"] == role:
            out[-1]["content"] = (out[-1]["content"] + " " + content).strip()
            continue
        out.append({"role": role, "content": content})
    if len(out) != len(rows):
        logger.info("seed normalized: {} → {} rows", len(rows), len(out))
    return out


class MemoryStore:
    """Thread-safe append/recent over one SQLite table."""

    def __init__(self, path: str = DB_PATH) -> None:
        self._lock = threading.Lock()
        # Set only once the volatility columns are known to exist: every fact query must be able to
        # run against a pre-migration DB, so the column set is checked, never assumed.
        self._facts_ttl = False
        try:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("PRAGMA busy_timeout=15000")   # the embed service writes memory_vec too
            # WAL: readers and the writer stop blocking each other. Without it the embedding sweep
            # collided with the voice path on every batch — «sweep error: database is locked», 11
            # times in three days, each one a batch of memories silently left unembedded.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, text TEXT)"
            )
            # Durable, NEVER-trimmed facts the user explicitly asked her to remember (via a [MEMO:]
            # block — audit item #4). Kept SEPARATE from `messages` (which is recency-trimmed to
            # _MAX_ROWS) so «запомни, где я припарковался» is not silently forgotten next session.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, text TEXT, active INTEGER DEFAULT 1)"
            )
            # Volatility columns, added in place so an existing DB migrates on start. Each ALTER is
            # separately guarded: on every start after the first they raise "duplicate column", and
            # that must be a silent no-op rather than a dead memory store. The schema is shared with
            # out-of-process readers, which is why the columns are only ever ADDED, never reordered.
            for _ddl in (
                "ALTER TABLE facts ADD COLUMN kind TEXT DEFAULT 'fact'",
                "ALTER TABLE facts ADD COLUMN ttl_secs REAL",
            ):
                try:
                    self._db.execute(_ddl)
                except Exception:  # noqa: BLE001 - already migrated
                    pass
            _cols = {r[1] for r in self._db.execute("PRAGMA table_info(facts)").fetchall()}
            self._facts_ttl = {"kind", "ttl_secs"} <= _cols
            # Durable TIMERS. `_set_timer` arms an in-memory call_later that dies with the socket,
            # while the app keeps showing a live countdown — so a backgrounded phone means a timer
            # that never rings. A row here survives the disconnect: overdue ones are announced on the
            # next connect, future ones re-armed. Deliberately NOT folded into `pending`, whose
            # age-based queue cannot express an absolute fire time or future-vs-overdue.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS timers ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, fire_ts REAL, label TEXT, "
                "fired INTEGER DEFAULT 0)"
            )
            # Durable PENDING-INTENTS queue — the cheap first step of "background presence". Anything
            # decided/finished while the user was AWAY (an [AGENT:] task completing, a follow-up she
            # couldn't deliver) is queued here and spoken on his NEXT connect («пока тебя не было…»),
            # so her initiative survives disconnection instead of dying with the socket.
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS pending ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, text TEXT, "
                "delivered INTEGER DEFAULT 0)"
            )
            # Durable, NEVER-trimmed semantic store: every message is copied here (emb NULL); the
            # isolated embed service fills emb in the background. semantic_recall searches it by
            # MEANING, breaking the 400-row recency window («что решали про GPU три недели назад»).
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS memory_vec ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, text TEXT, emb BLOB)"
            )
            self._db.commit()
            logger.info("Conversation memory at {}", path)
        except Exception:  # noqa: BLE001 - memory must never break the pipeline
            logger.exception("Memory DB unavailable — running without cross-session memory")
            self._db = None

    def append(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if self._db is None or not text:
            return
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO messages (ts, role, text) VALUES (?, ?, ?)",
                    (time.time(), role, text[:1000]),
                )
                self._db.execute(
                    "DELETE FROM messages WHERE id NOT IN "
                    "(SELECT id FROM messages ORDER BY id DESC LIMIT ?)",
                    (_MAX_ROWS,),
                )
                if _SEMANTIC:
                    # Copy into the durable semantic store (emb filled later by the embed service).
                    self._db.execute(
                        "INSERT INTO memory_vec (ts, role, text, emb) VALUES (?, ?, ?, NULL)",
                        (time.time(), role, text[:1000]),
                    )
                self._db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("Memory append failed")

    def recent_messages(self, count: int = 24, max_chars: int = 4200) -> list[dict]:
        """Last `count` exchanges as LLM context messages, oldest first, size-capped so the extra
        prompt never hurts TTFT. The budget must bite on the OLDEST turns: the last thing he said is
        the one a fresh session cannot afford to drop.

        БЫЛО 10 реплик по 220 символов — полторы килобайты на всё прошлое. Разговор при этом идёт
        обрывками (распознаватель режет фразы на куски), поэтому десять «реплик» — это часто пять
        минут беседы, и она входила в новую сессию, не помня даже, чем закончился разговор час
        назад. Двадцать четыре по 400 — это связная нить, и всё равно втрое меньше, чем занимает
        сам системный промпт."""
        if self._db is None:
            return []
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT role, text, ts FROM messages ORDER BY id DESC LIMIT ?", (count,)
                ).fetchall()
            out, used = [], 0
            now = time.time()
            for role, text, ts in rows:  # newest first; keep newest within budget
                snippet = (text or "")[:400]
                if used + len(snippet) > max_chars:
                    break
                used += len(snippet)
                # ВРЕМЯ У ВОСПОМИНАНИЯ. Без него прошлое — плоская стенограмма: она помнит, ЧТО
                # было сказано, но не КОГДА, и «мы же вчера это обсуждали» для неё неотличимо от
                # «ты сказал это минуту назад». Метка ставится только на старое (от двух часов):
                # на свежих репликах она была бы шумом внутри одного живого разговора.
                out.append({"role": role, "content": snippet,
                            "_mark": _ago_mark(now - float(ts or now))})
            out.reverse()
            # Метка ставится там, где время МЕНЯЕТСЯ, а не на каждой строке. Иначе «[вчера]»
            # повторяется у каждой реплики, а соседние реплики одной роли ещё и склеиваются
            # нормализацией — и получается «[вчера] … [вчера] …» внутри одного сообщения.
            # Один раз в начале блока читается как отбивка в дневнике; на каждой строке — как шум.
            last_mark = ""
            for m in out:
                mark = m.pop("_mark", "")
                if mark and mark != last_mark:
                    m["content"] = mark + m["content"]
                    last_mark = mark
            return _normalize_seed(out)
        except Exception:  # noqa: BLE001
            logger.exception("Memory read failed")
            return []

    def recent_history(self, count: int = 60, max_chars: int = 12_000) -> list[dict]:
        """Last `count` messages with (near-)full text, oldest first — rides in the `voices`
        hello reply so a fresh app install can hydrate its chat screen. Unlike
        `recent_messages` this keeps whole replies (rows are already capped at 1000 chars
        on insert), trimming oldest-first when the total budget is exceeded."""
        if self._db is None:
            return []
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT role, text FROM messages ORDER BY id DESC LIMIT ?", (count,)
                ).fetchall()
            out, used = [], 0
            for role, text in rows:  # newest first; keep newest within budget
                if used + len(text) > max_chars:
                    break
                used += len(text)
                out.append({"role": role, "content": text})
            out.reverse()
            return out
        except Exception:  # noqa: BLE001
            logger.exception("Memory read failed")
            return []


    def remember_fact(self, text: str) -> None:
        """Persist a durable fact (from a ``[MEMO:]`` block). Never trimmed; drops exact AND
        near-identical repeats (same words, different case/punctuation) — telling her the same thing
        twice must not double the block that rides in every system message. The dedup stays local
        arithmetic on purpose: no embedding call may sit on this write path."""
        text = (text or "").strip()[:500]
        if self._db is None or not text:
            return
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT text FROM facts WHERE active=1 ORDER BY id DESC LIMIT 200"
                ).fetchall()
                key = _fact_key(text)
                if any(_fact_key(r[0]) == key for r in rows if r and r[0]):
                    return
                if self._facts_ttl:
                    volatile = bool(_VOLATILE_RE.search(text))
                    self._db.execute(
                        "INSERT INTO facts (ts, text, active, kind, ttl_secs) "
                        "VALUES (?, ?, 1, ?, ?)",
                        (
                            time.time(),
                            text,
                            "volatile" if volatile else "stable",
                            _VOLATILE_TTL if volatile else None,
                        ),
                    )
                else:
                    self._db.execute(
                        "INSERT INTO facts (ts, text, active) VALUES (?, ?, 1)",
                        (time.time(), text),
                    )
                self._db.commit()
                logger.info("fact remembered: {!r}", text[:60])
        except Exception:  # noqa: BLE001
            logger.exception("remember_fact failed")

    def add_pending(self, text: str, kind: str = "note") -> None:
        """Queue an intent to deliver on the next connect (durable background presence)."""
        text = (text or "").strip()
        if self._db is None or not text:
            return
        try:
            with self._lock:
                self._db.execute(
                    "INSERT INTO pending (ts, kind, text, delivered) VALUES (?, ?, ?, 0)",
                    (time.time(), kind, text[:800]),
                )
                self._db.commit()
                logger.info("pending queued ({}): {!r}", kind, text[:60])
        except Exception:  # noqa: BLE001
            logger.exception("add_pending failed")

    def peek_pending(self, limit: int = 6) -> list[dict]:
        """Undelivered pending intents, oldest first — does NOT mark them (mark only after speaking)."""
        if self._db is None:
            return []
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, kind, text FROM pending WHERE delivered=0 ORDER BY id ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [{"id": r[0], "kind": r[1], "text": r[2]} for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("peek_pending failed")
            return []

    def pending_due(self, min_age_secs: float = 480.0, limit: int = 6) -> list[dict]:
        """Undelivered pending intents OLDER than min_age_secs — for the away/Telegram reach-out (she
        waits this long for him to reconnect and hear it by voice; only then messages Telegram)."""
        if self._db is None:
            return []
        cutoff = time.time() - min_age_secs
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, kind, text FROM pending WHERE delivered=0 AND ts <= ? "
                    "ORDER BY id ASC LIMIT ?",
                    (cutoff, limit),
                ).fetchall()
            return [{"id": r[0], "kind": r[1], "text": r[2]} for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("pending_due failed")
            return []

    def mark_pending_delivered(self, ids: list[int]) -> None:
        """Mark the given pending rows delivered — called only AFTER she actually speaks them."""
        if self._db is None or not ids:
            return
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE pending SET delivered=1 WHERE id IN (%s)" % ",".join("?" * len(ids)),
                    ids,
                )
                self._db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("mark_pending_delivered failed")

    def add_timer(self, fire_ts: float, label: str) -> int:
        """Persist an armed timer and return its row id — 0 when it could not be stored, which the
        caller must treat as "in-memory only", never as a reason to skip arming. Rows more than a
        day old are pruned here, and at most _TIMER_MAX_LIVE timers may be live at once."""
        label = (label or "").strip() or "время вышло"
        if self._db is None:
            return 0
        now = time.time()
        try:
            with self._lock:
                self._db.execute("DELETE FROM timers WHERE fire_ts < ?", (now - _TIMER_MAX_AGE,))
                live = self._db.execute(
                    "SELECT COUNT(*) FROM timers WHERE fired=0 AND fire_ts > ?", (now,)
                ).fetchone()
                if live and live[0] >= _TIMER_MAX_LIVE:
                    self._db.commit()
                    logger.warning("timer NOT stored — {} already live", _TIMER_MAX_LIVE)
                    return 0
                cur = self._db.execute(
                    "INSERT INTO timers (fire_ts, label, fired) VALUES (?, ?, 0)",
                    (float(fire_ts), label[:200]),
                )
                self._db.commit()
                rid = int(cur.lastrowid or 0)
                logger.info("timer stored #{} in {}s: {!r}", rid, int(fire_ts - now), label[:60])
                return rid
        except Exception:  # noqa: BLE001
            logger.exception("add_timer failed")
            return 0

    def due_timers(self, limit: int = _TIMER_MAX_LIVE) -> list[dict]:
        """Timers whose moment has passed while nobody was connected, oldest first — for ONE
        coalesced catch-up line on the next connect. Does NOT mark them (mark before speaking).
        Anything overdue by more than a day is never surfaced: it is news, not a timer."""
        if self._db is None:
            return []
        now = time.time()
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, fire_ts, label FROM timers "
                    "WHERE fired=0 AND fire_ts <= ? AND fire_ts >= ? ORDER BY fire_ts ASC LIMIT ?",
                    (now, now - _TIMER_MAX_AGE, limit),
                ).fetchall()
            return [{"id": r[0], "fire_ts": r[1], "label": r[2]} for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("due_timers failed")
            return []

    def future_timers(self, limit: int = _TIMER_MAX_LIVE) -> list[dict]:
        """Timers still ahead of us, soonest first — re-armed with call_later on connect, since the
        handles they were originally armed with died with the previous socket."""
        if self._db is None:
            return []
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT id, fire_ts, label FROM timers "
                    "WHERE fired=0 AND fire_ts > ? ORDER BY fire_ts ASC LIMIT ?",
                    (time.time(), limit),
                ).fetchall()
            return [{"id": r[0], "fire_ts": r[1], "label": r[2]} for r in rows]
        except Exception:  # noqa: BLE001
            logger.exception("future_timers failed")
            return []

    def mark_timer_fired(self, ids: int | list[int]) -> None:
        """Mark timers announced; takes one id or a whole catch-up batch. Call it BEFORE speaking:
        a marked row that is never spoken is merely silent, an unmarked row that was spoken and then
        lost to a reconnect is announced twice."""
        if self._db is None or ids is None:
            return
        seq = list(ids) if isinstance(ids, (list, tuple, set)) else [ids]
        try:
            seq = [int(i) for i in seq]
        except Exception:  # noqa: BLE001
            return
        if not seq:
            return
        try:
            with self._lock:
                self._db.execute(
                    "UPDATE timers SET fired=1 WHERE id IN (%s)" % ",".join("?" * len(seq)),
                    seq,
                )
                self._db.commit()
        except Exception:  # noqa: BLE001
            logger.exception("mark_timer_fired failed")

    def recall_block(self, limit: int = 8, max_chars: int = 700) -> str:
        """A compact, relative-time-tagged digest of recent NON-trivial exchanges, to ground a fresh
        session in *when* things were said (recall, item #16). '' if nothing useful. Read-only: it
        does NOT touch recent_messages/recent_history (which hydrate the app UI + profile learner)."""
        if self._db is None:
            return ""
        try:
            with self._lock:
                rows = self._db.execute(
                    "SELECT ts, role, text FROM messages ORDER BY id DESC LIMIT ?", (limit * 3,)
                ).fetchall()
        except Exception:  # noqa: BLE001
            return ""
        picked, used = [], 0
        for ts, role, text in rows:                 # newest first
            t = (text or "").strip()
            low = t.lower().strip(" .,!?…")
            if low in _TRIVIAL_SET or len(t.split()) <= 1:
                continue
            who = "ты" if role == "assistant" else _OWNER
            line = f"{_ago(ts)} {who}: {t[:120]}"
            if used + len(line) > max_chars:
                break
            used += len(line)
            picked.append(line)
            if len(picked) >= limit:
                break
        if not picked:
            return ""
        picked.reverse()                            # oldest→newest reads naturally
        return (
            "Из прошлых разговоров (для контекста, НЕ зачитывай вслух списком): "
            + " | ".join(picked)
        )

    def _embed_query(self, text: str):
        try:
            body = json.dumps({"texts": [text]}).encode()
            req = urllib.request.Request(
                _EMBED_URL + "/embed", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8) as r:
                v = json.load(r).get("vectors") or []
            return v[0] if v else None
        except Exception:  # noqa: BLE001
            return None

    def semantic_recall(self, query: str, k: int = 8, max_chars: int = 1400) -> str:
        """Top-k past messages most similar BY MEANING to `query` — breaks the 400-row recency window
        («что решали про GPU три недели назад»). '' if the embed service is down or nothing relevant."""
        if self._db is None or not (query or "").strip() or not _SEMANTIC:
            return ""
        qv = self._embed_query(query)
        if not qv:
            return ""
        try:
            import numpy as np
            q = np.asarray(qv, dtype="<f4")
            qbytes = q.nbytes
            with self._lock:
                rows = self._db.execute(
                    "SELECT text, emb FROM memory_vec WHERE emb IS NOT NULL ORDER BY id DESC LIMIT 8000"
                ).fetchall()
            rows = [r for r in rows if r[1] and len(r[1]) == qbytes and (r[0] or "").strip()]
            if not rows:
                return ""
            mat = np.frombuffer(b"".join(r[1] for r in rows), dtype="<f4").reshape(len(rows), -1)
            sims = mat @ q                        # rows + query are normalized → cosine similarity
            order = np.argsort(-sims)
            picked, used, seen = [], 0, set()
            for i in order[: k * 4]:
                # Порог 0.50 на многоязычной MiniLM — это «почти дословное совпадение»: перефраз
                # той же мысли («что там с зимней резиной» ↔ «надо шины поменять») набирает 0.42-0.48
                # и отсекался, то есть память молчала ровно там, где человек и ждёт «ты же помнишь».
                # 0.42 пускает перефраз, оставляя мусор (0.2-0.3) снаружи.
                if sims[i] < 0.42:                # relevance floor (precision over recall — no junk)
                    break
                t = (rows[i][0] or "").strip()
                key = t[:60].lower()
                if len(t.split()) <= 2 or key in seen or _is_meta_junk(t):
                    continue
                seen.add(key)
                line = t[:320]
                if used + len(line) > max_chars:
                    break
                used += len(line)
                picked.append(line)
                if len(picked) >= k:
                    break
            if not picked:
                return ""
            return "Из прошлых разговоров (по смыслу к текущему): " + " | ".join(picked)
        except Exception:  # noqa: BLE001
            logger.exception("semantic_recall failed")
            return ""

    def last_ts(self) -> float:
        """Unix ts of the most recent stored message (0.0 if none) — for «N минут назад» grounding."""
        if self._db is None:
            return 0.0
        try:
            with self._lock:
                row = self._db.execute("SELECT MAX(ts) FROM messages").fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def all_facts(self, limit: int = 40, max_chars: int = 1200) -> list[tuple[float, str]]:
        """Active durable facts as (ts, text), most-recent first — for injection into the system
        prompt. Two ceilings, both load-bearing: expired volatile facts are held back (never
        deleted), and the char budget is what stops an ALWAYS-ON block from growing with the table
        — 40 rows × 500 chars would be ~20 KB on every single turn."""
        if self._db is None:
            return []
        try:
            with self._lock:
                if self._facts_ttl:
                    rows = self._db.execute(
                        "SELECT ts, text FROM facts WHERE active=1 "
                        "AND (ttl_secs IS NULL OR ts + ttl_secs >= ?) "
                        "ORDER BY id DESC LIMIT ?",
                        (time.time(), limit),
                    ).fetchall()
                else:
                    rows = self._db.execute(
                        "SELECT ts, text FROM facts WHERE active=1 ORDER BY id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            out: list[tuple[float, str]] = []
            used = 0
            for ts, text in rows:  # newest first; keep newest within budget
                t = (text or "").strip()
                if not t:
                    continue
                used += len(t) + 16  # budget the RENDERED «(вчера) …; » form, not the bare text
                if used > max_chars:
                    break
                out.append((float(ts or 0.0), t))
            return out
        except Exception:  # noqa: BLE001
            logger.exception("all_facts failed")
            return []

    def facts_block(self, limit: int = 40, max_chars: int = 1200) -> str:
        """The durable facts as ONE dated, char-capped line for the SINGLE system message (a second
        system message is rejected by the Anthropic-backed proxy). '' when there is nothing to
        inject. Dated because in an undated pile last month's parking spot looks exactly as current
        as yesterday's."""
        facts = self.all_facts(limit=limit, max_chars=max_chars)
        if not facts:
            return ""
        return (
            "Уже сохранённые факты (помни и учитывай, НЕ зачитывай вслух списком): "
            + "; ".join(f"({_ago(ts)}) {t}" for ts, t in facts)
            + "."
        )


_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
