"""Self-learning: distill the conversation memory into a compact portrait of the user.

Runs daily (systemd timer). Reads the last N messages from memory.db, asks the local shim
(cheap Haiku) to update a short, plain-Russian profile — preferences, recurring topics, what
annoys him, what he already knows, how he likes to be spoken to — and writes it to
``user_profile.txt``. Every session then loads it via ``config._learned_profile()``, so she gets
to know him better over time.

The profile is written as FOUR labelled lines (``ТЕМЫ:`` / ``ПРЕДПОЧТЕНИЯ:`` / ``НЕ НАДО:`` /
``ОТКРЫТЫЕ НИТИ:``) so the reader can split them: the first three belong in the system prompt,
while «открытые нити» is not prompt material at all — it is one of the local reasons that must
exist before she is allowed to consider speaking first (see :func:`find_local_reason`).

Idempotent and safe: on any error it leaves the existing profile untouched.

Also imported IN-PROCESS by the realtime server for :func:`find_local_reason`, so module import
must stay free of DB, path and network side effects — everything the daily job needs is set up
inside :func:`main`.
"""

from __future__ import annotations

import json
import os
_OWNER = __import__("os").environ.get("PYATNITSA_OWNER", "хозяин")
import re
import sys
import urllib.request
from collections.abc import Sequence

SHIM_URL = os.environ.get("EDIT_SHIM_URL", "http://127.0.0.1:9090/v1/chat/completions")
PROFILE_PATH = os.environ.get("EDIT_PROFILE_PATH", "/opt/pyatnitsa/user_profile.txt")
MODEL = os.environ.get("EDIT_LEARN_MODEL", "haiku")


# -- local reason detection (imported by the realtime server) ----------------
# STOP-by-default is only real if something LOCAL and ALREADY TRUE has to exist before the model is
# ever asked whether to speak. A timer running out is not a reason; a thing he actually said and
# left open is. This is that test — pure arithmetic and compiled regex over a handful of short
# strings, no LLM, no network, no file or DB access, so the presence loop can call it on the
# realtime event loop every tick and skip the shim entirely when it returns "".

# Word-anchored on BOTH sides: as a bare prefix «надо» also fires on «надоело», «потом» on
# «потомки», and the whole gate degrades into "he said something".
_OPEN_LOOP_RE = re.compile(
    r"(?<!\w)("
    r"собирал\w*|собираюсь|надо|нужно|потом|попозже|завтра|на\s+неделе|не\s+забыть|"
    r"планиру\w*|доделать|дописать|дочитать|разобраться\s+с|хочу\s+сделать"
    r")(?!\w)",
    re.IGNORECASE,
)

# A sentence is only an open loop once it has had time to go unanswered — inside this window it is
# still the live conversation, and asking about it would be interrupting, not remembering.
_LOOP_MIN_AGE = float(os.environ.get("EDIT_PRESENCE_LOOP_AGE", "600"))
# Past this it is not an open loop any more, it is history.
_LOOP_MAX_AGE = float(os.environ.get("EDIT_PRESENCE_LOOP_MAXAGE", "86400"))
# The greeting path owns reconnects; an absence only counts as a reason here when it is long
# enough that mentioning it mid-session is still natural.
_ABSENCE_MIN = float(os.environ.get("EDIT_PRESENCE_ABSENCE_MIN", "28800"))
_REASON_MAX = 200   # the reason rides in a prompt; it must never become the prompt
_QUOTE_MAX = 120

# Frequent long words whose 5-char stems would otherwise make any fact "relevant" to any sentence.
_STOP_STEMS = frozenset({
    "котор", "сегод", "завтр", "поэто", "потом", "хорош", "спаси", "немно", "навер", "конеч",
    "прост", "значи", "сейча", "тольк", "почем", "давай", "больш", "мален", "нельз", "долже",
    "сдела", "делат", "говор", "смотр", "думат", "хочет", "хочеш", "нрави", "вообщ", "кстат",
    "всегд", "никог", "может", "можно", "будет", "нужно", "пожал", "слуша", "понял", "поним",
})


def looks_like_open_loop(text: str) -> bool:
    """True when a user line states an intention the conversation may have left open."""
    return bool(text) and bool(_OPEN_LOOP_RE.search(text))


def _clip(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


def _ago_hint(secs: float) -> str:
    if secs < 3600:
        return "недавно"
    if secs < 6 * 3600:
        return "несколько часов назад"
    return "ещё раньше"


def _stems(text: str) -> set[str]:
    """5-char stems of content words — crude, but it survives Russian inflection («машину» →
    «машин» matches «машина»), costs one split, and needs no dictionary."""
    return {
        w[:5]
        for w in re.findall(r"\w+", (text or "").lower())
        if len(w) >= 6 and w[:5] not in _STOP_STEMS
    }


def find_local_reason(
    now: float,
    *,
    open_loops: Sequence[tuple[float, str]] = (),
    pending: Sequence[str] = (),
    facts: Sequence[str] = (),
    recent_user_text: str = "",
    open_threads: str = "",
    absence_secs: float = 0.0,
    loop_min_age: float = _LOOP_MIN_AGE,
    loop_max_age: float = _LOOP_MAX_AGE,
    absence_min: float = _ABSENCE_MIN,
) -> str:
    """Something that actually happened and is worth mentioning, or "" when there is nothing.

    "" means DO NOT CONSULT the model at all — the caller must ``continue``. A non-empty return is
    a CANDIDATE, never a licence to speak: it is grounding for a decision the model still answers
    with STOP by default, and the regex over-captures by design («надо бы кофе»), so the caller
    must word it tentatively.

    ``now`` and the timestamps in ``open_loops`` must come from the SAME clock (``time.monotonic``
    in the presence loop). ``absence_secs`` is a duration, so it is clock-agnostic.

    Sources, most concrete first:
      * ``pending``   — texts of undelivered pending rows (owner-authored, no model discretion);
      * ``open_loops``— ``(ts, text)`` user finals, kept by the caller; a hit counts only once the
        sentence is between ``loop_min_age`` and ``loop_max_age`` old;
      * ``facts`` + ``recent_user_text`` — a durable fact that shares a content word with what he
        has just been talking about, i.e. one that became relevant on its own;
      * ``absence_secs`` — an absence long enough to be worth acknowledging;
      * ``open_threads`` — the «ОТКРЫТЫЕ НИТИ» line of the daily profile (weakest: a day old and
        not tied to this session, so it is the last thing tried).

    Pure: no I/O, no globals mutated, no clock read. Never raises — a malformed input yields "".
    """
    try:
        for text in pending:
            if text and text.strip():
                return _clip(f"для него отложено: {_clip(text, _QUOTE_MAX)}", _REASON_MAX)

        best_ts, best_text = 0.0, ""
        for ts, text in open_loops:
            age = now - ts
            if age < loop_min_age or age > loop_max_age:
                continue
            if ts >= best_ts and looks_like_open_loop(text):
                best_ts, best_text = ts, text
        if best_text:
            return _clip(
                f"{_ago_hint(now - best_ts)} он сказал: «{_clip(best_text, _QUOTE_MAX)}»",
                _REASON_MAX,
            )

        said = _stems(recent_user_text)
        if said:
            for fact in facts:
                if fact and len(fact) >= 12 and _stems(fact) & said:
                    return _clip(
                        f"он говорил об этом, а в памяти есть: «{_clip(fact, _QUOTE_MAX)}»",
                        _REASON_MAX,
                    )

        if absence_secs >= absence_min > 0:
            return _clip(
                f"его не было примерно {int(absence_secs // 3600)} ч — это первый разговор после "
                "перерыва",
                _REASON_MAX,
            )

        thread = (open_threads or "").strip(" —-\t")
        if thread:
            return _clip(f"незакрытые нити из его профиля: {thread}", _REASON_MAX)
    except Exception:  # noqa: BLE001 — a bad input must silence her, never break the loop
        return ""
    return ""


def _read_existing() -> str:
    try:
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _call_shim(system: str, user: str) -> str:
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "edit_mode": "typed",          # typed path → no voice-shortening, no TTS
        "edit_model": MODEL,
        "edit_effort": "low",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(SHIM_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


def main() -> int:
    sys.path.insert(0, os.path.dirname(__file__))
    from services.memory_store import get_store  # noqa: PLC0415 — cron-only; keep import cheap

    store = get_store()
    history = store.recent_history(count=200, max_chars=20_000)
    if len(history) < 6:
        print("not enough memory yet; skipping")
        return 0

    convo = "\n".join(
        f"{_OWNER if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
        for m in history
    )
    existing = _read_existing()

    # The labels are a CONTRACT with config._learned_profile(). Keep them byte-exact — a reader
    # that cannot find them falls back to injecting the whole blob (which is what happens today,
    # so a new label is picked up automatically and needs no reader change).
    system = (
        "Ты ведёшь досье-портрет пользователя для голосового ассистента. "
        "Выдай РОВНО пять строк, каждая начинается своей меткой, каждая не длиннее 220 "
        "символов:\n"
        "ТЕМЫ: повторяющиеся темы и дела, чем занимается, важные факты о нём и его жизни.\n"
        "ПРЕДПОЧТЕНИЯ: что любит, как предпочитает общение и тон, что уже знает и объяснять "
        "не надо.\n"
        "НЕ НАДО: что его раздражает, о чём не заговаривать, чего избегать.\n"
        "ОТКРЫТЫЕ НИТИ: незакрытые дела и намерения из разговоров — то, о чём уместно "
        "спросить позже.\n"
        # ПЯТАЯ СТРОКА — единственная, которая про НЕЁ, а не про него.
        #
        # Три предыдущие метки описывают человека; ассистент от этого умнее не становится, он лишь
        # лучше информирован. Память о СОБСТВЕННЫХ промахах — другое: это единственный механизм, из-за
        # которого завтрашний разговор может пройти иначе, чем вчерашний. В журнале лежат живые
        # поправки («я же попросил тебя другую спеть», «файлов нету», «не пришли»), и сегодня они
        # пропадают вместе с сессией — она наступает на те же грабли ровно потому, что не помнит,
        # что уже наступала.
        # БЕЗ РОДА И БЕЗ «Я» — не стилистика, а необходимость. Характер бывает и женский (Ксения),
        # и мужской (Джарвис), а строка пишется один раз на сутки и уезжает в промпт обоим. Первый
        # прогон это и показал: модель написала «я давал», «я предлагал» — и она заговорила бы о
        # себе в мужском роде. Правило в неопределённой форме одинаково верно для любого характера.
        "УРОКИ: за что " + _OWNER + " поправлял и чем был недоволен — каждый пункт КОРОТКИМ ПРАВИЛОМ в "
        "неопределённой форме, без «я» и без слов в мужском или женском роде. Например: "
        "«перепроверять факты перед ответом, а не повторять услышанное»; «на „что угодно“ — "
        "выбирать самой и делать, не переспрашивать». "
        # Урок должен быть про ПОВЕДЕНИЕ, а не про поломку. Первый прогон вынес правило «отправлять
        # не файлом, а текстом» — обходной путь эпохи, когда файлы не умели уходить. Их научили в тот
        # же день, и такое «знание» толкало бы её в противоположную сторону ещё месяцами: она ведь
        # видит только разговоры, а не список того, что с тех пор починили.
        "Уроки — ТОЛЬКО про поведение и тон. Если что-то не получилось из-за поломки или "
        "отсутствия возможности, это НЕ урок: такое чинят, и правило про обход быстро станет "
        "вредным. Не превращай в правило обходной путь.\n"
        "Пиши на русском, сухими тезисами через точку с запятой, БЕЗ вступлений, заголовков и "
        "пояснений. Не выдумывай — только то, что реально следует из разговоров. Сохраняй прежние "
        "верные факты, добавляй новые, убирай устаревшие. Если по метке сказать нечего — оставь "
        "строку с меткой и прочерком."
    )
    user = (
        (f"Текущий профиль:\n{existing}\n\n" if existing else "")
        + f"Недавние разговоры:\n{convo}\n\n"
        "Выдай ОБНОВЛЁННЫЙ профиль — ровно пять строк с метками, без пояснений."
    )

    try:
        profile = _call_shim(system, user)
    except Exception as exc:  # noqa: BLE001
        print(f"shim call failed: {exc}", file=sys.stderr)
        return 1

    profile = profile.strip()
    if not profile or len(profile) < 20:
        print("empty/short profile; keeping existing")
        return 0
    # 2200, а не 1600: УРОКИ — ПОСЛЕДНЯЯ строка, поэтому обрезка блоба всегда съедала именно её,
    # и первый прогон оборвался на «с». Строка, ради которой всё затевалось, не должна быть той,
    # что не поместилась.
    if len(profile) > 2200:
        profile = profile[:2200]

    tmp = PROFILE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(profile)
    os.replace(tmp, PROFILE_PATH)
    print(f"profile updated ({len(profile)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
