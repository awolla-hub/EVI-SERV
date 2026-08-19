"""Weekly recap — «о чём ты думал на этой неделе» digest to Telegram.

Runs from a systemd timer (Sunday evening): pulls the week's exchanges from the conversation
memory (SQLite), asks the local Claude shim to distil them into a warm short digest, and sends it
to the user's Telegram via the bot. Fails quietly — a missed digest must never break anything.
"""

import json
import os
_OWNER_GEN = os.environ.get("PYATNITSA_OWNER_GEN", "хозяина")
import sqlite3
import time
import urllib.request

DB = os.environ.get("MEMORY_DB", "/opt/pyatnitsa/memory.db")
SHIM = "http://127.0.0.1:9090/v1/chat/completions"
TG_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TG_CHAT_ID", "")


def week_messages() -> list[tuple[str, str]]:
    db = sqlite3.connect(DB)
    # Same 15 s busy_timeout as the server's MemoryStore: a live session writing a turn used to
    # make this read raise "database is locked" and the digest silently skipped that week.
    db.execute("PRAGMA busy_timeout=15000")
    rows = db.execute(
        "SELECT role, text FROM messages WHERE ts > ? ORDER BY id ASC LIMIT 300",
        (time.time() - 7 * 86400,),
    ).fetchall()
    db.close()
    return rows


def summarize(rows) -> str:
    convo = "\n".join(f"{r}: {t}" for r, t in rows)[:12000]
    body = {
        "model": "haiku",
        "messages": [
            {"role": "system", "content": (
                "Ты — Эдит, тёплый ассистент " + _OWNER_GEN + ". Составь короткий недельный дайджест по "
                "истории разговоров: о чём говорили, что он просил запомнить, какие дела/обещания "
                "звучали и что из этого стоит не забыть на следующей неделе. Пиши по-русски, "
                "по-доброму, маркированным списком, до 12 строк. Без выдумок — только из истории."
            )},
            {"role": "user", "content": convo or "История пуста."},
        ],
    }
    req = urllib.request.Request(
        SHIM, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"].strip()


def send_tg(text: str) -> None:
    if not (TG_TOKEN and TG_CHAT and text):
        return
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        data=json.dumps({"chat_id": TG_CHAT, "text": "🗓 Недельный дайджест от Эдит\n\n" + text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=30)


if __name__ == "__main__":
    try:
        rows = week_messages()
        if len(rows) < 4:
            print("too little history — skipping")
        else:
            send_tg(summarize(rows))
            print("digest sent")
    except Exception as e:  # noqa: BLE001
        print("recap failed:", e)
