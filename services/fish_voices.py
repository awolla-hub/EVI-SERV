"""Fish voice models: listing them, and making one out of his own voice.

WHY A SEPARATE MODULE. `fish_tts.py` is on the audio path and must stay boring — it synthesizes a
clause and gets out of the way. Everything here is control-plane: it runs on a worker thread from a
control message, never per clause, and a slow or failed call costs a settings screen rather than her
speech.

WHAT REPLACED WHAT. The clone used to be XTTS on a home 3070 reached through an SSH tunnel; when the
node was down she answered with silence. Fish trains a model in one call and hosts it, so the clone
survives the home machine being off — which is the whole reason the old one was removed.

PRIVATE, ALWAYS. Fish defaults `visibility` to **public**: a voice created without saying otherwise
is published to their library. Every model created here is explicitly private, and that is not a
preference — it is his voice.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from loguru import logger

_API = "https://api.fish.audio"
_TIMEOUT = 30.0
_TRAIN_TIMEOUT = 180.0     # training is the slow one; measured ~10 s, but it is not on the audio path

# The catalogue barely moves and every connect wants it, so it is fetched at most this often and
# shared by every session. `hello` must not pay a network round trip to populate a settings screen.
_CACHE_TTL = 900.0
_cache: dict[str, tuple[float, list[dict]]] = {}


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def list_voices(
    key: str,
    language: str = "ru",
    limit: int = 30,
    page: int = 1,
    tag: str = "",
    query: str = "",
    sort_by: str = "task_count",
) -> dict:
    """Голоса, которые стоит предложить. Возвращает `{"items": [...], "has_more": bool, "page": n}`.

    НИКОГДА не бросает: экран настроек, не дозвонившийся до Fish, должен показать те голоса, что
    уже знает, а не диалог с ошибкой поверх её лица.

    ЧТО ИЗМЕНИЛОСЬ И ПОЧЕМУ. Раньше отсюда уходило 30 голосов с четырьмя полями — при том, что в
    библиотеке 1002 русских, а Fish отдаёт про каждый теги (пол, возраст, характер), лайки и
    ОБРАЗЕЦ ЗВУЧАНИЯ. Выбор голоса без возможности его послушать — это выбор вслепую по названию,
    а названия там вроде «Спокойный женский голос» и «Меллстрой». Поэтому:

      * `page` — каталог листается, а не обрывается на первой странице;
      * `tag`/`query`/`sort_by` — фильтрация делается НА СТОРОНЕ FISH: тянуть тысячу строк, чтобы
        отфильтровать их на телефоне, дороже и медленнее, чем попросить нужное;
      * в строке появляются `tags`, `likes` и `sample` — по образцу приложение даёт послушать.

    Сортировка `score` у Fish — их собственная смесь популярности и качества; `task_count` — «чаще
    всего используют»; `created_at` — «новые». Остальное отвергается, чтобы в запрос нельзя было
    подставить произвольную строку.
    """
    if not key:
        return {"items": [], "has_more": False, "page": 1}
    if sort_by not in ("task_count", "score", "created_at"):
        sort_by = "task_count"
    page = max(1, int(page or 1))
    # Ключ кэша — ВСЕ параметры. Раньше ключом был только язык, и запрос с другим лимитом или
    # фильтром получал в ответ прошлый список: фильтр «мужские» тихо показывал бы женские.
    ckey = f"{language}|{limit}|{page}|{tag}|{query}|{sort_by}"
    hit = _cache.get(ckey)
    if hit and (time.monotonic() - hit[0]) < _CACHE_TTL:
        return hit[1]
    params = {
        "language": language,
        "page_size": max(1, min(100, limit)),
        "page_number": page,
        "sort_by": sort_by,
    }
    if tag:
        params["tag"] = tag
    if query:
        params["title"] = query
    url = f"{_API}/model?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers=_auth(key))
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            payload = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        # A stale list beats an empty one: the picker keeps working through a Fish blip.
        logger.warning("Fish: список голосов не получен ({})", exc)
        return hit[1] if hit else {"items": [], "has_more": False, "page": page}
    out = []
    for item in payload.get("items", []):
        vid = item.get("_id") or item.get("id")
        if not vid:
            continue
        # Образец: первый непустой URL. Он отдаётся Fish'ем публично (проверено: 200, audio/mpeg),
        # поэтому телефон играет его сам, без ключа и без проксирования через сервер.
        sample = ""
        for s in (item.get("samples") or []):
            if isinstance(s, dict) and s.get("audio"):
                sample = str(s["audio"])
                break
        out.append({
            "id": str(vid),
            "title": str(item.get("title") or "без названия")[:60],
            "languages": [str(x) for x in (item.get("languages") or [])],
            "uses": int(item.get("task_count") or 0),
            "likes": int(item.get("like_count") or 0),
            # Теги — это и есть будущий фильтр: female/male, young/middle-aged/old, calm/energetic…
            # Ограничены двенадцатью: у иных голосов их по двадцать, и на телефоне это простыня.
            "tags": [str(t)[:24] for t in (item.get("tags") or [])][:12],
            "sample": sample,
        })
    res = {"items": out, "has_more": bool(payload.get("has_more")), "page": page}
    if out:
        _cache[ckey] = (time.monotonic(), res)
    return res


def create_clone(key: str, wav_bytes: bytes, title: str = "Голос владельца") -> str | None:
    """Train a PRIVATE voice model on his recording. Returns the reference id, or None.

    `train_mode=fast` is the only mode their API documents, and it returns `state=trained`
    immediately — there is no polling loop to write here.
    """
    if not key or not wav_bytes:
        return None
    boundary = "----edit" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in (
        ("type", "tts"),
        ("title", title),
        ("train_mode", "fast"),
        # NOT a default worth trusting: see the module docstring.
        ("visibility", "private"),
        ("description", "Пятница: голос владельца"),
    ):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; "
            f'name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="voices"; '
        f'filename="voice.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode("utf-8")
        + wav_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{_API}/model",
        data=b"".join(parts),
        headers={**_auth(key), "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TRAIN_TIMEOUT) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()[:200].decode("utf-8", "replace")
        logger.warning("Fish: клон не создан, HTTP {} — {}", exc.code, body)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fish: клон не создан ({})", exc)
        return None
    vid = res.get("_id") or res.get("id")
    if not vid:
        logger.warning("Fish: ответ без id модели")
        return None
    logger.info("Fish: клон создан id={} state={} visibility={}",
                vid, res.get("state"), res.get("visibility"))
    return str(vid)
