# Пятница Realtime — план нативного голосового ассистента (RU, self-hosted)

> Цель: ассистент уровня «Джарвиса» на очках Ray-Ban Meta — ощущается нативным
> (sub-1.5s до первого звука, перебивание на полуслове, без обрывов), полностью
> русскоязычный, работает **из России без VPN**, на **своём** сервере, через
> **свой** LLM-прокси (claude-api.io). Клиент — существующее приложение OpenGlasses.
>
> Статус: план утверждён по итогам параллельного research (6 глубоких изысканий,
> июль 2026). Код пишется по этому документу.

---

## 1. Три жёстких ограничения, которые определяют всю архитектуру

1. **Россия глушит realtime-UDP.** С августа 2025 TSPU-DPB РКН throttl-ит/блокирует
   зашифрованный realtime-голос (WhatsApp/Telegram calls) по форме трафика —
   постоянный поток зашифрованного UDP (STUN/DTLS-SRTP) на зарубежный IP. WhatsApp
   полностью заблокирован 11.02.2026. **Вывод: WebRTC/LiveKit/VisionClaw = хрупкий
   транспорт из РФ.** Единственный устойчивый путь — **TLS-WebSocket на порту 443**
   к обычному домену. OpenGlasses уже так и делает для Gemini Live.

2. **LLM только через свой прокси.** claude-api.io — OpenAI-совместимый REST
   (`/v1/chat/completions`, 84+ модели, текст+vision, но НЕ realtime-websocket).
   → архитектура обязана быть **каскадной** (STT→LLM→TTS), тогда отсутствие
   realtime-API у прокси не мешает, а сам прокси втыкается как LLM-стадия.

3. **Русский — first-class.** Качество STT и TTS на русском — hard requirement.
   Значит компоненты выбираем по русскому бенчмарку, а не по «общей популярности».

---

## 2. Архитектурное решение

**Основной режим — каскад на Pipecat, self-hosted, WSS/443, Russian-first.**
**Вторичный режим (опция, A/B) — relay Gemini Live через свой сервер.**

```
 iPhone (РФ)                         Твой VPS (EU, вне РФ)
┌─────────────────┐   WSS :443     ┌──────────────────────────────────────────┐
│ OpenGlasses      │◀──Opus/PCM───▶│ Pipecat (один asyncio-процесс)             │
│  ├ AVAudioEngine │  16k up        │  Silero VAD ─ Smart Turn v3 (RU endpoint)  │
│  ├ DAT SDK (очки)│  24k down      │       │                                    │
│  ├ PyatnitsaRT   │                │  GigaAM v3 STT (RU, gigastt) ──partials──▶ │
│  │  session mgr  │                │       │                                    │
│  └ барже-ин лок. │                │  claude-api.io  (OpenAI-совмест., стрим)   │
└─────────────────┘                │       │  (sentence-boundary chunking)      │
                                    │  Silero v5_ru TTS (голос aidar) ──audio──▶ │
                                    └──────────────────────────────────────────┘
```

### Почему каскад, а не relay Gemini — первичный

| Критерий | Каскад (Pipecat) | Relay Gemini Live |
|---|---|---|
| Работает из РФ без VPN на телефоне | ✅ WSS/443 | ✅ (через сервер), но нужен Gemini-ключ (раз добыть под VPN) |
| Использует твой прокси claude-api.io | ✅ | ❌ (только Gemini) |
| Русский STT/TTS под контролем | ✅ лучший (GigaAM/Silero) | ⚠️ качество аудио Gemini не настраивается |
| Приватность (аудио на твоём железе) | ✅ | ❌ уходит в Google |
| TOS/легальность | ✅ чисто | ⚠️ relay realtime-ключа — серая зона |
| Расширяемость (свой Jarvis-бэкенд) | ✅ | ❌ чужой протокол |
| Латентность до первого звука | ~1.0–1.3с | ~0.5–0.8с |
| Новый код клиента | средне | почти ноль |

Relay быстрее по латентности, но проигрывает по всем стратегическим осям. Делаем
каскад основным, а Gemini-relay оставляем как **переключаемый режим** для сравнения
(в коде это почти бесплатно — `Config.geminiLiveWebSocketBaseURL` уже одна константа).

---

## 3. Выбор компонентов (с доказательной базой, июль 2026)

| Слой | Выбор | Почему (проверено) | Fallback |
|---|---|---|---|
| **Framework** | **Pipecat** v1.5 (BSD-2) | один asyncio-процесс, без media-сервера/Redis; FastAPIWebsocketTransport на 443; 25+ STT / 30+ TTS; OpenAI base_url → прокси as-is; ~1.0с median voice-to-voice на self-host | кастомный asyncio-сервер (позже, «eject») |
| **Транспорт** | **WSS :443 + Opus** (SILK-WB 16–24 кбит/с) | DPI-устойчиво; Opus = 10–16× меньше трафика, +6.5мс lookahead; на TCP FEC выключить | старт на PCM16, Opus фазой 2 |
| **VAD** | **Silero VAD** (в Pipecat) | стандарт, быстрый CPU | WebRTC VAD |
| **Endpointing** | **Smart Turn v3** (BSD-2) | ЕДИНСТВЕННЫЙ семантический детектор конца фразы **с русским** (93.67%), 12–60мс CPU; можно и на iPhone | Silero VAD-таймаут (хуже) |
| **STT** | **GigaAM v3** (Сбер, MIT) via **gigastt** (Rust+ONNX, стрим RNNT) | ~8.4% WER RU (multi-domain) vs 25.1% Whisper; пунктуация встроена; ~0.78с до 1-го partial на CPU | **T-one** (Т-Банк, Apache-2.0, нативный стрим) для телефонии/шума; sherpa-onnx zipformer-ru **на iPhone** (офлайн-деградация) |
| **LLM** | **claude-api.io** (OpenAI base_url, стрим) | твой прокси; модель под низкий first-token (claude-haiku / gemini-flash через прокси) | любая модель из 84+ |
| **TTS** | **Silero v5_ru** голос **`aidar`** (муж.), 24кГц | RTF 0.025 на 1 ядре CPU; ~200–400мс TTFA; встроенное ударение; 24кГц = совпадает с плеером приложения | **Cartesia Sonic** (188мс, RU, $0.03/мин) как premium; **F5-TTS_RUSSIAN** для макс. натуральности; ElevenLabs Flash — лучший RU, но санкции/гео (звать только с VPS) |

> ⚠️ Отсеяно по русскому бенчмарку: Piper (мужские RU-голоса ruslan/denis ломаются),
> OpenAI TTS (американский акцент на русском), Whisper на CPU (не стрим, разваливается
> на шуме), LiveKit turn-detector (нет русского), AssemblyAI (нет русского стрима).

---

## 4. Бюджет латентности (Москва → Франкфурт, RTT ~70–90мс)

Целевой **time-to-first-audio (TTFA)** — от момента, когда пользователь замолчал, до
первого звука ответа:

| Стадия | Бюджет | Приём оптимизации |
|---|---|---|
| Детекция конца фразы (Smart Turn v3) | ~260мс p50 | семантический endpoint вместо 600–1000мс VAD-таймаута |
| STT финализация | ~0мс (partials уже стримятся во время речи) | стрим RNNT, финал почти мгновенно после endpoint |
| Сеть (1 serial RTT РФ↔EU) | ~70–90мс | постоянный duplex-WS, pre-warm соединений |
| LLM first token (через прокси) | ~300–500мс | быстрая модель, `stream=true`, короткий системный промпт |
| TTS first audio (Silero) | ~100–200мс | синтез первого предложения по границе `.!?…` пока LLM ещё пишет |
| **ИТОГО TTFA** | **~0.9–1.3с** | под целевые 1.5с |

Честно: нативный s2s (Gemini/GPT Realtime) даёт ~0.4–0.5с. Мы даём ~1.0–1.3с — это
уверенный «разговорный» уровень (быстрее текущего wake-word-режима), но не «перебивает
на полуслове сам». Ниже 0.9с — только нативные speech-to-speech, их каскад не заменит.

---

## 5. Wire-протокол (клиент ↔ сервер)

Один постоянный duplex TLS-WebSocket, бинарные аудио-кадры + JSON управляющие
сообщения. Кадрирование — совместимо с текущим WS-клиентом OpenGlasses.

**Клиент → сервер:**
- `bin` аудио: Opus-кадры 20мс, 16кГц mono (или PCM16 в MVP). Стрим непрерывно.
- `{"type":"hello","session":"<uuid>","resume_seq":N,"codec":"opus|pcm16"}` — старт/resume.
- `{"type":"barge_in"}` — локально задетектили перебивание (мгновенная отмена).
- `{"type":"vision","image_b64":"..."}` — кадр с очков для vision-запроса.
- `{"type":"bye"}`.

**Сервер → клиент:**
- `bin` аудио: TTS-кадры 24кГц PCM16 (в плеер приложения напрямую) / Opus.
- `{"type":"partial","text":"...","seq":N}` — промежуточный ASR (для HUD).
- `{"type":"user_final","text":"..."}` / `{"type":"assistant_text","text":"..."}`.
- `{"type":"speaking_start"}` / `{"type":"speaking_end"}` — для UI и барже-ин AEC.
- `{"type":"interrupted"}` — подтверждение отмены (сбросить буфер плеера).
- `{"type":"pong","seq":N}` — heartbeat.

Порядок и seq-номера → resume после разрыва (см. §7).

---

## 6. Инженерия «нативности» (что делает Джарвиса Джарвисом)

Приоритет — **turn-taking, а не выбор модели** (главный вывод research).

1. **Семантический endpointing (Smart Turn v3).** Не ждём тупой таймаут молчания —
   модель понимает, закончил ли человек мысль. Экономит 300–700мс на каждой реплике.
2. **Speculative / eager generation.** По eager-EOT (за 150–250мс до финала) начинаем
   черновой LLM-запрос; если финал совпал — ответ уже готов. (Фаза 3.)
3. **Barge-in — local-first.** VAD на телефоне держит 250мс речи поверх TTS →
   мгновенно (<60мс) **сбрасываем буфер плеера** (дропаем очередь, не доигрываем) →
   шлём `barge_in` → сервер гасит TTS (flush) и отменяет LLM. Предусловие — AEC:
   собственный TTS не должен ретриггерить микрофон (у очков Ray-Ban эхо гасится в их
   5-мик DSP; на телефоне — `setVoiceProcessingEnabled`).
4. **Sentence-buffer chunking TTS.** Копим токены LLM, при первой границе `.!?…`
   отдаём предложение в TTS — синтез идёт параллельно генерации. ~1600мс → ~900мс.
5. **Prosody continuity.** Между чанками TTS сохраняем интонацию (Silero — общий
   контекст; Cartesia — `continuations`/`context_id`).
6. **Filler ack (опц.).** Микро-«угу/секунду» пока считается тяжёлый ответ — маскирует
   хвост латентности.

---

## 7. Надёжность (уровень Apple: «никогда не отваливается»)

- **Reconnect.** WS-TCP умирает при WiFi↔cellular. `NWPathMonitor` на iOS → мгновенный
  reconnect; сервер держит **session buffer ~30с + seq-номера** → resume без потери
  контекста разговора. Цель восстановления <1с (TLS 1.3 resumption).
- **Heartbeat.** ping/pong каждые 10с; мёртвое соединение — пересоздать.
- **Аудио-буферы.** Jitter-buffer переживает блипы 1–3с без разрыва сессии.
- **Супервизия.** Docker Compose + `restart: unless-stopped` (или systemd),
  healthcheck на каждый сервис (VAD/STT/TTS/оркестратор), авто-рестарт.
- **Fallback-цепочки.** STT: GigaAM → T-one → on-device zipformer. TTS: Silero →
  Cartesia. LLM: модель A → модель B через прокси. Прокси тормозит → короткий queue +
  «секунду…».
- **Мониторинг.** Uptime Kuma + Prometheus (per-stage латентности), алерт в Telegram.
- **RF-устойчивость.** WSS/443, обычный домен (не *.fly.dev), TLS 1.3. Никакого UDP.
  Опционально на своём сервере — маскировка под обычный HTTPS (уже так и есть).
- **Фоновый режим iOS.** `.audio` background mode + активная audio-session держат
  приложение живым; WS в фоне работает при активном аудио (не через background
  URLSession — она WS не тянет).

---

## 8. Реализация — СЕРВЕР (`pyatnitsa-server/`, Python)

Отдельный репозиторий/папка, деплой на VPS вне РФ. Pipecat-пайплайн:

```
pyatnitsa-server/
├─ pipeline.py         # Pipecat: transport(WSS) → VAD → SmartTurn → STT → LLM → TTS
├─ transport.py        # FastAPIWebsocketTransport + кастомный FrameSerializer
│                      #   (совместим с 16k-PCM/Opus протоколом OpenGlasses)
├─ services/
│  ├─ gigaam_stt.py    # обёртка gigastt (стрим RNNT, partials)
│  ├─ silero_tts.py    # Silero v5_ru, голос aidar, 24kHz, sentence chunking
│  └─ proxy_llm.py     # OpenAILLMService(base_url=claude-api.io) — стрим
├─ session.py          # resume-буфер (seq, 30с), состояние разговора, барже-ин
├─ config.py           # .env: PROXY_URL, PROXY_KEY, MODEL, TTS_VOICE, TLS…
├─ Dockerfile
├─ docker-compose.yml  # app + (опц.) prometheus/uptime-kuma; restart: unless-stopped
└─ deploy.md           # nginx TLS :443, Let's Encrypt, systemd/compose
```

Стадии Pipecat: `SileroVAD → LocalSmartTurnAnalyzerV3(ru) → GigaAMSTTService →
OpenAILLMService(base_url) → SileroTTSService`. Барже-ин — нативный interruption
Pipecat + `barge_in` от клиента. Первый запуск: CPU-only (см. §10).

---

## 9. Реализация — КЛИЕНТ (OpenGlasses, Swift)

Переиспользуем существующий Gemini Live путь (он уже WSS+PCM, 16к in / 24к out).

**Точки переиспользования (из разбора кода):**
- `Config.geminiLiveWebSocketBaseURL` — одна константа (уже есть) → рядом
  `customRealtimeServerURL` (из настроек, `wss://твой-домен/ws`).
- `GeminiLiveSessionManager` — эталон WS-сессии (audio in/out, reconnect, барже-ин)
  → клонируем в `PyatnitsaRealtimeSessionManager` под наш протокол (§5).
- `AVAudioEngine` capture + 24кГц плеер — без изменений.
- `AppMode` enum + `switchMode(to:)` → добавить `.customRealtime`.
- DAT SDK путь аудио с очков — не трогаем.

**Файлы:**
```
OpenGlasses/Sources/Services/Realtime/
├─ PyatnitsaRealtimeSessionManager.swift   # WS, протокол §5, resume, барже-ин
├─ PyatnitsaProtocol.swift                 # кодеки JSON/бинарь, seq
└─ OpusCodec.swift                         # Opus enc/dec (фаза 2; MVP — PCM16)
OpenGlasses/Sources/App/Views/
└─ RealtimeSettingsView.swift              # URL сервера, вкл/выкл, выбор голоса
```
+ правки: `AppMode.swift` (кейс `.customRealtime`), `Config.swift` (URL/токен),
`VoiceTab`/`ModelPickerSheet` (переключение режима), `TranscriptOverlay` (partials).

**Оценка:** ~600–900 LOC нового кода, ~10 точечных правок. Пересборка нашей отлаженной
цепочкой (XcodeGen → xcodebuild → devicectl).

---

## 10. Железо

- **Baseline (CPU-only, вписывается в обычный VPS):** 4 vCPU / 8 ГБ RAM.
  GigaAM v3 (gigastt, ONNX) + Silero v5_ru (RTF 0.025) + Smart Turn v3 (60мс CPU) +
  Pipecat — всё живёт на CPU. LLM — внешний (прокси), локальный GPU не нужен.
  Ожидаемо: TTFA ~1.0–1.3с. **С этого начинаем.**
- **GPU-uplift (опц.):** любая NVIDIA (даже T4/L4) срезает STT/turn до единиц мс →
  TTFA ближе к ~0.8с. Не обязательно для старта.
- **Гео:** VPS в Франкфурте/Хельсинки/Амстердаме (ниже RTT до РФ + Gemini-регион, если
  включим relay-режим).

---

## 11. Этапы (по возрастанию видимой пользы)

1. **M1 — «говорит по-русски, сквозь мой прокси».** Pipecat на VPS: WSS + Silero VAD +
   GigaAM STT + прокси-LLM + Silero TTS. Клиент: `.customRealtime` режим на клоне
   Gemini-сессии, PCM16. Push-to-talk (без wake). → первый живой диалог из РФ без VPN.
2. **M2 — нативное turn-taking.** Smart Turn v3 (RU endpoint) + барже-ин local-first +
   sentence-chunking TTS. → ощущается разговором, а не рацией.
3. **M3 — надёжность.** Reconnect + session resume + супервизия + мониторинг +
   fallback-цепочки. → «не отваливается».
4. **M4 — vision + wake.** Кадр с очков в vision-модель через прокси; wake «Пятница»
   поднимает realtime-сессию.
5. **M5 — Opus + полировка.** Opus-транспорт, filler-ack, prosody continuity, выбор
   голоса/модели в UI. Опц.: premium-TTS (Cartesia), on-device STT-деградация.
6. **M6 (опц.) — Gemini-relay режим** как A/B для сравнения натуральности.

---

## 12. Риски и митигации

| Риск | Митигация |
|---|---|
| RF DPI ужесточится и на WSS | WSS/443 неотличим от обычного HTTPS; обычный домен; TLS 1.3. При эскалации — свой обфусцирующий фронт на сервере. |
| GigaAM streaming (gigastt) — community-проект | T-one (нативный стрим, Apache) как параллельный fallback; на iPhone — zipformer-ru офлайн. |
| Латентность > ожидаемой на сотовой сети | pre-warm соединений, короткий системный промпт, быстрая LLM-модель; измеряем per-stage. |
| Silero голос недостаточно «премиум» | Cartesia Sonic / F5-TTS_RUSSIAN как premium-тир (переключатель). |
| Один VPS — единая точка отказа | healthcheck+рестарт; клиент авто-фоллбэк на текущий wake-word+прокси режим OpenGlasses. |
| ElevenLabs/санкции | не использовать как основной; если нужен — только исходящий вызов с не-РФ VPS, аккаунт не-РФ. |

---

## 13. Нужно от пользователя (блокирует старт кода)

1. **VPS:** ОС, vCPU/RAM, есть ли GPU, гео (страна). (Определяет тир STT/TTS.)
2. **Домен** под сервер (для TLS/443) — есть или взять.
3. **TTS-старт:** Silero (бесплатно, self-host, хорошо) — по умолчанию; premium
   (Cartesia/ElevenLabs) — позже.
4. **LLM-модель** для голоса через прокси (рекомендую самый быстрый вариант — уточним
   по first-token латентности твоего прокси).
