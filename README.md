<div align="center">

# 🖥 EVI Server

**Русскоязычный realtime-бэкенд «голос-в-голос» для ассистента EVI**
*Russian-first realtime voice backend · 面向俄语的实时语音后端*

[🇷🇺 Русский](#русский) · [🇬🇧 English](#english) · [🇨🇳 中文](#中文)

📱 [**Клиент · Client · 客户端**](https://github.com/awolla-hub/EVI) &nbsp;·&nbsp; 🖥 [**Сервер · Server · 服务器**](https://github.com/awolla-hub/EVI-SERV)

</div>

---

## Русский

Самостоятельно хостится, голос-в-голос бэкенд для [клиента EVI](https://github.com/awolla-hub/EVI). Каскад **STT → LLM → TTS** одним процессом [Pipecat](https://github.com/pipecat-ai/pipecat) за TLS-WebSocket на `:443` — `wss://` неотличим от обычного HTTPS, что важно там, где realtime-UDP (WebRTC) не работает. LLM смотрит в ваш OpenAI-совместимый прокси, готового realtime-вендора не нужно.

- 🎧 **STT:** GigaAM RNNT (лучшее качество для русского) или faster-whisper — выбор через env.
- 🧠 **LLM:** потоковый ответ через ваш прокси (OpenAI-совместимый).
- 🔊 **TTS:** Silero v5_ru — голоса `aidar` / `baya` / `kseniya` / `xenia` / `eugene`.
- 👁 **Зрение**, 🧩 **память** (SQLite), 📈 **самообучающийся профиль** пользователя.
- 🔐 **Секреты — в `.env`** (в репозитории только `.env.example`); имя пользователя и строка о создателе вынесены в окружение с нейтральными дефолтами.

**Быстрый старт**
```bash
cp .env.example .env          # впишите PROXY_API_KEY
docker compose up --build     # или: pip install -r requirements.txt && python server.py
```
Контейнер слушает `127.0.0.1:8080`; TLS терминирует nginx (см. `deploy.md`). Протокол WebSocket, конвейер и тонкая настройка STT/TTS — ниже в разделе **Technical reference**.

> 📱 **Клиент:** [awolla-hub/EVI](https://github.com/awolla-hub/EVI) — iOS-приложение с живой пиксельной комнатой.

---

## English

A self-hosted, voice-to-voice backend for the [EVI client](https://github.com/awolla-hub/EVI). It runs an **STT → LLM → TTS** cascade as a single [Pipecat](https://github.com/pipecat-ai/pipecat) process behind a TLS WebSocket on `:443` — `wss://` is indistinguishable from ordinary HTTPS, which matters where realtime UDP (WebRTC) is unreliable. The LLM stage points at your own OpenAI-compatible proxy, so no realtime vendor API is required.

- 🎧 **STT:** GigaAM RNNT (best Russian quality) or faster-whisper — env-selectable.
- 🧠 **LLM:** streaming replies via your proxy (OpenAI-compatible).
- 🔊 **TTS:** Silero v5_ru — voices `aidar` / `baya` / `kseniya` / `xenia` / `eugene`.
- 👁 **Vision**, 🧩 **memory** (SQLite), 📈 a **self-learning user profile**.
- 🔐 **Secrets live in `.env`** (only `.env.example` is committed); the user's name and creator line are moved to env with neutral defaults.

**Quick start**
```bash
cp .env.example .env          # set PROXY_API_KEY
docker compose up --build     # or: pip install -r requirements.txt && python server.py
```
The container listens on `127.0.0.1:8080`; put nginx TLS in front (see `deploy.md`). The WebSocket protocol, pipeline and STT/TTS tuning are in **Technical reference** below.

> 📱 **Client:** [awolla-hub/EVI](https://github.com/awolla-hub/EVI) — the iOS app with the living pixel room.

---

## 中文

一个自托管的“语音到语音”后端，配合 [EVI 客户端](https://github.com/awolla-hub/EVI) 使用。它以单个 [Pipecat](https://github.com/pipecat-ai/pipecat) 进程运行 **STT → LLM → TTS** 级联，置于 `:443` 上的 TLS WebSocket 之后——`wss://` 与普通 HTTPS 难以区分，这在实时 UDP（WebRTC）不可靠的环境中很重要。LLM 环节指向你自己的 OpenAI 兼容代理，因此无需任何实时厂商 API。

- 🎧 **STT：** GigaAM RNNT（俄语质量最佳）或 faster-whisper——通过环境变量选择。
- 🧠 **LLM：** 通过你的代理（OpenAI 兼容）流式返回。
- 🔊 **TTS：** Silero v5_ru——音色 `aidar` / `baya` / `kseniya` / `xenia` / `eugene`。
- 👁 **视觉**、🧩 **记忆**（SQLite）、📈 **自学习用户画像**。
- 🔐 **密钥保存在 `.env`**（仓库中只有 `.env.example`）；用户名与“创建者”文案已移至环境变量，并带中性默认值。

**快速开始**
```bash
cp .env.example .env          # 填入 PROXY_API_KEY
docker compose up --build     # 或：pip install -r requirements.txt && python server.py
```
容器监听 `127.0.0.1:8080`；由 nginx 终止 TLS（见 `deploy.md`）。WebSocket 协议、流水线与 STT/TTS 调优见下方 **Technical reference**。

> 📱 **客户端：** [awolla-hub/EVI](https://github.com/awolla-hub/EVI) —— 带“会呼吸的像素房间”的 iOS App。

---

## Technical reference

Pipeline: `transport.input → STT → [partial/final tap] → user context → LLM → [assistant-text tap] → TTS → [speaking tap] → transport.output`. Interruptions enabled for barge-in.

| Stage | Component | File |
|-------|-----------|------|
| Transport | `FastAPIWebsocketTransport` + custom serializer | `transport.py`, `server.py` |
| VAD | Silero VAD (via Pipecat) | `pipeline.py` |
| STT | GigaAM RNNT **or** faster-whisper (env-selectable) | `services/gigaam_stt.py` |
| LLM | `OpenAILLMService` → your proxy, streaming | `services/proxy_llm.py` |
| TTS | Silero v5_ru | `services/silero_tts.py` |

**Wire protocol** — one persistent WebSocket. Binary = audio (16 kHz in / 24 kHz out, mono PCM16 LE); text = JSON control (`hello`, `barge_in`, `ping`/`pong`, `partial`, `user_final`, `assistant_text`, `speaking_start`/`speaking_end`, `interrupted`). Full deploy notes (nginx + TLS) in [`deploy.md`](deploy.md).
