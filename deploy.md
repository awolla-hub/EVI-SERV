# Deploy — Пятница Realtime server

Target: a small VPS **outside Russia** (Frankfurt / Helsinki / Amsterdam for low
RTT to RU). Baseline is CPU-only: **4 vCPU / 8 GB RAM** is enough for GigaAM +
Silero + VAD + Pipecat. The app connects over `wss://your-domain/ws` on port 443,
which is indistinguishable from normal HTTPS to DPI.

## 0. Prerequisites

- A domain (e.g. `pyatnitsa.example.com`) with an A/AAAA record pointing at the
  VPS. Use a **real domain**, not `*.fly.dev` / `*.trycloudflare.com`.
- Docker + Docker Compose, or Python 3.12 + systemd.
- Ports 80 and 443 open.

## 1. Run the app container (loopback only)

```bash
git clone <your-repo> && cd pyatnitsa-server
cp .env.example .env
#   edit .env: PROXY_API_KEY, MODEL, STT_BACKEND, ...
#   AUTH_TOKEN is mandatory in production (openssl rand -hex 32): without it /ws
#   accepts anyone who learns the URL and replies with your conversation history.
docker compose up -d --build
curl http://127.0.0.1:8080/health      # {"status":"ok",...}
```

The container binds `127.0.0.1:8080`; nginx terminates TLS and reverse-proxies
to it. Nothing but nginx is exposed publicly. Cross-session memory (SQLite) and
the voice / self-learning profiles live in the `pyatnitsa-state` volume mounted
at `/opt/pyatnitsa`; set `PYATNITSA_STATE=/srv/pyatnitsa` in `.env` to bind a
host directory instead.

## 2. nginx + Let's Encrypt (TLS on :443)

Install nginx and certbot:

```bash
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/pyatnitsa`:

```nginx
server {
    listen 80;
    server_name pyatnitsa.example.com;
    # certbot fills in the HTTP-01 challenge here, then redirects to 443.
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    server_name pyatnitsa.example.com;

    # Managed by certbot (step below).
    ssl_certificate     /etc/letsencrypt/live/pyatnitsa.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/pyatnitsa.example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # WebSocket reverse proxy -> app container on loopback.
    location /ws {
        proxy_pass http://127.0.0.1:8080/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Long-lived audio streams: don't let nginx time them out.
        proxy_read_timeout  3600s;
        proxy_send_timeout  3600s;
        proxy_buffering     off;
    }

    location /health {
        proxy_pass http://127.0.0.1:8080/health;
    }
}
```

Enable and get a cert:

```bash
sudo ln -s /etc/nginx/sites-available/pyatnitsa /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d pyatnitsa.example.com   # provisions + auto-renews TLS
```

Verify from your laptop:

```bash
# Should upgrade to a WebSocket (101). Any wss client works; websocat example
# (the Bearer header is required whenever AUTH_TOKEN is set — otherwise 1008):
websocat -H "Authorization: Bearer $AUTH_TOKEN" wss://pyatnitsa.example.com/ws
```

## 3. Point the app at it

In OpenGlasses settings set the custom realtime server URL to:

```
wss://pyatnitsa.example.com/ws
```

and select the `.customRealtime` mode. The client streams 16 kHz PCM16 up and
plays 24 kHz PCM16 down — no app-side audio changes needed for M1.

## 4. Swapping STT / TTS in production

- **STT backend:** edit `.env` → `STT_BACKEND=gigaam` (best Russian) or
  `faster_whisper` (default, zero-setup). For GigaAM, uncomment `gigaam` in
  `requirements.txt`, rebuild, and set `GIGAAM_MODEL` to the id your installed
  release exposes (`rnnt` / `v2_rnnt` / `v3_rnnt`). Then
  `docker compose up -d --build`.
- **TTS voice:** edit `.env` → `TTS_SPEAKER` (`aidar`, `baya`, `kseniya`,
  `xenia`, `eugene`) and restart. No rebuild needed. Swapping the whole TTS
  engine (e.g. Cartesia) means editing `services/silero_tts.py` / `pipeline.py`.

## 5. Supervision & health

- `restart: unless-stopped` (compose) or a systemd unit keeps it alive.
- Docker `HEALTHCHECK` hits `/health`; `docker inspect` shows status.
- First start downloads models into the `model-cache` volume; watch
  `docker compose logs -f pyatnitsa` for "Silero TTS ready" / model load lines.
- Monitoring (Uptime Kuma / Prometheus) and fallback chains are M3.

## Alternative: systemd (no Docker)

```ini
# /etc/systemd/system/pyatnitsa.service
[Unit]
Description=Pyatnitsa Realtime server
After=network.target

[Service]
WorkingDirectory=/opt/pyatnitsa-server
Environment=PATH=/opt/pyatnitsa-server/.venv/bin
EnvironmentFile=/opt/pyatnitsa-server/.env
ExecStart=/opt/pyatnitsa-server/.venv/bin/python server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now pyatnitsa
```

nginx config is identical — it still proxies `:443/ws` → `127.0.0.1:8080/ws`.
