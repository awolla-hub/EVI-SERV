# Пятница Realtime server — CPU-only image.
FROM python:3.12-slim

# System deps:
#   git        -> torch.hub.load() clones the Silero models repo
#   ffmpeg     -> faster-whisper audio decoding
#   libsndfile1-> soundfile / torchaudio I/O (GigaAM)
#   build-essential -> occasional wheel compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching. Torch comes from the CPU
# wheel index declared inside requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# torch.hub cache (Silero TTS) + HF cache (faster-whisper) persist via volumes.
ENV TORCH_HOME=/root/.cache/torch \
    HF_HOME=/root/.cache/huggingface \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/health')" || exit 1

CMD ["python", "server.py"]
