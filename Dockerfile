FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FREEVIDEOAI_ENV_FILE=/app/config/runtime.env \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright \
    HF_HOME=/app/workspace_data/huggingface

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-noto-cjk libsndfile1 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
COPY krvoiceai/ ./krvoiceai/
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[cloud,local,publish]" && pip check
RUN python -m playwright install --with-deps chromium

# Local cache is optional. Without it the model is downloaded during build.
COPY docker/model-cache/ /opt/models/
RUN python -c "from pathlib import Path; from huggingface_hub import snapshot_download; p=Path('/opt/models/faster-whisper-small'); snapshot_download('Systran/faster-whisper-small', local_dir=str(p), allow_patterns=['model.bin','config.json','tokenizer.json','vocabulary.*','preprocessor_config.json']) if not (p/'model.bin').is_file() else None" \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('/opt/models/faster-whisper-small', device='cpu', compute_type='int8', local_files_only=True)"

COPY config/ ./config/
COPY scripts/musetalk_worker.py /app/scripts/musetalk_worker.py
COPY docker/default-user.yaml /opt/freevideoai/default-user.yaml
COPY docker/default-user.cpu.yaml /opt/freevideoai/default-user.cpu.yaml
COPY docker/entrypoint.web.sh /usr/local/bin/freevideoai-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/freevideoai-entrypoint \
    && chmod +x /usr/local/bin/freevideoai-entrypoint \
    && mkdir -p /app/workspace_data /app/output /app/config/voices /app/config/avatars /app/config/cookies

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"
ENTRYPOINT ["/usr/local/bin/freevideoai-entrypoint"]
CMD ["python", "-m", "krvoiceai.ui.cli", "web", "--host", "0.0.0.0", "--port", "8000"]
