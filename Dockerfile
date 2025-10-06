FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf \
    TRANSFORMERS_CACHE=/app/hf \
    MODEL_DIR=/app/model \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔽 빌드 타임에 모델 스냅샷
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='yjungs2/trained_klueBERT', local_dir='/app/model', local_dir_use_symlinks=False)"

COPY src ./src

EXPOSE 8080
CMD ["sh","-c","uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]