FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf \
    TRANSFORMERS_CACHE=/app/hf \
    MODEL_DIR=/app/model \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⬇️ 인라인 대신 스크립트로 스냅샷
COPY download_model.py .
RUN python download_model.py && rm download_model.py

COPY src ./src

EXPOSE 8080
CMD ["sh","-c","uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]