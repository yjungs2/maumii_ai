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

# git은 필요할 때만, 불필요하면 빼도 됩니다.
RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 스냅샷 스크립트 (이미지 빌드 시 모델을 /app/model 로 저장)
COPY download_model.py .
RUN HF_HUB_OFFLINE=0 python download_model.py && rm download_model.py


# 앱 소스
COPY src ./src

EXPOSE 8080

# uvicorn 실행 (PORT 환경변수 사용)
CMD ["sh","-c","uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]