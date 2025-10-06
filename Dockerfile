FROM python:3.12-slim

WORKDIR /app

# (선택) 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

# 파이썬 라이브러리
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . .

# Cloud Run이 넣어주는 PORT 사용 (기본 8080)
ENV PORT=8080
EXPOSE 8080

# JSON 배열에서 $PORT 확장 안 됨 → shell 형태로 실행해야 함
# app 모듈명이 main.py면 main:app, app.py면 app:app 로 맞춰주세요.
CMD ["sh","-c","uvicorn app:app --app-dir src --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
# 또는
# CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]