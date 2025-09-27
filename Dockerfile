FROM python:3.12-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 라이브러리 (PyTorch 설치시 필요)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . .

# FastAPI 실행 (Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]