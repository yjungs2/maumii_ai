import os, time, threading, logging, asyncio
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger("uvicorn")

app = FastAPI(title="Emotion Classification API")

# CPU 안정화
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")

# 모델 디렉토리 (이미지 빌드시 /app/model로 내려받아 이미지에 포함)
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")

# 전역 상태
tokenizer = None
model = None
clf = None
model_ready = False
_lock = threading.Lock()

emotion_labels = {
    0: "scared",
    1: "surprised",
    2: "angry",
    3: "sad",
    4: "calm",
    5: "happy",
    6: "disgust",
}

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

def _load_model():
    """앱 스타트 직후 백그라운드에서 모델 로딩 (런타임 외부 다운로드 금지)"""
    global tokenizer, model, clf, model_ready
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        mdl.to("cpu")
        pl = pipeline("text-classification", model=mdl, tokenizer=tok, device=-1)
        # 워밍업
        _ = pl("warmup")

        with _lock:
            tokenizer, model, clf = tok, mdl, pl
            model_ready = True
        print("[ai] model loaded and ready", flush=True)
    except Exception as e:
        print(f"[ai] model load failed: {e}", flush=True)

@app.on_event("startup")
def startup():
    # 백그라운드로 로딩 → uvicorn이 즉시 8080 리슨 가능
    threading.Thread(target=_load_model, daemon=True).start()

@app.get("/healthz")
def health():
    return {"ok": True, "ready": model_ready}

@app.post("/analyze")
async def analyze(req: TextRequest):
    # 모델 준비 대기(최대 30초)
    deadline = time.time() + 30
    while not model_ready and time.time() < deadline:
        await asyncio.sleep(0.2)
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    # ⬇️ 세마포어 제거: 요청마다 바로 스레드 실행
    raw = await asyncio.to_thread(clf, req.text, truncation=True)
    label = raw[0]["label"]
    idx = int(label.replace("LABEL_", "")) if isinstance(label, str) and "LABEL_" in label else int(label)
    return {"label": emotion_labels[idx], "score": raw[0]["score"]}

@app.post("/analyze/batch")
async def analyze_batch(req: BatchRequest):
    deadline = time.time() + 30
    while not model_ready and time.time() < deadline:
        await asyncio.sleep(0.2)
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    # ⬇️ 세마포어 제거: 배치도 바로 실행
    t0 = time.perf_counter()
    raw = await asyncio.to_thread(
        clf,
        req.texts,
        batch_size=min(int(os.getenv("BATCH_SIZE", "8")), max(1, len(req.texts))),
        truncation=True
    )
    dt = time.perf_counter() - t0
    logger.info(f"[BATCH DONE] took={dt:.2f}s size={len(req.texts)}")

    results = []
    for r in raw:
        label = r["label"]
        idx = int(label.replace("LABEL_", "")) if isinstance(label, str) and "LABEL_" in label else int(label)
        results.append({"label": emotion_labels[idx], "score": r["score"]})
    return {"results": results}