import asyncio
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os, logging, time
logger = logging.getLogger("uvicorn")

app = FastAPI(title="Emotion Classification API")

# -------- PyTorch 스레드 제한 (CPU 성능 안정화) --------
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# -------- 모델 로드 --------
# model_name = "dlckdfuf141/korean-emotion-kluebert-v2"
model_name = "yjungs2/trained_klueBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

emotion_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    # device=0  # GPU(MPS) 쓰려면 주석 해제
)

# -------- 감정 레이블 매핑 --------
emotion_labels = {
    0: "scared",
    1: "surprised",
    2: "angry",
    3: "sad",
    4: "calm",
    5: "happy",
    6: "disgust",
}

# -------- 요청 스키마 --------
class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

# -------- 세마포어로 동시 실행 제한 --------
SEM = asyncio.Semaphore(2)  # 동시에 최대 2개 추론 실행

# -------- 단건 API --------
@app.post("/analyze")
async def analyze(req: TextRequest):
    async with SEM:
        raw = await asyncio.to_thread(emotion_classifier, req.text)
    label = raw[0]["label"]
    idx = int(label.replace("LABEL_", "")) if isinstance(label, str) and "LABEL_" in label else int(label)
    return {"label": emotion_labels[idx], "score": raw[0]["score"]}

# -------- 배치 API --------
@app.post("/analyze/batch")
async def analyze_batch(req: BatchRequest):
    rid = id(req)
    logger.info(f"[ENTER pid={os.getpid()} rid={rid}] sem={SEM._value}")
    async with SEM:
        logger.info(f"[START pid={os.getpid()} rid={rid}] sem={SEM._value}")
        # 여러 문장을 한 번에 묶어 추론 → GPU/CPU 효율 높임
        t0 = time.perf_counter()
        raw = await asyncio.to_thread(
            emotion_classifier,
            req.texts,
            batch_size=8,
            truncation=True
        )
        dt = time.perf_counter() - t0
        logger.info(f"[DONE  pid={os.getpid()} rid={rid}] took={dt:.2f}s sem={SEM._value}")
    
    results = []
    for r in raw:
        label = r["label"]
        idx = int(label.replace("LABEL_", "")) if isinstance(label, str) and "LABEL_" in label else int(label)
        results.append({"label": emotion_labels[idx], "score": r["score"]})
    return {"results": results}