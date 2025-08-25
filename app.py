from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Emotion Classification API")

# Colab 에서 사용한 모델 그대로 로컬에서 로드
# 모델 및 토크나이저 로드
model_name = "dlckdfuf141/korean-emotion-kluebert-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 감정 분석 파이프라인 생성
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 감정 레이블 매핑
emotion_labels = {
    0: ("scared"),
    1: ("surprised"),
    2: ("angry"),
    3: ("sad"),
    4: ("calm"),
    5: ("happy"),
    6: ("disgust"),
}

# 요청 스키마 (입력 text)
class TextRequest(BaseModel):
    text: str
    # 요청으로 들어오는 text 필드가 str(문자열) 타입이어야 함
    # 즉, FastAPI에서 POST 요청을 보낼 때 JSON 형식으로 보내야 함

# 감정분석 API
@app.post("/analyze")
def analyze(req: TextRequest):
    raw_result = emotion_classifier(req.text)
    result = {
        "label": emotion_labels[raw_result[0]["label"]],
        "score": raw_result[0]["score"]
    }
    return result