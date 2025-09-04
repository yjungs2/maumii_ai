# GPU 학슴 및 저장 코드 (fine-tuning)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import os

# =====================
# 1. 데이터 준비
# =====================
# 예시 데이터셋 (실제는 CSV/JSON 불러와서 사용)
data = [
    {"text": "너무 기분이 좋아!", "label": 0},  # happy
    {"text": "오늘 너무 슬퍼...", "label": 1},  # sad
    {"text": "화가 난다!", "label": 2},       # angry
]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# =====================
# 2. 모델과 토크나이저 불러오기
# =====================
model_name = "dlckdfuf141/korean-emotion-kluebert-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# =====================
# 3. 토크나이징
# =====================
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# =====================
# 4. GPU / CPU 설정
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =====================
# 5. 학습 설정
# =====================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,                # GPU니까 3 epoch 정도
    per_device_train_batch_size=16,    # 대중적인 GPU에서 안전한 크기
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),    # GPU일 때만 Mixed Precision
    seed=42,
    save_total_limit=2                 # 저장본 최대 2개만 유지
)

# =====================
# 6. Trainer 정의
# =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# =====================
# 7. 학습 실행
# =====================
trainer.train()

# =====================
# 8. 학습된 모델 저장
# =====================
save_path = "./trained_model"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 학습된 모델이 {save_path} 에 저장되었습니다.")


# trained_model 폴더는 용량이 크기 때문에 Git에 올릴 때 주의
# Git LFS(Large File Storage) 사용
# Git LFS 설정 방법
# 학습한 모델을 trained_model/ 에 저장했다고 가정하면
# GIT BASH 에서 사용할 명령어
# Git LFS 설치 (한번만 하면 됨)
#  git lfs install
# 모델 파일 추적 (특히 .bin 파일)
# git lfs track "*.bin"
# gitignore 대신 .gitattributes 에 규칙이 추가됨
# git add .gitattributes
# 모델 폴더 추가
# git add trained_model/
# git commit -m "Add trained Huggingface model"
# git push origin main
# 만약 GitHub 용량 제한 때문에 불편하면 Hugging Face Hub 에 올려서 이를 받아와 사용하기