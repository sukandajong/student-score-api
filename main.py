import json
import pickle
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import create_model

# 1. อ่านการตั้งค่าจาก config.json
BASE_DIR = Path(__file__).parent
CONFIG = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
FEATURES = CONFIG["features"]
NAMES = [f["name"] for f in FEATURES]
LABELS = CONFIG.get("class_labels") or {}

# 2. ฟังก์ชันโหลดโมเดล (.pkl หรือ .joblib)
def load_file(name):
    path = BASE_DIR / name
    if name.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            return pickle.load(f)
    return joblib.load(path)

model = load_file(CONFIG["model_file"])
scaler = load_file(CONFIG["scaler_file"]) if CONFIG.get("scaler_file") else None

# 3. สร้าง Schema สำหรับรับ Input
InputData = create_model(
    "InputData",
    **{f["name"]: ((float if f["type"] == "number" else str), ...) for f in FEATURES},
)

# 4. สร้าง FastAPI App พร้อม CORS
app = FastAPI(title=CONFIG.get("title") or "Student Score API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_python(value):
    return value.item() if hasattr(value, "item") else value

def prepare(obj, X):
    return X if hasattr(obj, "feature_names_in_") else X.to_numpy()

def label_of(value):
    key = str(to_python(value))
    return LABELS.get(key, key)

@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__, "features": NAMES}

@app.get("/info")
def info():
    return {"features": FEATURES, "class_labels": LABELS, "example": CONFIG.get("example")}

@app.post("/predict")
def predict(data: InputData):
    # จัดคอลัมน์ให้ตรงกับตอนเทรน
    X = pd.DataFrame([data.model_dump()], columns=NAMES)
    
    # ปรับสเกลถ้ามี Scaler
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(prepare(scaler, X)), columns=NAMES)
        
    # ทำนายผล
    X_ready = prepare(model, X)
    prediction = to_python(model.predict(X_ready)[0])
    result = {"prediction": prediction}
    
    # กรณีเป็นโมเดลจำแนกประเภท (Classification)
    if hasattr(model, "classes_"):
        result["label"] = label_of(prediction)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_ready)[0]
        result["probabilities"] = {label_of(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}
        result["confidence"] = round(float(max(proba)), 4)
        
    return result
