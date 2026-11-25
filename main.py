import os
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Student Score API")

# อนุญาตให้เว็บหน้าบ้าน (Frontend) เรียก API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดไฟล์โมเดล
model_path = os.path.join(os.path.dirname(__file__), "Student_Academic_Performance.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# 1. หน้าแรก (Root) - แสดงหน้าเว็บฟอร์มจาก index.html
@app.get("/")
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path)

# 2. ตรวจสอบสถานะ API
@app.get("/health")
def health():
    return {"status": "ok"}

# 3. ข้อมูล Feature สำหรับทำนาย
@app.get("/info")
def info():
    return {
        "features": ["Study_Hours", "Attendance_Rate", "Assignments"],
        "example": {"Study_Hours": 5, "Attendance_Rate": 85, "Assignments": 8}
    }

# 4. โครงสร้างข้อมูลขาเข้า
class PredictRequest(BaseModel):
    Study_Hours: float
    Attendance_Rate: float
    Assignments: float

# 5. Endpoint ทำนายผลคะแนน
@app.post("/predict")
def predict(data: PredictRequest):
    features = np.array([[data.Study_Hours, data.Attendance_Rate, data.Assignments]])
    prediction = model.predict(features)
    return {"prediction": float(prediction[0])}
