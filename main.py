# -*- coding: utf-8 -*-
import pandas as pd
import joblib  # سنستخدم joblib لتحميل الموديل
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# 1. إنشاء التطبيق
app = FastAPI()

# 2. إعدادات CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. تحميل الموديل باستخدام joblib
# ملاحظة: تأكد أن ملف الموديل هو "best_student_grade_model.pkl" وموجود بجانب هذا الملف
model = joblib.load("best_student_grade_model.pkl")

# 4. تعريف هيكل البيانات المستقبلة
class InputData(BaseModel):
    Age: float
    Gender: float
    Ethnicity: float
    ParentalEducation: float
    StudyTimeWeekly: float
    Absences: float
    Tutoring: float
    ParentalSupport: float
    Extracurricular: float
    Sports: float
    Music: float
    Volunteering: float

# 5. تعريف هيكل البيانات المرسلة
class OutputData(BaseModel):
    prediction: str

# 6. نقطة التنبؤ (Endpoint)
@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    # تحويل البيانات إلى DataFrame
    df = pd.DataFrame([data.model_dump()])
    
    # إجراء التنبؤ باستخدام الموديل مباشرة (الذي هو عبارة عن Pipeline)
    # الموديلات المحفوظة من PyCaret تحتوي بداخلها على كل خطوات معالجة البيانات
    prediction = model.predict(df)

    # استخراج القيمة المتوقعة
    result = str(prediction[0])

    return {
        "prediction": result
    }

# 7. تشغيل السيرفر
if __name__ == "__main__":
    # ملاحظة: عند الرفع على Render، المنصة هي من تشغل uvicorn، هذا السطر للتجربة المحلية فقط
    uvicorn.run(app, host="0.0.0.0", port=8000)