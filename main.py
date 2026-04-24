# -*- coding: utf-8 -*-
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# 1. إنشاء التطبيق
app = FastAPI()

# 2. إعدادات CORS (خارج الدالة - في المكان الصحيح)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. تحميل الموديل (تأكد من أن الملف موجود في نفس المجلد)
# ملاحظة: إذا كان اسم الملف "best_student_grade_model.pkl" نكتب الاسم بدون .pkl
model = load_model("best_student_grade_model")

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
    
    # إجراء التنبؤ باستخدام PyCaret
    prediction_df = predict_model(model, data=df)

    # استخراج القيمة المتوقعة (تأكد من اسم العمود في PyCaret)
    # عادة يكون 'prediction_label' أو 'Label'
    result = str(prediction_df["prediction_label"].iloc[0])

    return {
        "prediction": result
    }

# 7. تشغيل السيرفر
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)