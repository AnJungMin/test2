# app/api/predict.py
from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
from app.inference import load_model, predict_image

router = APIRouter()

# 모델은 한 번 로드해두어 재사용 (추론 속도 향상)
model = load_model()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    preds = predict_image(image=image, model=model)
    return {"predictions": preds}
