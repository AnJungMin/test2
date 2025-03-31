from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from app.model.inference import load_model, predict

router = APIRouter()

# 경로는 Render 기준 상대경로로 조정
model = load_model("app/model_weight/MobileVit-XXS_model_SCALP_2025_03_31_15.pt")

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    predictions = predict(model, image)
    
    response = {
        "task_1_mise": predictions[0],
        "task_2_pizi": predictions[1],
        "task_3_mosa": predictions[2],
        "task_4_mono": predictions[3],
        "task_5_biddem": predictions[4],
        "task_6_talmo": predictions[5],
    }
    return response
