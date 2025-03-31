from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io

from app.model.inference import load_model, predict

router = APIRouter()

model = load_model("app/model_weight/MobileVit-XXS_model_SCALP_2025_03_31_15.pt")


@router.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = predict(model, image)

    return {
        "task_1_mise": results[0],
        "task_2_pizi": results[1],
        "task_3_mosa": results[2],
        "task_4_mono": results[3],
        "task_5_biddem": results[4],
        "task_6_talmo": results[5],
    }
