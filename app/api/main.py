from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI(
    title="Scalp Disease Classifier API",
    description="멀티태스크 기반 두피 질환 분류 API",
    version="1.0.0"
)

app.include_router(predict_router, prefix="/api")
