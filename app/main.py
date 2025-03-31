from fastapi import FastAPI
from app.predict import router as predict_router

app = FastAPI(title="Scalp Disease Classifier API")

app.include_router(predict_router, prefix="/api")
