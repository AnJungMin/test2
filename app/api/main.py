# app/api/main.py
from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI()

# predict 라우터를 "/api" 경로 아래에 포함
app.include_router(predict_router, prefix="/api", tags=["prediction"])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
