from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 🔥 추가
from app.api.predict import router as predict_router

app = FastAPI()

# 🔥 CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["https://aiopsfrontend.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# predict 라우터를 "/api" 경로 아래에 포함
app.include_router(predict_router, prefix="/api", tags=["prediction"])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
