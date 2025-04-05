import json
import os

# JSON 파일 경로
RECOMMENDATION_PATH = os.path.join("app", "recommendation", "recommendations.json")

# 파일 로드 (한번만 로드해서 재사용)
with open(RECOMMENDATION_PATH, "r", encoding="utf-8") as f:
    RECOMMENDATION_DATA = json.load(f)

def get_recommendations_by_disease(disease_name, top_k=3):
    """
    질환 이름에 따라 추천 제품 상위 K개 반환
    """
    if disease_name not in RECOMMENDATION_DATA:
        return []

    products = RECOMMENDATION_DATA[disease_name][:top_k]
    return [
        {
            "name": p["product_name"],
            "category": p["category"],
            "similarity": f"{p['similarity'] * 100:.2f}%"
        }
        for p in products
    ]
