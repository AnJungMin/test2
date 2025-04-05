import os
import torch
from torchvision import transforms
from PIL import Image
from app.model.model import MultiTaskMobileViT
from app.train.config import DEVICE
from app.recommendation.utils import get_recommendations_by_disease  # 추천 불러오기

# 모델 경로
MODEL_PATH = os.path.join("app", "model_weight", "MobileVit-XXS_2025_04_01_14_model.pt")

# 전처리
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모델 로드
def load_model():
    model = MultiTaskMobileViT(head_channels=64).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_params'])
    model.eval()
    return model

# 예측 함수
def predict_image(image=None, image_path=None, model=None):
    if image is None and image_path is not None:
        image = Image.open(image_path).convert("RGB")

    image = data_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

    task_names = ["미세각질", "피지과다", "모낭사이홍반", "모낭홍반/농포", "비듬", "탈모"]
    severity_labels = ["정상", "경증", "중등증", "중증"]

    raw_preds = []
    formatted_preds = []

    for task_name, head in zip(task_names, outputs):
        probs = torch.softmax(head, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item() * 100

        raw_preds.append(pred_class)

        result = {
            "disease": task_name,
            "severity": severity_labels[pred_class],
            "confidence": f"{confidence:.2f}%"
        }

        # 심각도에 따른 분기 처리
        if pred_class == 0:
            result["comment"] = "정상 범위입니다. 두피 상태가 양호합니다."
        elif pred_class in [1, 2]:
            result["recommendations"] = get_recommendations_by_disease(task_name)
        elif pred_class == 3:
            result["hospital_recommendation"] = "주변 피부과를 추천합니다. 위치 정보를 기반으로 제공합니다."

        formatted_preds.append(result)

    return {
        "raw_predictions": raw_preds,
        "results": formatted_preds
    }
