# app/inference.py
import os
import torch
from torchvision import transforms
from PIL import Image
from app.model.model import MultiTaskMobileViT  # app/model/model.py에서 가져옴
from app.train.config import DEVICE            # app/train/config.py에서 DEVICE 가져옴

# 학습된 모델 weight 경로 (app/model_weight 폴더 내의 모델 파일)
MODEL_PATH = os.path.join("app", "model_weight", "MobileVit-XXS_2025_04_01_14_model.pt")

# 이미지 전처리 (학습 시 사용한 전처리와 동일)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = MultiTaskMobileViT(head_channels=64).to(DEVICE)
    # weights_only=False 옵션을 추가하여 체크포인트 로드
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_params'])
    model.eval()
    return model

def predict_image(image=None, image_path=None, model=None):
   
    if image is None and image_path is not None:
        image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
    predictions = [head.argmax(dim=1).item() for head in outputs]
    return predictions
