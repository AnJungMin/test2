# app/inference.py
import os
import torch
from torchvision import transforms
from PIL import Image
from app.model.model import MultiTaskMobileViT  # 수정: app/model/model.py 경로 사용
from train.config import DEVICE

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
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_params'])
    model.eval()
    return model

def predict_image(image=None, image_path=None, model=None):
    """
    image: PIL.Image 객체 또는
    image_path: 이미지 파일 경로 (둘 중 하나 제공)
    model: 미리 로드된 모델 (없으면 load_model()을 호출)
    """
    if image is None and image_path is not None:
        image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
    predictions = [head.argmax(dim=1).item() for head in outputs]
    return predictions

# 개별 실행 테스트
if __name__ == '__main__':
    test_img = r"C:\Users\안정민\Desktop\MTL2\data\image\sample.jpg"  # 테스트 이미지 경로 (예시)
    model = load_model()
    preds = predict_image(image_path=test_img, model=model)
    print("Predictions:", preds)
