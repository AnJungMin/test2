import torch
from app.model.model import MultiTaskMobileViT
from app.core.transform import transform  # transform은 이미지 전처리
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weight_path: str):
    model = MultiTaskMobileViT(use_pretrained_backbone=True)  # True로 설정하여 pretrained 모델 로드
    state = torch.load(weight_path, map_location=DEVICE)  # weights_only 파라미터 제거
    model.load_state_dict(state["model_params"])  # 모델 파라미터만 로드
    model.to(DEVICE)
    model.eval()
    return model

def predict(model, image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
    predictions = [torch.argmax(output, dim=1).item() for output in outputs]
    return predictions
