import torch
from app.model.model import MultiTaskMobileViT
from torchvision import transforms
from PIL import Image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 모델 로드 함수
def load_model(weight_path):
    model = MultiTaskMobileViT(use_pretrained_backbone=True)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE)['model_params'])
    model.to(DEVICE)
    model.eval()
    return model

# 예측 함수
def predict(model, image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
    
    predictions = [torch.argmax(head_output, dim=1).item() for head_output in outputs]
    return predictions
