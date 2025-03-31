import torch
from PIL import Image
from app.model.model import MultiTaskMobileViT
from app.core.transform import transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weight_path: str):
    model = MultiTaskMobileViT(use_pretrained_backbone=True)
    state = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state["model_params"])
    model.to(DEVICE)
    model.eval()
    return model

def predict(model, image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
    predictions = [torch.argmax(output, dim=1).item() for output in outputs]
    return predictions
