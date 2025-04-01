# train/dataset.py
import os
import json
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from app.train.config import TRAIN_RATIO, BATCH_SIZE  # 수정된 경로

class MultiTaskDataset(Dataset):
    def __init__(self, img_dir, json_dir, transform=None):
        self.data = []
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data_part = json.load(f)
                if isinstance(data_part, dict):
                    self.data.append(data_part)
                else:
                    self.data.extend(data_part)
        
        # 라벨이 모두 0인 샘플 제거 (6개 질환 모두 0인 경우)
        filtered_data = []
        for sample in self.data:
            labels = [int(sample[f"value_{i+1}"]) for i in range(6)]
            if any(label != 0 for label in labels):
                filtered_data.append(sample)
        self.data = filtered_data
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.img_dir, sample["image_file_name"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = [int(sample[f"value_{i+1}"]) for i in range(6)]
        labels = torch.tensor(labels, dtype=torch.long)
        return image, labels

# 학습 시 사용한 전처리
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataloaders(image_path, label_path):
    dataset = MultiTaskDataset(image_path, label_path, transform=data_transforms)
    total_len = len(dataset)
    NoT = int(total_len * TRAIN_RATIO)
    NoV = int(total_len * (1 - TRAIN_RATIO) / 2)
    NoTest = total_len - NoT - NoV
    
    train_DS, val_DS, test_DS = random_split(dataset, [NoT, NoV, NoTest])
    train_DL = DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    val_DL = DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=False)
    test_DL = DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_DL, val_DL, test_DL
