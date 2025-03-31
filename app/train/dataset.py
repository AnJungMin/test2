import os
import json
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

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

        # 모든 질환이 0인 샘플 제거
        self.data = [
            sample for sample in self.data
            if any(int(sample[f"value_{i+1}"]) != 0 for i in range(6))
        ]

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
        return image, torch.tensor(labels, dtype=torch.long)
