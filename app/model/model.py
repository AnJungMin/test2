import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification

class MultiTaskMobileViT(nn.Module):
    def __init__(self, use_pretrained_backbone=True, head_channels=64):
        super(MultiTaskMobileViT, self).__init__()
        
        if use_pretrained_backbone:
            self.backbone = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")
            backbone_out_channels = 320
        else:
            raise NotImplementedError("Only pretrained MobileViT is currently supported.")

        # 각 질환별 분류 head
        self.fc_mise = self._make_head(backbone_out_channels, head_channels)
        self.fc_pizi = self._make_head(backbone_out_channels, head_channels)
        self.fc_mosa = self._make_head(backbone_out_channels, head_channels)
        self.fc_mono = self._make_head(backbone_out_channels, head_channels)
        self.fc_biddem = self._make_head(backbone_out_channels, head_channels)
        self.fc_talmo = self._make_head(backbone_out_channels, head_channels)

    def _make_head(self, in_channels, out_channels, num_classes=4):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x).last_hidden_state  # [B, C, H, W]
        return (
            self.fc_mise(features),
            self.fc_pizi(features),
            self.fc_mosa(features),
            self.fc_mono(features),
            self.fc_biddem(features),
            self.fc_talmo(features),
        )
