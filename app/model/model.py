import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification

class MultiTaskMobileViT(nn.Module):
    def __init__(self, backbone=None, use_pretrained_backbone=False, head_channels=64):
        super(MultiTaskMobileViT, self).__init__()

        if backbone is None:
            if use_pretrained_backbone:
                self.backbone = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")
                backbone_out_channels = 320
            else:
                self.backbone = DummyMobileViTXXS()
                backbone_out_channels = 128
        else:
            self.backbone = backbone
            backbone_out_channels = 320

        # 각 태스크별 classifier (각 head는 두 개의 Inverted Bottleneck Block 후 fc-layer 통과)
        self.fc_mise = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_pizi = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_mosa = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_mono = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_biddem = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_talmo = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)

    def _make_task_head(self, in_channels, head_channels, out_channels=4):
        return nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_channels, out_channels)
        )

    def forward(self, x):
        features = self.backbone(x).last_hidden_state
        mise_head = self.fc_mise(features)
        pizi_head = self.fc_pizi(features)
        mosa_head = self.fc_mosa(features)
        mono_head = self.fc_mono(features)
        biddem_head = self.fc_biddem(features)
        talmo_head = self.fc_talmo(features)
        return mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head
