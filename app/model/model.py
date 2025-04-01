# model/model.py
import torch
import torch.nn as nn
from transformers import MobileViTForImageClassification
from train.config import DEVICE

class MobileViTBackbone(nn.Module):
    def __init__(self, model_name="apple/mobilevit-xx-small"):
        super(MobileViTBackbone, self).__init__()
        self.model = MobileViTForImageClassification.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
            outputs = self.model(dummy_input)
            features = outputs.hidden_states[-1]
            self.out_channels = features.shape[1]
            
    def forward(self, x):
        outputs = self.model(x)
        features = outputs.hidden_states[-1]
        return features

class InvertedBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super(InvertedBottleneckBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.stride = stride
        self.use_residual = (self.stride == 1 and in_channels == out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
             
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out

class MultiTaskMobileViT(nn.Module):
    def __init__(self, head_channels=64):
        super(MultiTaskMobileViT, self).__init__()
        self.backbone = MobileViTBackbone()
        backbone_out_channels = self.backbone.out_channels
        self.fc_mise = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_pizi = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_mosa = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_mono = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_biddem = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        self.fc_talmo = self._make_task_head(backbone_out_channels, head_channels, out_channels=4)
        
    def _make_task_head(self, in_channels, head_channels, out_channels=4):
        return nn.Sequential(
            InvertedBottleneckBlock(in_channels=in_channels, out_channels=head_channels, expansion_factor=6, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(head_channels, out_channels)
        )
 
    def forward(self, x):
        features = self.backbone(x)
        mise_head = self.fc_mise(features)
        pizi_head = self.fc_pizi(features)
        mosa_head = self.fc_mosa(features)
        mono_head = self.fc_mono(features)
        biddem_head = self.fc_biddem(features)
        talmo_head = self.fc_talmo(features)
        return mise_head, pizi_head, mosa_head, mono_head, biddem_head, talmo_head

if __name__ == '__main__':
    model = MultiTaskMobileViT(head_channels=64).to(DEVICE)
    input_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    outputs = model(input_tensor)
    for i, out in enumerate(outputs):
        print(f"Head {i+1} output shape: {out.shape}")
