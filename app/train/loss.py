import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_pt = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-log_pt)

        if isinstance(self.alpha, (list, torch.Tensor)):
            if isinstance(self.alpha, list):
                self.alpha = torch.tensor(self.alpha, device=inputs.device)
            at = self.alpha[targets]
        else:
            at = self.alpha

        focal_loss = at * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
