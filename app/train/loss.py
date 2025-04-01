# train/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
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
        else:
            return focal_loss

def get_criterion_list(device):
    import torch
    # 예시: 각 태스크별 클래스 수 기반 샘플 수
    mise_counts = torch.tensor([74766., 5702., 7062., 2940.])
    pizi_counts = torch.tensor([17450., 36193., 31945., 4882.])
    mosa_counts = torch.tensor([28875., 39217., 16859., 5519.])
    mono_counts = torch.tensor([86221., 2841., 981., 427.])
    biddem_counts = torch.tensor([53865., 21415., 12287., 2903.])
    talmo_counts = torch.tensor([66884., 17549., 4959., 1078.])
    
    mise_alpha = (1.0 / mise_counts); mise_alpha = (mise_alpha / mise_alpha.sum()).to(device)
    pizi_alpha = (1.0 / pizi_counts); pizi_alpha = (pizi_alpha / pizi_alpha.sum()).to(device)
    mosa_alpha = (1.0 / mosa_counts); mosa_alpha = (mosa_alpha / mosa_alpha.sum()).to(device)
    mono_alpha = (1.0 / mono_counts); mono_alpha = (mono_alpha / mono_alpha.sum()).to(device)
    biddem_alpha = (1.0 / biddem_counts); biddem_alpha = (biddem_alpha / biddem_alpha.sum()).to(device)
    talmo_alpha = (1.0 / talmo_counts); talmo_alpha = (talmo_alpha / talmo_alpha.sum()).to(device)
    
    criterion_mise = FocalLoss(alpha=mise_alpha, gamma=2).to(device)
    criterion_pizi = FocalLoss(alpha=pizi_alpha, gamma=2).to(device)
    criterion_mosa = FocalLoss(alpha=mosa_alpha, gamma=2).to(device)
    criterion_mono = FocalLoss(alpha=mono_alpha, gamma=2).to(device)
    criterion_biddem = FocalLoss(alpha=biddem_alpha, gamma=2).to(device)
    criterion_talmo = FocalLoss(alpha=talmo_alpha, gamma=2).to(device)
    
    return [criterion_mise, criterion_pizi, criterion_mosa, criterion_mono, criterion_biddem, criterion_talmo]
