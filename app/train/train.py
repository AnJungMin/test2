import time
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from train.dataset import MultiTaskDataset
from train.loss import FocalLoss
from train.config import *

from app.model.model import MultiTaskMobileViT

# ----------------------------
# 전처리 구성
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------
# 데이터셋 로딩
# ----------------------------
total_ds = MultiTaskDataset(IMAGE_PATH, LABEL_PATH, transform)
n_train = int(len(total_ds) * TRAIN_RATIO)
n_val = int(len(total_ds) * 0.1)
n_test = len(total_ds) - n_train - n_val

train_ds, val_ds, test_ds = random_split(total_ds, [n_train, n_val, n_test])
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ----------------------------
# 모델 및 손실함수
# ----------------------------
model = MultiTaskMobileViT(use_pretrained_backbone=True).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

criterion_list = [FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE) for _ in range(6)]

# ----------------------------
# 에폭별 손실/정확도 기록용
# ----------------------------
def loss_epoch(model, dataloader, criterion_list, optimizer=None):
    model.train() if optimizer else model.eval()

    epoch_loss, correct = 0, 0
    task_loss = [0.0] * 6
    task_correct = [0] * 6
    total = len(dataloader.dataset)

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        outputs = model(x_batch)

        losses = [criterion(outputs[i], y_batch[:, i]) for i, criterion in enumerate(criterion_list)]
        total_loss = sum(losses)

        if optimizer:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        epoch_loss += total_loss.item() * x_batch.size(0)
        for i in range(6):
            task_loss[i] += losses[i].item() * x_batch.size(0)
            preds = torch.argmax(outputs[i], dim=1)
            task_correct[i] += (preds == y_batch[:, i]).sum().item()

    avg_loss = epoch_loss / total
    avg_task_loss = [tl / total for tl in task_loss]
    avg_acc = sum(task_correct) / (total * 6) * 100
    avg_task_acc = [tc / total * 100 for tc in task_correct]

    return avg_loss, avg_task_loss, avg_acc, avg_task_acc

# ----------------------------
# 전체 학습 루프
# ----------------------------
def Train():
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch}/{EPOCHS}] LR: {current_lr}")

        train_loss, train_task_loss, train_acc, train_task_acc = loss_epoch(model, train_dl, criterion_list, optimizer)
        val_loss, val_task_loss, val_acc, val_task_acc = loss_epoch(model, val_dl, criterion_list)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"Time: {round(time.time() - start_time)}s\n{'-'*50}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_params": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, SAVE_MODEL_PATH)
            print("✅ Model saved!")

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    Train()
