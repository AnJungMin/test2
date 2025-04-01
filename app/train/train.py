# train/train.py
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from train.config import DEVICE
from train.loss import get_criterion_list

def loss_epoch(model, dataloader, criterion_list, optimizer=None):
    epoch_loss = 0.0
    epoch_loss_components = [0.0] * 6  # 각 태스크별 손실 저장
    task_correct = [0] * 6
    total_correct = 0
    num_data = len(dataloader.dataset)
    
    c_mise, c_pizi, c_mosa, c_mono, c_biddem, c_talmo = criterion_list
    
    for x_batch, y_batch in tqdm(dataloader, leave=False):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        outputs = model(x_batch)
        loss_values = [
            c_mise(outputs[0], y_batch[:, 0]),
            c_pizi(outputs[1], y_batch[:, 1]),
            c_mosa(outputs[2], y_batch[:, 2]),
            c_mono(outputs[3], y_batch[:, 3]),
            c_biddem(outputs[4], y_batch[:, 4]),
            c_talmo(outputs[5], y_batch[:, 5]),
        ]
        loss = sum(loss_values)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        batch_size = x_batch.shape[0]
        epoch_loss += loss.item() * batch_size
        for i, lv in enumerate(loss_values):
            epoch_loss_components[i] += lv.item() * batch_size
        
        preds = torch.stack([o.argmax(dim=1) for o in outputs], dim=1)
        for task in range(6):
            correct = (preds[:, task] == y_batch[:, task]).sum().item()
            task_correct[task] += correct
            total_correct += correct
            
    epoch_loss /= num_data
    epoch_acc = total_correct / (num_data * 6) * 100
    task_acc = [(c / num_data) * 100 for c in task_correct]
    
    return epoch_loss, *epoch_loss_components, epoch_acc, task_acc

def Train(model, train_DL, val_DL, criterion_list, optimizer, EPOCH, save_model_path, save_history_path, **kwargs):
    scheduler = StepLR(optimizer, step_size=kwargs.get('LR_STEP', 3), gamma=kwargs.get('LR_GAMMA', 0.9)) if 'LR_STEP' in kwargs else None
    
    loss_history = {
        'train': {'total': [], 'mise': [], 'pizi': [], 'mosa': [], 'mono': [], 'biddem': [], 'talmo': []},
        'val': {'total': [], 'mise': [], 'pizi': [], 'mosa': [], 'mono': [], 'biddem': [], 'talmo': []}
    }
    acc_history = {
        'train': {'total': [], 'mise': [], 'pizi': [], 'mosa': [], 'mono': [], 'biddem': [], 'talmo': []},
        'val': {'total': [], 'mise': [], 'pizi': [], 'mosa': [], 'mono': [], 'biddem': [], 'talmo': []}
    }
    
    best_loss = float('inf')
    
    for epoch in range(EPOCH):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1}/{EPOCH}, current_LR = {current_lr}")
        
        # 학습 단계
        train_results = loss_epoch(model, train_DL, criterion_list, optimizer)
        train_loss, *train_loss_components, train_acc, train_task_acc = train_results
        
        loss_history['train']['total'].append(train_loss)
        loss_history['train']['mise'].append(train_loss_components[0])
        loss_history['train']['pizi'].append(train_loss_components[1])
        loss_history['train']['mosa'].append(train_loss_components[2])
        loss_history['train']['mono'].append(train_loss_components[3])
        loss_history['train']['biddem'].append(train_loss_components[4])
        loss_history['train']['talmo'].append(train_loss_components[5])
        
        acc_history['train']['total'].append(train_acc)
        acc_history['train']['mise'].append(train_task_acc[0])
        acc_history['train']['pizi'].append(train_task_acc[1])
        acc_history['train']['mosa'].append(train_task_acc[2])
        acc_history['train']['mono'].append(train_task_acc[3])
        acc_history['train']['biddem'].append(train_task_acc[4])
        acc_history['train']['talmo'].append(train_task_acc[5])
        
        # 검증 단계
        model.eval()
        with torch.no_grad():
            val_results = loss_epoch(model, val_DL, criterion_list)
        val_loss, *val_loss_components, val_acc, val_task_acc = val_results
        
        loss_history['val']['total'].append(val_loss)
        loss_history['val']['mise'].append(val_loss_components[0])
        loss_history['val']['pizi'].append(val_loss_components[1])
        loss_history['val']['mosa'].append(val_loss_components[2])
        loss_history['val']['mono'].append(val_loss_components[3])
        loss_history['val']['biddem'].append(val_loss_components[4])
        loss_history['val']['talmo'].append(val_loss_components[5])
        
        acc_history['val']['total'].append(val_acc)
        acc_history['val']['mise'].append(val_task_acc[0])
        acc_history['val']['pizi'].append(val_task_acc[1])
        acc_history['val']['mosa'].append(val_task_acc[2])
        acc_history['val']['mono'].append(val_task_acc[3])
        acc_history['val']['biddem'].append(val_task_acc[4])
        acc_history['val']['talmo'].append(val_task_acc[5])
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_params': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer,
                'scheduler': scheduler
            }, save_model_path)
            print('Model saved!')
        
        if scheduler is not None:
            scheduler.step()
        
        print(f"Train loss: {round(train_loss,5)}, Val loss: {round(val_loss,5)}")
        print(f"Train acc: {round(train_acc,1)}%, Val acc: {round(val_acc,1)}%, Time: {round(time.time()-start_time)} s")
        print("-"*20)
        
        model.train()
        
    torch.save({
        "loss_history": loss_history,
        "acc_history": acc_history,
        "EPOCH": EPOCH,
        "BATCH_SIZE": BATCH_SIZE,
        "TRAIN_RATIO": TRAIN_RATIO,
    }, save_history_path)
        
    return loss_history, acc_history

if __name__ == '__main__':
    from train.config import DEVICE, EPOCH, BATCH_SIZE, SAVE_MODEL_PATH, SAVE_HISTORY_PATH, LR, LR_STEP, LR_GAMMA, TRAIN_RATIO
    from train.dataset import get_dataloaders
    from train.loss import get_criterion_list
    from model.model import MultiTaskMobileViT
    import torch.optim as optim

    # 데이터 경로 (환경에 맞게 수정)
    image_path = r"C:\Users\안정민\Desktop\MTL2\data\image"
    label_path = r"C:\Users\안정민\Desktop\MTL2\data\label"
    train_DL, val_DL, test_DL = get_dataloaders(image_path, label_path)
    
    model = MultiTaskMobileViT(head_channels=64).to(DEVICE)
    criterion_list = get_criterion_list(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    Train(model, train_DL, val_DL, criterion_list, optimizer, EPOCH, SAVE_MODEL_PATH, SAVE_HISTORY_PATH, LR_STEP=LR_STEP, LR_GAMMA=LR_GAMMA)
