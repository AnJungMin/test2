# train/config.py
import os
import random
import numpy as np
import torch
from datetime import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 하이퍼파라미터
BATCH_SIZE = 64
LR = 1e-4
LR_STEP = 3
LR_GAMMA = 0.9
EPOCH = 1
TRAIN_RATIO = 0.8

# 모델 타입 및 저장 경로 (학습된 모델은 app/model_weight 폴더에 위치)
model_type = "MobileVit-XXS_" + datetime.now().strftime("%Y_%m_%d_%H")
SAVE_MODEL_PATH = os.path.join("app", "model_weight", f"{model_type}_model.pt")
SAVE_HISTORY_PATH = os.path.join("result", "history", f"{model_type}_history.pt")
SAVE_RESULT_CSV_PATH = os.path.join("result", "model_results.csv")

# 시드 설정 (재현성을 위해)
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
