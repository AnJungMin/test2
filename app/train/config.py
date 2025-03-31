import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 2
TRAIN_RATIO = 0.8
LR_STEP = 3
LR_GAMMA = 0.9

IMAGE_PATH = "data/image"
LABEL_PATH = "data/label"
SAVE_MODEL_PATH = "model_weight/latest.pt"
SAVE_HISTORY_PATH = "history/latest_history.pt"
