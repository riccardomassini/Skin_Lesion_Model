import torch
import os

TRAIN_DIR = "TRAIN_DATA/"
TEST_DIR = "TEST_DATA/"
PARAM_DIR = "MODEL_AND_PARAMETERS/"
METRICS_DIR = PARAM_DIR + "METRICS/"
LAST_MODELS_DIR = PARAM_DIR + "LAST_MODELS/"
BEST_MODELS_DIR = PARAM_DIR + "BEST_MODELS/"
MATRIX_DIR = PARAM_DIR + "CONFUSION_MATRIX/"
BASE_CLASSES = ['DERM', 'DERM_CLINIC', 'OTHER', 'PAN']

TARGET_CLASS_MAPPING = {
    'DERM': 0,
    'DERM_CLINIC': 1,
    'LOW_QUALITY': 2,
    'OTHER': 3,
    'PAN': 4
}

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 20

os.makedirs(PARAM_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LAST_MODELS_DIR, exist_ok=True)
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)