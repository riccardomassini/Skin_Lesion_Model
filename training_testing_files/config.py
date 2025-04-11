import torch
import os

TRAIN_DIR = "TRAIN_DATA/"
TEST_DIR = "TEST_DATA/"
PARAM_DIR = "MODEL_AND_PARAMETERS_V4/"
ACCLOSS_DIR = PARAM_DIR + "ACCURACY_AND_LOSS/"
MODELS_DIR = PARAM_DIR + "BEST_MODELS/"
MATRIX_DIR = PARAM_DIR + "CONFUSION_MATRIX/"
BASE_CLASSES = ['DERM', 'DERM_CLINIC', 'OTHER', 'PAN']

TARGET_CLASS_MAPPING = {
    'DERM': 0,
    'DERM_CLINIC': 1,
    'LOW_QUALITY': 2,
    'OTHER': 3,
    'PAN': 4
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(PARAM_DIR, exist_ok=True)
os.makedirs(ACCLOSS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MATRIX_DIR, exist_ok=True)