from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import torch
import json
import os

from training_testing_files.config import DEVICE, TRAIN_DIR, TEST_DIR, MODELS_DIR, ACCLOSS_DIR, BASE_CLASSES, TARGET_CLASS_MAPPING
from training_testing_files.data_preprocessing import get_preprocess
from manage_dataset.degrade import ProbabilisticDegradationDataset, DegradedImageTransform
from training_testing_files.train import train
from training_testing_files.test import test

def get_positive_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            else:
                print("Please enter a number greater than 0.")
        except ValueError:
            print("That's not a valid number. Please enter a valid integer.")

models_dict = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit": "vit_base_patch16_224",
    "swin": "swin_base_patch4_window7_224"
}

chose_dict = {
    "1": list(models_dict.keys())[0],
    "2": list(models_dict.keys())[1],
    "3": list(models_dict.keys())[2],
    "4": list(models_dict.keys())[3]
}

chose = ""

while chose not in chose_dict:
    print("Scelta del modello:\n1 - Resnet18\n2 - Resnet50\n3 - Vit\n4 - Swin")
    chose = input("Scelta: ")

selected_model = chose_dict[chose]
model_name = models_dict[selected_model]

batch = get_positive_int("Choose batch_size: ")
workers = get_positive_int("Choose num_workers: ")

model, train_preprocess, test_preprocess = get_preprocess(model_name)

degrader = DegradedImageTransform()

train_dataset = ProbabilisticDegradationDataset(
    root_dir=TRAIN_DIR,
    target_class_mapping=TARGET_CLASS_MAPPING,
    transform=train_preprocess,
    degrader=degrader
)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_preprocess)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=workers)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True, num_workers=workers)

paths = [s[0] for s in test_loader.dataset.samples]
paths = [os.path.basename(p) for p in paths]

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_acc = []
test_acc = []

train_loss = []
test_loss = []

# LOAD PARAMETERS
PATH = MODELS_DIR + f"{selected_model}.pth"
try:
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    print("File caricato")
except FileNotFoundError:
    print("File non trovato")

try:
    open(ACCLOSS_DIR + "accuracy_auroc.json", "r")
except FileNotFoundError:
    data = {
        "resnet18": {"acc": 0.0, "auroc": 0.0},
        "resnet50": {"acc": 0.0, "auroc": 0.0},
        "vit": {"acc": 0.0, "auroc": 0.0},
        "swin": {"acc": 0.0, "auroc": 0.0}
    }

    with open(ACCLOSS_DIR + "accuracy_auroc.json", "w") as file:
        json.dump(data, file, indent=4)

# for i in range(1):
    # Esegui il training
    # train(model, train_dataset, train_loader, criterion, optimizer, 1, selected_model, train_acc, train_loss)
    # Esegui il test
    # test(model, test_dataset, test_loader, criterion, 1, selected_model, test_acc, test_loss, train_acc, train_loss)