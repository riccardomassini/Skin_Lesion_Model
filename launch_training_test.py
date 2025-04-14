from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

from training_testing_files.config import DEVICE, TRAIN_DIR, TEST_DIR, LAST_MODELS_DIR, METRICS_DIR, TARGET_CLASS_MAPPING, EPOCHS
from training_testing_files.model_utils import get_user_inputs, initialize_model_and_metrics, load_checkpoint, save_checkpoint
from manage_dataset.degrade import ProbabilisticDegradationDataset, DegradedImageTransform
from training_testing_files.data_preprocessing import get_preprocess
from training_testing_files.train import train
from training_testing_files.test import test


models_dict = {
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit": "vit_base_patch16_224",
    "swin": "swin_base_patch4_window7_224"
}

selected_model, model_name, batch, workers = get_user_inputs(models_dict)
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

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True)

train_acc, train_loss, test_acc, test_loss, epoch_completed = load_checkpoint(model, optimizer, scheduler, LAST_MODELS_DIR + f"{selected_model}_checkpoint.pth")
remaining_epochs = EPOCHS - epoch_completed

initialize_model_and_metrics(model, selected_model, LAST_MODELS_DIR, METRICS_DIR, DEVICE)

for epoch in range(epoch_completed, EPOCHS):
    train(model, train_dataset, train_loader, criterion, optimizer, selected_model, train_acc, train_loss)
    epoch_loss = test(model, test_dataset, test_loader, criterion, selected_model, test_acc, test_loss, train_acc, train_loss)
    scheduler.step(epoch_loss)

    for param_group in optimizer.param_groups:
        print(f"📉 Learning rate corrente: {param_group['lr']}")

    save_checkpoint(model, optimizer, scheduler, train_acc, train_loss, test_acc, test_loss, epoch + 1, LAST_MODELS_DIR + f"{selected_model}_checkpoint.pth")