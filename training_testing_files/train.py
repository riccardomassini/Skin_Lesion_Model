from config import DEVICE, MATRIX_DIR
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train(model, train_dataset, train_loader, criterion, optimizer, selected_model, train_acc, train_loss):
    model.train()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    train_loss.append(epoch_loss)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    train_acc.append(accuracy)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auroc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, AUROC: {auroc:.4f}")

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Train {selected_model}')
    plt.tight_layout()
    plt.savefig(MATRIX_DIR + f"{selected_model}_train_confusion_matrix.png", dpi=300)
    plt.close()

    print("Finished Training")