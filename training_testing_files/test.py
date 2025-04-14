from config import DEVICE, MATRIX_DIR, BEST_MODELS_DIR, METRICS_DIR
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import json


def test(model, test_dataset, test_loader, criterion, selected_model, test_acc, test_loss, train_acc, train_loss):
    model.eval()

    # wrong_preds_dir = "MODEL_AND_PARAMETERS_V4/wrong_predictions/"
    # os.makedirs(wrong_preds_dir, exist_ok=True)
    # img_counter = 0

    def denormalize(tensor, mean, std):
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        return tensor * std + mean


    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            '''for j in range(inputs.size(0)):
                true_label = labels[j].item()
                pred_label = predicted[j].item()
                if true_label != pred_label:
                    img_tensor = inputs[j].unsqueeze(0).cpu()
                    img_denorm = denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img = img_denorm.squeeze(0).permute(1, 2, 0).numpy()

                    img = np.clip(img, 0, 1)
                    img_name = os.path.basename(paths[img_counter])
                    title = f"True: {test_dataset.classes[true_label]}, Pred: {test_dataset.classes[pred_label]}"
                    save_path = os.path.join(wrong_preds_dir, f"{img_name}")

                    if not os.path.exists(save_path):
                        plt.figure()
                        plt.imshow(img)
                        plt.title(title)
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(wrong_preds_dir, f"{img_name}"), dpi=400)
                        plt.close()

                img_counter += 1'''

    epoch_loss = running_loss / len(test_loader)
    test_loss.append(epoch_loss)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    test_acc.append(accuracy)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    try:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auroc = 0.0

    with open(METRICS_DIR + "accuracy_auroc.json", "r") as file:
        data = json.load(file)

    cm = confusion_matrix(all_labels, all_preds)

    old_acc_value = data[selected_model]["acc"]

    if accuracy > old_acc_value:
        data[selected_model]["acc"] = accuracy
        data[selected_model]["auroc"] = auroc

        with open(METRICS_DIR + "accuracy_auroc.json", "w") as file:
            json.dump(data, file, indent=4)

        best_model_path = BEST_MODELS_DIR + f"{selected_model}.pth"
        torch.save(model.state_dict(), best_model_path)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Test {selected_model}')
        plt.tight_layout()
        plt.savefig(MATRIX_DIR + f"{selected_model}_test_confusion_matrix.png", dpi=300)
        plt.close()

    print(f"Test Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, AUROC: {auroc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    epochs = list(range(1, len(train_acc) + 1))

    ax1.plot(epochs, train_acc, marker='o', linestyle='-', color='b', label='Train')
    ax1.plot(epochs, test_acc, marker='o', linestyle='-', color='r', label='Test')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy graph')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_loss, marker='o', linestyle='-', color='b', label='Train')
    ax2.plot(epochs, test_loss, marker='o', linestyle='-', color='r', label='Test')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss graph')
    ax2.legend()
    ax2.grid(True)
    fig.suptitle(f"{selected_model} - accuracy and loss", fontsize=16)
    plt.tight_layout()
    plt.savefig(METRICS_DIR + f"{selected_model}_accuracy_loss.png", dpi=300)
    plt.close()

    print("Finished Testing")

    return epoch_loss