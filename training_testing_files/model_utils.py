import os
import json
import torch


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


def get_user_inputs(models_dict):
    chose_dict = {
        "1": list(models_dict.keys())[0],
        "2": list(models_dict.keys())[1],
        "3": list(models_dict.keys())[2],
        "4": list(models_dict.keys())[3]
    }

    chose = ""
    while chose not in chose_dict:
        print("Choose a model:\n1 - Resnet18\n2 - Resnet50\n3 - Vit\n4 - Swin")
        chose = input("Model: ")

    selected_model = chose_dict[chose]
    model_name = models_dict[selected_model]

    batch = get_positive_int("Choose batch_size: ")
    workers = get_positive_int("Choose num_workers: ")

    return selected_model, model_name, batch, workers


def initialize_model_and_metrics(model, selected_model, last_models_dir, accloss_dir, device):
    metrics_path = os.path.join(accloss_dir, "accuracy_auroc.json")

    if not os.path.exists(metrics_path):
        data = {
            "resnet18": {"acc": 0.0, "auroc": 0.0},
            "resnet50": {"acc": 0.0, "auroc": 0.0},
            "vit": {"acc": 0.0, "auroc": 0.0},
            "swin": {"acc": 0.0, "auroc": 0.0}
        }
        with open(metrics_path, "w") as file:
            json.dump(data, file, indent=4)


def save_checkpoint(model, optimizer, scheduler, train_acc, train_loss, test_acc, test_loss, epoch, filename):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_acc": train_acc,
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "epoch": epoch
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, scheduler, filename):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        train_acc = checkpoint["train_acc"]
        train_loss = checkpoint["train_loss"]
        test_acc = checkpoint["test_acc"]
        test_loss = checkpoint["test_loss"]
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {filename}")
        return train_acc, train_loss, test_acc, test_loss, epoch
    else:
        print(f"No checkpoint found at {filename}")
        return [], [], [], [], 0