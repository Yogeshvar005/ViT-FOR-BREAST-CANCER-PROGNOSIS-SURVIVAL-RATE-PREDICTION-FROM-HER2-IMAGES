import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import csv
import optuna

# ✅ Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Paths
train_dir = "/Users/yogeshvar/Downloads/Breast Cancer/train_data_patch"
test_dir = "/Users/yogeshvar/Downloads/Breast Cancer/test_data_patch"

# ✅ Global variables
class_names = datasets.ImageFolder(train_dir).classes
num_classes = len(class_names)

# ✅ Metrics calculation
def calculate_metrics(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    per_class_metrics = []
    sensitivities, specificities = [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        per_class_metrics.append({
            "class": class_names[i],
            "sensitivity": sensitivity,
            "specificity": specificity
        })

    sensitivity = np.mean(sensitivities)
    specificity = np.mean(specificities)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    gmean = np.sqrt(sensitivity * specificity)

    return cm, sensitivity, specificity, precision, recall, f1, gmean, per_class_metrics

# ✅ Save metrics
def save_metrics_csv(model_name, cm, sensitivity, specificity, precision, recall, f1, gmean, per_class_metrics, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{model_name}_metrics.csv")

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Macro Sensitivity", sensitivity])
        writer.writerow(["Macro Specificity", specificity])
        writer.writerow(["Macro Precision", precision])
        writer.writerow(["Macro Recall", recall])
        writer.writerow(["Macro F1", f1])
        writer.writerow(["Macro G-Mean", gmean])
        writer.writerow([])
        writer.writerow(["Class", "Sensitivity", "Specificity"])
        for m in per_class_metrics:
            writer.writerow([m["class"], m["sensitivity"], m["specificity"]])

    print(f"✅ Metrics saved to {filepath}")

# ✅ Training + Evaluation
def train_and_evaluate(model, criterion, optimizer, train_loader, val_loader, model_name, epochs=30):
    out_dir = f"results/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    best_acc, patience, no_improve = 0, 5, 0
    train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = [], [], [], []

    scaler = torch.amp.GradScaler()


    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="mps"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # ✅ Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.amp.autocast(device_type="mps"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        val_loss /= total
        val_acc = 100. * correct / total

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} → Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc, no_improve = val_acc, 0
            torch.save(model.state_dict(), os.path.join(out_dir, f"{model_name}_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print("⏹️ Early stopping!")
                break

    # ✅ Metrics
    cm, sens, spec, prec, rec, f1, gmean, per_class_metrics = calculate_metrics(all_labels, all_preds, class_names)
    save_metrics_csv(model_name, cm, sens, spec, prec, rec, f1, gmean, per_class_metrics, out_dir)

    # ✅ Confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(out_dir, f"{model_name}_cm.png"))
    plt.close()

    # ✅ Accuracy curve
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend(); plt.title(f"{model_name} Accuracy")
    plt.savefig(os.path.join(out_dir, f"{model_name}_acc.png"))
    plt.close()

    # ✅ Loss curve
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend(); plt.title(f"{model_name} Loss")
    plt.savefig(os.path.join(out_dir, f"{model_name}_loss.png"))
    plt.close()

    return best_acc

# ✅ Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_categorical("lr", [1e-5, 5e-5, 1e-4, 5e-4])
    wd = trial.suggest_categorical("weight_decay", [1e-4, 5e-4, 1e-3])
    drop_path = trial.suggest_categorical("drop_path_rate", [0.05, 0.1, 0.2])
    label_smooth = trial.suggest_categorical("label_smoothing", [0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    # ✅ Data augmentations
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ✅ Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_data = datasets.ImageFolder(test_dir, transform=val_tfms)

    # Weighted sampler for imbalance
    class_counts = np.bincount(train_data.targets)
    weights = 1. / class_counts
    sample_weights = [weights[t] for t in train_data.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # ✅ Model
    model = timm.create_model("deit_base_distilled_patch16_224",
                              pretrained=True,
                              num_classes=num_classes,
                              drop_path_rate=drop_path).to(device)

    # ✅ Criterion & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Train & Eval
    acc = train_and_evaluate(model, criterion, optimizer, train_loader, val_loader,
                             model_name=f"ViT_Optuna_{trial.number}", epochs=30)

    return acc

# ✅ Run Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # increase to 50–100 for thorough search

print("Best trial:", study.best_trial.params)
