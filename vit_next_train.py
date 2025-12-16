import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import timm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import csv

# ✅ Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Paths
train_dir = "/Users/yogeshvar/Downloads/Breast Cancer/train_data_patch"
test_dir = "/Users/yogeshvar/Downloads/Breast Cancer/test_data_patch"

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
class_names = train_data.classes
num_classes = len(class_names)
print("Classes:", class_names)

# ✅ Weighted sampling (handle imbalance)
class_counts = np.bincount(train_data.targets)
weights = 1. / class_counts
sample_weights = [weights[t] for t in train_data.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

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

# ✅ Training loop
def train_and_evaluate(model, model_name):
    out_dir = f"results/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_acc, patience, no_improve = 0, 5, 0
    train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = [], [], [], []

    for epoch in range(20):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

        print(f"Epoch {epoch+1}/20 → Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
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

# ✅ Models
models_dict = {
    "CNN": nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128*28*28, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    ),
    "ResNet18": models.resnet18(weights="IMAGENET1K_V1"),
    "ViT": timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
}

# ✅ Modify ResNet18 for our dataset
models_dict["ResNet18"].fc = nn.Linear(models_dict["ResNet18"].fc.in_features, num_classes)

# ✅ Run training for each model
results = {}
for name, model in models_dict.items():
    print(f"\n🚀 Training {name}...")
    model = model.to(device)
    results[name] = train_and_evaluate(model, name)

print("\n✅ Training finished for all models!")
print(results)
