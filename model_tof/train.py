import os
import glob
import torch
import numpy as np

from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset import SweepDataset
from model import ObstacleCNN


JSON_DIR    = r"C:\Users\larap\OneDrive\Priloge\Namizje\FERI\projekt\AIprojekt-SafeSteps\labels"
TOF_FVZ     = 25
WINDOW_SEC  = 1.0
STRIDE_SEC  = 0.25
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3


all_files = glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True)
print(f"Total files: {len(all_files)}")


full_ds = SweepDataset(all_files, WINDOW_SEC, TOF_FVZ, STRIDE_SEC)

labels = np.array([y for _, y in full_ds.samples])

train_idx, val_idx = train_test_split(
    np.arange(len(full_ds.samples)),
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_ds = Subset(full_ds, train_idx)
val_ds   = Subset(full_ds, val_idx)

print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

train_labels = labels[train_idx]
counts = np.bincount(train_labels)

weights_per_class = 1.0 / np.sqrt(counts + 1e-6)
sample_weights = weights_per_class[train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ObstacleCNN(T=int(WINDOW_SEC * TOF_FVZ)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            val_loss += loss.item()
            val_correct += (pred.argmax(1) == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Train loss: {train_loss/len(train_loader):.4f} acc: {train_acc:.3f} | "
          f"Val loss: {val_loss/len(val_loader):.4f} acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved best model (val acc: {val_acc:.3f})")

print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")

LABEL_NAMES = [
    "no_obstacle",
    "obstacle_left",
    "obstacle_right",
    "obstacle_center",
    "obstacle_right_left"
]

all_preds, all_true = [], []

model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_true.extend(y.numpy())

print("\nClassification Report:")
print(classification_report(
    all_true,
    all_preds,
    target_names=LABEL_NAMES,
    labels=list(range(len(LABEL_NAMES))),
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_true, all_preds))