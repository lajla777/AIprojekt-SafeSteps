import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dataset import SweepDataset
from model import ObstacleCNN
from dataset import LABEL_IDX
import json as _json

JSON_DIR = r"C:\Users\larap\OneDrive\Priloge\Namizje\FERI\projekt\AIprojekt-SafeSteps\labels"
TOF_FVZ = 25       
WINDOW_SEC = 1.0      
STRIDE_SEC = 0.25     
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3

LABEL_NAMES = [
    "no_obstacle",
    "obstacle_left",
    "obstacle_right",
    "obstacle_center",
    "obstacle_right_left",
]

all_files = glob.glob(os.path.join(JSON_DIR, "**", "*.json"), recursive=True)
print(f"Total files: {len(all_files)}")

angle_stats = {name: {"mins": [], "maxs": [], "means": []} for name in LABEL_NAMES}
for path in all_files:
    with open(path) as f:
        segs = _json.load(f)
    for seg in segs:
        if not seg.get("angle_distance") or seg.get("label") not in LABEL_IDX:
            continue
        angles = [row[0] for row in seg["angle_distance"] if row[0] == row[0]]  # skip NaN
        if not angles:
            continue
        name = seg["label"]
        angle_stats[name]["mins"].append(min(angles))
        angle_stats[name]["maxs"].append(max(angles))
        angle_stats[name]["means"].append(sum(angles) / len(angles))

print("\nAngle range per class (across all segments):")
print(f"  {'Label':<25} {'min°':>7} {'max°':>7} {'mean°':>7} {'span°':>7}")
for name in LABEL_NAMES:
    s = angle_stats[name]
    if not s["mins"]:
        continue
    mn = np.mean(s["mins"])
    mx = np.mean(s["maxs"])
    avg = np.mean(s["means"])
    print(f"  {name:<25} {mn:>7.1f} {mx:>7.1f} {avg:>7.1f} {mx-mn:>7.1f}")
print()

full_ds = SweepDataset(
    all_files,
    window_sec=WINDOW_SEC,
    tof_fvz=TOF_FVZ,
    stride_sec=STRIDE_SEC,
)

labels = np.array([y for _, y in full_ds.samples])

train_idx, val_idx = train_test_split(
    np.arange(len(full_ds.samples)),
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

train_ds = Subset(full_ds, train_idx)
val_ds = Subset(full_ds, val_idx)

train_labels = labels[train_idx]
val_labels   = labels[val_idx]

print("\nSample counts per label:")
for i, name in enumerate(LABEL_NAMES):
    tr = int(np.sum(train_labels == i))
    va = int(np.sum(val_labels   == i))
    print(f"  {name:<25}  train: {tr:>4}  val: {va:>4}  total: {tr+va:>4}")
print(f"  {'TOTAL':<25}  train: {len(train_ds):>4}  val: {len(val_ds):>4}  total: {len(full_ds):>4}")
print()


counts = np.bincount(train_labels)
weights_per_class = 1.0 / np.sqrt(counts + 1e-6)
sample_weights = weights_per_class[train_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ObstacleCNN(num_classes=len(LABEL_NAMES)).to(device)


train_counts = np.bincount(train_labels, minlength=len(LABEL_NAMES)).astype(np.float32)
class_weights = 1.0 / np.sqrt(train_counts + 1e-6)
class_weights /= class_weights.sum()        
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights for loss:")
for name, w in zip(LABEL_NAMES, class_weights):
    print(f"  {name:<25} {w:.4f}")
print()

criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)


best_val_acc = 0.0

N = len(LABEL_NAMES)

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    label_correct = np.zeros(N, dtype=int)
    label_total   = np.zeros(N, dtype=int)

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss+= loss.item()
        preds_idx = pred.argmax(1)
        correct += (preds_idx == y).sum().item()
        total += y.size(0)
        for c in range(N):
            mask = (y == c)
            label_correct[c] += (preds_idx[mask] == c).sum().item()
            label_total[c] += mask.sum().item()

    train_acc = correct / total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    val_label_correct = np.zeros(N, dtype=int)
    val_label_total = np.zeros(N, dtype=int)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item()
            preds_idx = pred.argmax(1)
            val_correct += (preds_idx == y).sum().item()
            val_total += y.size(0)
            for c in range(N):
                mask = (y == c)
                val_label_correct[c] += (preds_idx[mask] == c).sum().item()
                val_label_total[c] += mask.sum().item()

    val_acc = val_correct / val_total

    print(
        f"\nEpoch {epoch+1:3d}/{EPOCHS} | "
        f"Train loss: {train_loss/len(train_loader):.4f}  acc: {train_acc:.3f} | "
        f"Val loss: {val_loss/len(val_loader):.4f}  acc: {val_acc:.3f}"
    )
    print(f"  {'Label':<25} {'Tr.ok/tot':>12} {'Tr.acc':>7}  {'Val.ok/tot':>12} {'Val.acc':>7}")
    for c, name in enumerate(LABEL_NAMES):
        tr_acc  = label_correct[c] / label_total[c]     if label_total[c]     else float("nan")
        val_acc_c = val_label_correct[c] / val_label_total[c] if val_label_total[c] else float("nan")
        print(
            f"  {name:<25} "
            f"{label_correct[c]:>5}/{label_total[c]:<5}  {tr_acc:>6.1%}  "
            f"{val_label_correct[c]:>5}/{val_label_total[c]:<5}  {val_acc_c:>6.1%}"
        )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f" Saved best model (val acc: {val_acc:.3f})")

    scheduler.step(val_acc)
    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr < LR:
        print(f" LR reduced to {current_lr:.2e}")

print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")

all_preds, all_true = [], []
model.eval()

with torch.no_grad():
    for x, y in val_loader:
        x    = x.to(device)
        preds = model(x).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y.numpy())

print("\nClassification Report:")
print(classification_report(
    all_true, all_preds,
    target_names=LABEL_NAMES,
    labels=list(range(len(LABEL_NAMES))),
    zero_division=0,
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_true, all_preds))