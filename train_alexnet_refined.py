# train_alexnet_refined.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, AlexNet_Weights
from torchmetrics.classification import MultilabelF1Score
from sklearn.metrics import classification_report
import pandas as pd

from data_prep import get_data_loaders, NUM_LABELS

# --- config ---
HF_DS        = "alkzar90/NIH-Chest-X-ray-dataset"
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_WORKERS  = 4
NUM_EPOCHS   = 15
PATIENCE     = 3
LR_HEAD      = 3e-4
LR_FEAT      = 3e-5
WEIGHT_DECAY = 1e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = "best_alexnet_refined.pth"
LOG_CSV      = "results/train_logs_refined.csv"

CLASS_NAMES = [
    "No Finding","Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax","Consolidation",
    "Edema","Emphysema","Fibrosis","Pleural Thickening","Hernia"
]

# --- data loaders ---
train_loader, val_loader, test_loader, _ = get_data_loaders(
    HF_DS,
    image_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# --- model & loss ---
model = alexnet(weights=AlexNet_Weights.DEFAULT)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_LABELS)
model.to(DEVICE)

# compute pos_weight
counts, total = torch.zeros(NUM_LABELS), 0
for _, y in train_loader:
    counts += y.sum(dim=0)
    total  += y.size(0)
neg = total - counts
pos_weight = (neg/(counts+1e-6)).clamp(max=10).to(DEVICE)
print("pos_weight:", pos_weight.tolist())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
params = [
    {"params": model.features.parameters(),   "lr": LR_FEAT},
    {"params": model.classifier.parameters(), "lr": LR_HEAD}
]
optimizer = optim.AdamW(params, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[LR_FEAT, LR_HEAD],
    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS
)
f1_metric = MultilabelF1Score(
    num_labels=NUM_LABELS, average="micro", threshold=0.5
).to(DEVICE)

# --- train/validate ---
train_losses, val_losses, val_f1s = [], [], []
best_val_f1, no_imp = 0.0, 0

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    run_loss = 0.0
    for imgs, targs in train_loader:
        imgs, targs = imgs.to(DEVICE), targs.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, targs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        run_loss += loss.item() * imgs.size(0)
    train_loss = run_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    run_val = 0.0
    f1_metric.reset()
    with torch.no_grad():
        for imgs, targs in val_loader:
            imgs, targs = imgs.to(DEVICE), targs.to(DEVICE)
            logits = model(imgs)
            run_val += criterion(logits, targs).item() * imgs.size(0)
            f1_metric.update(torch.sigmoid(logits), targs.int())
    val_loss = run_val / len(val_loader.dataset)
    val_f1   = f1_metric.compute().item()
    val_losses.append(val_loss)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1, no_imp = val_f1, 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✓ Saved new best (Val F1={val_f1:.4f})")
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print("→ Early stopping")
            break

# --- save logs ---
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
pd.DataFrame({
    "epoch":      list(range(1, len(train_losses)+1)),
    "train_loss": train_losses,
    "val_loss":   val_losses,
    "val_f1":     val_f1s
}).to_csv(LOG_CSV, index=False)
print("Saved training logs to", LOG_CSV)

# --- test evaluation ---
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
all_p, all_t = [], []

with torch.no_grad():
    for imgs, targs in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).cpu().int().numpy()
        all_p.append(preds)
        all_t.append(targs.numpy())

all_preds   = np.vstack(all_p)
all_targs   = np.vstack(all_t)

print("\nTest Classification Report:")
print(classification_report(all_targs, all_preds,
                            target_names=CLASS_NAMES,
                            zero_division=0))
