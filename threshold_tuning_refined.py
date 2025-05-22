# threshold_tuning_refined.py

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report
)
from data_prep import get_data_loaders   # your original loader
from train_alexnet_refined import CLASS_NAMES  # ensure these match your training script

# ── CONFIG ──
HF_DS_NAME = "alkzar90/NIH-Chest-X-ray-dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_WORKERS = 4
MODEL_PATH  = "best_alexnet_refined.pth"
OUTPUT_DIR  = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA & UNPACK NUM_LABELS ──
train_loader, val_loader, test_loader, NUM_LABELS = get_data_loaders(
    HF_DS_NAME,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS
)

# ── BUILD & LOAD MODEL ──
model = alexnet(weights=AlexNet_Weights.DEFAULT)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_LABELS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# ── COLLECT VALIDATION PROBS & TRUTHS ──
val_probs, val_true = [], []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        val_probs.append(torch.sigmoid(logits).cpu().numpy())
        val_true.append(targets.numpy())

val_probs = np.vstack(val_probs)
val_true  = np.vstack(val_true)

# ── FIND BEST THRESHOLD PER CLASS ──
best_thresholds = []
for i in range(NUM_LABELS):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 81):
        preds = (val_probs[:, i] >= t).astype(int)
        f1 = f1_score(val_true[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    best_thresholds.append(best_t)

print("Per‑class optimal thresholds:", [round(t,2) for t in best_thresholds])

# ── SAVE VAL PROBS / TRUTHS / THRESHOLDS ──
np.save(os.path.join(OUTPUT_DIR, "val_probs_refined.npy"), val_probs)
np.save(os.path.join(OUTPUT_DIR, "val_true_refined.npy"),  val_true)
with open(os.path.join(OUTPUT_DIR, "refined_thresholds.json"), "w") as f:
    json.dump(best_thresholds, f)

# ── EVALUATE ON TEST SET ──
test_probs, test_true = [], []
with torch.no_grad():
    for imgs, targets in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        test_probs.append(torch.sigmoid(logits).cpu().numpy())
        test_true.append(targets.numpy())

test_probs = np.vstack(test_probs)
test_true  = np.vstack(test_true)

# apply per‑class thresholds
test_pred = np.zeros_like(test_probs, dtype=int)
for i, t in enumerate(best_thresholds):
    test_pred[:, i] = (test_probs[:, i] >= t).astype(int)

# ── PRINT & SAVE METRICS ──
print("\nFinal Test Classification Report (refined):")
print(classification_report(
    test_true, test_pred,
    target_names=CLASS_NAMES,
    zero_division=0
))

# compute per-class precision/recall/f1/support
prec, rec, f1, sup = precision_recall_fscore_support(
    test_true, test_pred, zero_division=0
)
# compute per-class AUC when possible
auc_list = []
for i in range(NUM_LABELS):
    # if both classes present
    if np.unique(test_true[:, i]).size == 2:
        auc_list.append(roc_auc_score(test_true[:, i], test_probs[:, i]))
    else:
        auc_list.append(np.nan)

df = pd.DataFrame({
    "Class": CLASS_NAMES,
    "Precision": prec,
    "Recall":    rec,
    "F1-Score":  f1,
    "Support":   sup,
    "AUC":       auc_list
})
df.to_csv(os.path.join(OUTPUT_DIR, "alexnet_refined_metrics.csv"), index=False)
print(f"\nSaved metrics to {OUTPUT_DIR}/alexnet_refined_metrics.csv")
