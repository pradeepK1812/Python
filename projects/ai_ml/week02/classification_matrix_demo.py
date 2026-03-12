"""
classification_metrics_engineering_demo.py

Engineering-style evaluation of a binary classifier.

Works in headless environments (like GitHub Codespaces)
by saving plots instead of displaying them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

# ------------------------------------------------
# 0. Create plots directory
# ------------------------------------------------

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------------------------------------
# 1. Generate synthetic classification dataset
# ------------------------------------------------

X, y = make_classification(
    n_samples=1000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)

# ------------------------------------------------
# 2. Train / Test split
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ------------------------------------------------
# 3. Train classifier
# ------------------------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------------------------
# 4. Predictions
# ------------------------------------------------

y_pred = model.predict(X_test)

# Probabilities (needed for curves)
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------
# 5. Basic metrics
# ------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluation Metrics\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ------------------------------------------------
# 6. Confusion Matrix
# ------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

conf_matrix_path = os.path.join(PLOT_DIR, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
plt.close()

print("Saved:", conf_matrix_path)

# ------------------------------------------------
# 7. ROC Curve
# ------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

roc_path = os.path.join(PLOT_DIR, "roc_curve.png")
plt.savefig(roc_path)
plt.close()

print("Saved:", roc_path)

# ------------------------------------------------
# 8. Precision-Recall Curve
# ------------------------------------------------

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

pr_path = os.path.join(PLOT_DIR, "precision_recall_curve.png")
plt.savefig(pr_path)
plt.close()

print("Saved:", pr_path)

# ------------------------------------------------
# 9. Decision Threshold Tuning
# ------------------------------------------------

threshold = 0.3
y_pred_custom = (y_prob > threshold).astype(int)

print("\nMetrics with custom threshold =", threshold)

print("Accuracy :", accuracy_score(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall   :", recall_score(y_test, y_pred_custom))
print("F1 Score :", f1_score(y_test, y_pred_custom))

print("\nPlots saved in:", PLOT_DIR)
