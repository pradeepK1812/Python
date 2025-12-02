import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)


def demo_ml_metrics(test_size=0.3, seed=42):
    # 1. Generate synthetic binary classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=seed,
    )

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 3. Train logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 4. Predictions (class + probability)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1

    # 5. Scalar metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n----- ML METRIC RESULTS -----")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Curves: ROC and Precision–Recall
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

    # 7. Plot: Confusion Matrix, ROC, PR curve
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # (a) Confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) ROC curve
    ax = axes[1]
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Precision–Recall curve
    ax = axes[2]
    ax.plot(rec_curve, prec_curve, linewidth=2)
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = "ml_metrics_plots.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Plots saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    demo_ml_metrics()
