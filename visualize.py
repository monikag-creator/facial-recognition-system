"""
utils/visualize.py
──────────────────
Visualization helpers: plot confusion matrix, sample grid, and performance charts.
Requires: matplotlib, seaborn, numpy
"""

import os
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/seaborn not installed. Visualization disabled.")


def plot_sample_grid(dataset_path: str = "dataset/train", max_per_class: int = 5):
    """Display a grid of sample training images per class."""
    if not HAS_PLOT:
        return

    import cv2

    classes = sorted(os.listdir(dataset_path))
    classes = [c for c in classes if os.path.isdir(os.path.join(dataset_path, c))]
    n_classes = len(classes)

    if n_classes == 0:
        print("[WARN] No classes found.")
        return

    fig, axes = plt.subplots(n_classes, max_per_class,
                              figsize=(max_per_class * 2, n_classes * 2.2))
    fig.suptitle("Training Dataset — Sample Faces", fontsize=14, fontweight="bold", y=1.01)

    if n_classes == 1:
        axes = [axes]

    for row, cls in enumerate(classes):
        imgs = [f for f in os.listdir(os.path.join(dataset_path, cls))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))][:max_per_class]

        for col in range(max_per_class):
            ax = axes[row][col] if n_classes > 1 else axes[col]
            ax.axis("off")
            if col < len(imgs):
                img_path = os.path.join(dataset_path, cls, imgs[col])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap="gray")
                    if col == 0:
                        ax.set_ylabel(cls, fontsize=10, rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    out = "results/sample_grid.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] Sample grid saved → {out}")
    plt.show()


def plot_evaluation_results(results_path: str = "results/evaluation_results.json"):
    """Bar chart of per-class accuracy from evaluation JSON."""
    if not HAS_PLOT:
        return

    if not os.path.exists(results_path):
        print(f"[WARN] Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    per_class = data.get("per_class", {})
    names = list(per_class.keys())
    accs  = [per_class[n]["accuracy"] for n in names]

    colors = ["#22c55e" if a >= 80 else "#f59e0b" if a >= 60 else "#ef4444" for a in accs]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Per-Class Recognition Accuracy\n(Overall: {data.get('accuracy', 0)}%)",
                 fontsize=13, fontweight="bold")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, label="80% threshold")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    legend_items = [
        mpatches.Patch(color="#22c55e", label="≥ 80% (Good)"),
        mpatches.Patch(color="#f59e0b", label="60–79% (Fair)"),
        mpatches.Patch(color="#ef4444", label="< 60% (Needs more data)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)
    plt.tight_layout()

    out = "results/accuracy_chart.png"
    plt.savefig(out, dpi=150)
    print(f"[INFO] Accuracy chart saved → {out}")
    plt.show()


def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list):
    """Plot a confusion matrix heatmap."""
    if not HAS_PLOT:
        return

    from sklearn.metrics import confusion_matrix  # optional dependency

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = "results/confusion_matrix.png"
    plt.savefig(out, dpi=150)
    print(f"[INFO] Confusion matrix saved → {out}")
    plt.show()
