"""训练曲线与混淆矩阵的绘制。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境也能出图

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, output_path, title: str = "Training") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h = history.history if hasattr(history, "history") else history
    epochs = np.arange(1, len(h["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, h["accuracy"], label="Train", linewidth=2)
    axes[0].plot(epochs, h["val_accuracy"], label="Val", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, h["loss"], label="Train", linewidth=2)
    axes[1].plot(epochs, h["val_loss"], label="Val", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(y_true, y_pred, class_names, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (INT8 quantized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() * 0.55 if cm.max() else 0
    for row in range(n):
        for col in range(n):
            color = "white" if cm[row, col] > threshold else "black"
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path
