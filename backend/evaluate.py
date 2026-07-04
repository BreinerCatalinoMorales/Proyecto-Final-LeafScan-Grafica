import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

from utils import (load_config, get_device, get_section, ensure_dir, save_json)
from cnn_model import load_model, DEFAULT_CNN
from train import build_dataloaders, DEFAULT_TRAINING


def compute_classification_metrics(y_true,
                                    y_pred,
                                    class_names: list,
                                    positive_class: str = "diseased") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]), 4),
            "f1":        round(float(f1[i]), 4),
            "support":   int(support[i]),
        }

    fp = fn = tp = tn = 0
    if positive_class in class_names:
        pos = class_names.index(positive_class)
        tp = int(np.sum((y_pred == pos) & (y_true == pos)))
        fp = int(np.sum((y_pred == pos) & (y_true != pos)))
        fn = int(np.sum((y_pred != pos) & (y_true == pos)))
        tn = int(np.sum((y_pred != pos) & (y_true != pos)))

    return {
        "accuracy":         round(accuracy, 4),
        "macro_precision":  round(float(macro[0]), 4),
        "macro_recall":     round(float(macro[1]), 4),
        "macro_f1":         round(float(macro[2]), 4),
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_samples":        int(len(y_true)),
    }


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.logical_and(a, b).sum()
    union        = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


@torch.no_grad()
def evaluate_cnn(model, loader, device, class_names: list) -> dict:
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    image_paths = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)

        y_true.extend(targets.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    if hasattr(loader.dataset, "samples"):
        image_paths = [s[0] for s in loader.dataset.samples]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    metrics.update({
        "y_true":      y_true,
        "y_pred":      y_pred,
        "y_prob":      y_prob,
        "image_paths": image_paths,
    })
    return metrics


def plot_confusion_matrix(cm, class_names: list,
                          title: str = "Matriz de confusión",
                          save_path: str = None):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Verdadero")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Matriz de confusión guardada en {save_path}")
    return fig


def plot_metrics_per_class(metrics: dict, save_path: str = None):
    per_class  = metrics["per_class"]
    class_names = list(per_class.keys())
    metric_names = ["precision", "recall", "f1"]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    x     = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (met, col) in enumerate(zip(metric_names, colors)):
        vals = [per_class[c][met] for c in class_names]
        bars = ax.bar(x + i * width, vals, width, label=met.capitalize(),
                      color=col, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor")
    ax.set_title(
        f"CNN — Métricas por clase\n"
        f"Accuracy global: {metrics['accuracy']:.3f}  |  "
        f"Macro F1: {metrics['macro_f1']:.3f}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Métricas por clase guardadas en {save_path}")
    return fig


def plot_confidence_distribution(y_true: list, y_pred: list,
                                  y_prob: list, class_names: list,
                                  save_path: str = None):
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_prob  = np.asarray(y_prob)

    confidence = y_prob[np.arange(len(y_pred)), y_pred]
    correct    = y_true == y_pred

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)

    ax.hist(confidence[correct],  bins=bins, alpha=0.7,
            color="#2ecc71", label=f"Correctas ({correct.sum()})")
    ax.hist(confidence[~correct], bins=bins, alpha=0.7,
            color="#e74c3c", label=f"Incorrectas ({(~correct).sum()})")

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Umbral 0.5")
    ax.set_xlabel("Confianza del modelo")
    ax.set_ylabel("Número de imágenes")
    ax.set_title("CNN — Distribución de confianza (aciertos vs errores)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Distribución de confianza guardada en {save_path}")
    return fig


def plot_error_gallery(y_true: list, y_pred: list, y_prob: list,
                       image_paths: list, class_names: list,
                       max_errors: int = 12, save_path: str = None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if len(image_paths) == 0:
        print("[evaluate] No hay rutas de imágenes disponibles para la galería.")
        return None

    confidence = y_prob[np.arange(len(y_pred)), y_pred]
    wrong_idx  = np.where(y_true != y_pred)[0]

    if len(wrong_idx) == 0:
        print("[evaluate] ¡El modelo no cometió errores en test! No se genera galería.")
        return None

    wrong_idx = wrong_idx[np.argsort(confidence[wrong_idx])[::-1]]
    wrong_idx = wrong_idx[:max_errors]

    cols = 4
    rows = int(np.ceil(len(wrong_idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).flatten()

    for ax_i, idx in enumerate(wrong_idx):
        path = image_paths[idx]
        bgr  = cv2.imread(str(path))
        if bgr is None:
            axes[ax_i].axis("off")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        axes[ax_i].imshow(rgb)
        real  = class_names[y_true[idx]]
        pred  = class_names[y_pred[idx]]
        conf  = confidence[idx]
        axes[ax_i].set_title(
            f"Real: {real}\nPred: {pred} ({conf:.0%})",
            fontsize=8,
            color="red"
        )
        axes[ax_i].axis("off")

    for ax_i in range(len(wrong_idx), len(axes)):
        axes[ax_i].axis("off")

    fig.suptitle(
        f"CNN — Errores de clasificación ({len(wrong_idx)} de {len(y_true)} imágenes)",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[evaluate] Galería de errores guardada en {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación de la CNN en test")
    parser.add_argument("--weights", type=str, default="../models/cnn_weights.pth")
    args = parser.parse_args()

    config = load_config()
    device = get_device()

    model, meta = load_model(args.weights, config, device)
    class_names = meta["class_names"]

    ccfg = get_section(config, "cnn", DEFAULT_CNN)
    tcfg = get_section(config, "training", DEFAULT_TRAINING)
    loaders, _, _ = build_dataloaders(config, tcfg, ccfg)

    print(f"[evaluate] Evaluando CNN ({meta['backbone']}) sobre test...")
    metrics = evaluate_cnn(model, loaders["test"], device, class_names)

    print(f"  Accuracy : {metrics['accuracy']:.3f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.3f}")
    print(f"  FP: {metrics['fp']}  FN: {metrics['fn']}")

    out = str(ensure_dir(config["paths"]["outputs_metrics"]))

    summary = {k: v for k, v in metrics.items()
               if k not in ("y_true", "y_pred", "y_prob", "image_paths")}
    save_json(summary, f"{out}/cnn_evaluation.json")

    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        title="CNN — Matriz de confusión",
        save_path=f"{out}/cnn_confusion_matrix.png")

    plot_metrics_per_class(
        metrics,
        save_path=f"{out}/cnn_metrics_per_class.png")

    plot_confidence_distribution(
        metrics["y_true"], metrics["y_pred"],
        metrics["y_prob"], class_names,
        save_path=f"{out}/cnn_confidence_dist.png")

    plot_error_gallery(
        metrics["y_true"], metrics["y_pred"],
        metrics["y_prob"], metrics["image_paths"],
        class_names,
        save_path=f"{out}/cnn_error_gallery.png")

    print(f"\n[evaluate] Todos los resultados en: {out}")