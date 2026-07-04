import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import (load_config, set_seed, get_device, get_section,
                   ensure_dir, save_json, count_parameters)
from cnn_model import build_model, get_transforms, DEFAULT_CNN


DEFAULT_TRAINING = {
    "epochs":                  15,
    "batch_size":              32,
    "lr":                      1e-3,
    "weight_decay":            1e-4,
    "num_workers":             2,
    "early_stopping_patience": 5,
    "use_class_weights":       True,
}


def build_dataloaders(config: dict, tcfg: dict, ccfg: dict):
    splits_dir = Path(config["paths"]["data_splits"])
    input_size = ccfg["input_size"]
    mean, std  = ccfg["normalize_mean"], ccfg["normalize_std"]

    tf_train = get_transforms(input_size, mean, std, train=True)
    tf_eval  = get_transforms(input_size, mean, std, train=False)

    datasets = {}
    for split, tf in [("train", tf_train), ("val", tf_eval), ("test", tf_eval)]:
        path = splits_dir / split
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta '{path}'. "
                f"Verifica la estructura data/splits/{split}/<clase>/*.jpg"
            )
        datasets[split] = ImageFolder(str(path), transform=tf)

    class_names = datasets["train"].classes

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=tcfg["batch_size"],
                            shuffle=True, num_workers=tcfg["num_workers"]),
        "val":   DataLoader(datasets["val"], batch_size=tcfg["batch_size"],
                            shuffle=False, num_workers=tcfg["num_workers"]),
        "test":  DataLoader(datasets["test"], batch_size=tcfg["batch_size"],
                            shuffle=False, num_workers=tcfg["num_workers"]),
    }

    counts = np.bincount(datasets["train"].targets, minlength=len(class_names))
    return loaders, class_names, counts


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def train(config: dict = None, overrides: dict = None):
    config = config or load_config()
    overrides = overrides or {}

    seed = config.get("dataset", {}).get("random_seed", 42)
    set_seed(seed)

    ccfg = get_section(config, "cnn", DEFAULT_CNN)
    tcfg = get_section(config, "training", DEFAULT_TRAINING)
    tcfg.update({k: v for k, v in overrides.items() if v is not None})
    if overrides.get("backbone"):
        ccfg["backbone"] = overrides["backbone"]

    device = get_device()
    print(f"[train] Dispositivo: {device}")

    loaders, class_names, counts = build_dataloaders(config, tcfg, ccfg)
    ccfg["class_names"] = class_names
    print(f"[train] Clases: {class_names}  | imágenes por clase: {counts.tolist()}")

    model = build_model(
        backbone=ccfg["backbone"],
        num_classes=len(class_names),
        pretrained=ccfg["pretrained"],
        freeze_backbone=ccfg["freeze_backbone"],
        dropout=ccfg["dropout"],
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"[train] {ccfg['backbone']} — params totales: {total:,} | "
          f"entrenables: {trainable:,}")

    if tcfg["use_class_weights"] and counts.sum() > 0:
        weights = counts.sum() / (len(counts) * np.maximum(counts, 1))
        weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_t)
    else:
        criterion = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=tcfg["lr"],
                                 weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, patience = 0.0, 0
    weights_path = Path("models/cnn_weights.pth")
    ensure_dir(weights_path.parent)

    for epoch in range(1, tcfg["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion,
                                    optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion,
                                    optimizer, device, train=False)
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Época {epoch:02d}/{tcfg['epochs']} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc, patience = va_acc, 0
            checkpoint = {
                "model_state":    model.state_dict(),
                "backbone":       ccfg["backbone"],
                "num_classes":    len(class_names),
                "class_names":    class_names,
                "input_size":     ccfg["input_size"],
                "normalize_mean": ccfg["normalize_mean"],
                "normalize_std":  ccfg["normalize_std"],
                "best_val_acc":   best_val_acc,
                "history":        history,
            }
            torch.save(checkpoint, weights_path)
            print(f"    ✓ Mejor modelo guardado (val acc {best_val_acc:.3f})")
        else:
            patience += 1
            if patience >= tcfg["early_stopping_patience"]:
                print(f"  Early stopping en la época {epoch} "
                      f"(sin mejora en {patience} épocas).")
                break

    out_metrics = ensure_dir(config["paths"]["outputs_metrics"])
    save_json(history, str(Path(out_metrics) / "training_history.json"))
    plot_history(history, str(Path(out_metrics) / "training_curves.png"))

    meta = {
        "backbone":       ccfg["backbone"],
        "class_names":    class_names,
        "input_size":     ccfg["input_size"],
        "normalize_mean": ccfg["normalize_mean"],
        "normalize_std":  ccfg["normalize_std"],
        "device":         str(device),
    }
    print(f"\n[train] Entrenamiento terminado. Mejor val acc: {best_val_acc:.3f}")
    print(f"[train] Pesos en: {weights_path}")
    return model, meta, history


def plot_history(history: dict, save_path: str = None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], "-o", label="Train", markersize=3)
    ax1.plot(epochs, history["val_loss"], "-o", label="Validación", markersize=3)
    ax1.set_title("Pérdida (loss)")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "-o", label="Train", markersize=3)
    ax2.plot(epochs, history["val_acc"], "-o", label="Validación", markersize=3)
    ax2.set_title("Exactitud (accuracy)")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Entrenamiento con transfer learning", fontsize=12)
    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[train] Curvas guardadas en {save_path}")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento CNN (transfer learning)")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--backbone",   type=str,   default=None,
                        choices=["mobilenet_v2", "resnet18"])
    args = parser.parse_args()

    cfg = load_config()
    train(cfg, overrides=vars(args))