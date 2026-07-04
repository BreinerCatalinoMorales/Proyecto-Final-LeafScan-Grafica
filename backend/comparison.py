import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (load_config, get_device, get_section, ensure_dir, save_json)
from cnn_model import load_model, predict_image, get_cnn_disease_mask, DEFAULT_CNN
from evaluate import compute_classification_metrics, compute_iou
from classical_pipeline import run_classical_pipeline


DEFAULT_COMPARISON = {
    "classical_infection_threshold": 5.0,
    "gradcam_threshold":             0.5,
    "max_images_per_class":          None,
    "save_gradcam_images":           True,
    "gradcam_max_samples":           20,
}


def collect_test_images(config: dict, class_names: list,
                        max_per_class=None) -> list:
    test_dir = Path(config["paths"]["data_splits"]) / "test"
    samples = []
    extensions = {".jpg", ".jpeg", ".png"}

    for idx, name in enumerate(class_names):
        class_dir = test_dir / name
        if not class_dir.exists():
            print(f"[comparison] Aviso: no existe {class_dir}")
            continue
        imgs = sorted([p for p in class_dir.iterdir()
                       if p.suffix.lower() in extensions])
        if max_per_class:
            imgs = imgs[:max_per_class]
        for p in imgs:
            samples.append({"path": str(p), "true_idx": idx, "true_label": name})

    return samples


def save_gradcam_figure(original_path: str,
                         classical_mask: np.ndarray,
                         cnn_cam: np.ndarray,
                         true_label: str,
                         classical_pred: str,
                         cnn_pred: str,
                         iou: float,
                         save_path: str) -> None:
    bgr = cv2.imread(original_path)
    if bgr is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    mask_resized = cv2.resize(classical_mask, (w, h))
    cam_resized  = cv2.resize(cnn_cam, (w, h))

    cam_uint8  = (cam_resized * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay    = cv2.addWeighted(rgb, 0.55, heatmap_rgb, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"Original\nReal: {true_label}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask_resized, cmap="Greens")
    ok_c = "✓" if classical_pred == true_label else "✗"
    axes[1].set_title(
        f"K-means + Morfología (Jose)\nPred: {classical_pred} {ok_c}",
        fontsize=9,
        color="green" if classical_pred == true_label else "red"
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    ok_n = "✓" if cnn_pred == true_label else "✗"
    axes[2].set_title(
        f"Grad-CAM CNN\nPred: {cnn_pred} {ok_n}  |  IoU: {iou:.2f}",
        fontsize=9,
        color="green" if cnn_pred == true_label else "red"
    )
    axes[2].axis("off")

    plt.suptitle(
        f"{Path(original_path).name}",
        fontsize=9, color="gray"
    )
    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_summary(comparison: dict, save_path: str = None):
    s  = comparison["summary"]
    c  = s["classical"]
    n  = s["cnn"]
    cm = comparison["confusion_matrix"]
    class_names = cm["class_names"]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Comparativa: Pipeline Clásico (K-means + Morfología)  vs  CNN (Transfer Learning)",
        fontsize=13, fontweight="bold"
    )

    ax_table = fig.add_axes([0.03, 0.55, 0.94, 0.36])
    ax_table.axis("off")

    metrics_labels = ["Accuracy", "Precision", "Recall", "F1",
                      "Falsos +", "Falsos -", "ms/imagen"]
    classical_vals = [c["accuracy"], c["precision"], c["recall"], c["f1"],
                      c["fp"],       c["fn"],        c["avg_ms"]]
    cnn_vals       = [n["accuracy"], n["precision"], n["recall"], n["f1"],
                      n["fp"],       n["fn"],        n["avg_ms"]]

    table_data = []
    for label, cv, nv in zip(metrics_labels, classical_vals, cnn_vals):
        table_data.append([label, str(cv), str(nv)])

    table_data.append(["IoU medio", str(s["mean_iou"]), "—"])
    table_data.append(["Imágenes test", str(s["n_images"]), "—"])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Métrica", "Clásico (K-means)", f"CNN ({s['backbone']})"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for col in range(3):
        tbl[0, col].set_facecolor("#2c3e50")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    for row_i in range(1, 5):
        cv = float(classical_vals[row_i - 1])
        nv = float(cnn_vals[row_i - 1])
        best_col = 1 if cv >= nv else 2
        tbl[row_i, best_col].set_facecolor("#d5f5e3")

    for col_i, (method, cm_data) in enumerate([
        ("Clásico (K-means + Morfología)", cm["classical"]),
        (f"CNN ({s['backbone']})",          cm["cnn"]),
    ]):
        ax = fig.add_axes([0.08 + col_i * 0.5, 0.04, 0.38, 0.44])
        cm_arr = np.asarray(cm_data)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicho", fontsize=9)
        ax.set_ylabel("Verdadero", fontsize=9)
        ax.set_title(f"Matriz de confusión\n{method}", fontsize=9)
        thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 0.5
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                        fontsize=10,
                        color="white" if cm_arr[i, j] > thresh else "black")

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[comparison] Resumen visual guardado en {save_path}")
    return fig


def run_comparison(config: dict = None,
                   weights_path: str = "../models/cnn_weights.pth") -> dict:
    config = config or load_config()
    cmp_cfg = get_section(config, "comparison", DEFAULT_COMPARISON)
    device = get_device()

    model, meta = load_model(weights_path, config, device)
    class_names  = meta["class_names"]
    diseased_idx = class_names.index("diseased") if "diseased" in class_names else 1
    threshold    = cmp_cfg["classical_infection_threshold"]
    image_size   = tuple(config["preprocessing"]["image_size"])

    samples = collect_test_images(config, class_names,
                                  cmp_cfg["max_images_per_class"])
    if not samples:
        raise RuntimeError("No se encontraron imágenes de test en data/splits/test/")

    print(f"[comparison] Evaluando {len(samples)} imágenes de test...\n")

    save_gradcam  = cmp_cfg["save_gradcam_images"]
    gradcam_limit = cmp_cfg["gradcam_max_samples"]
    gradcam_dir   = ensure_dir(config["paths"]["outputs_gradcam"])
    gradcam_saved = 0

    y_true, y_classical, y_cnn = [], [], []
    classical_times, cnn_times, ious = [], [], []
    per_image = []

    for i, s in enumerate(samples):
        path, true_idx = s["path"], s["true_idx"]

        t0 = time.perf_counter()
        cls_result = run_classical_pipeline(path, config, verbose=False)
        classical_ms = (time.perf_counter() - t0) * 1000.0
        infection    = cls_result["metrics"]["final_infection_ratio"]
        classical_idx = diseased_idx if infection >= threshold else (1 - diseased_idx)

        t0 = time.perf_counter()
        cnn_pred = predict_image(model, path, meta, device=device)
        cnn_ms   = (time.perf_counter() - t0) * 1000.0
        cnn_idx  = cnn_pred["label_idx"]

        classical_mask = cls_result["disease_mask"]
        cnn_mask = get_cnn_disease_mask(model, path, meta,
                                        out_size=image_size,
                                        threshold=cmp_cfg["gradcam_threshold"],
                                        device=device)
        if classical_mask.shape[:2] != cnn_mask.shape[:2]:
            cnn_mask = cv2.resize(cnn_mask, classical_mask.shape[:2][::-1])
        iou = compute_iou(classical_mask, cnn_mask)

        if save_gradcam and gradcam_saved < gradcam_limit:
            from cnn_model import GradCAM, get_target_layer, get_transforms, _to_pil
            import torch
            tf = get_transforms(
                meta.get("input_size", 224),
                meta.get("normalize_mean"),
                meta.get("normalize_std"),
                train=False
            )
            x = tf(_to_pil(path)).unsqueeze(0).to(device)
            backbone = meta.get("backbone", "mobilenet_v2")
            cam_engine = GradCAM(model, get_target_layer(model, backbone))
            try:
                cam_raw = cam_engine(x)
            finally:
                cam_engine.remove()
            cam_resized = cv2.resize(cam_raw, image_size)

            fig_name = f"comp_{Path(path).stem}.png"
            save_gradcam_figure(
                original_path  = path,
                classical_mask = classical_mask,
                cnn_cam        = cam_resized,
                true_label     = class_names[true_idx],
                classical_pred = class_names[classical_idx],
                cnn_pred       = class_names[cnn_idx],
                iou            = iou,
                save_path      = str(gradcam_dir / fig_name),
            )
            gradcam_saved += 1

        y_true.append(true_idx)
        y_classical.append(classical_idx)
        y_cnn.append(cnn_idx)
        classical_times.append(classical_ms)
        cnn_times.append(cnn_ms)
        ious.append(iou)

        per_image.append({
            "image_name":      Path(path).name,
            "true_label":      class_names[true_idx],
            "classical_pred":  class_names[classical_idx],
            "cnn_pred":        class_names[cnn_idx],
            "infection_ratio": round(float(infection), 2),
            "cnn_confidence":  round(cnn_pred["confidence"], 4),
            "iou":             round(iou, 4),
            "classical_ms":    round(classical_ms, 2),
            "cnn_ms":          round(cnn_ms, 2),
            "classical_ok":    classical_idx == true_idx,
            "cnn_ok":          cnn_idx == true_idx,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  Procesadas {i+1}/{len(samples)}")

    m_classical = compute_classification_metrics(y_true, y_classical, class_names)
    m_cnn       = compute_classification_metrics(y_true, y_cnn, class_names)

    classical_fail = [r["image_name"] for r in per_image if not r["classical_ok"]]
    cnn_fail       = [r["image_name"] for r in per_image if not r["cnn_ok"]]
    disagreements  = [r["image_name"] for r in per_image
                      if r["classical_pred"] != r["cnn_pred"]]

    comparison = {
        "summary": {
            "n_images": len(samples),
            "classical": {
                "accuracy":  m_classical["accuracy"],
                "precision": m_classical["macro_precision"],
                "recall":    m_classical["macro_recall"],
                "f1":        m_classical["macro_f1"],
                "fp":        m_classical["fp"],
                "fn":        m_classical["fn"],
                "avg_ms":    round(float(np.mean(classical_times)), 2),
            },
            "cnn": {
                "accuracy":  m_cnn["accuracy"],
                "precision": m_cnn["macro_precision"],
                "recall":    m_cnn["macro_recall"],
                "f1":        m_cnn["macro_f1"],
                "fp":        m_cnn["fp"],
                "fn":        m_cnn["fn"],
                "avg_ms":    round(float(np.mean(cnn_times)), 2),
            },
            "mean_iou": round(float(np.mean(ious)), 4),
            "backbone": meta["backbone"],
        },
        "confusion_matrix": {
            "classical":   m_classical["confusion_matrix"],
            "cnn":         m_cnn["confusion_matrix"],
            "class_names": class_names,
        },
        "errors": {
            "classical_misclassified": classical_fail,
            "cnn_misclassified":       cnn_fail,
            "disagreements":           disagreements,
        },
        "per_image": per_image,
    }
    return comparison


def print_comparison_table(comparison: dict) -> None:
    s = comparison["summary"]
    c, n = s["classical"], s["cnn"]

    print("\n" + "=" * 60)
    print("  COMPARATIVA: MÉTODO CLÁSICO vs CNN")
    print("=" * 60)
    print(f"  Imágenes de test: {s['n_images']}   |   Backbone: {s['backbone']}")
    print("-" * 60)
    print(f"  {'Métrica':<16}{'Clásico':>14}{'CNN':>14}")
    print("-" * 60)
    rows = [
        ("Accuracy",  c["accuracy"],  n["accuracy"]),
        ("Precision", c["precision"], n["precision"]),
        ("Recall",    c["recall"],    n["recall"]),
        ("F1",        c["f1"],        n["f1"]),
        ("Falsos +",  c["fp"],        n["fp"]),
        ("Falsos -",  c["fn"],        n["fn"]),
        ("ms/imagen", c["avg_ms"],    n["avg_ms"]),
    ]
    for name, cv, nv in rows:
        print(f"  {name:<16}{cv:>14}{nv:>14}")
    print("-" * 60)
    print(f"  IoU medio (clásico vs Grad-CAM): {s['mean_iou']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparativa clásico vs CNN")
    parser.add_argument("--weights", type=str, default="../models/cnn_weights.pth")
    args = parser.parse_args()

    config = load_config()
    result = run_comparison(config, weights_path=args.weights)

    print_comparison_table(result)

    out_metrics = str(ensure_dir(config["paths"]["outputs_metrics"]))
    save_json(result, f"{out_metrics}/comparison.json")
    print(f"[comparison] JSON para el frontend en: {out_metrics}/comparison.json")

    plot_comparison_summary(
        result,
        save_path=f"{out_metrics}/comparison_summary.png"
    )