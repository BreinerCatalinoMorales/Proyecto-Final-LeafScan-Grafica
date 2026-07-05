from __future__ import annotations

import cv2
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def segment_kmeans_hsv(image: np.ndarray,
                       k: int = 3,
                       max_iter: int = 100,
                       attempts: int = 5) -> dict:
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, w    = img_hsv.shape[:2]
    pixels  = img_hsv.reshape(-1, 3).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS
    )

    labels  = labels.flatten().reshape(h, w)
    centers = centers.astype(np.uint8)

    segmented_hsv = centers[labels]
    segmented_img = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2BGR)

    cluster_masks = [
        (labels == i).astype(np.uint8) * 255
        for i in range(k)
    ]

    return {
        "labels":        labels,
        "centers":       centers,
        "segmented_img": segmented_img,
        "cluster_masks": cluster_masks,
    }


def identify_disease_cluster(centers: np.ndarray) -> int:
    disease_hue_center = 25.0
    scores = []

    for center in centers:
        hue, sat, val = float(center[0]), float(center[1]), float(center[2])
        hue_distance = abs(hue - disease_hue_center)
        score = sat - hue_distance * 2
        scores.append(score)

    return int(np.argmax(scores))


def has_distinct_healthy_cluster(centers: np.ndarray,
                                  disease_idx: int,
                                  healthy_hue_min: int = 36,
                                  healthy_hue_max: int = 95,
                                  min_sat: int = 40) -> bool:
    for i, center in enumerate(centers):
        if i == disease_idx:
            continue
        hue, sat = float(center[0]), float(center[1])
        if healthy_hue_min <= hue <= healthy_hue_max and sat >= min_sat:
            return True
    return False


def compute_cluster_metrics(labels: np.ndarray,
                             centers: np.ndarray,
                             disease_cluster_idx: int) -> dict:
    h, w         = labels.shape
    total_pixels = h * w

    metrics = {
        "total_pixels": total_pixels,
        "clusters": [],
        "disease_cluster_idx": disease_cluster_idx,
        "disease_ratio": 0.0,
    }

    for i, center in enumerate(centers):
        count = int(np.sum(labels == i))
        ratio = round(count / total_pixels * 100, 2)
        metrics["clusters"].append({
            "id": i,
            "pixel_count": count,
            "percentage": ratio,
            "hsv_center": center.tolist(),
            "is_disease": i == disease_cluster_idx,
        })
        if i == disease_cluster_idx:
            metrics["disease_ratio"] = ratio

    return metrics


def segment_by_threshold(image: np.ndarray, config: dict) -> dict:
    cfg = config["segmentation"]
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([cfg["disease_hue_min"],
                      cfg["disease_sat_min"],
                      cfg["disease_val_min"]], dtype=np.uint8)
    upper = np.array([cfg["disease_hue_max"], 255, 255], dtype=np.uint8)

    mask    = cv2.inRange(img_hsv, lower, upper)
    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 220]

    total   = image.shape[0] * image.shape[1]
    disease = int(np.sum(mask > 0))

    return {
        "mask": mask,
        "overlay": overlay,
        "disease_ratio": round(disease / total * 100, 2),
    }


def plot_segmentation_results(original: np.ndarray,
                               seg_result: dict,
                               metrics: dict,
                               save_path: str = None) -> plt.Figure:
    import matplotlib.pyplot as plt

    disease_idx  = metrics["disease_cluster_idx"]
    disease_mask = seg_result["cluster_masks"][disease_idx]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    seg_rgb = cv2.cvtColor(seg_result["segmented_img"], cv2.COLOR_BGR2RGB)
    axes[1].imshow(seg_rgb)
    axes[1].set_title(f"K-means segmentado (k={len(seg_result['centers'])})")
    axes[1].axis("off")

    axes[2].imshow(disease_mask, cmap="Reds")
    axes[2].set_title(
        f"Zona enferma — cluster {disease_idx}\n"
        f"Área: {metrics['disease_ratio']:.1f}%"
    )
    axes[2].axis("off")

    plt.suptitle("Segmentación K-means sin supervisión (espacio HSV)", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[segmentation] Resultado guardado en {save_path}")
    return fig


def plot_cluster_distribution(metrics: dict,
                               save_path: str = None) -> plt.Figure:
    import matplotlib.pyplot as plt

    clusters    = metrics["clusters"]
    labels_bar  = [f"Cluster {c['id']}" for c in clusters]
    percentages = [c["percentage"] for c in clusters]
    colors_bar  = [
        "#e74c3c" if c["is_disease"] else "#27ae60"
        for c in clusters
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels_bar, percentages, color=colors_bar, edgecolor="none")
    ax.set_ylabel("% de píxeles")
    ax.set_title("Distribución de píxeles por cluster")
    ax.set_ylim(0, 100)

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{pct:.1f}%",
                ha="center", fontsize=10)

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="Enfermedad"),
            plt.Rectangle((0, 0), 1, 1, color="#27ae60", label="Sano/fondo"),
        ],
        loc="upper right"
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[segmentation] Distribución guardada en {save_path}")
    return fig


def run_segmentation(image: np.ndarray, config: dict) -> dict:
    cfg = config["segmentation"]

    seg = segment_kmeans_hsv(
        image,
        k=cfg["kmeans_k"],
        max_iter=cfg["kmeans_max_iter"],
        attempts=cfg["kmeans_attempts"]
    )

    disease_idx = identify_disease_cluster(seg["centers"])
    metrics     = compute_cluster_metrics(seg["labels"],
                                          seg["centers"],
                                          disease_idx)
    metrics["has_healthy_reference"] = has_distinct_healthy_cluster(
        seg["centers"], disease_idx
    )

    return {**seg, "metrics": metrics}


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Uso: python segmentation.py <ruta_imagen>")
        sys.exit(1)

    config = load_config()
    img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, tuple(config["preprocessing"]["image_size"]))

    result = run_segmentation(img, config)
    metrics = result["metrics"]

    print(f"\nSegmentación K-means (k={config['segmentation']['kmeans_k']})")
    print(f"Cluster de enfermedad: {metrics['disease_cluster_idx']}")
    print(f"Área enferma estimada: {metrics['disease_ratio']:.2f}%")
    print("\nDetalle por cluster:")
    for c in metrics["clusters"]:
        tag = " ← ENFERMEDAD" if c["is_disease"] else ""
        print(f"  Cluster {c['id']}: {c['percentage']:.1f}% | "
              f"HSV centro: {c['hsv_center']}{tag}")

    plot_segmentation_results(img, result, metrics,
                               save_path="outputs/segmented/kmeans_result.png")
    plot_cluster_distribution(metrics,
                               save_path="outputs/metrics/cluster_dist.png")
    plt.show()
