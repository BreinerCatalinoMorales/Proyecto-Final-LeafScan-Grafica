import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def bgr_to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)


def compute_rgb_histograms(image: np.ndarray,
                           bins: int = 64) -> dict:
    img_rgb = bgr_to_rgb(image)
    histograms = {}
    for i, channel in enumerate(["R", "G", "B"]):
        hist, edges = np.histogram(
            img_rgb[:, :, i].ravel(),
            bins=bins,
            range=(0, 256)
        )
        histograms[channel] = {"hist": hist, "edges": edges}
    return histograms


def compute_hsv_histograms(image: np.ndarray,
                           hue_bins: int = 36,
                           sat_bins: int = 32,
                           val_bins: int = 32) -> dict:
    img_hsv = bgr_to_hsv(image)
    ranges = {"H": (0, 181), "S": (0, 256), "V": (0, 256)}
    bin_counts = {"H": hue_bins, "S": sat_bins, "V": val_bins}

    histograms = {}
    for i, channel in enumerate(["H", "S", "V"]):
        hist, edges = np.histogram(
            img_hsv[:, :, i].ravel(),
            bins=bin_counts[channel],
            range=ranges[channel]
        )
        histograms[channel] = {"hist": hist, "edges": edges}
    return histograms


def detect_disease_zones_by_color(image: np.ndarray,
                                   config: dict) -> dict:
    cfg = config["segmentation"]
    img_hsv = bgr_to_hsv(image)

    lower = np.array([cfg["disease_hue_min"],
                      cfg["disease_sat_min"],
                      cfg["disease_val_min"]], dtype=np.uint8)
    upper = np.array([cfg["disease_hue_max"], 255, 255], dtype=np.uint8)

    mask = cv2.inRange(img_hsv, lower, upper)

    overlay = image.copy()
    overlay[mask > 0] = [0, 0, 200]

    total_pixels   = image.shape[0] * image.shape[1]
    disease_pixels = int(np.sum(mask > 0))
    disease_ratio  = round(disease_pixels / total_pixels * 100, 2)

    return {
        "mask": mask,
        "overlay": overlay,
        "disease_ratio": disease_ratio,
        "disease_pixels": disease_pixels,
        "total_pixels": total_pixels,
    }


def compute_color_statistics(image: np.ndarray) -> dict:
    img_rgb = bgr_to_rgb(image)
    img_hsv = bgr_to_hsv(image)

    stats = {}
    for i, ch in enumerate(["R", "G", "B"]):
        stats[f"rgb_{ch}_mean"] = float(np.mean(img_rgb[:, :, i]))
        stats[f"rgb_{ch}_std"]  = float(np.std(img_rgb[:, :, i]))

    for i, ch in enumerate(["H", "S", "V"]):
        stats[f"hsv_{ch}_mean"] = float(np.mean(img_hsv[:, :, i]))
        stats[f"hsv_{ch}_std"]  = float(np.std(img_hsv[:, :, i]))

    return stats


def plot_rgb_histograms(image: np.ndarray,
                        title: str = "Histogramas RGB",
                        save_path: str = None) -> plt.Figure:
    hists = compute_rgb_histograms(image)
    colors = {"R": "red", "G": "green", "B": "blue"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    for ax, (channel, data) in zip(axes, hists.items()):
        centers = (data["edges"][:-1] + data["edges"][1:]) / 2
        ax.bar(centers, data["hist"], width=4,
               color=colors[channel], alpha=0.75, edgecolor="none")
        ax.set_title(f"Canal {channel}")
        ax.set_xlabel("Intensidad (0-255)")
        ax.set_ylabel("Frecuencia")
        ax.set_xlim(0, 255)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[color_analysis] Histograma RGB guardado en {save_path}")
    return fig


def plot_hsv_histograms(image: np.ndarray,
                        title: str = "Histogramas HSV",
                        save_path: str = None) -> plt.Figure:
    cfg_default = {"hue_bins": 36, "sat_bins": 32, "val_bins": 32}
    hists = compute_hsv_histograms(image,
                                   hue_bins=cfg_default["hue_bins"],
                                   sat_bins=cfg_default["sat_bins"],
                                   val_bins=cfg_default["val_bins"])
    channel_colors = {"H": "darkorange", "S": "mediumpurple", "V": "steelblue"}
    xlabels = {"H": "Hue (0-180)", "S": "Saturación (0-255)", "V": "Brillo (0-255)"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    for ax, (channel, data) in zip(axes, hists.items()):
        centers = (data["edges"][:-1] + data["edges"][1:]) / 2
        ax.bar(centers, data["hist"], width=centers[1] - centers[0],
               color=channel_colors[channel], alpha=0.80, edgecolor="none")
        ax.set_title(f"Canal {channel}")
        ax.set_xlabel(xlabels[channel])
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[color_analysis] Histograma HSV guardado en {save_path}")
    return fig


def plot_comparison(original: np.ndarray,
                    overlay: np.ndarray,
                    disease_ratio: float,
                    save_path: str = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(bgr_to_rgb(original))
    ax1.set_title("Imagen original")
    ax1.axis("off")

    ax2.imshow(bgr_to_rgb(overlay))
    ax2.set_title(f"Zona enferma detectada ({disease_ratio:.1f}%)")
    ax2.axis("off")

    plt.suptitle("Análisis de color — detección por HSV", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[color_analysis] Comparación guardada en {save_path}")
    return fig


def analyze_image(image: np.ndarray, config: dict) -> dict:
    cfg_ca = config["color_analysis"]

    rgb_hists  = compute_rgb_histograms(image, bins=cfg_ca["rgb_bins"])
    hsv_hists  = compute_hsv_histograms(image,
                                        hue_bins=cfg_ca["hsv_hue_bins"],
                                        sat_bins=cfg_ca["hsv_sat_bins"],
                                        val_bins=cfg_ca["hsv_val_bins"])
    stats      = compute_color_statistics(image)
    disease    = detect_disease_zones_by_color(image, config)

    return {
        "rgb_histograms":  rgb_hists,
        "hsv_histograms":  hsv_hists,
        "color_stats":     stats,
        "disease_mask":    disease["mask"],
        "disease_overlay": disease["overlay"],
        "disease_ratio":   disease["disease_ratio"],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python color_analysis.py <ruta_imagen>")
        sys.exit(1)

    config = load_config()
    img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, tuple(config["preprocessing"]["image_size"]))

    results = analyze_image(img, config)

    print(f"Porcentaje área enferma: {results['disease_ratio']}%")
    print("Estadísticas de color:")
    for k, v in results["color_stats"].items():
        print(f"  {k}: {v:.2f}")

    plot_rgb_histograms(img, save_path="outputs/metrics/hist_rgb.png")
    plot_hsv_histograms(img, save_path="outputs/metrics/hist_hsv.png")
    plot_comparison(img, results["disease_overlay"],
                    results["disease_ratio"],
                    save_path="outputs/segmented/color_detection.png")
    plt.show()
