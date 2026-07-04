import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_kernel(size: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    return cv2.getStructuringElement(shape, (size, size))


def apply_erosion(mask: np.ndarray,
                  kernel_size: int = 5,
                  iterations: int = 2) -> np.ndarray:
    kernel = get_kernel(kernel_size)
    return cv2.erode(mask, kernel, iterations=iterations)


def apply_dilation(mask: np.ndarray,
                   kernel_size: int = 5,
                   iterations: int = 2) -> np.ndarray:
    kernel = get_kernel(kernel_size)
    return cv2.dilate(mask, kernel, iterations=iterations)


def apply_opening(mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def apply_closing(mask: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def apply_full_morphology(mask: np.ndarray, config: dict) -> dict:
    cfg = config["morphology"]

    opened  = apply_opening(mask,  kernel_size=cfg["opening_kernel"])
    closed  = apply_closing(opened, kernel_size=cfg["closing_kernel"])
    eroded  = apply_erosion(closed,
                            kernel_size=cfg["kernel_size"],
                            iterations=cfg["erosion_iterations"])
    dilated = apply_dilation(eroded,
                             kernel_size=cfg["kernel_size"],
                             iterations=cfg["dilation_iterations"])

    return {
        "raw_mask": mask,
        "opened":   opened,
        "closed":   closed,
        "eroded":   eroded,
        "final":    dilated,
    }


def analyze_connected_components(mask: np.ndarray,
                                  min_area: int = 100) -> dict:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    image_area = mask.shape[0] * mask.shape[1]
    components = []
    total_area = 0

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x    = int(stats[i, cv2.CC_STAT_LEFT])
        y    = int(stats[i, cv2.CC_STAT_TOP])
        w    = int(stats[i, cv2.CC_STAT_WIDTH])
        h    = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx   = float(centroids[i][0])
        cy   = float(centroids[i][1])

        components.append({
            "id":       i,
            "area":     area,
            "bbox":     (x, y, w, h),
            "centroid": (cx, cy),
        })
        total_area += area

    return {
        "num_components":  len(components),
        "total_area":      total_area,
        "image_area":      image_area,
        "infection_ratio": round(total_area / image_area * 100, 2),
        "components":      components,
        "labeled_img":     labels.astype(np.int32),
    }


def draw_component_boxes(image: np.ndarray,
                          cc_result: dict) -> np.ndarray:
    annotated = image.copy()
    for comp in cc_result["components"]:
        x, y, w, h = comp["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h),
                       color=(0, 0, 255), thickness=2)
        cv2.putText(annotated,
                    f"#{comp['id']} {comp['area']}px",
                    (x, max(y - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 200), 1, cv2.LINE_AA)
    return annotated


def plot_morphology_steps(morph_result: dict,
                           save_path: str = None) -> plt.Figure:
    steps = [
        ("Máscara raw",   morph_result["raw_mask"]),
        ("Apertura",      morph_result["opened"]),
        ("Cierre",        morph_result["closed"]),
        ("Erosión",       morph_result["eroded"]),
        ("Final (dilatada)", morph_result["final"]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, (title, img) in zip(axes, steps):
        ax.imshow(img, cmap="Greens")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle("Secuencia de operaciones morfológicas", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[morphology] Pasos guardados en {save_path}")
    return fig


def plot_final_detection(original: np.ndarray,
                          annotated: np.ndarray,
                          cc_result: dict,
                          save_path: str = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title("Imagen original")
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax2.set_title(
        f"Manchas detectadas: {cc_result['num_components']}\n"
        f"Área infectada: {cc_result['infection_ratio']:.1f}%"
    )
    ax2.axis("off")

    plt.suptitle("Detección final tras morfología + componentes conectados", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[morphology] Detección final guardada en {save_path}")
    return fig


def run_morphology(mask: np.ndarray,
                   original_image: np.ndarray,
                   config: dict) -> dict:
    morph_result = apply_full_morphology(mask, config)
    cc_result    = analyze_connected_components(morph_result["final"],
                                                 min_area=100)
    annotated    = draw_component_boxes(original_image, cc_result)

    return {
        "morph_steps":    morph_result,
        "final_mask":     morph_result["final"],
        "cc_metrics":     cc_result,
        "annotated_img":  annotated,
    }


if __name__ == "__main__":
    import sys
    from segmentation import run_segmentation, identify_disease_cluster

    if len(sys.argv) < 2:
        print("Uso: python morphology.py <ruta_imagen>")
        sys.exit(1)

    config = load_config()
    img = cv2.imread(sys.argv[1])
    img = cv2.resize(img, tuple(config["preprocessing"]["image_size"]))

    seg_result   = run_segmentation(img, config)
    disease_idx  = seg_result["metrics"]["disease_cluster_idx"]
    raw_mask     = seg_result["cluster_masks"][disease_idx]

    result = run_morphology(raw_mask, img, config)
    cc     = result["cc_metrics"]

    print(f"\nMorfología aplicada correctamente.")
    print(f"Manchas individuales detectadas: {cc['num_components']}")
    print(f"Área infectada total: {cc['infection_ratio']:.2f}%")
    print(f"Área total en píxeles: {cc['total_area']}")

    plot_morphology_steps(result["morph_steps"],
                           save_path="outputs/segmented/morphology_steps.png")
    plot_final_detection(img, result["annotated_img"], cc,
                          save_path="outputs/segmented/final_detection.png")
    plt.show()
