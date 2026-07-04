import cv2
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return img


def resize_image(image: np.ndarray, size: tuple = (256, 256)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def apply_gaussian_blur(image: np.ndarray,
                        kernel_size: int = 5,
                        sigma: float = 1.0) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_highpass_filter(image: np.ndarray,
                          alpha: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    high_freq = cv2.subtract(image, blurred)
    sharpened = cv2.addWeighted(image, 1.0, high_freq, alpha, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_edge_detection(image: np.ndarray,
                         method: str = "canny") -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "canny":
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    elif method == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(np.clip(edges, 0, 255))
    elif method == "laplacian":
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.abs(edges))
    else:
        raise ValueError(f"Método de detección no reconocido: {method}")

    return edges


def normalize_image(image: np.ndarray,
                    mean: list = None,
                    std: list = None) -> np.ndarray:
    img_float = image.astype(np.float32) / 255.0

    if mean is not None and std is not None:
        img_rgb = img_float[:, :, ::-1]
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr  = np.array(std,  dtype=np.float32)
        img_norm = (img_rgb - mean_arr) / std_arr
        return img_norm

    return img_float


def preprocess_pipeline(image_path: str,
                        config: dict = None) -> dict:
    if config is None:
        config = load_config()

    cfg_pre = config["preprocessing"]
    size    = tuple(cfg_pre["image_size"])

    img = load_image(image_path)
    img = resize_image(img, size)

    blurred = apply_gaussian_blur(
        img,
        kernel_size=cfg_pre["gaussian_blur_kernel"],
        sigma=cfg_pre["gaussian_blur_sigma"]
    )

    sharpened = apply_highpass_filter(
        img,
        alpha=cfg_pre["sharpen_alpha"]
    )

    edges = apply_edge_detection(img, method="canny")

    normalized = normalize_image(
        img,
        mean=cfg_pre["normalize_mean"],
        std=cfg_pre["normalize_std"]
    )

    return {
        "original":   img,
        "blurred":    blurred,
        "sharpened":  sharpened,
        "edges":      edges,
        "normalized": normalized,
    }


def save_preprocessed(results: dict, output_dir: str, filename: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / f"{filename}_original.jpg"),   results["original"])
    cv2.imwrite(str(out / f"{filename}_blurred.jpg"),    results["blurred"])
    cv2.imwrite(str(out / f"{filename}_sharpened.jpg"),  results["sharpened"])
    cv2.imwrite(str(out / f"{filename}_edges.jpg"),      results["edges"])
    norm_vis = (results["normalized"] * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out / f"{filename}_normalized.jpg"), norm_vis)
    print(f"[preprocessing] Guardadas versiones de '{filename}' en {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python preprocessing.py <ruta_imagen>")
        sys.exit(1)

    config = load_config()
    results = preprocess_pipeline(sys.argv[1], config)

    print("Versiones generadas:", list(results.keys()))
    print("Tamaño imagen original:", results["original"].shape)

    save_preprocessed(results, config["paths"]["data_processed"], "prueba")
