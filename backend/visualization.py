import base64

import cv2
import numpy as np


def encode_image_b64(img: np.ndarray) -> str:
    if img.ndim == 2:
        encode_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        encode_img = img

    success, buf = cv2.imencode(".png", encode_img)
    if not success:
        raise ValueError("No se pudo codificar la imagen a PNG.")
    return base64.b64encode(buf).decode("utf-8")


def create_disease_overlay(original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    disease = mask > 0
    red_layer = np.zeros_like(original_bgr)
    red_layer[disease] = [0, 0, 220]
    result = cv2.addWeighted(original_bgr, 0.65, red_layer, 0.35, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (0, 210, 70), 2)
    return result


def create_gradcam_heatmap(original_bgr: np.ndarray, cam_float: np.ndarray) -> np.ndarray:
    h, w = original_bgr.shape[:2]
    cam_resized = cv2.resize(np.asarray(cam_float, dtype=np.float32), (w, h))
    cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
    return cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)


def create_gradcam_overlay(original_bgr: np.ndarray, cam_float: np.ndarray) -> np.ndarray:
    heatmap = create_gradcam_heatmap(original_bgr, cam_float)
    return cv2.addWeighted(original_bgr, 0.55, heatmap, 0.45, 0)


def resize_for_cnn(img_bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)


def build_histogram_data(img_bgr: np.ndarray, config: dict) -> dict:
    viz_cfg = (config or {}).get("visualization", {})
    bins_rgb = viz_cfg.get("hist_bins_rgb", 64)
    bins_h = viz_cfg.get("hist_bins_h", 36)
    bins_sv = viz_cfg.get("hist_bins_sv", 32)

    b, g, r = cv2.split(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    def hist(channel, bins, rng):
        counts, _ = np.histogram(channel, bins=bins, range=rng)
        return counts.tolist()

    return {
        "rgb": {
            "r": hist(r, bins_rgb, (0, 256)),
            "g": hist(g, bins_rgb, (0, 256)),
            "b": hist(b, bins_rgb, (0, 256)),
            "bins": bins_rgb,
        },
        "hsv": {
            "h": hist(h, bins_h, (0, 180)),
            "s": hist(s, bins_sv, (0, 256)),
            "v": hist(v, bins_sv, (0, 256)),
            "bins_h": bins_h,
            "bins_sv": bins_sv,
        },
    }


def prepare_classical_payload(result: dict, config: dict) -> dict:
    m = result["metrics"]
    original = result["original_img"]
    disease_mask = result["disease_mask"]

    overlay = create_disease_overlay(original, disease_mask)

    metrics = {
        "infection_ratio": round(float(m["final_infection_ratio"]), 2),
        "num_spots": int(m["num_disease_spots"]),
        "kmeans_disease_ratio": round(float(m["kmeans_disease_ratio"]), 2),
        "color_disease_ratio": round(float(m["color_disease_ratio"]), 2),
        "total_infected_pixels": int(m["total_infected_pixels"]),
        "total_pixels": int(m["total_pixels"]),
        "has_healthy_reference": bool(m.get("has_healthy_reference", True)),
    }

    images = {
        "original": encode_image_b64(original),
        "blurred": encode_image_b64(result["blurred_img"]),
        "segmented": encode_image_b64(result["segmented_img"]),
        "disease_mask": encode_image_b64(disease_mask),
        "annotated": encode_image_b64(result["annotated_img"]),
        "overlay": encode_image_b64(overlay),
    }

    histograms = build_histogram_data(original, config)

    return {
        "metrics": metrics,
        "images": images,
        "histograms": histograms,
    }
