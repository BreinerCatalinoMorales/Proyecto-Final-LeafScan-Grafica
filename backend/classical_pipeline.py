import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import json
from pathlib import Path

from preprocessing   import preprocess_pipeline
from color_analysis  import analyze_image as analyze_colors
from segmentation    import run_segmentation
from morphology      import run_morphology


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_classical_pipeline(image_path: str,
                            config: dict,
                            verbose: bool = True) -> dict:
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Pipeline clásico: {Path(image_path).name}")
        print(f"{'='*55}")

    if verbose:
        print("[1/4] Preprocesando imagen...")

    pre = preprocess_pipeline(image_path, config)
    img = pre["original"]
    img_blur = pre["blurred"]

    if verbose:
        print("[2/4] Analizando color...")

    color_result = analyze_colors(img_blur, config)

    if verbose:
        print("[3/4] Segmentando con K-means (sin supervisión)...")

    seg_result  = run_segmentation(img_blur, config)
    disease_idx = seg_result["metrics"]["disease_cluster_idx"]
    raw_mask    = seg_result["cluster_masks"][disease_idx]

    if verbose:
        print("[4/4] Aplicando morfología matemática...")

    morph_result = run_morphology(raw_mask, img, config)
    cc_metrics   = morph_result["cc_metrics"]

    result = {
        "original_img":   img,
        "blurred_img":    img_blur,
        "sharpened_img":  pre["sharpened"],
        "edges_img":      pre["edges"],
        "segmented_img":  seg_result["segmented_img"],
        "disease_mask":   morph_result["final_mask"],
        "annotated_img":  morph_result["annotated_img"],
        "color_overlay":  color_result["disease_overlay"],

        "metrics": {
            "color_disease_ratio":   color_result["disease_ratio"],
            "kmeans_disease_ratio":  seg_result["metrics"]["disease_ratio"],
            "final_infection_ratio": cc_metrics["infection_ratio"],
            "num_disease_spots":     cc_metrics["num_components"],
            "total_infected_pixels": cc_metrics["total_area"],
            "total_pixels":          cc_metrics["image_area"],
            "color_stats":           color_result["color_stats"],
            "kmeans_cluster_idx":    disease_idx,
            "has_healthy_reference": seg_result["metrics"]["has_healthy_reference"],
        },

        "preprocessing":  pre,
        "color_analysis": color_result,
        "segmentation":   seg_result,
        "morphology":     morph_result,

        "image_path": str(image_path),
        "image_name": Path(image_path).stem,
    }

    if verbose:
        m = result["metrics"]
        print(f"\n  Resultado:")
        print(f"  ├─ Área enferma (K-means):  {m['kmeans_disease_ratio']:.1f}%")
        print(f"  ├─ Área enferma (final):    {m['final_infection_ratio']:.1f}%")
        print(f"  └─ Manchas detectadas:      {m['num_disease_spots']}")

    return result


def run_on_dataset(image_dir: str,
                   config: dict,
                   max_images: int = None) -> list:
    extensions = {".jpg", ".jpeg", ".png"}
    images = sorted([
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in extensions
    ])

    if max_images:
        images = images[:max_images]

    if not images:
        print(f"[classical_pipeline] No se encontraron imágenes en {image_dir}")
        return []

    results = []
    for i, img_path in enumerate(images):
        print(f"\nProcesando [{i+1}/{len(images)}]: {img_path.name}")
        try:
            result = run_classical_pipeline(str(img_path), config, verbose=False)
            results.append(result)
        except Exception as e:
            print(f"  ⚠ Error en {img_path.name}: {e}")

    return results


def save_metrics_to_json(results: list, output_path: str) -> None:
    records = []
    for r in results:
        records.append({
            "image_name": r["image_name"],
            "image_path": r["image_path"],
            **r["metrics"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[classical_pipeline] Métricas guardadas en {output_path}")


def plot_pipeline_summary(result: dict,
                           save_path: str = None) -> plt.Figure:
    def to_rgb(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    m = result["metrics"]

    panels = [
        ("Original",           to_rgb(result["original_img"])),
        ("Filtro Gauss",       to_rgb(result["blurred_img"])),
        ("Bordes (Canny)",     to_rgb(result["edges_img"])),
        ("K-means",            to_rgb(result["segmented_img"])),
        ("Máscara enferma",    result["disease_mask"]),
        ("Detección final",    to_rgb(result["annotated_img"])),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (title, img) in zip(axes, panels):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="Greens")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"Pipeline clásico — {result['image_name']}\n"
        f"Manchas: {m['num_disease_spots']}  |  "
        f"Área infectada: {m['final_infection_ratio']:.1f}%",
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[classical_pipeline] Resumen guardado en {save_path}")
    return fig


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python classical_pipeline.py <ruta_imagen>")
        print("     python classical_pipeline.py <carpeta_de_imagenes>")
        sys.exit(1)

    config = load_config()
    target = Path(sys.argv[1])

    if target.is_dir():
        results = run_on_dataset(str(target), config)
        save_metrics_to_json(results,
                              config["paths"]["outputs_metrics"] + "classical_results.json")
        print(f"\nProcesadas {len(results)} imágenes.")
    else:
        result = run_classical_pipeline(str(target), config)
        plot_pipeline_summary(
            result,
            save_path=config["paths"]["outputs_segmented"] + f"{result['image_name']}_summary.png"
        )
        plt.show()
