import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from preprocessing      import preprocess_pipeline, load_config
from color_analysis     import analyze_image as analyze_colors, plot_rgb_histograms, plot_hsv_histograms
from segmentation       import run_segmentation, plot_segmentation_results, plot_cluster_distribution
from morphology         import run_morphology, plot_morphology_steps, plot_final_detection
from classical_pipeline import run_classical_pipeline, plot_pipeline_summary


def create_sample_leaf_image() -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    img[:] = [240, 245, 240]

    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (100, 115), 0, 0, 360, 255, -1)
    img[mask > 0] = [34, 139, 34]

    cv2.ellipse(img, (90, 100), (28, 20), 20, 0, 360, [30, 165, 210], -1)
    cv2.ellipse(img, (155, 80), (18, 14), 0,  0, 360, [20, 120, 160], -1)
    cv2.ellipse(img, (120, 170), (22, 16), 45, 0, 360, [15, 100, 140], -1)

    noise = np.random.randint(-12, 12, img.shape, dtype=np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def save_sample_image(path: str) -> str:
    img = create_sample_leaf_image()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)
    return path


def run_demo(image_path: str, config: dict) -> None:
    out_seg  = Path(config["paths"]["outputs_segmented"])
    out_met  = Path(config["paths"]["outputs_metrics"])
    out_seg.mkdir(parents=True, exist_ok=True)
    out_met.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  DEMO LOCAL — Pipeline Clásico de Detección de Enfermedades")
    print("="*60)
    print(f"  Imagen: {image_path}\n")

    print("► Etapa 1/4: Preprocesamiento")
    pre     = preprocess_pipeline(image_path, config)
    img     = pre["original"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels  = ["Original", "Gaussian Blur", "Filtro pasa alto", "Bordes (Canny)"]
    imgs    = [img, pre["blurred"], pre["sharpened"], pre["edges"]]
    for ax, lbl, im in zip(axes, labels, imgs):
        if len(im.shape) == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
    plt.suptitle("Etapa 1 — Preprocesamiento", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_seg / "etapa1_preprocesamiento.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Guardado: etapa1_preprocesamiento.png")

    print("► Etapa 2/4: Análisis de color")
    color_result = analyze_colors(pre["blurred"], config)
    fig_rgb = plot_rgb_histograms(img, title="Histogramas RGB")
    fig_rgb.savefig(str(out_met / "histograma_rgb.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_rgb)
    fig_hsv = plot_hsv_histograms(img, title="Histogramas HSV")
    fig_hsv.savefig(str(out_met / "histograma_hsv.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_hsv)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original"); ax1.axis("off")
    ax2.imshow(cv2.cvtColor(color_result["disease_overlay"], cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Detección por color HSV ({color_result['disease_ratio']:.1f}%)")
    ax2.axis("off")
    plt.suptitle("Etapa 2 — Análisis de color", fontsize=12)
    plt.tight_layout()
    fig2.savefig(str(out_seg / "etapa2_color.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Área enferma por color HSV: {color_result['disease_ratio']:.1f}%")
    print("  ✓ Guardados: histograma_rgb.png, histograma_hsv.png, etapa2_color.png")

    print("► Etapa 3/4: Segmentación K-means (sin supervisión)")
    seg_result  = run_segmentation(pre["blurred"], config)
    metrics_seg = seg_result["metrics"]
    disease_idx = metrics_seg["disease_cluster_idx"]

    fig_seg = plot_segmentation_results(img, seg_result, metrics_seg)
    fig_seg.savefig(str(out_seg / "etapa3_kmeans.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_seg)
    fig_dist = plot_cluster_distribution(metrics_seg)
    fig_dist.savefig(str(out_met / "cluster_distribucion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_dist)
    print(f"  ✓ Área enferma K-means: {metrics_seg['disease_ratio']:.1f}%")
    print(f"  ✓ Cluster de enfermedad: #{disease_idx}")
    print("  ✓ Guardados: etapa3_kmeans.png, cluster_distribucion.png")

    print("► Etapa 4/4: Morfología matemática + componentes conectados")
    raw_mask     = seg_result["cluster_masks"][disease_idx]
    morph_result = run_morphology(raw_mask, img, config)
    cc           = morph_result["cc_metrics"]

    fig_morph = plot_morphology_steps(morph_result["morph_steps"])
    fig_morph.savefig(str(out_seg / "etapa4_morfologia.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_morph)
    fig_det = plot_final_detection(img, morph_result["annotated_img"], cc)
    fig_det.savefig(str(out_seg / "etapa4_deteccion_final.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_det)
    print(f"  ✓ Manchas detectadas: {cc['num_components']}")
    print(f"  ✓ Área infectada final: {cc['infection_ratio']:.1f}%")
    print("  ✓ Guardados: etapa4_morfologia.png, etapa4_deteccion_final.png")

    result_full = run_classical_pipeline(image_path, config, verbose=False)
    fig_sum = plot_pipeline_summary(
        result_full,
        save_path=str(out_seg / "resumen_pipeline.png")
    )
    plt.close(fig_sum)

    print("\n" + "="*60)
    print("  RESUMEN DE RESULTADOS")
    print("="*60)
    print(f"  Detección por color HSV : {color_result['disease_ratio']:6.1f}%")
    print(f"  Segmentación K-means    : {metrics_seg['disease_ratio']:6.1f}%")
    print(f"  Área infectada final    : {cc['infection_ratio']:6.1f}%")
    print(f"  Manchas individuales    : {cc['num_components']}")
    print(f"\n  Archivos generados en:")
    print(f"    outputs/segmented/  — imágenes anotadas")
    print(f"    outputs/metrics/    — histogramas y gráficas")
    print("="*60)
    print("\n  ✅ Demo completada exitosamente.\n")


if __name__ == "__main__":
    config = load_config(
        os.path.join(os.path.dirname(__file__), "config.yaml")
    )

    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"Error: No se encontró la imagen '{image_path}'")
            sys.exit(1)
    else:
        image_path = "data/raw/sample_leaf.jpg"
        print("No se indicó imagen. Generando imagen sintética de prueba...")
        save_sample_image(image_path)

    run_demo(image_path, config)
