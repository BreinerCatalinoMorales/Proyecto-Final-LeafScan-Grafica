import argparse
import random
import shutil
import zipfile
from pathlib import Path

ZIP_DEFAULT     = "../data/raw/plantvillage-tomato-leaf-dataset.zip"
EXTRACT_DIR     = "../data/raw/plantvillage_extracted"
SPLITS_DIR      = "../data/splits"
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
RANDOM_SEED     = 42
IMAGE_EXTS      = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    if extract_to.exists():
        print(f"[prepare] Carpeta ya existe, omitiendo extracción: {extract_to}")
        return
    print(f"[prepare] Descomprimiendo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[prepare] Extraído en: {extract_to}")


def collect_source_folders(extract_dir: Path) -> dict:
    class_folders = {}
    for folder in sorted(extract_dir.rglob("*")):
        if not folder.is_dir():
            continue
        imgs = [f for f in folder.iterdir()
                if f.is_file() and f.suffix in IMAGE_EXTS]
        if not imgs:
            continue
        label = "healthy" if "healthy" in folder.name.lower() else "diseased"
        if label not in class_folders:
            class_folders[label] = []
        class_folders[label].append(folder)

    return class_folders


def collect_images(class_folders: dict) -> dict:
    result = {}
    for label, folders in class_folders.items():
        paths = []
        for folder in folders:
            imgs = [f for f in folder.iterdir()
                    if f.is_file() and f.suffix in IMAGE_EXTS]
            paths.extend(imgs)
            print(f"  {folder.name:45s}  →  {label}  ({len(imgs)} imgs)")
        result[label] = paths
    return result


def split_paths(paths: list, train_r: float, val_r: float, seed: int):
    lst = list(paths)
    random.seed(seed)
    random.shuffle(lst)
    n       = len(lst)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]


def copy_split(paths: list, dest: Path, label: str, split: str) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths):
        dst = dest / f"{label}_{split}_{i:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    return len(paths)


def prepare(zip_path: Path, max_per_class: int = None) -> None:

    extract_dir = Path(EXTRACT_DIR)
    extract_zip(zip_path, extract_dir)

    print("\n[prepare] Detectando carpetas de clases...")
    class_folders = collect_source_folders(extract_dir)

    if not class_folders:
        raise RuntimeError(
            "No se encontraron carpetas con imágenes dentro del ZIP.\n"
            f"Revisá el contenido de: {extract_dir}"
        )

    print("\n[prepare] Clasificando imágenes:")
    images = collect_images(class_folders)

    for label in ("healthy", "diseased"):
        if label not in images:
            raise RuntimeError(
                f"No se encontraron imágenes para la clase '{label}'.\n"
                "Revisá que el ZIP contenga una carpeta con 'healthy' en el nombre."
            )

    if max_per_class:
        for label in images:
            random.seed(RANDOM_SEED)
            random.shuffle(images[label])
            images[label] = images[label][:max_per_class]
        print(f"\n[prepare] Límite aplicado: {max_per_class} imágenes por clase")

    splits_dir = Path(SPLITS_DIR)
    print(f"\n[prepare] Generando splits en {splits_dir} ...")

    resumen = {}
    for label, paths in images.items():
        tr, va, te = split_paths(paths, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
        resumen[label] = {"train": len(tr), "val": len(va), "test": len(te)}

        copy_split(tr, splits_dir / "train" / label, label, "train")
        copy_split(va, splits_dir / "val"   / label, label, "val")
        copy_split(te, splits_dir / "test"  / label, label, "test")

    print("\n" + "=" * 52)
    print("  RESUMEN DEL DATASET")
    print("=" * 52)
    print(f"  {'Clase':<12} {'train':>8} {'val':>8} {'test':>8} {'total':>8}")
    print("-" * 52)
    for label, counts in resumen.items():
        total = sum(counts.values())
        print(f"  {label:<12} {counts['train']:>8} {counts['val']:>8} "
              f"{counts['test']:>8} {total:>8}")
    print("=" * 52)
    print(f"\n  Splits listos en: {splits_dir.resolve()}")
    print("  Siguiente paso:   python train.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepara el dataset PlantVillage para entrenamiento PyTorch"
    )
    parser.add_argument(
        "--zip",
        type=str,
        default=ZIP_DEFAULT,
        help="Ruta al ZIP descargado de Kaggle"
    )
    parser.add_argument(
        "--max-per-clase",
        type=int,
        default=None,
        dest="max_per_class",
        help="Límite de imágenes por clase (útil para prueba rápida, ej. 300)"
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"[ERROR] No se encontró el ZIP en: {zip_path.resolve()}")
        print("  Asegurate de poner el ZIP en data/raw/ o pasá la ruta con --zip")
        raise SystemExit(1)

    prepare(zip_path, max_per_class=args.max_per_class)