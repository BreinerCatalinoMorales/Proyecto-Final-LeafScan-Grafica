import os
import json
import time
import random
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import yaml
import torch


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_section(config: dict, name: str, defaults: dict) -> dict:
    section = dict(defaults)
    if config and isinstance(config.get(name), dict):
        section.update(config[name])
    return section


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(directory: str,
                extensions=(".jpg", ".jpeg", ".png")) -> list:
    ext = {e.lower() for e in extensions}
    return sorted([
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in ext
    ])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, output_path: str) -> None:
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"[utils] JSON guardado en {output_path}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model) -> tuple:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@contextmanager
def timer(name: str = "bloque"):
    start = time.perf_counter()
    yield
    print(f"[timer] {name}: {time.perf_counter() - start:.3f}s")


def measure_inference_time(fn, *args, n_warmup: int = 1, n_runs: int = 3, **kwargs):
    for _ in range(max(0, n_warmup)):
        fn(*args, **kwargs)

    times = []
    result = None
    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000.0)

    return result, float(np.mean(times))