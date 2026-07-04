import sys
import uuid
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from classical_pipeline import run_classical_pipeline, load_config
from visualization import prepare_classical_payload


CNN_AVAILABLE = False
_model  = None
_meta   = None
_device = None


def _try_load_cnn(config: dict) -> None:
    global CNN_AVAILABLE, _model, _meta, _device
    try:
        from utils import get_device
        from cnn_model import load_model
        weights = Path(__file__).parent / "models" / "cnn_weights.pth"
        if not weights.exists():
            print(f"[app] Pesos CNN no encontrados en {weights}")
            return
        _device = get_device()
        _model, _meta = load_model(str(weights), config, _device)
        CNN_AVAILABLE = True
        backbone = _meta.get("backbone", "?")
        print(f"[app] CNN cargada ({backbone}) en {_device}.")
    except Exception as e:
        print(f"[app] CNN no disponible: {e}")


app = Flask(__name__)
CORS(app)

CONFIG_PATH = Path(__file__).parent / "config.yaml"
config: dict = {}
try:
    config = load_config(str(CONFIG_PATH))
    _try_load_cnn(config)
except Exception as e:
    print(f"[app] Error al cargar configuración: {e}")

FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "leafscan"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _add_cnn_result(response: dict, img_path: str, classical_result: dict) -> None:
    try:
        from cnn_model import (predict_image, GradCAM, get_target_layer,
                               get_transforms, get_cnn_disease_mask, _to_pil)
        from visualization import (create_gradcam_overlay, create_gradcam_heatmap,
                                   resize_for_cnn, encode_image_b64)

        cnn_pred = predict_image(_model, img_path, _meta, device=_device)

        backbone   = _meta.get("backbone", "mobilenet_v2")
        input_size = _meta.get("input_size", 224)
        tf = get_transforms(
            input_size,
            _meta.get("normalize_mean"),
            _meta.get("normalize_std"),
            train=False,
        )
        x = tf(_to_pil(img_path)).unsqueeze(0).to(_device)
        cam_engine = GradCAM(_model, get_target_layer(_model, backbone))
        try:
            cam_raw = cam_engine(x)
        finally:
            cam_engine.remove()

        original = classical_result["original_img"]
        h, w = original.shape[:2]

        input_preview   = resize_for_cnn(original, input_size)
        heatmap         = create_gradcam_heatmap(original, cam_raw)
        gradcam_overlay = create_gradcam_overlay(original, cam_raw)
        disease_mask    = get_cnn_disease_mask(
            _model, img_path, _meta, out_size=(w, h), device=_device
        )

        response["cnn"] = {
            "label":      cnn_pred["label"],
            "label_idx":  cnn_pred["label_idx"],
            "confidence": round(float(cnn_pred["confidence"]), 4),
            "images": {
                "input":        encode_image_b64(input_preview),
                "heatmap":      encode_image_b64(heatmap),
                "disease_mask": encode_image_b64(disease_mask),
                "gradcam":      encode_image_b64(gradcam_overlay),
            },
        }
    except Exception as e:
        response["cnn"] = {"error": f"CNN disponible pero falló: {e}"}


@app.route("/")
def frontend_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def frontend_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "cnn_available": CNN_AVAILABLE,
        "backbone":      _meta.get("backbone") if _meta else None,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Campo 'image' no encontrado en la solicitud."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacío."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"Formato '{ext}' no soportado. Usa JPG o PNG."}), 400

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(str(tmp_path))

    try:
        classical_result  = run_classical_pipeline(str(tmp_path), config, verbose=False)
        classical_payload = prepare_classical_payload(classical_result, config)

        response = {
            "success":   True,
            "filename":  file.filename,
            "classical": classical_payload,
        }

        if CNN_AVAILABLE and _model is not None:
            _add_cnn_result(response, str(tmp_path), classical_result)

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": f"Error al procesar la imagen: {e}"}), 500

    tmp_path.unlink(missing_ok=True)
    return jsonify(response)


@app.route("/api/compare", methods=["GET"])
def compare():
    comparison_file = (
        Path(__file__).parent.parent / "outputs" / "metrics" / "comparison.json"
    )
    if not comparison_file.exists():
        return jsonify({
            "available": False,
            "error":     "Comparativa no disponible. Ejecuta comparison.py primero.",
        }), 404

    from utils import load_json
    data = load_json(str(comparison_file))
    return jsonify({"success": True, "data": data})


METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"
ALLOWED_METRIC_ASSETS = {
    "training_curves.png",
    "cnn_confusion_matrix.png",
    "comparison_summary.png",
    "cnn_error_gallery.png",
    "cnn_metrics_per_class.png",
    "cnn_confidence_dist.png",
}


@app.route("/api/compare/assets/<path:filename>")
def compare_asset(filename):
    if filename not in ALLOWED_METRIC_ASSETS:
        return jsonify({"error": "Recurso no permitido."}), 404
    return send_from_directory(METRICS_DIR, filename)


if __name__ == "__main__":
    print("\n  LeafScan — Backend Flask")
    print(f"  CNN: {'disponible ✓' if CNN_AVAILABLE else 'no disponible'}")
    print("  Servidor en http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)