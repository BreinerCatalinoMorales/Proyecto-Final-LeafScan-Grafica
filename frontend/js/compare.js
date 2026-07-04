const COMPARE_API_BASE = "https://proyecto-final-leafscan-grafica.onrender.com";

async function loadComparison() {
  const loadingEl = document.getElementById("compareLoading");
  const unavailableEl = document.getElementById("compareUnavailable");
  const contentEl = document.getElementById("compareContent");

  try {
    const res = await fetch(`${COMPARE_API_BASE}/api/compare`);
    if (!res.ok) throw new Error("Comparativa no disponible");

    const { data } = await res.json();
    renderComparison(data);

    loadingEl.classList.add("hidden");
    contentEl.classList.remove("hidden");
  } catch (e) {
    loadingEl.classList.add("hidden");
    unavailableEl.classList.remove("hidden");
  }
}

function renderComparison(data) {
  const { summary, errors } = data;
  const cls = summary.classical;
  const cnn = summary.cnn;

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const ms  = (v) => `${v.toFixed(1)} ms`;

  const rows = [
    ["Accuracy",           pct(cls.accuracy),  pct(cnn.accuracy)],
    ["Precisión",          pct(cls.precision), pct(cnn.precision)],
    ["Recall",             pct(cls.recall),    pct(cnn.recall)],
    ["F1-score",           pct(cls.f1),        pct(cnn.f1)],
    ["Falsos positivos",   cls.fp,             cnn.fp],
    ["Falsos negativos",   cls.fn,             cnn.fn],
    ["Velocidad promedio", ms(cls.avg_ms),     ms(cnn.avg_ms)],
  ];

  document.getElementById("compareTableBody").innerHTML = rows
    .map(([label, clsVal, cnnVal]) => `
      <tr>
        <td>${label}</td>
        <td>${clsVal}</td>
        <td>${cnnVal}</td>
      </tr>
    `)
    .join("");

  document.getElementById("compareIou").textContent =
    `IoU promedio entre ambos métodos: ${pct(summary.mean_iou)} · ` +
    `evaluado sobre ${summary.n_images} imágenes de test.`;

  document.getElementById("compareErrors").textContent =
    `${errors.classical_misclassified.length} error(es) del pipeline clásico · ` +
    `${errors.cnn_misclassified.length} error(es) de la CNN · ` +
    `${errors.disagreements.length} caso(s) donde ambos métodos discrepan.`;

  document.getElementById("imgTrainingCurves").src =
    `${COMPARE_API_BASE}/api/compare/assets/training_curves.png`;
  document.getElementById("imgConfusionMatrix").src =
    `${COMPARE_API_BASE}/api/compare/assets/cnn_confusion_matrix.png`;
  document.getElementById("imgCompareSummary").src =
    `${COMPARE_API_BASE}/api/compare/assets/comparison_summary.png`;
  document.getElementById("imgErrorGallery").src =
    `${COMPARE_API_BASE}/api/compare/assets/cnn_error_gallery.png`;
}

loadComparison();
