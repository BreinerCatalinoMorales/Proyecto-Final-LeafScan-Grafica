const raw = sessionStorage.getItem("leafscan_result");

if (!raw) {
  document.getElementById("noResult").classList.remove("hidden");
  document.getElementById("resultContent").classList.add("hidden");
} else {
  const result = JSON.parse(raw);
  renderResults(result);
}

function renderResults(result) {
  document.getElementById("noResult").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");

  const cls = result.classical;
  const m   = cls.metrics;

  const filenameEl = document.getElementById("resultFilename");
  if (filenameEl) filenameEl.textContent = result.filename || "imagen analizada";

  const infected    = m.infection_ratio > 5;
  const verdictEl   = document.getElementById("verdict");
  const verdictIcon = document.getElementById("verdictIcon");
  const verdictText = document.getElementById("verdictText");
  const verdictSub  = document.getElementById("verdictSub");

  if (infected) {
    verdictEl.className = "verdict-box verdict-diseased";
    verdictIcon.textContent = "⚠️";
    verdictText.textContent = "Hoja ENFERMA";
    verdictSub.textContent  = `${m.infection_ratio}% del área infectada · ${m.num_spots} mancha(s) detectada(s)`;
  } else {
    verdictEl.className = "verdict-box verdict-healthy";
    verdictIcon.textContent = "✅";
    verdictText.textContent = "Hoja SANA";
    verdictSub.textContent  = `Solo ${m.infection_ratio}% de área posiblemente afectada`;
  }

  document.getElementById("metInfection").textContent = `${m.infection_ratio}%`;
  document.getElementById("metSpots").textContent     = m.num_spots;
  document.getElementById("metKmeans").textContent    = `${m.kmeans_disease_ratio}%`;
  document.getElementById("metColor").textContent     = `${m.color_disease_ratio}%`;

  document.getElementById("noHealthyRefWarning")
    .classList.toggle("hidden", m.has_healthy_reference !== false);

  const imgMap = {
    imgOriginal:    cls.images.original,
    imgBlurred:     cls.images.blurred,
    imgSegmented:   cls.images.segmented,
    imgMask:        cls.images.disease_mask,
    imgAnnotated:   cls.images.annotated,
    imgOverlay:     cls.images.overlay,
  };

  for (const [id, b64] of Object.entries(imgMap)) {
    const el = document.getElementById(id);
    if (el && b64) el.src = `data:image/png;base64,${b64}`;
  }

  const cnnSection = document.getElementById("cnnSection");
  if (result.cnn && !result.cnn.error) {
    const cnn = result.cnn;
    cnnSection.classList.remove("hidden");

    document.getElementById("cnnLabel").textContent =
      cnn.label === "diseased" ? "Enferma" : "Sana";
    document.getElementById("cnnConfidence").textContent =
      `${(cnn.confidence * 100).toFixed(1)}%`;
    document.getElementById("cnnLabelBadge").className =
      `cnn-badge ${cnn.label === "diseased" ? "badge-diseased" : "badge-healthy"}`;

    const cnnImgMap = {
      imgCnnInput:   cnn.images?.input,
      imgCnnHeatmap: cnn.images?.heatmap,
      imgCnnMask:    cnn.images?.disease_mask,
      imgGradcam:    cnn.images?.gradcam,
    };

    for (const [id, b64] of Object.entries(cnnImgMap)) {
      const el = document.getElementById(id);
      if (el && b64) el.src = `data:image/png;base64,${b64}`;
    }
  } else {
    cnnSection.classList.add("hidden");
  }

  if (cls.histograms && typeof renderCharts === "function") {
    renderCharts(cls.histograms);
  }
}

const newBtn = document.getElementById("newAnalysisBtn");
if (newBtn) {
  newBtn.addEventListener("click", () => {
    sessionStorage.removeItem("leafscan_result");
    window.location.href = "index.html#analizar";
  });
}
