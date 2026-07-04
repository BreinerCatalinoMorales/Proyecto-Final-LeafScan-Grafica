function makeLabels(bins, max) {
  const step = max / bins;
  return Array.from({ length: bins }, (_, i) => Math.round(i * step));
}

function buildBarChart(canvasId, datasets, labels, title) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  if (canvas._chartInstance) {
    canvas._chartInstance.destroy();
  }

  canvas._chartInstance = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: { duration: 400 },
      plugins: {
        legend: { position: "top", labels: { boxWidth: 14, font: { size: 11 } } },
        title:  { display: true, text: title, font: { size: 13, weight: "600" } },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 12, font: { size: 10 } },
          grid:  { display: false },
        },
        y: {
          ticks: { font: { size: 10 } },
          grid:  { color: "rgba(0,0,0,0.05)" },
        },
      },
    },
  });
}

function renderCharts(histograms) {
  const { rgb, hsv } = histograms;

  buildBarChart(
    "chartRGB",
    [
      { label: "R", data: rgb.r, backgroundColor: "rgba(231,76,60,0.55)",  borderColor: "#e74c3c", borderWidth: 1 },
      { label: "G", data: rgb.g, backgroundColor: "rgba(39,174,96,0.55)",  borderColor: "#27ae60", borderWidth: 1 },
      { label: "B", data: rgb.b, backgroundColor: "rgba(52,152,219,0.55)", borderColor: "#3498db", borderWidth: 1 },
    ],
    makeLabels(rgb.bins, 256),
    "Histograma RGB"
  );

  buildBarChart(
    "chartHSV",
    [
      { label: "Hue",        data: hsv.h, backgroundColor: "rgba(155,89,182,0.6)",  borderColor: "#9b59b6", borderWidth: 1 },
      { label: "Saturación", data: hsv.s, backgroundColor: "rgba(230,126,34,0.55)", borderColor: "#e67e22", borderWidth: 1 },
      { label: "Brillo (V)", data: hsv.v, backgroundColor: "rgba(52,73,94,0.45)",   borderColor: "#34495e", borderWidth: 1 },
    ],
    makeLabels(hsv.bins_sv, 256),
    "Histograma HSV"
  );
}
