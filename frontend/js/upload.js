const API_BASE = "https://proyecto-final-leafscan-grafica.onrender.com";

const uploadZone    = document.getElementById("uploadZone");
const fileInput     = document.getElementById("fileInput");
const uploadPreview = document.getElementById("uploadPreview");
const previewImg    = document.getElementById("previewImg");
const previewName   = document.getElementById("previewName");
const changeImgBtn  = document.getElementById("changeImg");
const uploadActions = document.getElementById("uploadActions");
const analyzeBtn    = document.getElementById("analyzeBtn");
const uploadLoading = document.getElementById("uploadLoading");
const uploadError   = document.getElementById("uploadError");
const errorMsg      = document.getElementById("errorMsg");

let selectedFile = null;

function showEl(el)  { el.classList.remove("hidden"); }
function hideEl(el)  { el.classList.add("hidden"); }

function showError(msg) {
  errorMsg.textContent = msg;
  showEl(uploadError);
}

function clearError() {
  hideEl(uploadError);
  errorMsg.textContent = "";
}

function handleFile(file) {
  if (!file) return;

  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Formato no soportado. Usa JPG o PNG.");
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showError("El archivo es demasiado grande. Máximo 15 MB.");
    return;
  }

  clearError();
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    hideEl(uploadZone);
    showEl(uploadPreview);
    showEl(uploadActions);
  };
  reader.readAsDataURL(file);
}

uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

changeImgBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  hideEl(uploadPreview);
  hideEl(uploadActions);
  showEl(uploadZone);
  clearError();
});

uploadZone.addEventListener("dragover", e => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

["dragleave", "dragend"].forEach(evt =>
  uploadZone.addEventListener(evt, () => uploadZone.classList.remove("drag-over"))
);

uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    showError("Por favor selecciona una imagen primero.");
    return;
  }

  clearError();
  hideEl(uploadActions);
  hideEl(uploadPreview);
  showEl(uploadLoading);
  analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Error desconocido del servidor.");
    }

    sessionStorage.setItem("leafscan_result", JSON.stringify(data));
    window.location.href = "results.html";

  } catch (err) {
    hideEl(uploadLoading);
    showEl(uploadActions);
    showEl(uploadPreview);
    analyzeBtn.disabled = false;

    if (err.message.includes("Failed to fetch") || err.name === "TypeError") {
      showError(
        "No se pudo conectar con el servidor. Asegúrate de que Flask esté corriendo: " +
        "cd backend && python app.py"
      );
    } else {
      showError(`Error: ${err.message}`);
    }
  }
});
