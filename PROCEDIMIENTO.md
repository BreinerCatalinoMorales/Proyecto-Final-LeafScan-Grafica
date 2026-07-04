# Procedimiento para ejecutar el avance
## LeafScan — Detector de Enfermedades en Hojas

---

## Estructura de archivos esperada

Antes de empezar verificar que el proyecto tiene esta estructura:

```
plant_disease_web/
├── backend/
│   ├── config.yaml
│   ├── requirements.txt
│   ├── preprocessing.py
│   ├── color_analysis.py
│   ├── segmentation.py
│   ├── morphology.py
│   ├── classical_pipeline.py
│   └── demo_local.py
│
└── frontend/
    ├── index.html
    ├── results.html
    ├── about.html
    └── css/
        └── styles.css
```

---

## Parte 1 — Demo local del pipeline clásico

### Paso 1: Crear entorno virtual (recomendado)

```bash
# Desde la raíz del proyecto
python -m venv venv

# Activar en Windows:
venv\Scripts\activate

# Activar en Mac / Linux:
source venv/bin/activate
```

### Paso 2: Instalar dependencias

```bash
cd backend
pip install -r requirements.txt
```

> Si hay problemas con PyTorch, instalarlo por separado desde https://pytorch.org/get-started/locally/

### Paso 3: Ejecutar la demo

**Opción A — Sin imagen propia (genera una hoja sintética automáticamente):**
```bash
python demo_local.py
```

**Opción B — Con tu propia imagen de hoja:**
```bash
python demo_local.py ruta/a/tu/imagen.jpg
```

### Paso 4: Ver los resultados

La demo genera automáticamente estos archivos en `outputs/`:

```
outputs/
├── segmented/
│   ├── etapa1_preprocesamiento.png     ← Filtros aplicados
│   ├── etapa2_color.png               ← Detección por color HSV
│   ├── etapa3_kmeans.png              ← Segmentación K-means
│   ├── etapa4_morfologia.png          ← Pasos morfológicos
│   ├── etapa4_deteccion_final.png     ← Manchas marcadas
│   └── resumen_pipeline.png           ← Vista general de todo el pipeline
│
└── metrics/
    ├── histograma_rgb.png             ← Histogramas R, G, B
    ├── histograma_hsv.png             ← Histogramas H, S, V
    └── cluster_distribucion.png       ← Distribución de clusters K-means
```

En consola verás un resumen como este:
```
============================================================
  RESUMEN DE RESULTADOS
============================================================
  Detección por color HSV :   18.4%
  Segmentación K-means    :   21.7%
  Área infectada final    :   19.2%
  Manchas individuales    :   4
============================================================
  ✅ Demo completada exitosamente.
```

---

## Parte 2 — Ver la página web

La web del avance es estática (no necesita servidor).

### Opción A — Abrir directamente en el navegador

1. Ir a la carpeta `frontend/`
2. Hacer doble clic en `index.html`
3. Se abre directamente en el navegador

### Opción B — Servidor local simple (recomendado para evitar errores de CORS)

```bash
cd frontend

# Con Python 3:
python -m http.server 8080

# Luego abrir en el navegador:
# http://localhost:8080
```

### Agregar imágenes reales a la galería (opcional para el avance)

1. Ejecutar `demo_local.py` para generar los resultados
2. Copiar las imágenes de `backend/outputs/segmented/` a `frontend/img/`
3. En `results.html`, reemplazar cada bloque `gallery-placeholder` por:

```html
<img src="img/etapa1_preprocesamiento.png" alt="Preprocesamiento" style="border-radius:12px; width:100%;" />
```

---

## Resumen rápido de comandos

```bash
# 1. Instalar
cd backend
pip install -r requirements.txt

# 2. Ejecutar demo local
python demo_local.py

# 3. Ver resultados en outputs/

# 4. Abrir web
cd ../frontend
python -m http.server 8080
# → http://localhost:8080
```

---

## Problemas frecuentes

| Error | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: cv2` | OpenCV no instalado | `pip install opencv-python` |
| `ModuleNotFoundError: yaml` | PyYAML no instalado | `pip install PyYAML` |
| `FileNotFoundError: config.yaml` | Ejecutar demo desde fuera de `backend/` | `cd backend` antes de ejecutar |
| La web no carga estilos | Abrir `index.html` sin servidor | Usar `python -m http.server 8080` |
| Imagen no se lee | Formato no soportado | Usar `.jpg` o `.png` |

---

## Lo que se muestra en el avance vs lo que viene después

| Funcionalidad | Avance | Entrega final |
|---|---|---|
| Pipeline clásico local (demo) | ✅ Funcionando | ✅ |
| Página web del proyecto | ✅ Estática | ✅ Dinámica con backend |
| Subida de imagen desde la web | ❌ No aún | ✅ |
| CNN entrenada | 🔄 En proceso | ✅ |
| Comparativa clásico vs CNN | ❌ No aún | ✅ |
| Grad-CAM y mapas de activación | ❌ No aún | ✅ |
| API Flask conectada al frontend | ❌ No aún | ✅ |

---

*LeafScan · Computación Gráfica · 2026*
*Espinoza · Huanca · Catalino*
