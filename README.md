# Backend — Clasificación Multietiqueta de Enfermedades Autoinmunes

API REST construida con **FastAPI** que sirve un modelo **Gated MLP (PyTorch)** pre-entrenado para clasificar **7 enfermedades autoinmunes** a partir de **14 variables clínicas**.

> **Importante:** Este es un sistema de predicción **multietiqueta** (sigmoides independientes), no multiclase. Un paciente puede tener varias enfermedades activas simultáneamente.

---

## Requisitos previos

- **Python 3.10+**
- Los artefactos del modelo entrenado en la carpeta `artifacts/`:
  - `model.pt` — pesos del modelo (state_dict)
  - `preprocessor.joblib` — ColumnTransformer pre-ajustado
  - `label_binarizer.joblib` — MultiLabelBinarizer con las 7 clases
  - `config.yaml` — configuración de arquitectura
  - `input_schema.json` — esquema de las 14 variables de entrada
  - `decision_rule.json` — umbral de compatibilidad (0.5)

---

## Instalación

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Ejecución

```bash
# Desarrollo (con recarga automática)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Producción
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

La documentación Swagger queda disponible en: **http://localhost:8000/docs**

---

## Endpoints

Endpoints originales en español (sin prefijo):

| Método | Ruta        | Descripción                                      |
|--------|-------------|--------------------------------------------------|
| GET    | `/salud`    | Estado del servicio y verificación del modelo     |
| GET    | `/esquema`  | Esquema de variables de entrada y enfermedades    |
| POST   | `/predecir` | Predicción multietiqueta completa                 |
| POST   | `/explicar` | Explicación LIME *(estructura lista, pendiente)*  |

Endpoints en inglés bajo `/api/v1`, consumidos por el frontend React:

| Método | Ruta                                | Descripción                                              |
|--------|-------------------------------------|-----------------------------------------------------------|
| GET    | `/health`                           | Alias en inglés de `/salud`                               |
| POST   | `/api/v1/predict`                   | Predicción multietiqueta (payload en inglés)               |
| POST   | `/api/v1/predict-with-explanation`  | Predicción + explicación LIME en una sola llamada           |
| POST   | `/api/v1/explain`                   | Explicación LIME real de un paciente                        |
| GET    | `/api/v1/model-info`                | Metadatos del modelo (sin métricas: no hay modelo entrenado real ni test set) |
| GET    | `/api/v1/feature-importance`        | Importancia global de variables por enfermedad (LIME, cacheada) |
| POST   | `/api/v1/datasets/upload`           | Sube un dataset (.csv/.xlsx/.xls), persiste metadata en sqlite |
| GET    | `/api/v1/datasets/history`          | Historial de datasets subidos                               |
| DELETE | `/api/v1/datasets/{id}`             | Elimina un dataset (archivo + registro)                     |
| GET    | `/api/v1/dashboard/summary`         | Totales de predicciones realizadas                          |
| GET    | `/api/v1/dashboard/disease-distribution` | Frecuencia de cada enfermedad como diagnóstico principal |
| GET    | `/api/v1/dashboard/timeline`        | Predicciones por día                                        |

**Nota sobre explicabilidad**: no existe la librería `shap` instalada ni un dataset de entrenamiento real (los artefactos en `artifacts/` son sintéticos, ver `generate_test_artifacts.py`). El campo `shap_values` de `/api/v1/explain` y `/api/v1/predict-with-explanation` reutiliza los pesos calculados por LIME como aproximación — no son valores SHAP reales. Los datos de `/api/v1/dashboard/*` y `/api/v1/datasets/*` se persisten en `data/app.db` (sqlite) y `data/datasets/` (carpeta ignorada por git).

---

## Ejemplo de petición

### `POST /predecir`

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": 0,
    "ESR": 30.5,
    "CRP": 12.0,
    "RF": 20.0,
    "Anti-CCP": 18.0,
    "HLA-B27": 1,
    "ANA": 1,
    "Anti-Ro": 0,
    "Anti-La": 0,
    "Anti-dsDNA": 1,
    "Anti-Sm": 1,
    "C3": 0.7,
    "C4": 0.1
  }'
```

### Respuesta esperada (ejemplo)

```json
{
  "diagnostico_principal": "Systemic Lupus Erythematosus",
  "probabilidad_principal": 0.94,
  "perfil_compatibles": [
    {"enfermedad": "Systemic Lupus Erythematosus", "probabilidad": 0.94},
    {"enfermedad": "Sjögren's Syndrome", "probabilidad": 0.61}
  ],
  "todas_las_probabilidades": [
    {"enfermedad": "Systemic Lupus Erythematosus", "probabilidad": 0.94},
    {"enfermedad": "Sjögren's Syndrome", "probabilidad": 0.61},
    {"enfermedad": "Rheumatoid Arthritis", "probabilidad": 0.23},
    {"enfermedad": "Reactive Arthritis", "probabilidad": 0.12},
    {"enfermedad": "Psoriatic Arthritis", "probabilidad": 0.08},
    {"enfermedad": "Ankylosing Spondylitis", "probabilidad": 0.05},
    {"enfermedad": "Normal", "probabilidad": 0.02}
  ],
  "umbral_usado": 0.5,
  "advertencia": "Resultado de apoyo. Los perfiles con más de una enfermedad son sospechas a descartar por el profesional, no coexistencias confirmadas. No sustituye el juicio clínico."
}
```

### Petición con datos parciales (campos faltantes)

```bash
curl -X POST http://localhost:8000/predecir \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 32,
    "Gender": 1,
    "ESR": null,
    "CRP": null,
    "RF": null,
    "Anti-CCP": null,
    "HLA-B27": 1,
    "ANA": 0,
    "Anti-Ro": null,
    "Anti-La": null,
    "Anti-dsDNA": null,
    "Anti-Sm": null,
    "C3": null,
    "C4": null
  }'
```

---

## Variables de entrada

### Continuas (numéricas)
| Variable  | Descripción                          |
|-----------|--------------------------------------|
| Age       | Edad del paciente                    |
| ESR       | Velocidad de sedimentación (mm/h)    |
| CRP       | Proteína C reactiva (mg/L)           |
| RF        | Factor reumatoide (UI/mL)            |
| Anti-CCP  | Anticuerpos anti-CCP                 |
| C3        | Complemento C3 (g/L)                |
| C4        | Complemento C4 (g/L)                |

### Binarias (0 o 1)
| Variable   | Descripción                         |
|------------|-------------------------------------|
| Gender     | Género (0=F, 1=M)                  |
| HLA-B27    | HLA-B27                            |
| ANA        | Anticuerpos antinucleares           |
| Anti-Ro    | Anti-Ro/SSA                         |
| Anti-La    | Anti-La/SSB                         |
| Anti-dsDNA | Anti-dsDNA                          |
| Anti-Sm    | Anti-Sm                             |

---

## Enfermedades de salida

1. Ankylosing Spondylitis
2. Normal
3. Psoriatic Arthritis
4. Reactive Arthritis
5. Rheumatoid Arthritis
6. Sjögren's Syndrome
7. Systemic Lupus Erythematosus

---

## Estructura del proyecto

```
backend/
├── artifacts/                # Artefactos del modelo (ya provistos)
│   ├── model.pt
│   ├── preprocessor.joblib
│   ├── label_binarizer.joblib
│   ├── config.yaml
│   ├── input_schema.json
│   └── decision_rule.json
├── app/
│   ├── __init__.py
│   ├── main.py               # App FastAPI, CORS, lifespan
│   ├── model.py               # RedAutoinmuneGated + carga de artefactos
│   ├── schemas.py             # Modelos Pydantic entrada/salida
│   ├── inference.py           # Lógica: preprocesar → predecir → perfil
│   └── routers/
│       ├── __init__.py
│       ├── predict.py         # /predecir, /explicar
│       └── health.py          # /salud, /esquema
├── requirements.txt
└── README.md
```
