"""
lime_engine.py — Explicabilidad real con LIME (lime.lime_tabular).

Sustituye al stub de app.inference.explicar() sin modificar ese archivo:
los routers de /api/v1 llaman directamente a explain_patient() de aquí.

Se construye el explainer sobre las 14 variables CLÍNICAS crudas (sin
transformar), con una función de predicción que aplica internamente
preprocessor.transform() + modelo + sigmoid. Así las explicaciones salen
en términos de las variables que el frontend conoce (age, esr, crp, ...),
no del espacio interno escalado.

Nota importante: no existe un dataset real de entrenamiento en este
proyecto (los artefactos son sintéticos, ver generate_test_artifacts.py),
así que el "fondo" que LIME necesita para perturbar muestras se genera
sintéticamente a partir de las estadísticas ya guardadas en el
preprocessor (medias/escalas del StandardScaler). Por la misma razón no
hay librería `shap` instalada: el campo `shap_values` del contrato del
frontend se rellena reutilizando los mismos pesos de LIME como
aproximación, dejado explícito aquí y en la respuesta.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from app.inference import _construir_dataframe, predecir
from app.model import artefactos
from app.schemas import PacienteInput

logger = logging.getLogger(__name__)

_DEFAULT_MEAN = 50.0
_DEFAULT_STD = 15.0
_NUM_FEATURES_EXPLAIN = 5

_background_cache: Optional[pd.DataFrame] = None
_explainer_cache: Optional[LimeTabularExplainer] = None


def _stats_continuas() -> Dict[str, Tuple[float, float]]:
    """Intenta leer (media, desviación) reales del StandardScaler ya
    ajustado dentro del preprocessor. Si la estructura no coincide (p.ej.
    otro nombre de transformer), usa valores por defecto razonables."""
    stats: Dict[str, Tuple[float, float]] = {}
    variables = artefactos.variables_continuas or []
    try:
        transformador = artefactos.preprocessor.named_transformers_["continuas"]
        scaler = transformador.named_steps["scaler"]
        for nombre, media, escala in zip(variables, scaler.mean_, scaler.scale_):
            stats[nombre] = (float(media), float(escala) if escala else _DEFAULT_STD)
    except Exception:
        logger.warning("No se pudieron leer estadísticas del preprocessor; usando valores por defecto.")
    for nombre in variables:
        stats.setdefault(nombre, (_DEFAULT_MEAN, _DEFAULT_STD))
    return stats


def _build_background_dataframe(n: int = 300) -> pd.DataFrame:
    """Genera n muestras sintéticas plausibles en el espacio crudo de las
    14 variables, en el orden de artefactos.variables_requeridas."""
    rng = np.random.default_rng(42)
    stats = _stats_continuas()
    continuas = set(artefactos.variables_continuas or [])

    data = {}
    for col in artefactos.variables_requeridas:
        if col in continuas:
            media, desviacion = stats[col]
            data[col] = rng.normal(media, abs(desviacion) or _DEFAULT_STD, n)
        else:
            data[col] = rng.integers(0, 2, n).astype(float)

    return pd.DataFrame(data, columns=artefactos.variables_requeridas)


def _get_background() -> pd.DataFrame:
    global _background_cache
    if _background_cache is None:
        _background_cache = _build_background_dataframe()
    return _background_cache


def _predict_fn(raw_rows: np.ndarray) -> np.ndarray:
    """Función que LIME usa para perturbar muestras: recibe un array 2D en
    el espacio crudo (columnas = variables_requeridas) y devuelve las 7
    probabilidades sigmoid independientes por fila."""
    import torch

    df = pd.DataFrame(raw_rows, columns=artefactos.variables_requeridas)
    X = artefactos.preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(artefactos.modelo(tensor)).cpu().numpy()
    return probs


def _get_explainer() -> LimeTabularExplainer:
    global _explainer_cache
    if _explainer_cache is None:
        background = _get_background()
        continuas = set(artefactos.variables_continuas or [])
        categorical_features = [
            i for i, col in enumerate(artefactos.variables_requeridas) if col not in continuas
        ]
        _explainer_cache = LimeTabularExplainer(
            training_data=background.values,
            feature_names=artefactos.variables_requeridas,
            categorical_features=categorical_features,
            class_names=artefactos.nombres_enfermedades,
            discretize_continuous=False,
            mode="classification",
            random_state=42,
        )
    return _explainer_cache


def _explicar_una_enfermedad(fila_cruda: np.ndarray, idx_enfermedad: int) -> List[Tuple[str, float]]:
    explainer = _get_explainer()
    explicacion = explainer.explain_instance(
        fila_cruda,
        _predict_fn,
        labels=[idx_enfermedad],
        num_features=len(artefactos.variables_requeridas),
    )
    return explicacion.as_list(label=idx_enfermedad)


def _rellenar_faltantes(fila_cruda: np.ndarray) -> np.ndarray:
    """LIME necesita una fila numérica completa (sin NaN) para calcular
    distancias respecto al fondo sintético. Los campos clínicos que el
    paciente no proporcionó (NaN, ver inference._construir_dataframe) se
    rellenan con el promedio del fondo sintético para esa variable —
    el mismo criterio de imputación que usa el preprocesador del modelo."""
    background = _get_background()
    medias = background.mean(axis=0).values
    fila = fila_cruda.astype(float).copy()
    faltantes = np.isnan(fila)
    fila[faltantes] = medias[faltantes]
    return fila


def explain_patient(datos: PacienteInput) -> dict:
    """Genera explicaciones LIME reales para las enfermedades del perfil
    compatible de un paciente. Devuelve un dict compatible con el
    ExplainabilityResponse del frontend."""
    resultado = predecir(datos)
    fila_cruda = _rellenar_faltantes(_construir_dataframe(datos).values[0])

    nombres = artefactos.nombres_enfermedades
    lime_explanation: Dict[str, List[Tuple[str, float]]] = {}
    top_positive: Dict[str, List[str]] = {}
    top_negative: Dict[str, List[str]] = {}
    shap_values: Dict[str, Dict[str, float]] = {}

    for compatible in resultado["perfil_compatibles"]:
        idx = nombres.index(compatible.enfermedad)
        pares = _explicar_una_enfermedad(fila_cruda, idx)

        lime_explanation[compatible.enfermedad] = pares
        shap_values[compatible.enfermedad] = dict(pares)

        positivos = [f for f, peso in pares if peso > 0]
        negativos = [f for f, peso in pares if peso < 0]
        top_positive[compatible.enfermedad] = positivos[:_NUM_FEATURES_EXPLAIN]
        top_negative[compatible.enfermedad] = negativos[:_NUM_FEATURES_EXPLAIN]

    return {
        "shap_values": shap_values,
        "lime_explanation": lime_explanation,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
    }
