"""
feature_importance.py — Importancia global de variables por enfermedad,
para GET /api/v1/feature-importance (usado por la página Explainability).

A diferencia de lime_engine.explain_patient (explicación de UN paciente),
aquí se promedian los pesos de LIME sobre un lote de muestras sintéticas
para obtener una importancia "global" aproximada por enfermedad. Se
calcula perezosamente (primera llamada) y se cachea en memoria, ya que es
determinística mientras no cambien los artefactos del modelo.
"""

import logging
from typing import Dict

import numpy as np

from app.lime_engine import _explicar_una_enfermedad, _get_background
from app.model import artefactos

logger = logging.getLogger(__name__)

_N_MUESTRAS = 50
_importance_cache: Dict[str, Dict[str, float]] = {}


def compute_global_importance() -> Dict[str, Dict[str, float]]:
    global _importance_cache
    if _importance_cache:
        return _importance_cache

    logger.info("Calculando importancia global de variables (primera llamada, puede tardar)…")
    background = _get_background().values[:_N_MUESTRAS]
    nombres = artefactos.nombres_enfermedades

    resultado: Dict[str, Dict[str, float]] = {}
    for idx, enfermedad in enumerate(nombres):
        acumulado: Dict[str, list] = {col: [] for col in artefactos.variables_requeridas}
        for fila in background:
            pares = _explicar_una_enfermedad(fila, idx)
            for feature, peso in pares:
                # Las variables binarias vienen como "Gender=1"/"Gender=0"
                # (LIME las trata como categóricas); se normaliza al nombre
                # base para poder promediar entre muestras con valores distintos.
                nombre_base = feature.split("=")[0]
                acumulado[nombre_base].append(peso)
        resultado[enfermedad] = {
            feature: float(np.mean(pesos)) if pesos else 0.0
            for feature, pesos in acumulado.items()
        }

    _importance_cache = resultado
    logger.info("✅ Importancia global calculada y cacheada.")
    return _importance_cache
