"""
inference.py — Lógica de predicción multietiqueta.

Secuencia exacta:
1. Construir DataFrame con las columnas en el orden de input_schema.
2. preprocessor.transform(df)  — NUNCA fit_transform.
3. Tensor float32 → modelo en eval() + no_grad() → 7 logits.
4. torch.sigmoid → 7 probabilidades INDEPENDIENTES (NO softmax).
5. argmax → diagnóstico principal.
6. umbral → perfil de compatibles (el principal siempre se incluye).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from app.model import artefactos
from app.schemas import EnfermedadProbabilidad, PacienteInput

logger = logging.getLogger(__name__)

# Advertencia legal que acompaña cada predicción
ADVERTENCIA = (
    "Resultado de apoyo. Los perfiles con más de una enfermedad son sospechas a "
    "descartar por el profesional, no coexistencias confirmadas. No sustituye el "
    "juicio clínico."
)


def _construir_dataframe(datos: PacienteInput) -> pd.DataFrame:
    """Construye un DataFrame de 1 fila con las columnas en el orden exacto
    definido por input_schema.json (variables_requeridas)."""
    # Convertir el modelo Pydantic a dict usando los alias originales
    datos_dict = datos.model_dump(by_alias=True)

    fila: Dict[str, Optional[float]] = {}
    for col in artefactos.variables_requeridas:
        fila[col] = datos_dict.get(col)

    df = pd.DataFrame([fila], columns=artefactos.variables_requeridas)
    return df


def _preprocesar(df: pd.DataFrame) -> torch.Tensor:
    """Aplica el ColumnTransformer pre-ajustado y devuelve un tensor float32."""
    X = artefactos.preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return torch.tensor(X, dtype=torch.float32)


def _inferir(tensor: torch.Tensor) -> np.ndarray:
    """Pasa el tensor por el modelo (eval + no_grad) y devuelve probabilidades
    sigmoid INDEPENDIENTES (NO softmax)."""
    with torch.no_grad():
        logits = artefactos.modelo(tensor)          # (1, 7)
        probabilidades = torch.sigmoid(logits)       # sigmoide independiente
    return probabilidades.cpu().numpy()[0]           # shape (7,)


def predecir(datos: PacienteInput) -> dict:
    """Pipeline completo de predicción. Devuelve un dict listo para serializar."""
    # 1-2. Construir DataFrame
    df = _construir_dataframe(datos)
    logger.debug("DataFrame construido: %s", df.to_dict(orient="records"))

    # 3. Preprocesar
    tensor = _preprocesar(df)

    # 4-5. Inferir (sigmoid independiente)
    probs = _inferir(tensor)

    # Redondear a 4 decimales (convertir a float Python para evitar ruido float32)
    probs = [round(float(p), 4) for p in probs]

    nombres = artefactos.nombres_enfermedades
    umbral = artefactos.umbral

    # 7. Diagnóstico principal = argmax
    idx_principal = int(np.argmax(probs))
    diagnostico_principal = nombres[idx_principal]
    prob_principal = probs[idx_principal]

    # 8. Perfil de compatibles (>= umbral + siempre el principal)
    compatibles: List[EnfermedadProbabilidad] = []
    for i, (nombre, p) in enumerate(zip(nombres, probs)):
        if p >= umbral or i == idx_principal:
            compatibles.append(EnfermedadProbabilidad(enfermedad=nombre, probabilidad=p))

    # Ordenar compatibles de mayor a menor probabilidad
    compatibles.sort(key=lambda x: x.probabilidad, reverse=True)

    # 9. Todas las probabilidades ordenadas de mayor a menor
    todas = [
        EnfermedadProbabilidad(enfermedad=n, probabilidad=p)
        for n, p in zip(nombres, probs)
    ]
    todas.sort(key=lambda x: x.probabilidad, reverse=True)

    return {
        "diagnostico_principal": diagnostico_principal,
        "probabilidad_principal": prob_principal,
        "perfil_compatibles": compatibles,
        "todas_las_probabilidades": todas,
        "umbral_usado": umbral,
        "advertencia": ADVERTENCIA,
    }


def explicar(datos: PacienteInput) -> dict:
    """Genera explicaciones LIME para las enfermedades del perfil compatible.

    TODO: Integrar lime.lime_tabular.LimeTabularExplainer para producir
    explicaciones reales. Actualmente devuelve la estructura de respuesta
    con un placeholder indicando que LIME no está integrado aún.
    """
    # Primero obtenemos la predicción normal para saber el perfil
    resultado_prediccion = predecir(datos)

    explicaciones = []
    for enfermedad_prob in resultado_prediccion["perfil_compatibles"]:
        explicaciones.append({
            "enfermedad": enfermedad_prob.enfermedad,
            "a_favor": [],
            "en_contra": [],
        })

    return {
        "diagnostico_principal": resultado_prediccion["diagnostico_principal"],
        "explicaciones": explicaciones,
        "advertencia": (
            "Explicación LIME pendiente de integración. La estructura de "
            "respuesta está lista; falta conectar lime.lime_tabular."
        ),
    }
