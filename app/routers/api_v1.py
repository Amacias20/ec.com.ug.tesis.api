"""
api_v1.py — Endpoints en inglés bajo /api/v1 que consume el frontend React
ya construido (Diagnosis, ModelInfo, Explainability).

Reutiliza la lógica existente de app.inference y app.lime_engine; solo
traduce entrada/salida (ver app.adapters) y registra cada predicción real
en el historial (app.db) para alimentar el dashboard de analítica.
"""

import logging

from fastapi import APIRouter

from app import db, feature_importance, lime_engine
from app.adapters import (
    ExplainabilityResponse,
    ModelInfoResponse,
    PatientInputEN,
    PredictionResponse,
    missing_features,
    to_paciente_input,
    to_prediction_response,
)
from app.inference import predecir
from app.model import artefactos

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["API v1"])


@router.post("/predict", response_model=PredictionResponse, summary="Predicción multietiqueta (EN)")
def predict(paciente: PatientInputEN):
    paciente_es = to_paciente_input(paciente)
    resultado = predecir(paciente_es)
    db.log_prediction(resultado)
    return to_prediction_response(resultado, missing_features(paciente))


@router.post(
    "/predict-with-explanation",
    summary="Predicción + explicación LIME en una sola llamada",
)
def predict_with_explanation(paciente: PatientInputEN):
    paciente_es = to_paciente_input(paciente)
    resultado = predecir(paciente_es)
    db.log_prediction(resultado)
    prediction = to_prediction_response(resultado, missing_features(paciente))
    explanation = lime_engine.explain_patient(paciente_es)
    return {"prediction": prediction, "explanation": explanation}


@router.post("/explain", response_model=ExplainabilityResponse, summary="Explicación LIME de un paciente")
def explain(paciente: PatientInputEN):
    paciente_es = to_paciente_input(paciente)
    return lime_engine.explain_patient(paciente_es)


@router.get("/model-info", response_model=ModelInfoResponse, summary="Información del modelo")
def model_info():
    nombre_modelo = "desconocido"
    if artefactos.config:
        nombre_modelo = str(artefactos.config.get("nombre_modelo", "desconocido"))

    n_labels = len(artefactos.nombres_enfermedades or [])
    return ModelInfoResponse(
        model_name=nombre_modelo,
        input_dim=int(artefactos.config.get("input_dim")) if artefactos.config else 0,
        n_labels=n_labels,
        label_names=artefactos.nombres_enfermedades or [],
        feature_names=artefactos.variables_requeridas or [],
        thresholds=[artefactos.umbral] * n_labels,
    )


@router.get("/feature-importance", summary="Importancia global de variables por enfermedad")
def feature_importance_endpoint():
    return feature_importance.compute_global_importance()
