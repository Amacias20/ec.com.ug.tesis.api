"""
api.py — Consolidated English endpoints consumed by the React frontend.
(Diagnosis, ModelInfo, Explainability, Health).
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud, feature_importance, lime_engine
from app.database import get_db
from app.inference import predict
from app.model import artifacts
from app.schemas import (
    ExplainabilityResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientInput,
    PredictionResponse,
    SchemaResponse,
    ThresholdsUpdate,
    ThresholdsResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["API v1"])


@router.post("/predict", response_model=PredictionResponse, summary="Multi-label Prediction")
def predict_endpoint(patient: PatientInput, db: Session = Depends(get_db)):
    result = predict(patient, db)
    crud.create_patient_with_predictions(db, patient, result)
    return result


@router.post(
    "/predict-with-explanation",
    summary="Prediction + LIME explanation in a single call",
)
def predict_with_explanation(patient: PatientInput, db: Session = Depends(get_db)):
    result = predict(patient, db)
    crud.create_patient_with_predictions(db, patient, result)
    explanation = lime_engine.explain_patient(patient, db)
    return {"prediction": result, "explanation": explanation}


@router.post("/explain", response_model=ExplainabilityResponse, summary="LIME explanation of a patient")
def explain_endpoint(patient: PatientInput, db: Session = Depends(get_db)):
    return lime_engine.explain_patient(patient, db)


@router.get("/model-info", response_model=ModelInfoResponse, summary="Model Information")
def model_info(db: Session = Depends(get_db)):
    model_name = "unknown"
    if artifacts.config:
        model_name = str(
            artifacts.config.get("nombre_seleccion")
            or artifacts.config.get("nombre_modelo")
            or artifacts.config.get("arquitectura")
            or "unknown"
        )

    n_labels = len(artifacts.disease_names or [])
    
    db_thresholds = crud.get_thresholds_dict(db)
    if not db_thresholds:
        db_thresholds = artifacts.decision_rule.get("umbrales_optimos", {})
        
    thresholds = []
    for d in (artifacts.disease_names or []):
        thresholds.append(float(db_thresholds.get(d, artifacts.threshold)))

    return ModelInfoResponse(
        model_name=model_name,
        input_dim=int(artifacts.config.get("input_dim")) if artifacts.config else 0,
        n_labels=n_labels,
        label_names=artifacts.disease_names or [],
        feature_names=artifacts.required_features or [],
        thresholds=thresholds,
    )


@router.get("/feature-importance", summary="Global feature importance per disease")
def feature_importance_endpoint():
    return feature_importance.compute_global_importance()
@router.get("/health", response_model=HealthResponse, summary="Service health status")
def health():
    """Verifies that the service is active and the model was loaded correctly."""
    version = "unknown"
    if artifacts.config:
        version = str(
            artifacts.config.get("nombre_seleccion")
            or artifacts.config.get("nombre_modelo")
            or artifacts.config.get("arquitectura")
            or "unknown"
        )
    return HealthResponse(
        status="ok",
        model_loaded=artifacts.loaded,
        model_version=version,
    )


@router.get("/schema", response_model=SchemaResponse, summary="Input schema")
def schema_endpoint():
    """Returns the input features schema and the output diseases
    so the frontend knows which fields to present to the user."""
    return SchemaResponse(
        continuous_features=artifacts.continuous_features or [],
        binary_features=artifacts.binary_features or [],
        required_features=artifacts.required_features or [],
        output_diseases=artifacts.disease_names or [],
    )

@router.get("/thresholds", response_model=ThresholdsResponse, summary="Get current optimal thresholds")
def get_thresholds(db: Session = Depends(get_db)):
    db_thresholds = crud.get_thresholds_dict(db)
    if not db_thresholds:
        db_thresholds = artifacts.decision_rule.get("umbrales_optimos", {})
    return ThresholdsResponse(thresholds=db_thresholds)

@router.put("/thresholds", summary="Update optimal thresholds")
def update_thresholds(payload: ThresholdsUpdate, db: Session = Depends(get_db)):
    crud.upsert_thresholds(db, payload.thresholds)
    return {"status": "success", "message": "Umbrales actualizados correctamente"}
