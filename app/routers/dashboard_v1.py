"""
dashboard_v1.py — Analítica de predicciones, bajo /api/v1/dashboard.

Endpoints nuevos (sin equivalente previo en el frontend): alimentan la
página Dashboard.tsx con datos agregados del historial de predicciones
reales guardado en sqlite (ver app.db).
"""

from fastapi import APIRouter

from app import db
from app.model import artefactos

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


@router.get("/summary", summary="Totales de predicciones realizadas")
def summary():
    resumen = db.dashboard_summary()
    modelo_usado = ""
    if artefactos.config:
        modelo_usado = str(artefactos.config.get("nombre_modelo", "desconocido"))
    return {
        **resumen,
        "threshold_used": artefactos.umbral,
        "model_used": modelo_usado,
    }


@router.get("/disease-distribution", summary="Frecuencia de cada enfermedad como diagnóstico principal")
def disease_distribution():
    return db.dashboard_disease_distribution()


@router.get("/timeline", summary="Predicciones por día")
def timeline():
    return db.dashboard_timeline()
