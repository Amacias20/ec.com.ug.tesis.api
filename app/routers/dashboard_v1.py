"""
dashboard.py — Prediction analytics, under /api/v1/dashboard.

Endpoints feeding the Dashboard.tsx page with aggregated data
from the patient predictions history stored in PostgreSQL.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.model import artifacts

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


@router.get("/summary", summary="Total predictions made")
def summary(db: Session = Depends(get_db)):
    resume = crud.dashboard_summary(db)
    model_used = ""
    if artifacts.config:
        model_used = str(artifacts.config.get("nombre_modelo", "unknown"))
    return {
        **resume,
        "threshold_used": artifacts.threshold,
        "model_used": model_used,
    }


@router.get("/disease-distribution", summary="Frequency of each disease as primary diagnosis")
def disease_distribution(db: Session = Depends(get_db)):
    return crud.dashboard_disease_distribution(db)


@router.get("/timeline", summary="Predictions per day")
def timeline(db: Session = Depends(get_db)):
    return crud.dashboard_timeline(db)
