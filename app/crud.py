"""
crud.py — Create / Read / Delete operations for patients and predictions.

All functions receive an active SQLAlchemy Session from the FastAPI dependency.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import func, cast, Date
from sqlalchemy.orm import Session

from app.models import Patient, PredictionRecord
from app.schemas import PatientInput


def _binary_to_int(val: Optional[str]) -> Optional[int]:
    """Converts 'Positive'/'Negative' strings to 1/0."""
    if val is None:
        return None
    return 1 if val == "Positive" else 0


def create_patient_with_predictions(
    db: Session,
    patient_input: PatientInput,
    prediction_result: dict,
) -> Patient:
    """Persists a patient and all 7 disease prediction records in one transaction."""

    data = patient_input.model_dump()

    patient = Patient(
        age=data["age"],
        gender=1 if data["gender"] == "Male" else 0,
        esr=data.get("esr"),
        crp=data.get("crp"),
        rf=data.get("rf"),
        anti_ccp=data.get("anti_ccp"),
        hla_b27=_binary_to_int(data.get("hla_b27")),
        ana=_binary_to_int(data.get("ana")),
        anti_ro=_binary_to_int(data.get("anti_ro")),
        anti_la=_binary_to_int(data.get("anti_la")),
        anti_dsdna=_binary_to_int(data.get("anti_dsdna")),
        anti_sm=_binary_to_int(data.get("anti_sm")),
        c3=data.get("c3"),
        c4=data.get("c4"),
        # Denormalized summary
        primary_diagnosis=prediction_result["primary_diagnosis"],
        primary_probability=prediction_result["primary_probability"],
        overlap_syndrome_detected=prediction_result["overlap_syndrome_detected"],
        model_used=prediction_result.get("model_used", "unknown"),
    )

    db.add(patient)
    db.flush()  # get patient.id without committing

    threshold = prediction_result["threshold_used"]
    primary = prediction_result["primary_diagnosis"]

    for item in prediction_result["all_probabilities"]:
        record = PredictionRecord(
            patient_id=patient.id,
            disease_name=item["disease"],
            probability=item["probability"],
            is_positive=item["probability"] >= threshold,
            threshold_used=threshold,
            is_primary=(item["disease"] == primary),
        )
        db.add(record)

    db.commit()
    db.refresh(patient)
    return patient


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
def get_patients(db: Session, skip: int = 0, limit: int = 20) -> tuple[list[Patient], int]:
    """Returns a paginated list of patients (most recent first) and total count."""
    total = db.query(func.count(Patient.id)).scalar()
    patients = (
        db.query(Patient)
        .order_by(Patient.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return patients, total


def get_patient(db: Session, patient_id: UUID) -> Optional[Patient]:
    """Returns a single patient with its prediction records."""
    return db.query(Patient).filter(Patient.id == patient_id).first()


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------
def delete_patient(db: Session, patient_id: UUID) -> bool:
    """Deletes a patient and cascades to prediction records. Returns True if found."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient is None:
        return False
    db.delete(patient)
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Dashboard aggregations
# ---------------------------------------------------------------------------
def dashboard_summary(db: Session) -> dict:
    """Total predictions, overlap count, and last prediction timestamp."""
    total = db.query(func.count(Patient.id)).scalar() or 0
    overlap_count = (
        db.query(func.count(Patient.id))
        .filter(Patient.overlap_syndrome_detected == True)
        .scalar()
        or 0
    )
    last_at = db.query(func.max(Patient.created_at)).scalar()
    return {
        "total_predictions": total,
        "overlap_syndrome_count": overlap_count,
        "last_prediction_at": last_at.isoformat() if last_at else None,
    }


def dashboard_disease_distribution(db: Session) -> dict[str, int]:
    """Frequency of each disease as primary diagnosis."""
    rows = (
        db.query(Patient.primary_diagnosis, func.count(Patient.id).label("total"))
        .group_by(Patient.primary_diagnosis)
        .order_by(func.count(Patient.id).desc())
        .all()
    )
    return {row.primary_diagnosis: row.total for row in rows if row.primary_diagnosis}


def dashboard_timeline(db: Session) -> list[dict]:
    """Predictions per day."""
    rows = (
        db.query(
            cast(Patient.created_at, Date).label("day"),
            func.count(Patient.id).label("total"),
        )
        .group_by("day")
        .order_by("day")
        .all()
    )
    return [{"date": str(row.day), "count": row.total} for row in rows]
