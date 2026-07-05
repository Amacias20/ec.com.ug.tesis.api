"""
patients.py — Patient history CRUD, under /api/v1/patients.

Provides paginated listing, detail view, and deletion of patient
evaluations stored in PostgreSQL.
"""

import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db
from app.schemas import PaginatedPatients, PatientDetail, PatientRecord

router = APIRouter(prefix="/api/v1/patients", tags=["Patients"])


@router.get("", response_model=PaginatedPatients, summary="Paginated patient history")
def list_patients(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
):
    """Returns a paginated list of patient evaluations, most recent first."""
    skip = (page - 1) * size
    patients, total = crud.get_patients(db, skip=skip, limit=size)
    return PaginatedPatients(
        items=[PatientRecord.model_validate(p) for p in patients],
        total=total,
        page=page,
        size=size,
        pages=math.ceil(total / size) if total > 0 else 0,
    )


@router.get("/{patient_id}", response_model=PatientDetail, summary="Patient detail with predictions")
def get_patient(patient_id: UUID, db: Session = Depends(get_db)):
    """Returns a single patient with all clinical features and prediction records."""
    patient = crud.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientDetail.model_validate(patient)


@router.delete("/{patient_id}", summary="Delete a patient record")
def delete_patient(patient_id: UUID, db: Session = Depends(get_db)):
    """Deletes a patient and cascades to all prediction records."""
    deleted = crud.delete_patient(db, patient_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"message": "Patient deleted successfully"}
