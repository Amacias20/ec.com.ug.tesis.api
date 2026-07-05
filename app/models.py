"""
models.py — SQLAlchemy ORM models (code-first).

Tables are auto-created via Base.metadata.create_all() in main.py.

Tables:
  patients           — 14 clinical features + metadata for each evaluation.
  prediction_records — One row per disease per prediction (7 rows per patient).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Patient(Base):
    """Stores the clinical input features submitted for each evaluation."""

    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Clinical features — continuous
    age = Column(Float, nullable=False)
    gender = Column(Integer, nullable=False, comment="1=Male, 0=Female")
    esr = Column(Float, nullable=True, comment="Erythrocyte sedimentation rate (mm/h)")
    crp = Column(Float, nullable=True, comment="C-reactive protein (mg/L)")
    rf = Column(Float, nullable=True, comment="Rheumatoid factor (IU/mL)")
    anti_ccp = Column(Float, nullable=True, comment="Anti-CCP antibodies")
    c3 = Column(Float, nullable=True, comment="Complement C3 (g/L)")
    c4 = Column(Float, nullable=True, comment="Complement C4 (g/L)")

    # Clinical features — binary
    hla_b27 = Column(Integer, nullable=True, comment="0=Negative, 1=Positive")
    ana = Column(Integer, nullable=True, comment="0=Negative, 1=Positive")
    anti_ro = Column(Integer, nullable=True, comment="0=Negative, 1=Positive")
    anti_la = Column(Integer, nullable=True, comment="0=Negative, 1=Positive")
    anti_dsdna = Column(Integer, nullable=True, comment="0=Negative, 1=Positive")
    anti_sm = Column(Integer, nullable=True, comment="0=Negative, 1=Positive")

    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Aggregated result fields (denormalized for quick queries)
    primary_diagnosis = Column(String(100), nullable=True)
    primary_probability = Column(Float, nullable=True)
    overlap_syndrome_detected = Column(Boolean, default=False)
    model_used = Column(String(100), nullable=True)

    # Relationship
    predictions = relationship(
        "PredictionRecord",
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="PredictionRecord.probability.desc()",
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, age={self.age}, primary={self.primary_diagnosis})>"


class PredictionRecord(Base):
    """One row per disease per prediction — 7 rows per patient evaluation."""

    __tablename__ = "prediction_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)

    disease_name = Column(String(100), nullable=False)
    probability = Column(Float, nullable=False)
    is_positive = Column(Boolean, nullable=False, comment="probability >= threshold")
    threshold_used = Column(Float, nullable=False)
    is_primary = Column(Boolean, default=False, comment="True if this is the argmax disease")

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    patient = relationship("Patient", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<PredictionRecord(disease={self.disease_name}, p={self.probability:.3f})>"
