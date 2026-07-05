"""
schemas.py — Pydantic input and output models for the API.

- PatientInput: 14 clinical variables.
- PredictionResponse: multi-label prediction result.
- PatientRecord / PatientDetail: history response models.
"""

from datetime import datetime
from typing import List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field

Binary = Literal["Positive", "Negative"]

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    """Clinical data of a patient. All fields are optional because the
    preprocessor imputes missing values."""

    first_name: str = Field(..., description="Patient's first name")
    last_name: str = Field(..., description="Patient's last name")
    age: float = Field(..., description="Patient's age")
    gender: Literal["Male", "Female"] = Field(..., description="Patient's gender")
    esr: Optional[float] = Field(None, description="Erythrocyte sedimentation rate (mm/h)")
    crp: Optional[float] = Field(None, description="C-reactive protein (mg/L)")
    rf: Optional[float] = Field(None, description="Rheumatoid factor (IU/mL)")
    anti_ccp: Optional[float] = Field(None, description="Anti-CCP antibodies")
    hla_b27: Optional[Binary] = Field(None, description="HLA-B27")
    ana: Optional[Binary] = Field(None, description="Antinuclear antibodies")
    anti_ro: Optional[Binary] = Field(None, description="Anti-Ro/SSA")
    anti_la: Optional[Binary] = Field(None, description="Anti-La/SSB")
    anti_dsdna: Optional[Binary] = Field(None, description="Anti-dsDNA")
    anti_sm: Optional[Binary] = Field(None, description="Anti-Sm")
    c3: Optional[float] = Field(None, description="Complement C3 (g/L)")
    c4: Optional[float] = Field(None, description="Complement C4 (g/L)")


# ---------------------------------------------------------------------------
# Output — Prediction
# ---------------------------------------------------------------------------
class DiagnosisPrediction(BaseModel):
    disease: str
    probability: float
    is_positive: bool
    threshold_used: float


class PredictionResponse(BaseModel):
    predictions: List[DiagnosisPrediction]
    overlap_syndrome_detected: bool
    missing_features: List[str]
    model_used: str


class ExplainabilityResponse(BaseModel):
    shap_values: dict = Field(default_factory=dict)
    lime_explanation: dict = Field(default_factory=dict)
    top_positive_features: dict = Field(default_factory=dict)
    top_negative_features: dict = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    model_name: str
    input_dim: int
    n_labels: int
    label_names: List[str]
    feature_names: List[str]
    thresholds: List[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class SchemaResponse(BaseModel):
    continuous_features: List[str]
    binary_features: List[str]
    required_features: List[str]
    output_diseases: List[str]


# ---------------------------------------------------------------------------
# Output — Patient History (PostgreSQL)
# ---------------------------------------------------------------------------
class PredictionRecordOut(BaseModel):
    """One disease prediction row."""
    id: UUID
    disease_name: str
    probability: float
    is_positive: bool
    threshold_used: float
    is_primary: bool

    model_config = {"from_attributes": True}


class PatientRecord(BaseModel):
    """Lightweight patient row for list views."""
    id: UUID
    first_name: str
    last_name: str
    age: float
    gender: int
    primary_diagnosis: Optional[str] = None
    primary_probability: Optional[float] = None
    overlap_syndrome_detected: bool = False
    model_used: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PatientDetail(PatientRecord):
    """Full patient with all clinical features + predictions."""
    esr: Optional[float] = None
    crp: Optional[float] = None
    rf: Optional[float] = None
    anti_ccp: Optional[float] = None
    hla_b27: Optional[int] = None
    ana: Optional[int] = None
    anti_ro: Optional[int] = None
    anti_la: Optional[int] = None
    anti_dsdna: Optional[int] = None
    anti_sm: Optional[int] = None
    c3: Optional[float] = None
    c4: Optional[float] = None
    predictions: List[PredictionRecordOut] = []

    model_config = {"from_attributes": True}


class PaginatedPatients(BaseModel):
    """Paginated response for patient history."""
    items: List[PatientRecord]
    total: int
    page: int
    size: int
    pages: int
