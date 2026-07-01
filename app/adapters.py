"""
adapters.py — Traducción entre el contrato en inglés que espera el frontend
(/api/v1/*) y los modelos Pydantic en español ya existentes (app.schemas).

No modifica app/inference.py ni app/schemas.py: solo traduce entrada y salida
para que la lógica de predicción original se reutilice sin cambios.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.model import artefactos
from app.schemas import PacienteInput

Binario = Literal["Positive", "Negative"]

# Nombres de campo del contrato en inglés, en el mismo orden que las 14
# variables clínicas. Se usan tanto para detectar valores faltantes como
# para poblar PacienteInput.
_CAMPOS_OPCIONALES = [
    "esr", "crp", "rf", "anti_ccp", "hla_b27", "ana",
    "anti_ro", "anti_la", "anti_dsdna", "anti_sm", "c3", "c4",
]


class PatientInputEN(BaseModel):
    """Datos clínicos de un paciente en el formato que envía el frontend."""

    age: float
    gender: Literal["Male", "Female"]
    esr: Optional[float] = None
    crp: Optional[float] = None
    rf: Optional[float] = None
    anti_ccp: Optional[float] = None
    hla_b27: Optional[Binario] = None
    ana: Optional[Binario] = None
    anti_ro: Optional[Binario] = None
    anti_la: Optional[Binario] = None
    anti_dsdna: Optional[Binario] = None
    anti_sm: Optional[Binario] = None
    c3: Optional[float] = None
    c4: Optional[float] = None


def _binario_a_int(valor: Optional[Binario]) -> Optional[int]:
    if valor is None:
        return None
    return 1 if valor == "Positive" else 0


def to_paciente_input(datos: PatientInputEN) -> PacienteInput:
    """Convierte el payload en inglés al esquema interno en español."""
    return PacienteInput(
        Age=datos.age,
        Gender=1 if datos.gender == "Male" else 0,
        ESR=datos.esr,
        CRP=datos.crp,
        RF=datos.rf,
        **{
            "Anti-CCP": datos.anti_ccp,
            "HLA-B27": _binario_a_int(datos.hla_b27),
            "Anti-Ro": _binario_a_int(datos.anti_ro),
            "Anti-La": _binario_a_int(datos.anti_la),
            "Anti-dsDNA": _binario_a_int(datos.anti_dsdna),
            "Anti-Sm": _binario_a_int(datos.anti_sm),
        },
        ANA=_binario_a_int(datos.ana),
        C3=datos.c3,
        C4=datos.c4,
    )


def missing_features(datos: PatientInputEN) -> List[str]:
    """Nombres (en inglés) de los campos opcionales que llegaron como null."""
    valores = datos.model_dump()
    return [campo for campo in _CAMPOS_OPCIONALES if valores.get(campo) is None]


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


class ModelInfoResponse(BaseModel):
    model_name: str
    input_dim: int
    n_labels: int
    label_names: List[str]
    feature_names: List[str]
    thresholds: List[float]


class ExplainabilityResponse(BaseModel):
    shap_values: dict = Field(default_factory=dict)
    lime_explanation: dict = Field(default_factory=dict)
    top_positive_features: dict = Field(default_factory=dict)
    top_negative_features: dict = Field(default_factory=dict)


def to_prediction_response(resultado: dict, faltantes: List[str]) -> PredictionResponse:
    """Traduce el dict devuelto por app.inference.predecir() al contrato
    PredictionResponse que consume Diagnosis.tsx."""
    umbral = resultado["umbral_usado"]

    predictions = [
        DiagnosisPrediction(
            disease=item.enfermedad,
            probability=item.probabilidad,
            is_positive=item.probabilidad >= umbral,
            threshold_used=umbral,
        )
        for item in resultado["todas_las_probabilidades"]
    ]

    compatibles_sin_normal = [
        c for c in resultado["perfil_compatibles"] if c.enfermedad != "Normal"
    ]
    overlap_syndrome_detected = len(compatibles_sin_normal) > 1

    modelo_usado = ""
    if artefactos.config:
        modelo_usado = str(artefactos.config.get("nombre_modelo", "desconocido"))

    return PredictionResponse(
        predictions=predictions,
        overlap_syndrome_detected=overlap_syndrome_detected,
        missing_features=faltantes,
        model_used=modelo_usado,
    )
