"""
schemas.py — Modelos Pydantic de entrada y salida para la API.

- PacienteInput: 14 variables clínicas (todas opcionales para reflejar la
  realidad clínica de que no siempre se piden todos los exámenes).
- PrediccionOutput: resultado completo de la predicción multietiqueta.
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------
class PacienteInput(BaseModel):
    """Datos clínicos de un paciente. Todos los campos son opcionales (null
    permitido) porque el preprocesador imputa valores faltantes."""

    # Variables continuas
    Age: Optional[float] = Field(None, description="Edad del paciente")
    ESR: Optional[float] = Field(None, description="Velocidad de sedimentación globular (mm/h)")
    CRP: Optional[float] = Field(None, description="Proteína C reactiva (mg/L)")
    RF: Optional[float] = Field(None, description="Factor reumatoide (UI/mL)")
    Anti_CCP: Optional[float] = Field(None, alias="Anti-CCP", description="Anticuerpos anti-CCP")
    C3: Optional[float] = Field(None, description="Complemento C3 (g/L)")
    C4: Optional[float] = Field(None, description="Complemento C4 (g/L)")

    # Variables binarias (0 o 1)
    Gender: Optional[int] = Field(None, description="Género (0=F, 1=M)")
    HLA_B27: Optional[int] = Field(None, alias="HLA-B27", description="HLA-B27 (0 o 1)")
    ANA: Optional[int] = Field(None, description="Anticuerpos antinucleares (0 o 1)")
    Anti_Ro: Optional[int] = Field(None, alias="Anti-Ro", description="Anti-Ro/SSA (0 o 1)")
    Anti_La: Optional[int] = Field(None, alias="Anti-La", description="Anti-La/SSB (0 o 1)")
    Anti_dsDNA: Optional[int] = Field(None, alias="Anti-dsDNA", description="Anti-dsDNA (0 o 1)")
    Anti_Sm: Optional[int] = Field(None, alias="Anti-Sm", description="Anti-Sm (0 o 1)")

    model_config = {"populate_by_name": True}

    @field_validator(
        "Gender", "HLA_B27", "ANA", "Anti_Ro", "Anti_La", "Anti_dsDNA", "Anti_Sm",
        mode="before",
    )
    @classmethod
    def validar_binarios(cls, v, info):
        if v is None:
            return v
        if v not in (0, 1):
            raise ValueError(
                f"El campo '{info.field_name}' debe ser 0 o 1, se recibió: {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------
class EnfermedadProbabilidad(BaseModel):
    enfermedad: str
    probabilidad: float


class PrediccionOutput(BaseModel):
    diagnostico_principal: str
    probabilidad_principal: float
    perfil_compatibles: List[EnfermedadProbabilidad]
    todas_las_probabilidades: List[EnfermedadProbabilidad]
    umbral_usado: float
    advertencia: str


class ExplicacionVariable(BaseModel):
    variable: str
    peso: float


class ExplicacionEnfermedad(BaseModel):
    enfermedad: str
    a_favor: List[ExplicacionVariable]
    en_contra: List[ExplicacionVariable]


class ExplicacionOutput(BaseModel):
    diagnostico_principal: str
    explicaciones: List[ExplicacionEnfermedad]
    advertencia: str


# ---------------------------------------------------------------------------
# Respuestas de /salud y /esquema
# ---------------------------------------------------------------------------
class SaludResponse(BaseModel):
    estado: str
    modelo_cargado: bool
    version_modelo: str


class EsquemaResponse(BaseModel):
    variables_continuas: List[str]
    variables_binarias: List[str]
    variables_requeridas: List[str]
    enfermedades_salida: List[str]
