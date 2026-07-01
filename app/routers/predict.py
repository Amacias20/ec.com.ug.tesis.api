"""
predict.py — Endpoints de predicción y explicación.

POST /predecir  → predicción multietiqueta completa.
POST /explicar  → explicación LIME (estructura lista, integración pendiente).
"""

from fastapi import APIRouter

from app.inference import explicar, predecir
from app.schemas import ExplicacionOutput, PacienteInput, PrediccionOutput

router = APIRouter(tags=["Predicción"])


@router.post(
    "/predecir",
    response_model=PrediccionOutput,
    summary="Predicción multietiqueta",
)
def endpoint_predecir(paciente: PacienteInput):
    """Recibe las variables clínicas de un paciente y devuelve:
    - Diagnóstico principal (enfermedad con mayor probabilidad).
    - Perfil de enfermedades compatibles (probabilidad ≥ umbral).
    - Las 7 probabilidades independientes ordenadas de mayor a menor.

    Todos los campos son opcionales (null); el preprocesador imputa faltantes.
    """
    return predecir(paciente)


@router.post(
    "/explicar",
    response_model=ExplicacionOutput,
    summary="Explicación de la predicción (LIME)",
)
def endpoint_explicar(paciente: PacienteInput):
    """Misma entrada que /predecir. Devuelve la explicación LIME de la
    predicción: para cada enfermedad del perfil compatible, las variables
    que más contribuyeron a favor y en contra.

    > **Nota**: La integración completa de LIME está pendiente (TODO).
    > La estructura de respuesta está lista.
    """
    return explicar(paciente)
