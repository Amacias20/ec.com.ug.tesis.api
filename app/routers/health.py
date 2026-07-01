"""
health.py — Endpoints de salud y esquema.

GET /salud   → verifica que el servicio está vivo y el modelo cargó.
GET /esquema → devuelve las listas de variables y enfermedades para el frontend.
"""

from fastapi import APIRouter

from app.model import artefactos
from app.schemas import EsquemaResponse, SaludResponse

router = APIRouter(tags=["Estado"])


@router.get("/salud", response_model=SaludResponse, summary="Estado del servicio")
def salud():
    """Verifica que el servicio está activo y que el modelo se cargó correctamente."""
    version = ""
    if artefactos.config:
        version = artefactos.config.get("nombre_modelo", artefactos.config.get("nombre", "desconocido"))
    return SaludResponse(
        estado="ok",
        modelo_cargado=artefactos.cargado,
        version_modelo=str(version),
    )


@router.get("/esquema", response_model=EsquemaResponse, summary="Esquema de entrada")
def esquema():
    """Devuelve el esquema de variables de entrada y las enfermedades de salida
    para que el frontend sepa qué campos presentar al usuario."""
    return EsquemaResponse(
        variables_continuas=artefactos.variables_continuas or [],
        variables_binarias=artefactos.variables_binarias or [],
        variables_requeridas=artefactos.variables_requeridas or [],
        enfermedades_salida=artefactos.nombres_enfermedades or [],
    )
