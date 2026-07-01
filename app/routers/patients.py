from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(prefix="/pacientes", tags=["Pacientes"])

class PacienteResponse(BaseModel):
    id: int
    nombre: str
    edad: int
    genero: int
    fecha_registro: str

# Mock data
PACIENTES = [
    {"id": 1, "nombre": "Juan Perez", "edad": 45, "genero": 1, "fecha_registro": "2023-01-15"},
    {"id": 2, "nombre": "Maria Lopez", "edad": 38, "genero": 0, "fecha_registro": "2023-02-20"},
    {"id": 3, "nombre": "Carlos Sanchez", "edad": 50, "genero": 1, "fecha_registro": "2023-03-10"},
]

@router.get("", response_model=List[PacienteResponse], summary="Obtener lista de pacientes")
def listar_pacientes():
    """Devuelve una lista de pacientes (mocked)."""
    return PACIENTES

@router.get("/{paciente_id}", response_model=Optional[PacienteResponse], summary="Obtener paciente por ID")
def obtener_paciente(paciente_id: int):
    """Devuelve los detalles de un paciente específico."""
    for p in PACIENTES:
        if p["id"] == paciente_id:
            return p
    return None
