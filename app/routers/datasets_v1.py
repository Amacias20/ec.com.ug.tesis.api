"""
datasets_v1.py — Gestión de datasets subidos, bajo /api/v1/datasets.

Guarda cada archivo en disco (data/datasets/) con un nombre único para
evitar colisiones/sobrescrituras, y su metadata en sqlite (app.db). El
conteo de filas se calcula con pandas y es "best effort": si el archivo no
se puede parsear, igual se acepta la subida con total_rows = None en vez
de fallar toda la petición.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/datasets", tags=["Datasets"])

EXTENSIONES_PERMITIDAS = {".csv", ".xlsx", ".xls"}


class DatasetInfo(BaseModel):
    id: int
    filename: str
    upload_date: str
    size_bytes: int
    total_rows: Optional[int] = None


def _contar_filas(path: Path) -> Optional[int]:
    try:
        if path.suffix.lower() == ".csv":
            return len(pd.read_csv(path))
        return len(pd.read_excel(path))
    except Exception:
        logger.warning("No se pudo calcular total_rows para %s", path.name)
        return None


@router.post("/upload", summary="Subir un dataset (.csv, .xlsx, .xls)")
async def upload_dataset(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no permitida: {extension}. Use .csv, .xlsx o .xls",
        )

    nombre_unico = f"{uuid.uuid4().hex}{extension}"
    destino = db.DATASETS_DIR / nombre_unico

    contenido = await file.read()
    with open(destino, "wb") as f:
        f.write(contenido)

    total_rows = _contar_filas(destino)
    db.insert_dataset(
        filename=file.filename,
        stored_path=str(destino),
        size_bytes=len(contenido),
        total_rows=total_rows,
    )

    return {"message": "Dataset subido correctamente", "filename": file.filename}


@router.get("/history", response_model=list[DatasetInfo], summary="Historial de datasets subidos")
def dataset_history():
    return db.list_datasets()


@router.delete("/{dataset_id}", summary="Eliminar un dataset")
def delete_dataset(dataset_id: int):
    dataset = db.delete_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    stored_path = dataset["stored_path"]
    if stored_path and os.path.exists(stored_path):
        os.remove(stored_path)

    return {"message": "Dataset eliminado correctamente"}
