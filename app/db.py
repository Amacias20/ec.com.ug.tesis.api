"""
db.py — Persistencia ligera con sqlite3 (stdlib, sin ORM).

Guarda:
- prediction_history: un registro por cada predicción real hecha vía
  /api/v1/predict o /api/v1/predict-with-explanation, usado por los
  endpoints de /api/v1/dashboard/*.
- datasets: metadata de los archivos subidos vía /api/v1/datasets/upload.

Se usan conexiones cortas (abrir/cerrar por llamada); es suficiente para
un proceso FastAPI de un solo worker en un proyecto de tesis.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"
DATASETS_DIR = DATA_DIR / "datasets"


def _conectar() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crea las carpetas y tablas necesarias si no existen todavía."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    with _conectar() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                primary_diagnosis TEXT NOT NULL,
                primary_probability REAL NOT NULL,
                overlap_syndrome INTEGER NOT NULL,
                threshold_used REAL NOT NULL,
                predictions_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                total_rows INTEGER
            )
            """
        )


# ---------------------------------------------------------------------------
# Historial de predicciones (para el dashboard)
# ---------------------------------------------------------------------------
def log_prediction(resultado: dict) -> None:
    """Registra una predicción real (dict devuelto por inference.predecir)."""
    compatibles_sin_normal = [
        c for c in resultado["perfil_compatibles"] if c.enfermedad != "Normal"
    ]
    overlap = len(compatibles_sin_normal) > 1

    predictions_json = json.dumps(
        [
            {"disease": item.enfermedad, "probability": item.probabilidad}
            for item in resultado["todas_las_probabilidades"]
        ]
    )

    with _conectar() as conn:
        conn.execute(
            """
            INSERT INTO prediction_history
                (created_at, primary_diagnosis, primary_probability,
                 overlap_syndrome, threshold_used, predictions_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                resultado["diagnostico_principal"],
                resultado["probabilidad_principal"],
                1 if overlap else 0,
                resultado["umbral_usado"],
                predictions_json,
            ),
        )


def dashboard_summary() -> dict:
    with _conectar() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(overlap_syndrome), 0) AS overlap_count,
                   MAX(created_at) AS last_prediction_at
            FROM prediction_history
            """
        ).fetchone()
    return {
        "total_predictions": row["total"],
        "overlap_syndrome_count": row["overlap_count"],
        "last_prediction_at": row["last_prediction_at"],
    }


def dashboard_disease_distribution() -> Dict[str, int]:
    with _conectar() as conn:
        rows = conn.execute(
            """
            SELECT primary_diagnosis, COUNT(*) AS total
            FROM prediction_history
            GROUP BY primary_diagnosis
            ORDER BY total DESC
            """
        ).fetchall()
    return {row["primary_diagnosis"]: row["total"] for row in rows}


def dashboard_timeline() -> List[dict]:
    with _conectar() as conn:
        rows = conn.execute(
            """
            SELECT date(created_at) AS day, COUNT(*) AS total
            FROM prediction_history
            GROUP BY day
            ORDER BY day
            """
        ).fetchall()
    return [{"date": row["day"], "count": row["total"]} for row in rows]


# ---------------------------------------------------------------------------
# Datasets subidos
# ---------------------------------------------------------------------------
def insert_dataset(
    filename: str, stored_path: str, size_bytes: int, total_rows: Optional[int]
) -> int:
    with _conectar() as conn:
        cursor = conn.execute(
            """
            INSERT INTO datasets (filename, stored_path, upload_date, size_bytes, total_rows)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                filename,
                stored_path,
                datetime.now(timezone.utc).isoformat(),
                size_bytes,
                total_rows,
            ),
        )
        return cursor.lastrowid


def list_datasets() -> List[dict]:
    with _conectar() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, upload_date, size_bytes, total_rows
            FROM datasets
            ORDER BY upload_date DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_dataset(dataset_id: int) -> Optional[dict]:
    with _conectar() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_dataset(dataset_id: int) -> Optional[dict]:
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return None
    with _conectar() as conn:
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return dataset
