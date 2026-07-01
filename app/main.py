"""
main.py — Punto de entrada de la aplicación FastAPI.

- Carga los artefactos UNA SOLA VEZ al arrancar (lifespan).
- Habilita CORS para el frontend React (p.ej. localhost:5173).
- Registra los routers de /salud, /esquema, /predecir y /explicar.
- Documentación Swagger disponible en /docs.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.model import artefactos
from app.routers import api_v1, dashboard_v1, datasets_v1, health, predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: carga de artefactos al iniciar
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga todos los artefactos (modelo, preprocesador, esquema, etc.)
    una sola vez al arrancar el servidor. Si falta algún archivo crítico,
    la aplicación falla con un mensaje explícito en vez de arrancar rota."""
    logger.info("🚀 Iniciando carga de artefactos…")
    try:
        artefactos.cargar()
    except RuntimeError as e:
        logger.critical("❌ Error fatal al cargar artefactos: %s", e)
        raise
    init_db()
    logger.info("✅ Servidor listo para recibir peticiones.")
    yield
    logger.info("🛑 Servidor apagándose.")


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de Clasificación Multietiqueta — Enfermedades Autoinmunes",
    description=(
        "Sirve un modelo Gated MLP (PyTorch) pre-entrenado para clasificar "
        "7 enfermedades autoinmunes a partir de 14 variables clínicas. "
        "Predicción **multietiqueta** con sigmoides independientes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — permite que el frontend React en otro puerto consuma la API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:3000",   # CRA / Next dev
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(api_v1.router)
app.include_router(datasets_v1.router)
app.include_router(dashboard_v1.router)
