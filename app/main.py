"""
main.py — FastAPI application entry point.

- Loads ML artifacts ONCE at startup (lifespan).
- Creates PostgreSQL tables via SQLAlchemy (code-first).
- Enables CORS for the React frontend (e.g. localhost:5173).
- Registers the consolidated API routers.
- Swagger documentation available at /docs.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.model import artifacts
from app.routers import api, dashboard_v1, patients

# Import models so Base.metadata knows about them
import app.models  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load artifacts + create tables on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all ML artifacts (model, preprocessor, schema, etc.)
    once when starting the server and ensures PostgreSQL tables exist."""
    logger.info("🚀 Starting artifacts load…")
    try:
        artifacts.load()
    except RuntimeError as e:
        logger.critical("❌ Fatal error loading artifacts: %s", e)
        raise

    # Code-first: create tables if they don't exist yet
    logger.info("🗄️  Ensuring PostgreSQL tables exist…")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables ready.")

    logger.info("✅ Server ready to receive requests.")
    yield
    logger.info("🛑 Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multi-label Classification API — Autoimmune Diseases",
    description=(
        "Serves a pre-trained Gated MLP (PyTorch) model to classify "
        "7 autoimmune diseases from 14 clinical variables. "
        "**Multi-label** prediction with independent sigmoids."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allows the React frontend on another port to consume the API
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
app.include_router(api.router)
app.include_router(patients.router)
app.include_router(dashboard_v1.router)
