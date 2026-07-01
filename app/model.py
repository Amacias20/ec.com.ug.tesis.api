"""
model.py — Definición de la arquitectura RedAutoinmuneGated y carga de artefactos.

La clase RedAutoinmuneGated DEBE ser idéntica byte-por-byte a la usada durante
el entrenamiento; de lo contrario, torch.load_state_dict() fallará.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ruta base de artefactos (relativa a la raíz del proyecto)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# Arquitectura EXACTA del modelo entrenado (NO modificar)
# ---------------------------------------------------------------------------
class RedAutoinmuneGated(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, dropout=0.2):
        super(RedAutoinmuneGated, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        g = torch.sigmoid(self.gate(h))
        h = h * g
        h = self.dropout(F.leaky_relu(self.bn1(self.fc2(h))))
        h = F.leaky_relu(self.bn2(self.fc3(h)))
        return self.output(h)


# ---------------------------------------------------------------------------
# Contenedor de artefactos cargados (singleton en memoria)
# ---------------------------------------------------------------------------
class ArtifactosModelo:
    """Almacena todos los artefactos necesarios cargados una sola vez al inicio."""

    def __init__(self) -> None:
        self.modelo: Optional[RedAutoinmuneGated] = None
        self.preprocessor: Optional[Any] = None
        self.label_binarizer: Optional[Any] = None
        self.config: Optional[Dict[str, Any]] = None
        self.input_schema: Optional[Dict[str, Any]] = None
        self.decision_rule: Optional[Dict[str, Any]] = None
        self.nombres_enfermedades: Optional[List[str]] = None
        self.variables_requeridas: Optional[List[str]] = None
        self.variables_continuas: Optional[List[str]] = None
        self.variables_binarias: Optional[List[str]] = None
        self.umbral: float = 0.5
        self.cargado: bool = False

    def cargar(self) -> None:
        """Carga todos los artefactos desde disco. Lanza RuntimeError si falta algo."""
        logger.info("Cargando artefactos desde %s …", ARTIFACTS_DIR)

        # --- config.yaml ---
        config_path = ARTIFACTS_DIR / "config.yaml"
        if not config_path.exists():
            raise RuntimeError(f"Falta artefacto crítico: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        logger.info("  config.yaml cargado: %s", self.config)

        # --- input_schema.json ---
        schema_path = ARTIFACTS_DIR / "input_schema.json"
        if not schema_path.exists():
            raise RuntimeError(f"Falta artefacto crítico: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.input_schema = json.load(f)
        self.variables_requeridas = self.input_schema["variables_requeridas"]
        self.variables_continuas = self.input_schema["variables_continuas"]
        self.variables_binarias = self.input_schema["variables_binarias"]
        logger.info("  input_schema.json cargado (%d variables)", len(self.variables_requeridas))

        # --- decision_rule.json ---
        rule_path = ARTIFACTS_DIR / "decision_rule.json"
        if not rule_path.exists():
            raise RuntimeError(f"Falta artefacto crítico: {rule_path}")
        with open(rule_path, "r", encoding="utf-8") as f:
            self.decision_rule = json.load(f)
        self.umbral = float(self.decision_rule.get("umbral", 0.5))
        logger.info("  decision_rule.json cargado (umbral=%.2f)", self.umbral)

        # --- preprocessor.joblib ---
        prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
        if not prep_path.exists():
            raise RuntimeError(f"Falta artefacto crítico: {prep_path}")
        self.preprocessor = joblib.load(prep_path)
        logger.info("  preprocessor.joblib cargado")

        # --- label_binarizer.joblib ---
        lb_path = ARTIFACTS_DIR / "label_binarizer.joblib"
        if not lb_path.exists():
            raise RuntimeError(f"Falta artefacto crítico: {lb_path}")
        self.label_binarizer = joblib.load(lb_path)
        self.nombres_enfermedades = list(self.label_binarizer.classes_)
        logger.info("  label_binarizer.joblib cargado: %s", self.nombres_enfermedades)

        # --- model.pt (state_dict) ---
        model_path = ARTIFACTS_DIR / "model.pt"
        if not model_path.exists():
            raise RuntimeError(f"Falta artefacto crítico: {model_path}")

        input_dim = int(self.config["input_dim"])
        num_classes = int(self.config["num_classes"])
        hidden_dim = int(self.config.get("hidden_dim", 64))
        dropout = float(self.config.get("dropout", 0.2))

        self.modelo = RedAutoinmuneGated(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.modelo.load_state_dict(state_dict)
        self.modelo.eval()  # BatchNorm en modo inferencia
        logger.info("  model.pt cargado y puesto en eval()")

        self.cargado = True
        logger.info("✅ Todos los artefactos cargados correctamente.")


# Instancia global (se carga en el lifespan de FastAPI)
artefactos = ArtifactosModelo()
