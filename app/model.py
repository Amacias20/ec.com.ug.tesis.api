"""
model.py — Definition of the RedAutoinmuneGated architecture and artifact loading.

The RedAutoinmuneGated class MUST be byte-for-byte identical to the one used during
training; otherwise, torch.load_state_dict() will fail.
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
# Base path for artifacts (relative to project root)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


# ---------------------------------------------------------------------------
# EXACT architecture of the trained model (DO NOT modify)
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
# Container for loaded artifacts (singleton in memory)
# ---------------------------------------------------------------------------
class ModelArtifacts:
    """Stores all necessary artifacts loaded once at startup."""

    def __init__(self) -> None:
        self.model: Optional[RedAutoinmuneGated] = None
        self.preprocessor: Optional[Any] = None
        self.label_binarizer: Optional[Any] = None
        self.config: Optional[Dict[str, Any]] = None
        self.input_schema: Optional[Dict[str, Any]] = None
        self.decision_rule: Optional[Dict[str, Any]] = None
        self.disease_names: Optional[List[str]] = None
        self.required_features: Optional[List[str]] = None
        self.continuous_features: Optional[List[str]] = None
        self.binary_features: Optional[List[str]] = None
        self.threshold: float = 0.5
        self.loaded: bool = False

    def load(self) -> None:
        """Loads all artifacts from disk. Raises RuntimeError if something is missing."""
        logger.info("Loading artifacts from %s …", ARTIFACTS_DIR)

        # --- config.yaml ---
        config_path = ARTIFACTS_DIR / "config.yaml"
        if not config_path.exists():
            raise RuntimeError(f"Missing critical artifact: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        logger.info("  config.yaml loaded: %s", self.config)

        # --- input_schema.json ---
        schema_path = ARTIFACTS_DIR / "input_schema.json"
        if not schema_path.exists():
            raise RuntimeError(f"Missing critical artifact: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.input_schema = json.load(f)
        self.required_features = self.input_schema["variables_requeridas"]
        self.continuous_features = self.input_schema["variables_continuas"]
        self.binary_features = self.input_schema["variables_binarias"]
        logger.info("  input_schema.json loaded (%d features)", len(self.required_features))

        # --- decision_rule.json ---
        rule_path = ARTIFACTS_DIR / "decision_rule.json"
        if not rule_path.exists():
            raise RuntimeError(f"Missing critical artifact: {rule_path}")
        with open(rule_path, "r", encoding="utf-8") as f:
            self.decision_rule = json.load(f)
        self.threshold = float(self.decision_rule.get("umbral", 0.5))
        logger.info("  decision_rule.json loaded (threshold=%.2f)", self.threshold)

        # --- preprocessor.joblib ---
        prep_path = ARTIFACTS_DIR / "preprocessor.joblib"
        if not prep_path.exists():
            raise RuntimeError(f"Missing critical artifact: {prep_path}")
        self.preprocessor = joblib.load(prep_path)
        logger.info("  preprocessor.joblib loaded")

        # --- label_binarizer.joblib ---
        lb_path = ARTIFACTS_DIR / "label_binarizer.joblib"
        if not lb_path.exists():
            raise RuntimeError(f"Missing critical artifact: {lb_path}")
        self.label_binarizer = joblib.load(lb_path)
        self.disease_names = list(self.label_binarizer.classes_)
        logger.info("  label_binarizer.joblib loaded: %s", self.disease_names)

        # --- model.pt (state_dict) ---
        model_path = ARTIFACTS_DIR / "model.pt"
        if not model_path.exists():
            raise RuntimeError(f"Missing critical artifact: {model_path}")

        input_dim = int(self.config["input_dim"])
        num_classes = int(self.config["num_classes"])
        hidden_dim = int(self.config.get("hidden_dim", 64))
        dropout = float(self.config.get("dropout", 0.2))

        self.model = RedAutoinmuneGated(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # BatchNorm in inference mode
        logger.info("  model.pt loaded and set to eval()")

        self.loaded = True
        logger.info("✅ All artifacts loaded successfully.")


# Global instance (loaded in FastAPI lifespan)
artifacts = ModelArtifacts()
