"""
feature_importance.py — Global feature importance per disease,
for GET /api/v1/feature-importance (used by the Explainability page).

Loads the pre-computed global importance from artifacts/global_importance.json
to avoid blocking the API server with heavy LIME calculations on the fly.
"""

import json
import logging
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
_importance_cache: Dict[str, Dict[str, float]] = {}


def compute_global_importance() -> Dict[str, Dict[str, float]]:
    global _importance_cache
    if _importance_cache:
        return _importance_cache

    json_path = ARTIFACTS_DIR / "global_importance.json"
    if not json_path.exists():
        logger.warning(f"Global importance file not found at {json_path}")
        return {}

    logger.info("Loading pre-computed global feature importance...")
    with open(json_path, "r", encoding="utf-8") as f:
        _importance_cache = json.load(f)
        
    logger.info("✅ Global importance loaded.")
    return _importance_cache
