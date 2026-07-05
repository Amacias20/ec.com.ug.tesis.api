"""
feature_importance.py — Global feature importance per disease,
for GET /api/v1/feature-importance (used by the Explainability page).

Unlike lime_engine.explain_patient (explanation of ONE patient),
here LIME weights are averaged over a batch of synthetic samples
to obtain an approximate "global" importance per disease. It is
calculated lazily (first call) and cached in memory, as it is
deterministic as long as the model artifacts do not change.
"""

import logging
from typing import Dict

import numpy as np

from app.lime_engine import _explain_one_disease, _get_background
from app.model import artifacts

logger = logging.getLogger(__name__)

_N_SAMPLES = 50
_importance_cache: Dict[str, Dict[str, float]] = {}


def compute_global_importance() -> Dict[str, Dict[str, float]]:
    global _importance_cache
    if _importance_cache:
        return _importance_cache

    logger.info("Computing global feature importance (first call, may take a while)…")
    background = _get_background().values[:_N_SAMPLES]
    names = artifacts.disease_names

    result: Dict[str, Dict[str, float]] = {}
    for idx, disease in enumerate(names):
        accumulated: Dict[str, list] = {col: [] for col in artifacts.required_features}
        for row in background:
            pairs = _explain_one_disease(row, idx)
            for feature, weight in pairs:
                # Binary variables come as "Gender=1"/"Gender=0"
                # (LIME treats them as categorical); normalized to base name
                # to be able to average between samples with different values.
                base_name = feature.split("=")[0]
                accumulated[base_name].append(weight)
        result[disease] = {
            feature: float(np.mean(weights)) if weights else 0.0
            for feature, weights in accumulated.items()
        }

    _importance_cache = result
    logger.info("✅ Global importance computed and cached.")
    return _importance_cache
