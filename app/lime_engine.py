"""
lime_engine.py — Real explainability with LIME (lime.lime_tabular).

Substitutes the stub app.inference.explain() without modifying it:
the /api/v1 routers call explain_patient() directly from here.

The explainer is built on the 14 raw CLINICAL variables (untransformed),
with a prediction function that internally applies
preprocessor.transform() + model + sigmoid. This way the explanations come out
in terms of the variables the frontend knows (age, esr, crp, ...),
not the internally scaled space.

Important note: there is no real training dataset in this
project (the artifacts are synthetic, see generate_test_artifacts.py),
so the "background" that LIME needs to perturb samples is generated
synthetically from the statistics already saved in the
preprocessor (means/scales of the StandardScaler). For the same reason there is
no `shap` library installed: the `shap_values` field of the frontend contract
is populated by reusing the same LIME weights as an
approximation, left explicit here and in the response.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from app.inference import _build_dataframe, predict
from app.model import artifacts
from app.schemas import PatientInput

logger = logging.getLogger(__name__)

_DEFAULT_MEAN = 50.0
_DEFAULT_STD = 15.0
_NUM_FEATURES_EXPLAIN = 5

_background_cache: Optional[pd.DataFrame] = None
_explainer_cache: Optional[LimeTabularExplainer] = None


def _continuous_stats() -> Dict[str, Tuple[float, float]]:
    """Tries to read real (mean, std) from the StandardScaler already
    fitted inside the preprocessor. If the structure doesn't match (e.g.
    another transformer name), it uses reasonable default values."""
    stats: Dict[str, Tuple[float, float]] = {}
    variables = artifacts.continuous_features or []
    try:
        transformer = artifacts.preprocessor.named_transformers_["continuas"]
        scaler = transformer.named_steps["scaler"]
        for name, mean, scale in zip(variables, scaler.mean_, scaler.scale_):
            stats[name] = (float(mean), float(scale) if scale else _DEFAULT_STD)
    except Exception:
        logger.warning("Could not read statistics from preprocessor; using default values.")
    for name in variables:
        stats.setdefault(name, (_DEFAULT_MEAN, _DEFAULT_STD))
    return stats


def _build_background_dataframe(n: int = 300) -> pd.DataFrame:
    """Generates n plausible synthetic samples in the raw space of the
    14 variables, in the order of artifacts.required_features."""
    rng = np.random.default_rng(42)
    stats = _continuous_stats()
    continuous = set(artifacts.continuous_features or [])

    data = {}
    for col in artifacts.required_features:
        if col in continuous:
            mean, std_dev = stats[col]
            data[col] = rng.normal(mean, abs(std_dev) or _DEFAULT_STD, n)
        else:
            data[col] = rng.integers(0, 2, n).astype(float)

    return pd.DataFrame(data, columns=artifacts.required_features)


def _get_background() -> pd.DataFrame:
    global _background_cache
    if _background_cache is None:
        _background_cache = _build_background_dataframe()
    return _background_cache


def _predict_fn(raw_rows: np.ndarray) -> np.ndarray:
    """Function that LIME uses to perturb samples: receives a 2D array in
    the raw space (columns = required_features) and returns the 7
    independent sigmoid probabilities per row."""
    import torch

    df = pd.DataFrame(raw_rows, columns=artifacts.required_features)
    X = artifacts.preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(artifacts.model(tensor)).cpu().numpy()
    return probs


def _get_explainer() -> LimeTabularExplainer:
    global _explainer_cache
    if _explainer_cache is None:
        warnings.filterwarnings(
            "ignore",
            message="Prediction probabilties do not sum to 1",
            category=UserWarning,
            module="lime.lime_tabular"
        )
        background = _get_background()
        continuous = set(artifacts.continuous_features or [])
        categorical_features = [
            i for i, col in enumerate(artifacts.required_features) if col not in continuous
        ]
        _explainer_cache = LimeTabularExplainer(
            training_data=background.values,
            feature_names=artifacts.required_features,
            categorical_features=categorical_features,
            class_names=artifacts.disease_names,
            discretize_continuous=False,
            mode="classification",
            random_state=42,
        )
    return _explainer_cache


def _explain_one_disease(raw_row: np.ndarray, disease_idx: int) -> List[Tuple[str, float]]:
    explainer = _get_explainer()
    explanation = explainer.explain_instance(
        raw_row,
        _predict_fn,
        labels=[disease_idx],
        num_features=len(artifacts.required_features),
    )
    return explanation.as_list(label=disease_idx)


def _fill_missing(raw_row: np.ndarray) -> np.ndarray:
    """LIME needs a complete numeric row (without NaN) to calculate
    distances with respect to the synthetic background. The clinical fields that the
    patient did not provide (NaN, see inference._build_dataframe) are
    filled with the average of the synthetic background for that variable —
    the same imputation criterion used by the model's preprocessor."""
    background = _get_background()
    means = background.mean(axis=0).values
    row = raw_row.astype(float).copy()
    missing = np.isnan(row)
    row[missing] = means[missing]
    return row


from sqlalchemy.orm import Session

def explain_patient(patient: PatientInput, db: Session = None) -> dict:
    """Generates real LIME explanations for the diseases in a patient's
    compatible profile. Returns a dict compatible with the frontend's
    ExplainabilityResponse."""
    result = predict(patient, db)
    raw_row = _fill_missing(_build_dataframe(patient).values[0])

    names = artifacts.disease_names
    lime_explanation: Dict[str, List[Tuple[str, float]]] = {}
    top_positive: Dict[str, List[str]] = {}
    top_negative: Dict[str, List[str]] = {}
    shap_values: Dict[str, Dict[str, float]] = {}

    for compatible in result["predictions"]:
        if not compatible.get("is_positive"):
            continue
        idx = names.index(compatible["disease"])
        pairs = _explain_one_disease(raw_row, idx)

        lime_explanation[compatible["disease"]] = pairs
        shap_values[compatible["disease"]] = dict(pairs)

        positives = [f for f, weight in pairs if weight > 0]
        negatives = [f for f, weight in pairs if weight < 0]
        top_positive[compatible["disease"]] = positives[:_NUM_FEATURES_EXPLAIN]
        top_negative[compatible["disease"]] = negatives[:_NUM_FEATURES_EXPLAIN]

    return {
        "shap_values": shap_values,
        "lime_explanation": lime_explanation,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
    }
