"""
inference.py — Multi-label prediction logic.

Exact sequence:
1. Build DataFrame with columns in the order of input_schema.
2. preprocessor.transform(df) — NEVER fit_transform.
3. Tensor float32 → model in eval() + no_grad() → 7 logits.
4. torch.sigmoid → 7 INDEPENDENT probabilities (NOT softmax).
5. argmax → primary diagnosis.
6. threshold → compatible profile (the primary is always included).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from app.model import artifacts
from app.schemas import PatientInput

logger = logging.getLogger(__name__)

# Legal warning accompanying each prediction
WARNING = (
    "Supporting result. Profiles with more than one disease are suspicions to be "
    "ruled out by the professional, not confirmed coexistences. It does not replace "
    "clinical judgment."
)

# Optional fields defined in adapters previously, now we can use them directly
OPTIONAL_FEATURES = [
    "esr", "crp", "rf", "anti_ccp", "hla_b27", "ana",
    "anti_ro", "anti_la", "anti_dsdna", "anti_sm", "c3", "c4"
]

def _build_dataframe(data: PatientInput) -> pd.DataFrame:
    """Builds a 1-row DataFrame with the exact column order defined
    by input_schema.json (required_features)."""
    # Convert Pydantic model to dict
    data_dict = data.model_dump()
    
    # We need to map the English keys from PatientInput to the ML feature names
    # which are in artifacts.required_features.
    # We can do this using a mapping.
    
    # The Pydantic model uses English fields. The ML model expects specific names.
    # Let's map PatientInput fields to ML model features if they differ.
    
    # The required features from input_schema are:
    # ['Age', 'Gender', 'ESR', 'CRP', 'RF', 'Anti-CCP', 'HLA-B27', 'ANA', 'Anti-Ro', 'Anti-La', 'Anti-dsDNA', 'Anti-Sm', 'C3', 'C4']
    
    def binary_to_int(val):
        if val is None:
            return None
        return 1 if val == "Positive" else 0

    mapped_data = {
        "Age": data_dict.get("age"),
        "Gender": 1 if data_dict.get("gender") == "Male" else 0,
        "ESR": data_dict.get("esr"),
        "CRP": data_dict.get("crp"),
        "RF": data_dict.get("rf"),
        "Anti-CCP": data_dict.get("anti_ccp"),
        "HLA-B27": binary_to_int(data_dict.get("hla_b27")),
        "ANA": binary_to_int(data_dict.get("ana")),
        "Anti-Ro": binary_to_int(data_dict.get("anti_ro")),
        "Anti-La": binary_to_int(data_dict.get("anti_la")),
        "Anti-dsDNA": binary_to_int(data_dict.get("anti_dsdna")),
        "Anti-Sm": binary_to_int(data_dict.get("anti_sm")),
        "C3": data_dict.get("c3"),
        "C4": data_dict.get("c4"),
    }

    row: Dict[str, Optional[float]] = {}
    for col in artifacts.required_features:
        row[col] = mapped_data.get(col)

    df = pd.DataFrame([row], columns=artifacts.required_features)
    # Force numeric dtype to handle Nones correctly for the preprocessor
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _preprocess(df: pd.DataFrame) -> torch.Tensor:
    """Applies the pre-fitted ColumnTransformer and returns a float32 tensor."""
    X = artifacts.preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return torch.tensor(X, dtype=torch.float32)


def _infer(tensor: torch.Tensor) -> np.ndarray:
    """Passes the tensor through the model (eval + no_grad) and returns
    INDEPENDENT sigmoid probabilities (NOT softmax)."""
    with torch.no_grad():
        logits = artifacts.model(tensor)          # (1, 7)
        probabilities = torch.sigmoid(logits)       # independent sigmoid
    return probabilities.cpu().numpy()[0]           # shape (7,)


def missing_features(data: PatientInput) -> List[str]:
    """English names of optional features that arrived as null."""
    data_dict = data.model_dump()
    return [field for field in OPTIONAL_FEATURES if data_dict.get(field) is None]


def predict(data: PatientInput) -> dict:
    """Complete prediction pipeline. Returns a dict ready to serialize."""
    # 1-2. Build DataFrame
    df = _build_dataframe(data)
    logger.debug("DataFrame built: %s", df.to_dict(orient="records"))

    # 3. Preprocess
    tensor = _preprocess(df)

    # 4-5. Infer (independent sigmoid)
    probs = _infer(tensor)

    # Round to 4 decimals
    probs = [round(float(p), 4) for p in probs]

    names = artifacts.disease_names
    threshold = artifacts.threshold

    # 7. Primary diagnosis = argmax
    primary_idx = int(np.argmax(probs))
    primary_diagnosis = names[primary_idx]
    primary_prob = probs[primary_idx]

    # 8. Compatible profile (>= threshold + always the primary)
    compatible_diseases = []
    for i, (name, p) in enumerate(zip(names, probs)):
        if p >= threshold or i == primary_idx:
            compatible_diseases.append({"disease": name, "probability": p})

    # Sort compatibles from highest to lowest probability
    compatible_diseases.sort(key=lambda x: x["probability"], reverse=True)

    # 9. All probabilities sorted from highest to lowest
    all_probs = [
        {"disease": n, "probability": p}
        for n, p in zip(names, probs)
    ]
    all_probs.sort(key=lambda x: x["probability"], reverse=True)
    
    # Overlap syndrome detected?
    compatibles_without_normal = [
        c for c in compatible_diseases if c["disease"] != "Normal"
    ]
    overlap_syndrome_detected = len(compatibles_without_normal) > 1

    # Model name
    model_used = "unknown"
    if artifacts.config:
        model_used = str(artifacts.config.get("nombre_modelo", "unknown"))

    return {
        "primary_diagnosis": primary_diagnosis,
        "primary_probability": primary_prob,
        "compatible_profile": compatible_diseases,
        "all_probabilities": all_probs,
        "overlap_syndrome_detected": overlap_syndrome_detected,
        "threshold_used": threshold,
        "missing_features": missing_features(data),
        "model_used": model_used,
        "warning": WARNING,
    }


def explain(data: PatientInput) -> dict:
    """Generates LIME explanations for the diseases in the compatible profile.
    
    TODO: Integrate lime.lime_tabular.LimeTabularExplainer.
    Currently returns the response structure with a placeholder.
    """
    # First we get the normal prediction to know the profile
    prediction_result = predict(data)

    explanations = []
    for disease_prob in prediction_result["compatible_profile"]:
        explanations.append({
            "disease": disease_prob["disease"],
            "positive_evidence": [],
            "negative_evidence": [],
        })

    return {
        "primary_diagnosis": prediction_result["primary_diagnosis"],
        "explanations": explanations,
        "warning": (
            "LIME explanation pending integration. Response structure "
            "is ready; need to connect lime.lime_tabular."
        ),
    }
