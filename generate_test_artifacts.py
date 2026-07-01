"""
Script auxiliar para generar artefactos sintéticos de prueba.
Esto permite arrancar y probar el backend sin los artefactos reales del entrenamiento.

¡SOLO PARA DESARROLLO! En producción, reemplaza estos archivos con los reales.
"""

import sys
import os

# Agregar el directorio raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from app.model import RedAutoinmuneGated

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ---------- 1. model.pt ----------
print("Generando model.pt (pesos aleatorios)…")
modelo = RedAutoinmuneGated(input_dim=14, num_classes=7, hidden_dim=64, dropout=0.2)
torch.save(modelo.state_dict(), os.path.join(ARTIFACTS_DIR, "model.pt"))
print("  ✅ model.pt guardado")

# ---------- 2. preprocessor.joblib ----------
print("Generando preprocessor.joblib…")

variables_continuas = ["Age", "ESR", "CRP", "RF", "Anti-CCP", "C3", "C4"]
variables_binarias = ["Gender", "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm"]

# Simular datos para ajustar el preprocesador
import pandas as pd

np.random.seed(42)
n = 200
data = {}
for col in variables_continuas:
    data[col] = np.random.randn(n) * 10 + 50
for col in variables_binarias:
    data[col] = np.random.randint(0, 2, n).astype(float)

df_fake = pd.DataFrame(data)

# El ColumnTransformer que probablemente se usó en el entrenamiento
preprocessor = ColumnTransformer(
    transformers=[
        ("continuas", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), variables_continuas),
        ("binarias", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), variables_binarias),
    ],
    remainder="drop",
)

# Columnas en el orden de variables_requeridas
variables_requeridas = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP",
    "HLA-B27", "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm",
    "C3", "C4",
]
df_ordenado = df_fake[variables_requeridas]
preprocessor.fit(df_ordenado)

joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
print("  ✅ preprocessor.joblib guardado")

# ---------- 3. label_binarizer.joblib ----------
print("Generando label_binarizer.joblib…")
mlb = MultiLabelBinarizer()
mlb.fit([
    ["Ankylosing Spondylitis", "Normal", "Psoriatic Arthritis",
     "Reactive Arthritis", "Rheumatoid Arthritis",
     "Sjögren's Syndrome", "Systemic Lupus Erythematosus"]
])
joblib.dump(mlb, os.path.join(ARTIFACTS_DIR, "label_binarizer.joblib"))
print("  ✅ label_binarizer.joblib guardado")

print("\n🎉 Todos los artefactos sintéticos generados en artifacts/")
print("   ⚠️  Recuerda reemplazarlos con los reales para producción.")
