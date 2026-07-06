# Model Card — CDSS multietiqueta para enfermedades autoinmunes

## Resumen
Modelo: Gated MLP (selección automática por evidencia estadística).
Tarea: CLASIFICACIÓN MULTIETIQUETA — 7 sigmoides independientes (BCEWithLogitsLoss),
una por enfermedad. La arquitectura permite activar varias enfermedades a la vez.
Dataset: Rheumatic and Autoimmune Disease Dataset (Mahdi et al., 2025), 12,085 pacientes.

## Selección de arquitectura
Candidata por F1 Macro (Nested CV): Gated MLP (F1=0.7864).
  - MLP vs Gated MLP: p=0.0000 (significativa)
  - Gated MLP vs Residual: p=0.0000 (significativa)
Empates estadísticos: ['Gated MLP'].
Ganador (menor complejidad entre empates): Gated MLP (12551 parámetros).

## Datos y partición
- 14 variables clínicas/laboratorio + indicadores de ausencia.
- 70% train / 15% validation / 15% test, estratificada, semilla 42.
- Clase minoritaria: Reactive Arthritis (~4.3%).

## Métricas (test bloqueado, multietiqueta)
- Hamming Loss   : 0.0743
- Subset Accuracy: 0.5609
- F1 Micro       : 0.7906
- F1 Macro       : 0.7827
- ROC-AUC Macro  : 0.9817

## Perfil multietiqueta
- Umbral: 0.50 sobre cada sigmoide.
- Cobertura del perfil: 0.9818.
- Promedio de etiquetas por paciente: 1.48.
- LIMITACIÓN: el dataset asigna una sola enfermedad por paciente; los perfiles con
  más de una etiqueta se reportan como sospechas a descartar, no como coexistencias
  confirmadas. La capacidad multietiqueta es arquitectónica y validada en el límite
  que permiten los datos.

## Calibración
- Brier medio (por etiqueta): 0.0520
- ECE medio (por etiqueta)  : 0.0615

## Limitaciones documentadas
- Datos faltantes MNAR (informados por sospecha clínica); se usan indicadores de ausencia.
- Riesgo de circularidad (incorporation bias): varios predictores forman parte de los
  criterios diagnósticos.
- Sin validación externa (solo validación interna).
- Etiquetas multietiqueta sin verdad de terreno de coexistencia.

## Reproducibilidad
Semilla global: 42. Versiones en environment.json. Partición en split_indices.json.
