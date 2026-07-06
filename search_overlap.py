"""
Targeted overlap search - uses known overlap region and varies around it.
Much faster than brute-force grid search.
"""
import sys
sys.path.append('c:/Users/damacias/Videos/tesis/ec.com.ug.tesis.api')
from app.model import ModelArtifacts
import pandas as pd
import torch
import numpy as np

artifacts = ModelArtifacts()
artifacts.load()
names = artifacts.disease_names

def get_probs(age, gender, esr, crp, rf, ccp, hla, ana, ro, la, dsdna, sm, c3, c4):
    df = pd.DataFrame([{
        'Age': age, 'Gender': gender, 'ESR': esr, 'CRP': crp,
        'RF': rf, 'Anti-CCP': ccp, 'HLA-B27': hla, 'ANA': ana,
        'Anti-Ro': ro, 'Anti-La': la, 'Anti-dsDNA': dsdna,
        'Anti-Sm': sm, 'C3': c3, 'C4': c4
    }])
    X = artifacts.preprocessor.transform(df)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    t = torch.tensor(X, dtype=torch.float32)
    artifacts.model.eval()
    return torch.sigmoid(artifacts.model(t)).cpu().detach().numpy()[0]

# Targeted test cases based on known overlap region + variations
test_cases = []

# We know RF=80, CCP=30, ANA=1, dsDNA=1 gives AR+LES overlap
# Now vary age, gender, ESR, CRP, other markers around that
for gender_val, gender_name in [(0, 'Femenino'), (1, 'Masculino')]:
    for age in [25, 30, 35, 40, 45, 50, 55, 60]:
        for esr in [10, 20, 30, 40, 50, 60, 70, 80]:
            for crp in [3, 5, 10, 15, 20, 30, 40]:
                for rf in [0, 20, 40, 60, 80, 100, 120]:
                    for ccp in [0, 15, 30, 50, 80]:
                        # Only test combos with ANA+dsDNA (known overlap trigger)
                        for ana, dsdna in [(1, 1), (1, 0), (0, 1)]:
                            for ro in [0, 1]:
                                for la in [0, 1]:
                                    for sm in [0, 1]:
                                        for hla in [0, 1]:
                                            for c3 in [50, 80, 110]:
                                                for c4 in [10, 20, 30]:
                                                    test_cases.append((
                                                        age, gender_val, gender_name,
                                                        esr, crp, rf, ccp,
                                                        hla, ana, ro, la, dsdna, sm,
                                                        c3, c4
                                                    ))

print("Total de combinaciones a evaluar: {}".format(len(test_cases)))

# If still too many, sample randomly
if len(test_cases) > 200000:
    np.random.seed(42)
    indices = np.random.choice(len(test_cases), 200000, replace=False)
    test_cases = [test_cases[i] for i in indices]
    print("Reducido a {} combinaciones por muestreo aleatorio".format(len(test_cases)))

results = []
for i, tc in enumerate(test_cases):
    age, gender_val, gender_name, esr, crp, rf, ccp, hla, ana, ro, la, dsdna, sm, c3, c4 = tc
    p = get_probs(age, gender_val, esr, crp, rf, ccp, hla, ana, ro, la, dsdna, sm, c3, c4)
    positives = [(n, float(v)) for n, v in zip(names, p) if v >= 0.5]
    if len(positives) >= 2:
        results.append({
            'gender': gender_name, 'age': age,
            'esr': esr, 'crp': crp, 'rf': rf, 'ccp': ccp,
            'hla': hla, 'ana': ana, 'ro': ro, 'la': la,
            'dsdna': dsdna, 'sm': sm, 'c3': c3, 'c4': c4,
            'positives': positives,
            'score': sum(v for _, v in positives),
            'min_prob': min(v for _, v in positives)
        })
    if (i + 1) % 50000 == 0:
        print("Progreso: {}/{} evaluaciones, {} solapamientos...".format(i + 1, len(test_cases), len(results)))

print("Solapamientos encontrados: {}".format(len(results)))

# Deduplicate: keep unique disease combos per gender, top score
seen = set()
unique = []
for r in sorted(results, key=lambda x: x['min_prob'], reverse=True):
    combo_key = (r['gender'], tuple(sorted(n for n, _ in r['positives'])))
    if combo_key not in seen:
        seen.add(combo_key)
        unique.append(r)

print("Casos unicos (por combo enfermedades+genero): {}".format(len(unique)))

# Write TXT
output_path = 'c:/Users/damacias/Videos/tesis/ec.com.ug.tesis.api/casos_solapamiento.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('=' * 80 + '\n')
    f.write('   CASOS DE PRUEBA CON SOLAPAMIENTO (Umbral = 0.50)\n')
    f.write('   Modelo: Gated MLP - Red Autoinmune Multietiqueta\n')
    f.write('=' * 80 + '\n\n')
    f.write('Total de combinaciones unicas encontradas: {}\n\n'.format(len(unique)))

    for i, r in enumerate(unique, 1):
        gender = r['gender']
        hla_str = 'Positivo' if r['hla'] else 'Negativo'
        ana_str = 'Positivo' if r['ana'] else 'Negativo'
        ro_str = 'Positivo' if r['ro'] else 'Negativo'
        la_str = 'Positivo' if r['la'] else 'Negativo'
        dsdna_str = 'Positivo' if r['dsdna'] else 'Negativo'
        sm_str = 'Positivo' if r['sm'] else 'Negativo'
        hla_b27_str = 'Positivo' if r['hla'] else 'Negativo'

        f.write('-' * 60 + '\n')
        f.write('CASO #{} ({})\n'.format(i, gender))
        f.write('-' * 60 + '\n')
        f.write('  Edad: {}\n'.format(r['age']))
        f.write('  Genero: {}\n'.format(gender))
        f.write('  ESR: {}   CRP: {}\n'.format(r['esr'], r['crp']))
        f.write('  RF: {}   Anti-CCP: {}\n'.format(r['rf'], r['ccp']))
        f.write('  HLA-B27: {}   ANA: {}\n'.format(hla_b27_str, ana_str))
        f.write('  Anti-Ro: {}   Anti-La: {}\n'.format(ro_str, la_str))
        f.write('  Anti-dsDNA: {}   Anti-Sm: {}\n'.format(dsdna_str, sm_str))
        f.write('  C3: {}   C4: {}\n\n'.format(r['c3'], r['c4']))
        f.write('  RESULTADO ESPERADO:\n')
        for n, v in sorted(r['positives'], key=lambda x: x[1], reverse=True):
            f.write('    -> {}: {:.1f}% [POSITIVO]\n'.format(n, v * 100))
        f.write('\n')

    f.write('=' * 80 + '\n')
    f.write('FIN DEL DOCUMENTO\n')
    f.write('=' * 80 + '\n')

print("Archivo generado en: {}".format(output_path))
