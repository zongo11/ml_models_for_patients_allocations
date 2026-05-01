import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

hospitals_df = pd.read_excel("Book1.xlsx")
patients_df = pd.read_excel("patients_1000_ULTRA_COMPLET.xlsx")

print("Hospitals columns:", hospitals_df.columns.tolist())
# print("Patients columns:", patients_df.columns.tolist())

RESOURCE_MAPPINGS = {
    'Lit': {'p': 'Lits_totaux', 'h': 'Lits totaux'},
    'Rea_Lit': {'p': 'Rea_lits', 'h': 'Ra (lits)'},
    'Medecin': {'p': 'Medecins', 'h': 'Mdecins'},
    'Infirmier': {'p': 'Infirmiers', 'h': 'Infirmiers'},
    'Urgentiste': {'p': 'Urgentistes', 'h': 'Urgentistes'},
    'Reanimateur': {'p': 'Reanimateurs', 'h': 'Ranimateurs'},
    'Anesth_Rea': {'p': 'Anesth_Rea', 'h': 'Anesth.-Ra'},
    'Pneumo': {'p': 'Pneumo', 'h': 'Pneumo'},
    'Cardio': {'p': 'Cardio', 'h': 'Cardio'},
    'Neuro': {'p': 'Neuro', 'h': 'Neuro'},
    'Interniste': {'p': 'Internistes', 'h': 'Internistes'},
    'Chirurgien': {'p': 'Chirurgiens', 'h': 'Chirurgiens'},
    'Pediatre': {'p': 'Pediatres', 'h': 'Pdiatres'},
    'Biologiste': {'p': 'Biologistes', 'h': 'Biologistes'},
    'Radiologue': {'p': 'Radiologues', 'h': 'Radiologues'},
    'O2': {'p': 'O2', 'h': 'O'},
    'Ventilateur': {'p': 'Ventilateur', 'h': 'Ventilateurs'},
    'Sang': {'p': 'Sang', 'h': 'Sang (units)'},
    'Moniteur': {'p': 'Moniteurs', 'h': 'Moniteurs'},
    'Defibrillateur': {'p': 'Defibrillateur', 'h': 'Dfibrillateurs'},
    'Monitoring': {'p': 'Monitoring', 'h': 'Moniteurs'},
    'Labo': {'p': 'Labo', 'h': 'Labo'},
    'Imagerie': {'p': 'Imagerie', 'h': 'Imagerie'}
}

def hospital_can_handle(patient, hospital):
    for res, keys in RESOURCE_MAPPINGS.items():
        p_val = patient.get(keys['p'], 0)
        h_val = hospital.get(keys['h'], 0)
        if pd.isna(p_val): p_val = 0
        if pd.isna(h_val): h_val = 0
        if h_val < p_val:
            return False
    return True

hospitals = hospitals_df.to_dict(orient="records")
patients = patients_df.to_dict(orient="records")

can_handle_counts = []
for p in patients:
    count = sum(1 for h in hospitals if hospital_can_handle(p, h))
    can_handle_counts.append(count)

print("\nHospital data types:")
print(hospitals_df.dtypes)

print("\nUnique values in 'Réanimateurs':")
print(hospitals_df['Réanimateurs'].unique())
