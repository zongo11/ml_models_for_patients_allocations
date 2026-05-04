import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Modèle d'Allocation de Patients avec XGBoost\n",
            "Ce notebook vise à préparer les données et à entraîner un modèle XGBoost pour allouer les patients vers le meilleur hôpital en fonction de leurs besoins en ressources et de la distance géographique."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import xgboost as xgb\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.metrics import accuracy_score, classification_report\n",
            "import math\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "import matplotlib.pyplot as plt"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Chargement des données"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Chargement des données\n",
            "df_hopitaux = pd.read_excel('Book1.xlsx')\n",
            "df_patients = pd.read_excel('patients_1000_ULTRA_COMPLET.xlsx')\n",
            "\n",
            "print('Dimensions des hôpitaux :', df_hopitaux.shape)\n",
            "print('Dimensions des patients :', df_patients.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Définition des règles d'allocation et création des labels\n",
            "Nous définissons une fonction pour calculer la distance géographique. Ensuite, nous allouons chaque patient à l'hôpital le plus proche qui dispose des ressources nécessaires."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Fonction pour calculer la distance de Haversine entre deux points GPS\n",
            "def haversine(lat1, lon1, lat2, lon2):\n",
            "    R = 6371 # Rayon de la terre en km\n",
            "    phi1 = math.radians(lat1)\n",
            "    phi2 = math.radians(lat2)\n",
            "    delta_phi = math.radians(lat2 - lat1)\n",
            "    delta_lambda = math.radians(lon2 - lon1)\n",
            "    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2\n",
            "    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))\n",
            "    return R * c\n",
            "\n",
            "# Mapping des ressources (Patient -> Hôpital)\n",
            "RESOURCE_MAPPINGS = {\n",
            "    'Lit': {'p': 'Lits_totaux', 'h': 'Lits totaux'},\n",
            "    'Rea_Lit': {'p': 'Rea_lits', 'h': 'Réa (lits)'},\n",
            "    'Medecin': {'p': 'Medecins', 'h': 'Médecins'},\n",
            "    'Infirmier': {'p': 'Infirmiers', 'h': 'Infirmiers'},\n",
            "    'Urgentiste': {'p': 'Urgentistes', 'h': 'Urgentistes'},\n",
            "    'Reanimateur': {'p': 'Reanimateurs', 'h': 'Réanimateurs'},\n",
            "    'Anesth_Rea': {'p': 'Anesth_Rea', 'h': 'Anesth.-Réa'},\n",
            "    'Pneumo': {'p': 'Pneumo', 'h': 'Pneumo'},\n",
            "    'Cardio': {'p': 'Cardio', 'h': 'Cardio'},\n",
            "    'Neuro': {'p': 'Neuro', 'h': 'Neuro'},\n",
            "    'Interniste': {'p': 'Internistes', 'h': 'Internistes'},\n",
            "    'Chirurgien': {'p': 'Chirurgiens', 'h': 'Chirurgiens'},\n",
            "    'Pediatre': {'p': 'Pediatres', 'h': 'Pédiatres'},\n",
            "    'Biologiste': {'p': 'Biologistes', 'h': 'Biologistes'},\n",
            "    'Radiologue': {'p': 'Radiologues', 'h': 'Radiologues'},\n",
            "    'O2': {'p': 'O2', 'h': 'O₂'},\n",
            "    'Ventilateur': {'p': 'Ventilateur', 'h': 'Ventilateurs'},\n",
            "    'Sang': {'p': 'Sang', 'h': 'Sang (unités)'},\n",
            "    'Moniteur': {'p': 'Moniteurs', 'h': 'Moniteurs'},\n",
            "    'Defibrillateur': {'p': 'Defibrillateur', 'h': 'Défibrillateurs'},\n",
            "    'Monitoring': {'p': 'Monitoring', 'h': 'Moniteurs'},\n",
            "    'Labo': {'p': 'Labo', 'h': 'Labo'},\n",
            "    'Imagerie': {'p': 'Imagerie', 'h': 'Imagerie'}\n",
            "}\n",
            "\n",
            "# Nettoyage : forcer la conversion en nombre pour les comparaisons\n",
            "for key, mapping in RESOURCE_MAPPINGS.items():\n",
            "    col_p = mapping['p']\n",
            "    col_h = mapping['h']\n",
            "    if col_p in df_patients.columns:\n",
            "        df_patients[col_p] = pd.to_numeric(df_patients[col_p], errors='coerce').fillna(0)\n",
            "    if col_h in df_hopitaux.columns:\n",
            "        df_hopitaux[col_h] = pd.to_numeric(df_hopitaux[col_h], errors='coerce').fillna(0)\n",
            "\n",
            "def allouer_hopital(patient, df_hopitaux):\n",
            "    hopitaux_eligibles = []\n",
            "    \n",
            "    # 1. Filtrer les hôpitaux qui ont les ressources nécessaires\n",
            "    for idx, hopital in df_hopitaux.iterrows():\n",
            "        est_eligible = True\n",
            "        for key, mapping in RESOURCE_MAPPINGS.items():\n",
            "            col_patient = mapping['p']\n",
            "            col_hopital = mapping['h']\n",
            "            \n",
            "            # Vérification uniquement si les deux colonnes existent\n",
            "            if col_patient in patient and col_hopital in hopital:\n",
            "                besoin = patient[col_patient]\n",
            "                capacite = hopital[col_hopital]\n",
            "                \n",
            "                # Conversion robuste pour éviter les erreurs de type string/int\n",
            "                try:\n",
            "                    besoin = float(besoin)\n",
            "                except:\n",
            "                    besoin = 0.0\n",
            "                \n",
            "                try:\n",
            "                    capacite = float(capacite)\n",
            "                except:\n",
            "                    capacite = 0.0\n",
            "                \n",
            "                if besoin > 0 and capacite < besoin:\n",
            "                    est_eligible = False\n",
            "                    break\n",
            "        \n",
            "        if est_eligible:\n",
            "            hopitaux_eligibles.append(hopital)\n",
            "    \n",
            "    # 2. Gérer le cas où aucun n'est éligible (par défaut, on assigne au plus grand hôpital)\n",
            "    if not hopitaux_eligibles:\n",
            "        return df_hopitaux.iloc[0]['Hôpital']\n",
            "    \n",
            "    # 3. Parmi les éligibles, trouver le plus proche\n",
            "    distances = []\n",
            "    for h in hopitaux_eligibles:\n",
            "        dist = haversine(patient['Latitude'], patient['Longitude'], h['Lat'], h['Long'])\n",
            "        distances.append((h['Hôpital'], dist))\n",
            "        \n",
            "    distances.sort(key=lambda x: x[1])\n",
            "    return distances[0][0]\n",
            "\n",
            "print(\"Génération de l'allocation pour tous les patients...\")\n",
            "df_patients['Hopital_Alloue'] = df_patients.apply(lambda row: allouer_hopital(row, df_hopitaux), axis=1)\n",
            "print(\"Répartition des allocations :\")\n",
            "print(df_patients['Hopital_Alloue'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Préparation des données pour XGBoost\n",
            "Encodage de la cible (Hôpital) et sélection des caractéristiques (Features)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Sélection des features (X)\n",
            "# On prend les données vitales, l'urgence, les coordonnées géographiques\n",
            "features_cols = ['ESI', 'SpO2', 'FC', 'FR', 'GCS', 'Latitude', 'Longitude'] \n",
            "\n",
            "# On ajoute les ressources demandées par le patient comme features additionnelles\n",
            "ressources_cols = [v['p'] for k, v in RESOURCE_MAPPINGS.items() if v['p'] in df_patients.columns]\n",
            "features_cols.extend(ressources_cols)\n",
            "\n",
            "# Retirer les doublons potentiels\n",
            "features_cols = list(set(features_cols))\n",
            "\n",
            "X = df_patients[features_cols].copy()\n",
            "\n",
            "# Remplacement des valeurs manquantes par 0\n",
            "X = X.fillna(0)\n",
            "\n",
            "# Encodage de la Target (y)\n",
            "label_encoder = LabelEncoder()\n",
            "y = label_encoder.fit_transform(df_patients['Hopital_Alloue'])\n",
            "\n",
            "print(\"Shape de X :\", X.shape)\n",
            "print(\"Classes cibles (Hôpitaux) :\", label_encoder.classes_)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Entraînement du modèle XGBoost"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Division en ensemble d'entraînement et de test (80% / 20%)\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "\n",
            "# Initialisation du modèle\n",
            "model = xgb.XGBClassifier(\n",
            "    objective='multi:softmax',\n",
            "    num_class=len(label_encoder.classes_),\n",
            "    eval_metric='mlogloss',\n",
            "    random_state=42\n",
            ")\n",
            "\n",
            "print(\"Entraînement du modèle en cours...\")\n",
            "model.fit(X_train, y_train)\n",
            "print(\"Entraînement terminé !\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Évaluation du modèle\n",
            "Vérifions les performances globales du modèle sur les données de test."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Prédictions sur l'ensemble de test\n",
            "y_pred = model.predict(X_test)\n",
            "\n",
            "# Calcul de la précision\n",
            "accuracy = accuracy_score(y_test, y_pred)\n",
            "print(f\"Précision (Accuracy) : {accuracy * 100:.2f}%\\n\")\n",
            "\n",
            "# Rapport détaillé\n",
            "noms_classes = [str(c) for c in label_encoder.classes_]\n",
            "\n",
            "# On gère le cas où certaines classes ne seraient pas dans le test set\n",
            "classes_in_test = np.unique(y_test)\n",
            "target_names = [noms_classes[i] for i in classes_in_test]\n",
            "\n",
            "print(\"Rapport de classification :\")\n",
            "print(classification_report(y_test, y_pred, target_names=target_names))\n",
            "\n",
            "# Visualisation des variables les plus importantes pour les décisions de l'IA\n",
            "fig, ax = plt.subplots(figsize=(10, 8))\n",
            "xgb.plot_importance(model, max_num_features=15, ax=ax, title=\"Importance des Variables\")\n",
            "plt.show()"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("xgboost_patient_allocation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook 'xgboost_patient_allocation.ipynb' a été créé avec succès !")
