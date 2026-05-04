"""
Système d'allocation intelligente de patients vers les hôpitaux.
Composants :
  - HospitalState       : ressources dynamiques d'un hôpital
  - PatientQueue        : file de priorité FIFO par niveau ESI
  - QuarantineZone      : buffer pour patients non alloués (réinjection)
  - AllocationEngine    : moteur principal DNN + règles métier
"""

from __future__ import annotations

import heapq
import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constantes de colonnes
# ---------------------------------------------------------------------------

PATIENT_FEATURES = [
    "ESI", "SpO2", "FC", "FR", "GCS",
    "O2", "Ventilateur", "Sang", "Defibrillateur", "Monitoring",
    "Reanimation", "Labo", "Imagerie",
    "Medecins", "Infirmiers", "Urgentistes", "Reanimateurs", "Anesth_Rea",
    "Pneumo", "Cardio", "Neuro", "Internistes", "Chirurgiens", "Pediatres",
    "Biologistes", "Radiologues", "Moniteurs", "Rea_lits", "Lits_totaux",
    "Latitude", "Longitude",
]  # 31 features

HOSPITAL_RESOURCE_COLS = [
    "Médecins", "Infirmiers", "Urgentistes", "Réanimateurs", "Anesth.-Réa",
    "Pneumo", "Cardio", "Neuro", "Internistes", "Chirurgiens", "Pédiatres",
    "Biologistes", "Radiologues", "Labo", "Imagerie",
    "O₂", "Ventilateurs", "Moniteurs", "Défibrillateurs", "Sang (unités)",
    "Réa (lits)", "Lits totaux",
]  # 22 features  (+1 distance => 23 total pour le DNN)

RESOURCE_MAPPINGS: Dict[str, Dict[str, str]] = {
    "Lit":          {"p": "Lits_totaux",   "h": "Lits totaux"},
    "Rea_Lit":      {"p": "Rea_lits",      "h": "Réa (lits)"},
    "Medecin":      {"p": "Medecins",      "h": "Médecins"},
    "Infirmier":    {"p": "Infirmiers",    "h": "Infirmiers"},
    "Urgentiste":   {"p": "Urgentistes",   "h": "Urgentistes"},
    "Reanimateur":  {"p": "Reanimateurs",  "h": "Réanimateurs"},
    "Anesth_Rea":   {"p": "Anesth_Rea",   "h": "Anesth.-Réa"},
    "Pneumo":       {"p": "Pneumo",        "h": "Pneumo"},
    "Cardio":       {"p": "Cardio",        "h": "Cardio"},
    "Neuro":        {"p": "Neuro",         "h": "Neuro"},
    "Interniste":   {"p": "Internistes",   "h": "Internistes"},
    "Chirurgien":   {"p": "Chirurgiens",   "h": "Chirurgiens"},
    "Pediatre":     {"p": "Pediatres",     "h": "Pédiatres"},
    "Biologiste":   {"p": "Biologistes",   "h": "Biologistes"},
    "Radiologue":   {"p": "Radiologues",   "h": "Radiologues"},
    "O2":           {"p": "O2",            "h": "O₂"},
    "Ventilateur":  {"p": "Ventilateur",   "h": "Ventilateurs"},
    "Sang":         {"p": "Sang",          "h": "Sang (unités)"},
    "Moniteur":     {"p": "Moniteurs",     "h": "Moniteurs"},
    "Defibrillateur": {"p": "Defibrillateur", "h": "Défibrillateurs"},
    "Monitoring":   {"p": "Monitoring",    "h": "Moniteurs"},
    "Labo":         {"p": "Labo",          "h": "Labo"},
    "Imagerie":     {"p": "Imagerie",      "h": "Imagerie"},
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en km entre deux coordonnées GPS."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def patient_needs(patient_features: Dict) -> Dict[str, float]:
    """Convertit les besoins patient en colonnes hôpital (agrégation si doublon)."""
    acc: Dict[str, float] = defaultdict(float)
    for spec in RESOURCE_MAPPINGS.values():
        p_col, h_col = spec["p"], spec["h"]
        val = patient_features.get(p_col, 0)
        try:
            acc[h_col] += float(val)
        except (TypeError, ValueError):
            pass
    return dict(acc)


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    """Représente un patient avec ses données cliniques."""
    id: int
    features: Dict          # colonnes issues du DataFrame patients
    arrival_time: float = field(default_factory=time.time)

    @property
    def esi(self) -> int:
        return int(self.features.get("ESI", 5))

    @property
    def latitude(self) -> float:
        return float(self.features.get("Latitude", 0))

    @property
    def longitude(self) -> float:
        return float(self.features.get("Longitude", 0))

    def to_feature_vector(self) -> np.ndarray:
        return np.array(
            [float(self.features.get(f, 0)) for f in PATIENT_FEATURES],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# File de priorité par ESI (FIFO à niveau ESI égal)
# ---------------------------------------------------------------------------

class PatientQueue:
    """
    Min-heap trié par (ESI, arrival_counter).
    ESI 1 = critique = sort en premier.
    """

    def __init__(self):
        self._heap: list = []
        self._counter: int = 0

    def push(self, patient: Patient) -> None:
        heapq.heappush(self._heap, (patient.esi, self._counter, patient))
        self._counter += 1

    def pop(self) -> Optional[Patient]:
        if self._heap:
            _, _, p = heapq.heappop(self._heap)
            return p
        return None

    def peek_esi(self) -> Optional[int]:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


# ---------------------------------------------------------------------------
# Zone de quarantaine (buffer dynamique)
# ---------------------------------------------------------------------------

class QuarantineZone:
    """
    Buffer de patients qui n'ont pu être alloués.
    Réinjection déclenchée manuellement ou après mise à jour des ressources.
    """

    def __init__(self):
        self._queue = PatientQueue()
        self.log: List[Dict] = []          # historique des entrées
        self._lock = threading.Lock()

    def admit(self, patient: Patient, reason: str = "no_feasible_hospital") -> None:
        with self._lock:
            self._queue.push(patient)
            self.log.append({
                "patient_id":    patient.id,
                "ESI":           patient.esi,
                "reason":        reason,
                "admitted_at":   time.time(),
            })

    def drain(self) -> List[Patient]:
        """Extrait tous les patients dans l'ordre de priorité pour réinjection."""
        with self._lock:
            patients: List[Patient] = []
            while self._queue:
                patients.append(self._queue.pop())
            return patients

    def __len__(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.log) if self.log else pd.DataFrame()


# ---------------------------------------------------------------------------
# État dynamique d'un hôpital
# ---------------------------------------------------------------------------

class HospitalState:
    """
    Suivi en temps réel des ressources disponibles d'un hôpital.
    Thread-safe (verrou par hôpital).
    """

    SATURATION_THRESHOLD = 0.95   # 95 % d'utilisation des lits → saturé

    def __init__(self, row: pd.Series):
        self.name: str = str(row["Hôpital"])
        self.lat: float = float(row.get("Lat", 0))
        self.lon: float = float(row.get("Long", 0))
        self._max: Dict[str, float] = {
            c: float(row.get(c, 0)) for c in HOSPITAL_RESOURCE_COLS
        }
        self._current: Dict[str, float] = dict(self._max)
        self._lock = threading.Lock()
        self.patients_allocated: int = 0

    # --- Propriétés ---

    @property
    def saturation(self) -> float:
        max_lits = self._max.get("Lits totaux", 1) or 1
        curr_lits = self._current.get("Lits totaux", 0)
        return 1.0 - curr_lits / max_lits

    @property
    def is_saturated(self) -> bool:
        return self.saturation >= self.SATURATION_THRESHOLD

    def available(self, col: str) -> float:
        return self._current.get(col, 0.0)

    # --- Opérations ---

    def can_satisfy(self, needs: Dict[str, float]) -> bool:
        """Vérifie si les ressources actuelles couvrent les besoins."""
        with self._lock:
            for h_col, need in needs.items():
                if need > 0 and self._current.get(h_col, 0) < need:
                    return False
        return True

    def consume(self, needs: Dict[str, float]) -> bool:
        """Consomme les ressources. Retourne False si impossible."""
        with self._lock:
            # Double-check sous verrou
            for h_col, need in needs.items():
                if need > 0 and self._current.get(h_col, 0) < need:
                    return False
            for h_col, need in needs.items():
                if h_col in self._current:
                    self._current[h_col] = max(0.0, self._current[h_col] - need)
            self.patients_allocated += 1
        return True

    def release(self, needs: Dict[str, float]) -> None:
        """Libère les ressources quand le patient quitte l'hôpital."""
        with self._lock:
            for h_col, need in needs.items():
                if h_col in self._current:
                    self._current[h_col] = min(
                        self._max[h_col],
                        self._current[h_col] + need,
                    )

    def resource_vector(self) -> np.ndarray:
        """Vecteur de ressources actuelles (pour le DNN)."""
        return np.array(
            [self._current.get(c, 0.0) for c in HOSPITAL_RESOURCE_COLS],
            dtype=np.float32,
        )

    def status(self) -> Dict:
        return {
            "hospital":            self.name,
            "saturation_%":        round(self.saturation * 100, 1),
            "is_saturated":        self.is_saturated,
            "lits_disponibles":    int(self._current.get("Lits totaux", 0)),
            "patients_allocated":  self.patients_allocated,
        }


# ---------------------------------------------------------------------------
# Moteur d'allocation
# ---------------------------------------------------------------------------

class AllocationEngine:
    """
    Moteur principal :
      1. Reçoit les patients depuis la file principale (ESI FIFO).
      2. Utilise le DNN pour scorer chaque hôpital.
      3. Vérifie la faisabilité en temps réel (ressources actuelles).
      4. Si aucun hôpital → quarantaine.
      5. Réinjection automatique depuis la quarantaine à chaque libération.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scaler_p,           # StandardScaler patients
        scaler_h,           # StandardScaler hôpitaux
        label_encoder,      # LabelEncoder noms hôpitaux
        hospitals_df: pd.DataFrame,
        device: torch.device,
    ):
        self.model = model
        self.model.eval()
        self.scaler_p = scaler_p
        self.scaler_h = scaler_h
        self.label_encoder = label_encoder
        self.device = device

        # État dynamique des hôpitaux
        self.hospitals: Dict[str, HospitalState] = {
            row["Hôpital"]: HospitalState(row)
            for _, row in hospitals_df.iterrows()
        }

        # Files
        self.main_queue = PatientQueue()
        self.quarantine = QuarantineZone()

        # Logs
        self.allocation_log: List[Dict] = []
        self.stats = {"allocated": 0, "quarantined": 0, "reinjected": 0, "failed_reinjection": 0}

        # Verrou global pour les réinjections
        self._reinject_lock = threading.Lock()

    # --- Scoring DNN ---

    def _score(self, p_vec: np.ndarray, hosp: HospitalState) -> float:
        """Score DNN pour la paire (patient, hôpital)."""
        dist = haversine(p_vec[PATIENT_FEATURES.index("Latitude")],
                         p_vec[PATIENT_FEATURES.index("Longitude")],
                         hosp.lat, hosp.lon)
        h_vec = np.append(hosp.resource_vector(), dist)   # 22 ressources + distance = 23

        p_norm = self.scaler_p.transform([p_vec])[0]
        h_norm = self.scaler_h.transform([h_vec])[0]

        x = torch.FloatTensor(np.concatenate([p_norm, h_norm])).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.model(x).item())

    # --- Allocation d'un patient ---

    def _try_allocate(self, patient: Patient) -> Optional[str]:
        """
        Tente d'allouer un patient.
        Retourne le nom de l'hôpital choisi, ou None (→ quarantaine).
        """
        p_vec = patient.to_feature_vector()
        needs = patient_needs(patient.features)

        # Scorer tous les hôpitaux non saturés
        scores: List[Tuple[float, str]] = []
        for name, h_state in self.hospitals.items():
            if h_state.is_saturated:
                continue
            if not h_state.can_satisfy(needs):
                continue
            score = self._score(p_vec, h_state)
            scores.append((score, name))

        if not scores:
            return None   # Pas d'hôpital disponible

        # Trier par score décroissant (DNN préfère l'hôpital optimal)
        scores.sort(reverse=True)

        for score, name in scores:
            h_state = self.hospitals[name]
            if h_state.consume(needs):
                self.allocation_log.append({
                    "patient_id":  patient.id,
                    "ESI":         patient.esi,
                    "hospital":    name,
                    "dnn_score":   round(score, 4),
                    "saturation":  round(h_state.saturation * 100, 1),
                    "allocated_at": time.time(),
                })
                self.stats["allocated"] += 1
                return name

        return None

    def admit(self, patient: Patient) -> None:
        """Ajoute un patient à la file principale."""
        self.main_queue.push(patient)

    def process_next(self) -> Tuple[Optional[Patient], Optional[str]]:
        """
        Traite le prochain patient de la file.
        Retourne (patient, hôpital_assigné) ou (patient, None) si quarantaine.
        """
        patient = self.main_queue.pop()
        if patient is None:
            return None, None

        hospital = self._try_allocate(patient)
        if hospital is None:
            self.quarantine.admit(patient, reason="no_feasible_hospital")
            self.stats["quarantined"] += 1

        return patient, hospital

    def process_all(self) -> List[Dict]:
        """Traite toute la file principale. Retourne le log d'allocation."""
        while self.main_queue:
            self.process_next()
        return self.allocation_log

    # --- Réinjection depuis la quarantaine ---

    def reinject(self) -> Dict[str, int]:
        """
        Tente de réallouer tous les patients en quarantaine.
        Déclenché automatiquement après libération de ressources.
        """
        with self._reinject_lock:
            waiting = self.quarantine.drain()
            reinjected, failed = 0, 0

            for patient in waiting:
                hospital = self._try_allocate(patient)
                if hospital is not None:
                    reinjected += 1
                    self.stats["reinjected"] += 1
                else:
                    # Toujours pas faisable → retour en quarantaine
                    self.quarantine.admit(patient, reason="reinjection_failed")
                    failed += 1
                    self.stats["failed_reinjection"] += 1

            return {"reinjected": reinjected, "still_waiting": failed}

    def release_patient(self, patient: Patient, hospital_name: str) -> Dict[str, int]:
        """
        Libère les ressources d'un patient sorti d'un hôpital,
        puis déclenche automatiquement une réinjection depuis la quarantaine.
        """
        if hospital_name in self.hospitals:
            needs = patient_needs(patient.features)
            self.hospitals[hospital_name].release(needs)

        # Réinjection automatique
        return self.reinject()

    # --- Tableau de bord ---

    def dashboard(self) -> pd.DataFrame:
        """Vue d'ensemble de l'état de tous les hôpitaux."""
        return pd.DataFrame([h.status() for h in self.hospitals.values()])

    def allocation_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.allocation_log)

    def global_stats(self) -> Dict:
        return {
            **self.stats,
            "en_attente_file":      len(self.main_queue),
            "en_quarantaine":       len(self.quarantine),
            "hopitaux_satures":     sum(1 for h in self.hospitals.values() if h.is_saturated),
        }
