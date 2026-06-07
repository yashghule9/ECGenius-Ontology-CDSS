"""
ECGenius — scripts/generate_mock_aiims_data.py
===============================================
Generates a clinically realistic synthetic patient_history.csv for
development, demo, and evaluation without real AIIMS data.

NOT random noise — every symptom-disease association follows:
  • ACC/AHA clinical guidelines
  • Framingham risk factor correlations
  • Standard triage presentations

Disease profiles encoded (prevalence-weighted, 15 labels):
  NSR, AF, STEMI, NSTEMI, LVH, VF, VT, SVT,
  LBBB, RBBB, Ischemia, ST_Elevation, ST_Depression, TWI, PVC

Output:
  data/raw/aiims/patient_history.csv    — one row per patient
  data/raw/aiims/metadata.csv           — demographics

Usage:
    python scripts/generate_mock_aiims_data.py --n 500 --seed 42
    python scripts/generate_mock_aiims_data.py --n 200 --seed 42 --output /custom/path.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Disease profile definition ────────────────────────────────────────────────

@dataclass
class DiseaseProfile:
    """
    Encodes the clinical presentation of one diagnosis.
    All probabilities are conditional P(feature | disease).
    Missing value rates simulate real hospital data noise.
    """
    name:         str
    prevalence:   float    # fraction of dataset

    # Age distribution (years)
    age_mean:  float
    age_std:   float
    age_min:   float = 18.0
    age_max:   float = 90.0

    # P(male)
    p_male: float = 0.55

    # Symptoms: P(present | disease). Key = symptom name.
    symptoms: dict[str, float] = field(default_factory=dict)

    # Risk factors: P(present | disease)
    risk_factors: dict[str, float] = field(default_factory=dict)

    # Vitals: (mean, std, min, max)  — None = not relevant / skip
    vitals: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    # P(vital is missing) — simulates hospital noise
    vital_missing_rate: float = 0.10

    # P(symptom field is missing)
    symptom_missing_rate: float = 0.08


# ── Clinical profile table ────────────────────────────────────────────────────
# Sources:
#   STEMI: 2013 AHA/ACC STEMI Guideline
#   AF: 2019 AHA/ACC Atrial Fibrillation Guideline
#   HF/LVH: 2022 AHA/ACC Heart Failure Guideline
#   VF/VT: 2017 AHA/ACC VA Guideline
#   SVT: 2015 ACC/AHA/HRS SVT Guideline
#   Bundle blocks: AHA Scientific Statement 2010

DISEASE_PROFILES: list[DiseaseProfile] = [

    DiseaseProfile(
        name="NSR", prevalence=0.18,
        age_mean=45, age_std=18, p_male=0.50,
        symptoms={
            "chest_pain": 0.05, "palpitations": 0.05,
            "dyspnea": 0.06, "syncope": 0.02,
            "fatigue": 0.10, "diaphoresis": 0.02,
            "nausea": 0.05, "dizziness": 0.06,
        },
        risk_factors={
            "htn": 0.30, "dm": 0.10, "cad": 0.08,
            "smoking": 0.20, "obesity": 0.25,
            "hyperlipidemia": 0.22, "family_hx_cvd": 0.15,
        },
        vitals={
            "sbp": (122, 14, 90, 150),
            "dbp": (78,  9, 60, 100),
            "hr":  (72,  10, 55, 95),
            "spo2":(98,  1,  94, 100),
            "rr":  (16,  2,  12, 20),
        },
    ),

    DiseaseProfile(
        name="AF", prevalence=0.14,
        age_mean=68, age_std=12, p_male=0.55,
        symptoms={
            "palpitations": 0.78,  # hallmark
            "dyspnea":      0.52,
            "fatigue":      0.48,
            "dizziness":    0.30,
            "chest_pain":   0.18,
            "syncope":      0.08,
            "diaphoresis":  0.06,
            "nausea":       0.10,
        },
        risk_factors={
            "htn":          0.70,  # strongest modifiable RF for AF
            "dm":           0.25,
            "cad":          0.30,
            "smoking":      0.22,
            "obesity":      0.35,
            "hyperlipidemia":0.40,
            "family_hx_cvd":0.20,
            "alcohol":      0.18,
            "osa":          0.28,  # obstructive sleep apnea
        },
        vitals={
            "sbp":  (132, 18, 90, 180),
            "dbp":  (82,  12, 55, 110),
            "hr":   (108, 25, 60, 160),  # irregularly irregular, variable rate
            "spo2": (96,  2,  88, 100),
            "rr":   (18,  3,  12, 28),
        },
    ),

    DiseaseProfile(
        name="STEMI", prevalence=0.08,
        age_mean=60, age_std=13, p_male=0.72,
        symptoms={
            "chest_pain":   0.92,  # severe, crushing, radiating
            "diaphoresis":  0.72,
            "nausea":       0.60,
            "dyspnea":      0.58,
            "dizziness":    0.35,
            "syncope":      0.12,
            "palpitations": 0.25,
            "fatigue":      0.40,
        },
        risk_factors={
            "htn":          0.65,
            "dm":           0.32,
            "cad":          0.55,   # prior CAD is strongest predictor
            "smoking":      0.48,
            "obesity":      0.38,
            "hyperlipidemia":0.58,
            "family_hx_cvd":0.42,
            "prior_mi":     0.28,
        },
        vitals={
            "sbp":  (104, 22, 60, 160),  # often hypotensive in anterior STEMI
            "dbp":  (68,  16, 40, 105),
            "hr":   (98,  20, 50, 140),  # can be tachycardic or bradycardic (inferior MI)
            "spo2": (94,  3,  84, 99),
            "rr":   (20,  4,  14, 32),
        },
        vital_missing_rate=0.05,   # emergency — vitals usually recorded
    ),

    DiseaseProfile(
        name="NSTEMI", prevalence=0.10,
        age_mean=64, age_std=12, p_male=0.65,
        symptoms={
            "chest_pain":   0.85,
            "dyspnea":      0.62,
            "diaphoresis":  0.45,
            "nausea":       0.38,
            "fatigue":      0.50,
            "dizziness":    0.28,
            "palpitations": 0.18,
            "syncope":      0.06,
        },
        risk_factors={
            "htn":          0.62,
            "dm":           0.36,
            "cad":          0.58,
            "smoking":      0.42,
            "obesity":      0.36,
            "hyperlipidemia":0.55,
            "family_hx_cvd":0.40,
            "prior_mi":     0.32,
        },
        vitals={
            "sbp":  (118, 20, 75, 175),
            "dbp":  (74,  14, 45, 108),
            "hr":   (88,  18, 48, 130),
            "spo2": (95,  2,  87, 100),
            "rr":   (19,  4,  12, 30),
        },
    ),

    DiseaseProfile(
        name="LVH", prevalence=0.08,
        age_mean=58, age_std=14, p_male=0.58,
        symptoms={
            "dyspnea":      0.50,  # exertional primarily
            "chest_pain":   0.20,
            "fatigue":      0.45,
            "palpitations": 0.22,
            "dizziness":    0.20,
            "syncope":      0.08,
            "diaphoresis":  0.08,
            "nausea":       0.05,
        },
        risk_factors={
            "htn":          0.85,  # LVH is direct consequence of sustained HTN
            "dm":           0.30,
            "cad":          0.28,
            "smoking":      0.25,
            "obesity":      0.42,
            "hyperlipidemia":0.38,
            "family_hx_cvd":0.32,
        },
        vitals={
            "sbp":  (148, 22, 110, 210),  # typically hypertensive
            "dbp":  (92,  14, 65, 130),
            "hr":   (78,  12, 55, 105),
            "spo2": (97,  2,  90, 100),
            "rr":   (17,  3,  12, 24),
        },
    ),

    DiseaseProfile(
        name="VF", prevalence=0.03,
        age_mean=62, age_std=15, p_male=0.70,
        symptoms={
            "syncope":      0.95,  # loss of consciousness = VF defining presentation
            "chest_pain":   0.40,
            "palpitations": 0.30,
            "dyspnea":      0.25,
            "diaphoresis":  0.35,
            "nausea":       0.20,
            "dizziness":    0.50,
            "fatigue":      0.30,
        },
        risk_factors={
            "htn":          0.55,
            "dm":           0.32,
            "cad":          0.65,
            "smoking":      0.40,
            "prior_mi":     0.52,
            "cardiomyopathy":0.38,
            "hyperlipidemia":0.48,
            "family_hx_cvd":0.38,
        },
        vitals={
            "sbp":  (70,  30, 0,  120),   # near-collapse or absent
            "dbp":  (45,  20, 0,  80),
            "hr":   (180, 60, 0,  300),   # VF = chaotic, unmeasurable
            "spo2": (80,  10, 50, 95),
            "rr":   (8,   8,  0,  30),    # agonal breathing
        },
        vital_missing_rate=0.25,   # emergency — chaos
    ),

    DiseaseProfile(
        name="VT", prevalence=0.04,
        age_mean=58, age_std=15, p_male=0.65,
        symptoms={
            "palpitations": 0.82,
            "dizziness":    0.65,
            "syncope":      0.38,
            "chest_pain":   0.35,
            "dyspnea":      0.42,
            "diaphoresis":  0.30,
            "fatigue":      0.38,
            "nausea":       0.18,
        },
        risk_factors={
            "cad":          0.62,
            "prior_mi":     0.50,
            "cardiomyopathy":0.35,
            "htn":          0.45,
            "dm":           0.28,
            "hyperlipidemia":0.40,
        },
        vitals={
            "sbp":  (95,  25, 50, 150),
            "dbp":  (62,  18, 30, 100),
            "hr":   (165, 30, 120, 250),
            "spo2": (93,  4,  82, 99),
            "rr":   (22,  5,  14, 36),
        },
    ),

    DiseaseProfile(
        name="SVT", prevalence=0.06,
        age_mean=38, age_std=16, p_male=0.42,  # SVT more common in young women
        symptoms={
            "palpitations": 0.90,  # abrupt onset palpitations = classic SVT
            "dizziness":    0.55,
            "chest_pain":   0.28,
            "dyspnea":      0.35,
            "syncope":      0.12,
            "fatigue":      0.30,
            "diaphoresis":  0.20,
            "nausea":       0.22,
        },
        risk_factors={
            "young_age":    0.70,
            "caffeine":     0.40,
            "anxiety":      0.35,
            "alcohol":      0.22,
            "htn":          0.20,
            "hyperthyroid": 0.08,
        },
        vitals={
            "sbp":  (108, 18, 75, 150),
            "dbp":  (68,  12, 45, 100),
            "hr":   (175, 25, 140, 230),
            "spo2": (97,  2,  90, 100),
            "rr":   (18,  4,  12, 28),
        },
    ),

    DiseaseProfile(
        name="LBBB", prevalence=0.06,
        age_mean=66, age_std=13, p_male=0.60,
        symptoms={
            "fatigue":      0.45,
            "dyspnea":      0.40,
            "chest_pain":   0.22,
            "palpitations": 0.18,
            "dizziness":    0.15,
            "syncope":      0.08,
            "diaphoresis":  0.08,
            "nausea":       0.06,
        },
        risk_factors={
            "htn":          0.60,
            "cad":          0.48,
            "heart_failure":0.42,
            "dm":           0.28,
            "hyperlipidemia":0.42,
            "prior_mi":     0.30,
        },
        vitals={
            "sbp":  (128, 20, 90, 175),
            "dbp":  (80,  12, 55, 110),
            "hr":   (78,  14, 50, 110),
            "spo2": (95,  3,  86, 100),
            "rr":   (18,  3,  12, 26),
        },
    ),

    DiseaseProfile(
        name="RBBB", prevalence=0.06,
        age_mean=55, age_std=16, p_male=0.58,
        symptoms={
            "dyspnea":      0.28,  # often incidental
            "chest_pain":   0.15,
            "fatigue":      0.25,
            "palpitations": 0.12,
            "dizziness":    0.10,
            "syncope":      0.05,
            "diaphoresis":  0.06,
            "nausea":       0.05,
        },
        risk_factors={
            "htn":          0.35,
            "cad":          0.22,
            "pe":           0.12,  # acute RBBB can be PE
            "dm":           0.18,
            "obesity":      0.28,
            "hyperlipidemia":0.28,
        },
        vitals={
            "sbp":  (125, 16, 90, 165),
            "dbp":  (79,  10, 55, 105),
            "hr":   (74,  12, 52, 100),
            "spo2": (97,  2,  90, 100),
            "rr":   (16,  3,  12, 22),
        },
    ),

    DiseaseProfile(
        name="Ischemia", prevalence=0.06,
        age_mean=61, age_std=13, p_male=0.62,
        symptoms={
            "chest_pain":   0.65,
            "dyspnea":      0.40,
            "fatigue":      0.42,
            "diaphoresis":  0.28,
            "nausea":       0.22,
            "dizziness":    0.22,
            "palpitations": 0.15,
            "syncope":      0.05,
        },
        risk_factors={
            "cad":          0.70,
            "htn":          0.60,
            "dm":           0.38,
            "smoking":      0.45,
            "hyperlipidemia":0.55,
            "family_hx_cvd":0.42,
            "prior_mi":     0.30,
        },
        vitals={
            "sbp":  (120, 20, 80, 170),
            "dbp":  (76,  14, 50, 108),
            "hr":   (86,  18, 50, 120),
            "spo2": (95,  3,  85, 100),
            "rr":   (18,  4,  12, 28),
        },
    ),

    DiseaseProfile(
        name="PVC", prevalence=0.05,
        age_mean=48, age_std=18, p_male=0.52,
        symptoms={
            "palpitations": 0.72,  # "skipped beat" sensation
            "fatigue":      0.32,
            "dizziness":    0.20,
            "chest_pain":   0.12,
            "dyspnea":      0.15,
            "syncope":      0.04,
            "diaphoresis":  0.05,
            "nausea":       0.08,
        },
        risk_factors={
            "caffeine":     0.50,
            "anxiety":      0.38,
            "htn":          0.32,
            "cad":          0.20,
            "alcohol":      0.22,
            "dm":           0.15,
            "smoking":      0.25,
        },
        vitals={
            "sbp":  (120, 14, 90, 155),
            "dbp":  (78,  10, 58, 100),
            "hr":   (72,  12, 52, 100),
            "spo2": (98,  1,  94, 100),
            "rr":   (15,  2,  12, 20),
        },
    ),

    DiseaseProfile(
        name="TWI", prevalence=0.03,
        age_mean=57, age_std=14, p_male=0.58,
        symptoms={
            "chest_pain":   0.38,
            "dyspnea":      0.32,
            "fatigue":      0.40,
            "palpitations": 0.18,
            "dizziness":    0.18,
            "diaphoresis":  0.15,
            "nausea":       0.12,
            "syncope":      0.06,
        },
        risk_factors={
            "cad":          0.48,
            "htn":          0.52,
            "dm":           0.28,
            "hyperlipidemia":0.42,
            "prior_mi":     0.22,
            "smoking":      0.32,
        },
        vitals={
            "sbp":  (124, 18, 88, 168),
            "dbp":  (79,  12, 55, 108),
            "hr":   (82,  16, 52, 115),
            "spo2": (96,  2,  88, 100),
            "rr":   (17,  3,  12, 25),
        },
    ),

    DiseaseProfile(
        name="ST_Elevation", prevalence=0.02,
        age_mean=58, age_std=14, p_male=0.70,
        symptoms={
            "chest_pain":   0.88,
            "diaphoresis":  0.62,
            "nausea":       0.50,
            "dyspnea":      0.48,
            "dizziness":    0.30,
            "palpitations": 0.20,
            "syncope":      0.10,
            "fatigue":      0.35,
        },
        risk_factors={
            "cad":          0.60,
            "htn":          0.58,
            "dm":           0.30,
            "smoking":      0.45,
            "prior_mi":     0.25,
            "hyperlipidemia":0.52,
        },
        vitals={
            "sbp":  (102, 24, 60, 160),
            "dbp":  (66,  18, 40, 105),
            "hr":   (100, 22, 55, 145),
            "spo2": (93,  4,  82, 99),
            "rr":   (21,  5,  14, 34),
        },
    ),

    DiseaseProfile(
        name="ST_Depression", prevalence=0.03,
        age_mean=63, age_std=12, p_male=0.60,
        symptoms={
            "chest_pain":   0.70,
            "dyspnea":      0.45,
            "fatigue":      0.45,
            "diaphoresis":  0.32,
            "dizziness":    0.25,
            "nausea":       0.22,
            "palpitations": 0.15,
            "syncope":      0.06,
        },
        risk_factors={
            "cad":          0.65,
            "htn":          0.60,
            "dm":           0.35,
            "smoking":      0.40,
            "prior_mi":     0.28,
            "hyperlipidemia":0.52,
        },
        vitals={
            "sbp":  (118, 20, 78, 172),
            "dbp":  (75,  14, 48, 108),
            "hr":   (90,  18, 52, 128),
            "spo2": (94,  3,  84, 99),
            "rr":   (19,  4,  12, 30),
        },
    ),
]

# Normalise prevalences to sum to 1
_total_prev = sum(p.prevalence for p in DISEASE_PROFILES)
for _p in DISEASE_PROFILES:
    _p.prevalence /= _total_prev


# ── Generator ─────────────────────────────────────────────────────────────────

class MockAIIMSGenerator:
    """
    Generates one synthetic patient row per call.
    Medical correlations are enforced — not random.
    """

    ALL_SYMPTOMS = [
        "chest_pain", "palpitations", "dyspnea", "syncope",
        "fatigue", "diaphoresis", "nausea", "dizziness",
    ]
    ALL_RISK_FACTORS = [
        "htn", "dm", "cad", "smoking", "obesity", "hyperlipidemia",
        "family_hx_cvd", "prior_mi", "cardiomyopathy", "alcohol",
        "osa", "pe", "caffeine", "anxiety", "young_age",
        "hyperthyroid", "heart_failure",
    ]
    ALL_VITALS = ["sbp", "dbp", "hr", "spo2", "rr"]

    def __init__(self, rng: np.random.Generator):
        self.rng       = rng
        self._profiles = DISEASE_PROFILES
        self._weights  = np.array([p.prevalence for p in self._profiles])

    def _choose_disease(self) -> DiseaseProfile:
        idx = self.rng.choice(len(self._profiles), p=self._weights)
        return self._profiles[idx]

    def _sample_bool(self, prob: float, missing_rate: float) -> Optional[str]:
        if self.rng.random() < missing_rate:
            return ""           # missing value
        return "1" if self.rng.random() < prob else "0"

    def _sample_vital(
        self, mean: float, std: float, vmin: float, vmax: float,
        missing_rate: float,
    ) -> str:
        if self.rng.random() < missing_rate:
            return ""           # missing value
        val = float(np.clip(self.rng.normal(mean, std), vmin, vmax))
        return f"{val:.1f}"

    def generate_patient(self, patient_id: str) -> dict:
        profile = self._choose_disease()

        # Demographics
        age    = float(np.clip(self.rng.normal(profile.age_mean, profile.age_std),
                               profile.age_min, profile.age_max))
        gender = "M" if self.rng.random() < profile.p_male else "F"

        row: dict[str, str] = {
            "patient_id":      patient_id,
            "primary_label":   profile.name,
            "age":             f"{age:.0f}",
            "gender":          gender,
        }

        # Symptoms
        for sym in self.ALL_SYMPTOMS:
            prob = profile.symptoms.get(sym, 0.05)   # 5% base rate
            row[f"sym_{sym}"] = self._sample_bool(prob, profile.symptom_missing_rate)

        # Risk factors
        for rf in self.ALL_RISK_FACTORS:
            prob = profile.risk_factors.get(rf, 0.05)
            row[f"rf_{rf}"] = self._sample_bool(prob, profile.symptom_missing_rate)

        # Vitals
        for vital in self.ALL_VITALS:
            if vital in profile.vitals:
                mean, std, vmin, vmax = profile.vitals[vital]
                row[vital] = self._sample_vital(mean, std, vmin, vmax,
                                                profile.vital_missing_rate)
            else:
                row[vital] = ""

        return row

    def generate_dataset(self, n: int) -> list[dict]:
        return [
            self.generate_patient(f"AIIMS{i+1:05d}")
            for i in range(n)
        ]


# ── Converter: raw CSV → pipeline patient dict ────────────────────────────────

def csv_row_to_patient_dict(row: dict) -> dict:
    """
    Convert one row from patient_history.csv to the patient dict format
    expected by history_encoder.py and run_pipeline.py.

    Output format:
        {
            "symptoms":     {"chest_pain": True/False, ...},
            "risk_factors": {"htn": True/False, ...},
            "vitals":       {"sbp": 120.0, "hr": 78.0, ...},
        }

    Missing values (empty string) → key omitted (encoder treats as absent).
    """
    patient: dict[str, dict] = {"symptoms": {}, "risk_factors": {}, "vitals": {}}

    for key, val in row.items():
        if val == "":
            continue    # missing — omit (do not assume False)

        if key.startswith("sym_"):
            sym_name = key[4:]
            patient["symptoms"][sym_name] = bool(int(val))

        elif key.startswith("rf_"):
            rf_name = key[3:]
            patient["risk_factors"][rf_name] = bool(int(val))

        elif key in ("sbp", "dbp", "hr", "spo2", "rr"):
            try:
                patient["vitals"][key] = float(val)
            except ValueError:
                pass

    return patient


# ── Writer ────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {output_path}  ({len(rows)} rows)")


def write_metadata(rows: list[dict], output_path: Path) -> None:
    meta_fields = ["patient_id", "primary_label", "age", "gender"]
    meta_rows   = [{k: r[k] for k in meta_fields} for r in rows]
    write_csv(meta_rows, output_path)


def print_summary(rows: list[dict]) -> None:
    from collections import Counter
    label_counts = Counter(r["primary_label"] for r in rows)
    ages  = [float(r["age"]) for r in rows if r["age"]]
    print("\n  Disease distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        bar = "#" * (count * 40 // len(rows))
        print(f"    {label:18s}  {count:4d}  {bar}")
    if ages:
        print(f"\n  Age: mean={np.mean(ages):.1f}  std={np.std(ages):.1f}  "
              f"range=[{min(ages):.0f},{max(ages):.0f}]")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic AIIMS patient history (clinically correlated)"
    )
    parser.add_argument("--n",      type=int, default=500,
                        help="Number of patients to generate (default 500)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument("--output", type=str,
                        default="data/raw/aiims/patient_history.csv",
                        help="Output CSV path")
    parser.add_argument("--metadata", type=str,
                        default="data/raw/aiims/metadata.csv")
    args = parser.parse_args()

    rng  = np.random.default_rng(args.seed)
    gen  = MockAIIMSGenerator(rng)

    print(f"\n  Generating {args.n} synthetic AIIMS patients (seed={args.seed})...")
    rows = gen.generate_dataset(args.n)

    output   = PROJECT_ROOT / args.output
    metadata = PROJECT_ROOT / args.metadata

    write_csv(rows, output)
    write_metadata(rows, metadata)
    print_summary(rows)

    print("  Usage in pipeline:")
    print(f"    python run_pipeline.py --mock --patient-csv {output}")
    print()


if __name__ == "__main__":
    main()
