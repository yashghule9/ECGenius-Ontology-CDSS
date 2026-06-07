"""
ECGenius — run_demo_aiims.py
==============================
End-to-end demo pipeline for clinicians.
Reads one patient from mock (or real) AIIMS data and runs the full
ECGenius system: model → ontology → rules → history → fusion → output.

Usage:
    # Full mock (no model, no real ECG needed):
    python run_demo_aiims.py --demo-aiims

    # With real model but mock ECG:
    python run_demo_aiims.py --demo-aiims --model models/checkpoints/lightv2_p1_fold2

    # With specific patient from generated CSV:
    python run_demo_aiims.py --demo-aiims --patient-csv data/raw/aiims/patient_history.csv --patient-id AIIMS00042

    # Custom patient JSON (for demo):
    python run_demo_aiims.py --demo-aiims --patient '{"symptoms":{"chest_pain":true,"dyspnea":true},"risk_factors":{"htn":true,"cad":true},"vitals":{"sbp":85,"hr":108,"spo2":93}}'

Generates first if no CSV exists:
    python scripts/generate_mock_aiims_data.py --n 500

This script does NOT modify any existing module.
It orchestrates the existing pipeline via its public API.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ECGenius.demo")


# ── Embedded mock disease scenarios (no CSV needed) ───────────────────────────

DEMO_SCENARIOS = {
    "stemi": {
        "label": "STEMI — Anterior MI (Crushing chest pain + Cardiogenic shock)",
        "patient": {
            "symptoms": {
                "chest_pain":   True,
                "diaphoresis":  True,
                "nausea":       True,
                "dyspnea":      True,
                "dizziness":    True,
            },
            "risk_factors": {
                "cad":       True,
                "htn":       True,
                "smoking":   True,
                "prior_mi":  True,
                "dm":        True,
            },
            "vitals": {"sbp": 82, "hr": 115, "spo2": 91},
        },
        "mock_model_output": {
            "STEMI": 0.91, "ST_Elevation": 0.88,
            "NSTEMI": 0.18, "NSR": 0.12, "VT": 0.08,
            "AF": 0.06, "LVH": 0.14, "LBBB": 0.12,
            "VF": 0.04, "SVT": 0.05, "RBBB": 0.06,
            "Ischemia": 0.35, "ST_Depression": 0.10,
            "TWI": 0.14, "PVC": 0.07,
        },
    },

    "af": {
        "label": "Atrial Fibrillation — Uncontrolled rate (Palpitations + Hypertensive)",
        "patient": {
            "symptoms": {
                "palpitations": True,
                "dyspnea":      True,
                "fatigue":      True,
                "dizziness":    True,
                "chest_pain":   False,
            },
            "risk_factors": {
                "htn":          True,
                "dm":           False,
                "alcohol":      True,
                "obesity":      True,
                "osa":          True,
            },
            "vitals": {"sbp": 138, "hr": 128, "spo2": 96},
        },
        "mock_model_output": {
            "AF": 0.88, "NSR": 0.06,
            "STEMI": 0.04, "NSTEMI": 0.08, "LVH": 0.32,
            "VF": 0.02, "VT": 0.05, "SVT": 0.12,
            "LBBB": 0.08, "RBBB": 0.06,
            "Ischemia": 0.10, "ST_Elevation": 0.04,
            "ST_Depression": 0.12, "TWI": 0.08, "PVC": 0.18,
        },
    },

    "lvh": {
        "label": "LVH — Hypertensive Heart Disease (Exertional dyspnea + HTN)",
        "patient": {
            "symptoms": {
                "dyspnea":      True,
                "fatigue":      True,
                "chest_pain":   False,
                "palpitations": True,
                "dizziness":    False,
            },
            "risk_factors": {
                "htn":          True,
                "dm":           True,
                "obesity":      True,
                "hyperlipidemia":True,
                "smoking":      False,
            },
            "vitals": {"sbp": 162, "hr": 82, "spo2": 97},
        },
        "mock_model_output": {
            "LVH": 0.84, "NSR": 0.55,
            "AF": 0.22, "STEMI": 0.06, "NSTEMI": 0.12,
            "LBBB": 0.18, "RBBB": 0.08,
            "VT": 0.04, "VF": 0.02, "SVT": 0.08,
            "Ischemia": 0.20, "ST_Elevation": 0.06,
            "ST_Depression": 0.18, "TWI": 0.22, "PVC": 0.15,
        },
    },

    "vt": {
        "label": "Ventricular Tachycardia — Hemodynamically unstable",
        "patient": {
            "symptoms": {
                "palpitations": True,
                "syncope":      True,
                "dizziness":    True,
                "chest_pain":   True,
                "dyspnea":      True,
                "diaphoresis":  True,
            },
            "risk_factors": {
                "prior_mi":     True,
                "cad":          True,
                "cardiomyopathy": True,
                "htn":          True,
            },
            "vitals": {"sbp": 78, "hr": 188, "spo2": 88},
        },
        "mock_model_output": {
            "VT": 0.89, "VF": 0.28,
            "STEMI": 0.22, "AF": 0.08,
            "NSR": 0.04, "NSTEMI": 0.14,
            "LVH": 0.20, "LBBB": 0.16,
            "SVT": 0.06, "RBBB": 0.04,
            "Ischemia": 0.28, "ST_Elevation": 0.18,
            "ST_Depression": 0.14, "TWI": 0.12, "PVC": 0.35,
        },
    },
}


# ── CSV patient loader ────────────────────────────────────────────────────────

def load_patient_from_csv(csv_path: Path, patient_id: str) -> dict:
    """Load one patient row from patient_history.csv → pipeline patient dict."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("patient_id") == patient_id:
                return _row_to_patient_dict(row)
    raise ValueError(f"patient_id '{patient_id}' not found in {csv_path}")


def load_random_patient(csv_path: Path, seed: int = 0) -> tuple[str, dict]:
    """Load a random patient from patient_history.csv."""
    import random
    random.seed(seed)
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = random.choice(rows)
    pid = row.get("patient_id", "UNKNOWN")
    return pid, _row_to_patient_dict(row)


def _row_to_patient_dict(row: dict) -> dict:
    patient: dict = {"symptoms": {}, "risk_factors": {}, "vitals": {}}
    for key, val in row.items():
        if val == "":
            continue
        if key.startswith("sym_"):
            patient["symptoms"][key[4:]] = bool(int(val))
        elif key.startswith("rf_"):
            patient["risk_factors"][key[3:]] = bool(int(val))
        elif key in ("sbp", "dbp", "hr", "spo2", "rr"):
            try:
                patient["vitals"][key] = float(val)
            except ValueError:
                pass
    return patient


# ── Auto-generate if CSV missing ──────────────────────────────────────────────

def _ensure_mock_csv(csv_path: Path, n: int = 200) -> None:
    if csv_path.exists():
        return
    logger.info("No patient CSV found — generating %d mock patients...", n)
    import subprocess
    script = PROJECT_ROOT / "scripts" / "generate_mock_aiims_data.py"
    subprocess.run(
        [sys.executable, str(script), "--n", str(n), "--output", str(csv_path)],
        check=True,
    )


# ── Output formatter ──────────────────────────────────────────────────────────

_CONF_ICON = {
    "CONFIRMED":  "★★ CONFIRMED  ",
    "PROBABLE":   "★  PROBABLE   ",
    "POSSIBLE":   "?  POSSIBLE   ",
    "INCIDENTAL": "~  INCIDENTAL ",
}
_TIER_COLOR = {1: "!!! CRITICAL", 2: "  ! URGENT  ", 3: "    ROUTINE "}


def print_demo_header(scenario_label: str, patient_id: str) -> None:
    width = 62
    print("\n" + "=" * width)
    print("  ECGenius — AIIMS Demo")
    print(f"  Patient : {patient_id}")
    print(f"  Case    : {scenario_label}")
    print("=" * width)


def print_patient_summary(patient: dict) -> None:
    print("\n  PATIENT PRESENTATION")
    print("  " + "-" * 40)
    syms = [k for k, v in patient.get("symptoms", {}).items() if v]
    risks = [k for k, v in patient.get("risk_factors", {}).items() if v]
    vitals = patient.get("vitals", {})
    print(f"  Symptoms     : {', '.join(syms) if syms else 'None reported'}")
    print(f"  Risk factors : {', '.join(risks) if risks else 'None reported'}")
    if vitals:
        vital_str = "  ".join(f"{k.upper()}={v}" for k, v in vitals.items())
        print(f"  Vitals       : {vital_str}")


def print_results(fused_results: list, critical_alerts: list) -> None:
    print("\n  RANKED DIFFERENTIAL DIAGNOSIS")
    print("  " + "-" * 58)

    active = [r for r in fused_results if not r.is_suppressed]
    suppressed = [r for r in fused_results if r.is_suppressed]

    for i, r in enumerate(active, 1):
        icon = _CONF_ICON.get(r.confidence_label, "  ")
        tier_str = _TIER_COLOR.get(r.tier, "")
        bar_len  = int(r.score_final * 30)
        bar      = "█" * bar_len + "░" * (30 - bar_len)
        print(f"\n  {i:2d}. [{icon}] {r.label_name}  ({r.label_id})")
        print(f"       {tier_str}  |  Score: {r.score_final:.3f}  [{bar}]")
        print(f"       AI={r.pai:.3f}  Sym={r.s_symptom:.3f}  "
              f"Risk={r.s_risk:.3f}  Rule={r.s_rule:.3f}")
        print(f"       Action:  {r.default_action}")
        if r.snomed_ct:
            print(f"       SNOMED:  {r.snomed_ct}  |  ICD-10: {r.icd10}")
        if r.supporting:
            print(f"       Evidence: {' | '.join(r.supporting[:4])}")
        print(f"       Path:    {' → '.join(r.hierarchy)}")

    if suppressed:
        print(f"\n  --- Suppressed diagnoses (shown for transparency) ---")
        for r in suppressed[:3]:
            print(f"  {r.label_id:16s}  pai={r.pai:.3f}  score={r.score_final:.3f}  [SUPPRESSED]")

    if critical_alerts:
        print("\n" + "!" * 62)
        for alert in critical_alerts:
            print(f"  !!! CRITICAL ALERT: {alert.label_name}")
            print(f"      Confidence: {alert.confidence_label}  |  Score: {alert.score_final:.3f}")
            print(f"      ACTION:     {alert.default_action}")
        print("!" * 62)


# ── Main demo runner ──────────────────────────────────────────────────────────

def run_demo(
    scenario:      str    = "stemi",
    patient_csv:   str    = None,
    patient_id:    str    = None,
    patient_json:  str    = None,
    model_path:    str    = None,
    output_json:   str    = None,
    threshold:     float  = 0.10,
    ontology_dir:  str    = "ontology/",
    history_dir:   str    = "history_module/",
    rules_dir:     str    = "rules_engine/",
) -> dict:

    # ── Resolve patient ────────────────────────────────────────────────────
    pid = patient_id or "DEMO-001"

    if patient_json:
        try:
            patient = json.loads(patient_json)
            pid     = "CUSTOM"
            scenario_label = "Custom patient JSON"
            model_probs    = None
        except json.JSONDecodeError as e:
            logger.error("Invalid --patient JSON: %s", e)
            sys.exit(1)

    elif patient_csv:
        csv_path = PROJECT_ROOT / patient_csv
        _ensure_mock_csv(csv_path)
        if patient_id:
            patient = load_patient_from_csv(csv_path, patient_id)
        else:
            pid, patient = load_random_patient(csv_path)
        scenario_label = f"From CSV (patient_id={pid})"
        model_probs    = None

    else:
        # Use embedded scenario
        if scenario not in DEMO_SCENARIOS:
            valid = list(DEMO_SCENARIOS.keys())
            logger.error("Unknown scenario '%s'. Choose from: %s", scenario, valid)
            sys.exit(1)
        sc             = DEMO_SCENARIOS[scenario]
        patient        = sc["patient"]
        scenario_label = sc["label"]
        model_probs    = sc["mock_model_output"]

    # ── Model probabilities ────────────────────────────────────────────────
    if model_probs is None:
        if model_path:
            try:
                from inference.run_model_on_physionet import load_lightv2, _load_labels
                labels = _load_labels()
                model  = load_lightv2(PROJECT_ROOT / model_path, n_labels=len(labels))
                # Use a synthetic zero-mean ECG (for demo — replace with real ECG)
                import torch
                dummy = torch.zeros(1, 12, 5000)
                with torch.no_grad():
                    logits = model(dummy)
                    probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()
                model_probs = {label: float(p) for label, p in zip(labels, probs)}
                logger.info("Model inference on synthetic ECG completed.")
            except Exception as e:
                logger.warning("Model load failed (%s) — using STEMI mock output.", e)
                model_probs = DEMO_SCENARIOS["stemi"]["mock_model_output"]
        else:
            # Default to STEMI scenario mock output for demo
            model_probs = DEMO_SCENARIOS.get(scenario, DEMO_SCENARIOS["stemi"])["mock_model_output"]

    # Apply threshold
    model_probs = {k: v for k, v in model_probs.items() if v >= threshold}

    # ── Print header + patient summary ─────────────────────────────────────
    print_demo_header(scenario_label, pid)
    print_patient_summary(patient)

    # ── Run pipeline stages ────────────────────────────────────────────────
    print("\n  [1/4] Ontology mapping...")
    from inference.ontology_mapper import OntologyMapper
    mapper  = OntologyMapper(
        ontology_dir=str(PROJECT_ROOT / ontology_dir),
        threshold=threshold,
    )
    results = mapper.map(model_probs)

    print(f"  [2/4] Rule engine ({len(results)} candidates)...")
    from rules_engine.rule_executor import RuleExecutor
    executor = RuleExecutor(rules_dir=str(PROJECT_ROOT / rules_dir), strict=False)
    results, derived_log = executor.execute(results, patient, mapper)

    print("  [3/4] Patient history encoding...")
    from history_module.history_encoder import HistoryEncoder
    encoder = HistoryEncoder(history_module_dir=str(PROJECT_ROOT / history_dir))
    label_ids      = [r.label_id for r in results]
    history_deltas = encoder.encode_all(label_ids, patient)

    print("  [4/4] Decision fusion...")
    from inference.decision_fusion import DecisionFusion
    fusion_out = DecisionFusion().fuse(
        results, history_deltas, derived_log, patient, patient_id=pid
    )

    # ── Print results ──────────────────────────────────────────────────────
    print_results(fusion_out.results, fusion_out.critical_alerts)

    # ── Build output payload ───────────────────────────────────────────────
    payload = fusion_out.to_dict()
    payload["demo_scenario"]  = scenario
    payload["patient_input"]  = patient
    payload["model_probs"]    = model_probs

    if output_json:
        out_path = PROJECT_ROOT / output_json
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved full output → {out_path}")

    print(f"\n  Top diagnosis : {fusion_out.top_diagnosis.label_name if fusion_out.top_diagnosis else 'None'}")
    print(f"  Score         : {fusion_out.top_diagnosis.score_final:.3f} [{fusion_out.top_diagnosis.confidence_label}]" if fusion_out.top_diagnosis else "")
    print()

    return payload


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ECGenius — End-to-end AIIMS demo for clinical presentation"
    )
    parser.add_argument("--demo-aiims", action="store_true", required=True,
                        help="Enable AIIMS demo mode (required flag)")
    parser.add_argument("--scenario", type=str, default="stemi",
                        choices=list(DEMO_SCENARIOS.keys()),
                        help="Built-in clinical scenario to demo (default: stemi)")
    parser.add_argument("--patient-csv", type=str, default=None,
                        help="Path to patient_history.csv (generated or real)")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Specific patient_id to load from CSV")
    parser.add_argument("--patient", type=str, default=None,
                        help="Patient history as JSON string")
    parser.add_argument("--model", type=str, default=None,
                        help="LightV2 checkpoint path (omit for mock probabilities)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full output to JSON file")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--ontology",  type=str, default="ontology/")
    parser.add_argument("--history",   type=str, default="history_module/")
    parser.add_argument("--rules",     type=str, default="rules_engine/")

    args = parser.parse_args()

    run_demo(
        scenario     = args.scenario,
        patient_csv  = args.patient_csv,
        patient_id   = args.patient_id,
        patient_json = args.patient,
        model_path   = args.model,
        output_json  = args.output,
        threshold    = args.threshold,
        ontology_dir = args.ontology,
        history_dir  = args.history,
        rules_dir    = args.rules,
    )


if __name__ == "__main__":
    main()
