"""
ECGenius — scripts/run_full_pipeline_batch.py
===============================================
THE MISSING CONNECTOR.

Reads:
  1. data/processed/model_probabilities_aiims.csv   (AI Pai per record)
  2. data/raw/aiims/patient_history.csv             (symptoms, risk factors, vitals)
  3. data/processed/splits/test_ids.txt             (which records to evaluate)

Runs for each record:
  ontology_mapper → rule_executor → history_encoder → decision_fusion

Writes:
  evaluation/results/pipeline_outputs.json   (full ranked DDx per patient)
  evaluation/results/pipeline_summary.csv    (one row per patient: top label, score, tier1 status)

Usage:
    # Full run (model probabilities + patient history):
    python scripts/run_full_pipeline_batch.py \\
        --probabilities data/processed/model_probabilities_aiims.csv \\
        --history       data/raw/aiims/patient_history.csv \\
        --split         data/processed/splits/test_ids.txt \\
        --output        evaluation/results/pipeline_outputs.json

    # PhysioNet (no symptoms — ECG-only mode):
    python scripts/run_full_pipeline_batch.py \\
        --probabilities data/processed/model_probabilities.csv \\
        --split         data/processed/splits/test_ids.txt \\
        --output        evaluation/results/pipeline_outputs_physionet.json

    # Test with mock data (no real files needed):
    python scripts/run_full_pipeline_batch.py --mock
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,   # quiet during batch — only show errors
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ECGenius.batch")

LABELS_15 = [
    "AF", "NSR", "STEMI", "NSTEMI", "LVH", "VF", "VT", "SVT",
    "LBBB", "RBBB", "Ischemia", "ST_Elevation", "ST_Depression", "TWI", "PVC",
]
TIER1_LABELS = {"STEMI", "VF", "VT", "NSTEMI", "ST_Elevation"}

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_probabilities(csv_path: Path) -> dict[str, dict[str, float]]:
    """
    Load model_probabilities.csv.
    Returns {record_id: {label: probability, ...}}
    """
    result: dict[str, dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid  = row.get("record_id", "")
            probs = {}
            for label in LABELS_15:
                if label in row and row[label] != "":
                    try:
                        probs[label] = float(row[label])
                    except ValueError:
                        pass
            if rid and probs:
                result[rid] = probs
    logger.info("Loaded probabilities for %d records", len(result))
    return result


def load_patient_history(csv_path: Optional[Path]) -> dict[str, dict]:
    """
    Load patient_history.csv.
    Returns {patient_id: patient_dict} in the format history_encoder expects.

    Columns starting with sym_  → symptoms
    Columns starting with rf_   → risk_factors
    Columns sbp/dbp/hr/spo2/rr  → vitals
    Empty values = not recorded (NOT the same as False)
    """
    if csv_path is None or not csv_path.exists():
        logger.warning("No patient_history.csv found — running in ECG-only mode.")
        return {}

    result: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("patient_id", row.get("record_id", ""))
            if not pid:
                continue

            patient: dict = {"symptoms": {}, "risk_factors": {}, "vitals": {}}

            for key, val in row.items():
                if val == "" or val is None:
                    continue   # missing ≠ absent — skip entirely

                if key.startswith("sym_"):
                    try:
                        patient["symptoms"][key[4:]] = bool(int(val))
                    except (ValueError, TypeError):
                        pass

                elif key.startswith("rf_"):
                    try:
                        patient["risk_factors"][key[3:]] = bool(int(val))
                    except (ValueError, TypeError):
                        pass

                elif key in ("sbp", "dbp", "hr", "spo2", "rr"):
                    try:
                        patient["vitals"][key] = float(val)
                    except (ValueError, TypeError):
                        pass

            result[pid] = patient

    logger.info("Loaded patient history for %d patients", len(result))
    return result


def load_split_ids(split_path: Optional[Path]) -> Optional[list[str]]:
    if split_path is None or not split_path.exists():
        return None
    ids = [l.strip() for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    logger.info("Split file: %d record IDs", len(ids))
    return ids


# ── Pipeline runner (one patient) ─────────────────────────────────────────────

def _make_empty_patient() -> dict:
    return {"symptoms": {}, "risk_factors": {}, "vitals": {}}


class PipelineRunner:
    """
    Loads pipeline modules once, runs them for each patient efficiently.
    All module imports are deferred to __init__ so batch calls are fast.
    """

    def __init__(self, ontology_dir: str, history_dir: str, rules_dir: str,
                 threshold: float = 0.10):
        self.threshold = threshold

        from inference.ontology_mapper import OntologyMapper
        from rules_engine.rule_executor import RuleExecutor
        from history_module.history_encoder import HistoryEncoder
        from inference.decision_fusion import DecisionFusion

        self.mapper   = OntologyMapper(ontology_dir=ontology_dir, threshold=threshold)
        self.executor = RuleExecutor(rules_dir=rules_dir, strict=False)
        self.encoder  = HistoryEncoder(history_module_dir=history_dir)
        self.fusion   = DecisionFusion()

    def run_one(
        self,
        record_id:   str,
        model_probs: dict[str, float],
        patient:     dict,
    ) -> dict:
        """
        Run the full 4-stage pipeline for one record.
        Returns a serialisable dict of the pipeline output.
        """
        # Filter by threshold
        probs_filtered = {k: v for k, v in model_probs.items() if v >= self.threshold}

        if not probs_filtered:
            return {
                "record_id":     record_id,
                "error":         "all probabilities below threshold",
                "top_diagnosis": None,
                "differential":  [],
                "critical_alerts": [],
                "has_history":   bool(patient.get("symptoms") or patient.get("vitals")),
            }

        try:
            # Stage 1: Ontology mapping
            ontology_results = self.mapper.map(probs_filtered)

            # Stage 2: Rule engine
            ontology_results, derived_log = self.executor.execute(
                ontology_results, patient, self.mapper
            )

            # Stage 3: History encoder
            label_ids = [r.label_id for r in ontology_results]
            history_deltas = self.encoder.encode_all(label_ids, patient)

            # Stage 4: Decision fusion
            fusion_out = self.fusion.fuse(
                ontology_results, history_deltas, derived_log,
                patient, patient_id=record_id,
            )

            # Serialise output
            active = [r for r in fusion_out.results if not r.is_suppressed]
            suppressed = [r for r in fusion_out.results if r.is_suppressed]

            def _result_row(r) -> dict:
                return {
                    "rank":             r.rank,
                    "label_id":         r.label_id,
                    "label_name":       r.label_name,
                    "pai":              round(r.pai, 4),
                    "score_final":      round(r.score_final, 4),
                    "s_ai":            round(r.s_ai, 4),
                    "s_symptom":       round(r.s_symptom, 4),
                    "s_risk":          round(r.s_risk, 4),
                    "s_rule":          round(r.s_rule, 4),
                    "confidence_label": r.confidence_label,
                    "tier":             r.tier,
                    "is_tier1":         r.tier == 1,
                    "default_action":   r.default_action,
                    "supporting":       r.supporting,
                    "contradicting":    r.contradicting,
                    "is_suppressed":    r.is_suppressed,
                }

            top = fusion_out.top_diagnosis
            return {
                "record_id":     record_id,
                "error":         None,
                "top_diagnosis": {
                    "label_id":         top.label_id,
                    "score_final":      round(top.score_final, 4),
                    "confidence_label": top.confidence_label,
                    "tier":             top.tier,
                } if top else None,
                "differential":    [_result_row(r) for r in active],
                "suppressed":      [_result_row(r) for r in suppressed],
                "critical_alerts": [
                    {"label_id": r.label_id, "score_final": round(r.score_final, 4),
                     "confidence_label": r.confidence_label}
                    for r in fusion_out.critical_alerts
                ],
                "derived_log":   derived_log,
                "has_history":   bool(patient.get("symptoms") or patient.get("vitals")),
                "n_symptoms":    sum(1 for v in patient.get("symptoms", {}).values() if v),
                "n_risk_factors":sum(1 for v in patient.get("risk_factors", {}).values() if v),
                "n_vitals":      len(patient.get("vitals", {})),
            }

        except Exception as e:
            logger.error("Pipeline failed for record %s: %s", record_id, e, exc_info=True)
            return {
                "record_id":     record_id,
                "error":         str(e),
                "top_diagnosis": None,
                "differential":  [],
                "critical_alerts": [],
                "has_history":   False,
            }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(
    probabilities:  dict[str, dict[str, float]],
    patient_history: dict[str, dict],
    record_ids:     list[str],
    ontology_dir:   str,
    history_dir:    str,
    rules_dir:      str,
    threshold:      float = 0.10,
) -> list[dict]:
    """
    Run the full pipeline for every record in record_ids.
    Returns list of pipeline output dicts.
    """
    runner  = PipelineRunner(ontology_dir, history_dir, rules_dir, threshold)
    results = []
    errors  = []
    ecg_only = 0
    t0 = time.time()

    total = len(record_ids)
    print(f"\n  Running full pipeline on {total} records...")
    print(f"  {'─'*58}")

    for idx, record_id in enumerate(record_ids, 1):
        if record_id not in probabilities:
            logger.warning("No probabilities for record %s — skipping.", record_id)
            errors.append(record_id)
            continue

        model_probs = probabilities[record_id]

        # Look up patient history: try record_id first, then patient_id field
        # (they may differ when one patient has multiple recordings)
        patient = patient_history.get(record_id, patient_history.get(record_id.split("_")[0], {}))
        if not patient:
            ecg_only += 1
            patient = _make_empty_patient()

        output = runner.run_one(record_id, model_probs, patient)
        results.append(output)

        # Progress print every 50 records
        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t0
            rate    = idx / elapsed if elapsed > 0 else 0
            eta     = (total - idx) / rate if rate > 0 else 0
            top_lbl = (output.get("top_diagnosis") or {}).get("label_id", "—")
            print(f"  [{idx:4d}/{total}]  {record_id:20s}  top={top_lbl:16s}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n  Done. {len(results)} records processed in {elapsed:.1f}s")
    print(f"  ECG-only (no history): {ecg_only}  |  Errors: {len(errors)}")
    if errors:
        print(f"  Failed records: {errors[:10]}{'...' if len(errors)>10 else ''}")

    return results


# ── Output writers ────────────────────────────────────────────────────────────

def write_pipeline_outputs(results: list[dict], output_path: Path) -> None:
    """
    Write full pipeline output as JSON (for evaluation_evaluate_system.py).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_records":  len(results),
        "n_ecg_only": sum(1 for r in results if not r.get("has_history")),
        "n_errors":   sum(1 for r in results if r.get("error")),
        "records":    results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Full output → {output_path}")


def write_pipeline_summary_csv(results: list[dict], output_path: Path) -> None:
    """
    Write one-row-per-patient summary CSV (easy to open in Excel/Pandas).

    Columns:
      record_id | top_label | top_score | top_confidence |
      top_tier  | has_tier1_alert | n_critical_alerts | has_history |
      label_1..label_15 (system-predicted confidence label per diagnosis)
    """
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id", "top_label", "top_score", "top_confidence", "top_tier",
        "has_tier1_alert", "n_critical_alerts", "has_history",
        "n_symptoms", "n_risk_factors", "n_vitals", "error",
    ] + [f"sys_{l}" for l in LABELS_15]   # system confidence per label

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            top = r.get("top_diagnosis") or {}

            # Build per-label confidence map from differential
            label_conf: dict[str, str] = {}
            for entry in r.get("differential", []) + r.get("suppressed", []):
                lid = entry.get("label_id", "")
                if lid:
                    label_conf[lid] = entry.get("confidence_label", "")

            row = {
                "record_id":        r.get("record_id", ""),
                "top_label":        top.get("label_id", ""),
                "top_score":        top.get("score_final", ""),
                "top_confidence":   top.get("confidence_label", ""),
                "top_tier":         top.get("tier", ""),
                "has_tier1_alert":  int(len(r.get("critical_alerts", [])) > 0),
                "n_critical_alerts": len(r.get("critical_alerts", [])),
                "has_history":      int(r.get("has_history", False)),
                "n_symptoms":       r.get("n_symptoms", 0),
                "n_risk_factors":   r.get("n_risk_factors", 0),
                "n_vitals":         r.get("n_vitals", 0),
                "error":            r.get("error") or "",
            }
            for label in LABELS_15:
                row[f"sys_{label}"] = label_conf.get(label, "NOT_IN_DDx")

            writer.writerow(row)

    print(f"  Summary CSV → {output_path}")


# ── Mock mode ─────────────────────────────────────────────────────────────────

def _make_mock_data(n: int = 20) -> tuple[dict, dict, list]:
    """Generate mock inputs for testing without real files."""
    import numpy as np
    rng = np.random.default_rng(42)

    mock_labels = LABELS_15
    mock_probs  = {}
    mock_hist   = {}
    mock_ids    = []

    # Mock disease scenarios
    scenarios = [
        ("STEMI",   {"STEMI":0.88,"ST_Elevation":0.82,"NSTEMI":0.22},
                    {"symptoms":{"chest_pain":True,"diaphoresis":True},
                     "risk_factors":{"cad":True,"htn":True},"vitals":{"sbp":84,"hr":110,"spo2":91}}),
        ("AF",      {"AF":0.86,"NSR":0.08},
                    {"symptoms":{"palpitations":True,"dyspnea":True},
                     "risk_factors":{"htn":True},"vitals":{"hr":128}}),
        ("NSR",     {"NSR":0.92,"AF":0.04},
                    {}),
        ("VT",      {"VT":0.84,"VF":0.22},
                    {"symptoms":{"syncope":True,"palpitations":True},
                     "risk_factors":{"prior_mi":True},"vitals":{"sbp":78,"hr":188}}),
        ("LVH",     {"LVH":0.78,"NSR":0.55},
                    {"symptoms":{"dyspnea":True},"risk_factors":{"htn":True},
                     "vitals":{"sbp":165}}),
    ]

    for i in range(n):
        rid = f"MOCK{i+1:05d}"
        scenario_name, base_probs, patient = scenarios[i % len(scenarios)]
        # Add noise
        probs = {}
        for label in LABELS_15:
            base = base_probs.get(label, rng.uniform(0.02, 0.12))
            probs[label] = float(np.clip(base + rng.normal(0, 0.03), 0.01, 0.99))
        mock_probs[rid] = probs
        mock_hist[rid]  = patient
        mock_ids.append(rid)

    return mock_probs, mock_hist, mock_ids


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run full ECGenius pipeline on all records → pipeline_outputs.json"
    )
    parser.add_argument("--probabilities", type=str,
                        default="data/processed/model_probabilities_aiims.csv",
                        help="model_probabilities CSV from run_model_on_physionet.py")
    parser.add_argument("--history", type=str,
                        default="data/raw/aiims/patient_history.csv",
                        help="patient_history.csv with symptoms, risk factors, vitals")
    parser.add_argument("--split", type=str,
                        default=None,
                        help="Path to split_ids.txt (e.g. test_ids.txt). "
                             "If omitted, processes ALL records in probabilities CSV.")
    parser.add_argument("--output", type=str,
                        default="evaluation/results/pipeline_outputs.json")
    parser.add_argument("--summary", type=str,
                        default="evaluation/results/pipeline_summary.csv")
    parser.add_argument("--ontology",  type=str, default="ontology/")
    parser.add_argument("--history-dir", type=str, default="history_module/")
    parser.add_argument("--rules",     type=str, default="rules_engine/")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--mock",      action="store_true",
                        help="Run on built-in mock data (no files needed)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ECGenius — Full Pipeline Batch Runner")
    print("=" * 60)

    if args.mock:
        print("\n  MOCK MODE — using synthetic data")
        probabilities, patient_history, record_ids = _make_mock_data(n=20)
    else:
        prob_path = PROJECT_ROOT / args.probabilities
        hist_path = PROJECT_ROOT / args.history if args.history else None
        split_path = PROJECT_ROOT / args.split if args.split else None

        if not prob_path.exists():
            print(f"\n  ERROR: probabilities file not found: {prob_path}")
            print(f"  Run first:  python inference/run_model_on_physionet.py ...")
            sys.exit(1)

        print(f"\n  Probabilities : {prob_path}")
        print(f"  History       : {hist_path or 'NOT PROVIDED (ECG-only mode)'}")
        print(f"  Split         : {split_path or 'ALL records'}")

        probabilities   = load_probabilities(prob_path)
        patient_history = load_patient_history(hist_path)
        split_ids       = load_split_ids(split_path)
        record_ids      = split_ids if split_ids else list(probabilities.keys())

    # Resolve dirs relative to project root
    ontology_dir  = str(PROJECT_ROOT / args.ontology)
    history_dir   = str(PROJECT_ROOT / args.history_dir)
    rules_dir     = str(PROJECT_ROOT / args.rules)

    # Run
    results = run_batch(
        probabilities   = probabilities,
        patient_history = patient_history,
        record_ids      = record_ids,
        ontology_dir    = ontology_dir,
        history_dir     = history_dir,
        rules_dir       = rules_dir,
        threshold       = args.threshold,
    )

    # Write outputs
    output_path  = PROJECT_ROOT / args.output
    summary_path = PROJECT_ROOT / args.summary

    write_pipeline_outputs(results, output_path)
    write_pipeline_summary_csv(results, summary_path)

    print(f"\n  Next step:")
    print(f"  python evaluation/evaluate_system.py \\")
    print(f"      --pipeline-output {output_path} \\")
    print(f"      --annotations     data/raw/aiims/annotations.csv")
    print()


if __name__ == "__main__":
    main()
