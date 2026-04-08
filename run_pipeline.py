"""
ECGenius — run_pipeline.py
===========================
Complete end-to-end pipeline runner.

Run this to verify the entire system works before connecting
the real model and real ECG data.

Usage:
    # Test with mock signal (no model.pt needed)
    python run_pipeline.py --mock

    # Test with a real .mat or .npy ECG file
    python run_pipeline.py --ecg path/to/ecg.mat --fs 500

    # Test with a real model checkpoint
    python run_pipeline.py --ecg path/to/ecg.mat --checkpoint models/checkpoints/best_model.pt

    # Save output JSON
    python run_pipeline.py --mock --output results/output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import numpy as np

# ── Make sure project root is on sys.path ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ECGenius.Pipeline")


# ── Mock model output (used when --mock flag is set) ──────────────────────
MOCK_MODEL_OUTPUT = {
    "STEMI":        0.74,
    "AF":           0.42,
    "NSR":          0.61,
    "ST_Elevation": 0.80,
    "LVH":          0.38,
    "VF":           0.19,
}

# ── Sample patient history ─────────────────────────────────────────────────
SAMPLE_PATIENT = {
    "symptoms": {
        "chest_pain":   True,
        "palpitations": False,
        "syncope":      False,
        "dizziness":    False,
        "dyspnea":      True,
        "diaphoresis":  True,
    },
    "risk_factors": {
        "htn":       True,
        "dm":        False,
        "cad":       True,
        "young_age": False,
        "smoking":   True,
        "prior_mi":  False,
    },
    "vitals": {
        "sbp":  88,
        "hr":   102,
        "spo2": 94,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    ecg_path:       str | None = None,
    source_fs:      int        = 500,
    checkpoint:     str | None = None,
    mock:           bool       = False,
    patient:        dict       = None,
    ontology_dir:   str        = "ontology/",
    history_dir:    str        = "history_module/",
    rules_dir:      str        = "rules_engine/",
    label_encoder:  str        = "data/processed/labels/label_encoder.json",
    output_path:    str | None = None,
) -> dict:

    patient = patient or SAMPLE_PATIENT
    print("\n" + "="*60)
    print("  ECGenius — Full Pipeline Run")
    print("="*60)

    # ── STEP 1: Preprocess ─────────────────────────────────────────────────
    print("\n[1/6] Preprocessing ECG signal...")
    from inference.preprocess import ECGPreprocessor
    preprocessor = ECGPreprocessor()

    if mock:
        logger.info("MOCK MODE — generating synthetic 12-lead signal")
        raw_signal = np.random.randn(12, 5000).astype(np.float32)
        ecg_clean  = preprocessor.process(raw_signal, source_fs=500)
    else:
        if ecg_path is None:
            raise ValueError("Provide --ecg path or use --mock flag.")
        ecg_clean = preprocessor.load_and_process(ecg_path, source_fs=source_fs)

    print(f"    Signal shape: {ecg_clean.shape}  dtype: {ecg_clean.dtype}")

    # ── STEP 2: Model inference ────────────────────────────────────────────
    print("\n[2/6] Running model inference...")

    if mock or checkpoint is None:
        logger.info("Using MOCK model output — no checkpoint loaded.")
        model_output = MOCK_MODEL_OUTPUT
    else:
        from inference.model_inference import ECGModelInference
        model = ECGModelInference(
            checkpoint_path=checkpoint,
            label_encoder_path=label_encoder,
        )
        model_output = model.predict(ecg_clean)

    print(f"    Labels predicted (above threshold): {len(model_output)}")
    for label, prob in sorted(model_output.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {label:15s} {prob:.3f}  {bar}")

    # ── STEP 3: Ontology mapping ───────────────────────────────────────────
    print("\n[3/6] Mapping through ontology layer...")
    from inference.ontology_mapper import OntologyMapper
    mapper  = OntologyMapper(ontology_dir=ontology_dir)
    results = mapper.map(model_output)

    print(f"    OntologyResult objects: {len(results)}")
    for r in results:
        sup = "[SUPPRESSED]" if r.is_suppressed else f"Tier {r.tier}"
        print(f"    {r.label_id:15s} pai={r.pai:.3f}  {sup}  "
              f"hierarchy: {' > '.join(r.hierarchy)}")

    # ── STEP 4: Rule engine ────────────────────────────────────────────────
    print("\n[4/6] Running rule engine...")
    from rules_engine.rule_executor import RuleExecutor
    executor = RuleExecutor(rules_dir=rules_dir, strict=False)
    results, derived_log = executor.execute(results, patient, mapper)

    print(f"    Rules applied. Derived log entries: {len(derived_log)}")
    for entry in derived_log:
        print(f"    {entry}")

    # ── STEP 5: History encoding ───────────────────────────────────────────
    print("\n[5/6] Encoding patient history...")
    from history_module.history_encoder import HistoryEncoder
    encoder      = HistoryEncoder(history_dir=history_dir)

    # Validate patient data
    warnings = encoder.validate_patient(patient)
    for w in warnings:
        logger.warning("Patient validation: %s", w)

    label_ids      = [r.label_id for r in results]
    history_deltas = encoder.encode_all(label_ids, patient)

    print(f"    History deltas computed for {len(history_deltas)} labels:")
    for lid, delta in history_deltas.items():
        if delta.score_delta != 0 or delta.tier_delta != 0:
            print(f"    {lid:15s} score_delta={delta.score_delta:+.3f}  "
                  f"tier_delta={delta.tier_delta:+d}")
            for ev in delta.evidence:
                print(f"               {ev}")

    # ── STEP 6: Decision fusion ────────────────────────────────────────────
    print("\n[6/6] Decision fusion + explainability...")
    from inference.decision_fusion import DecisionFusion
    from inference.explainability  import Explainability

    fusion  = DecisionFusion()
    fused   = fusion.fuse(results, history_deltas, derived_log, patient)

    xai          = Explainability(include_suppressed=True)
    explanations = xai.explain_all(fused, derived_log)
    payload      = xai.to_ui_payload(explanations)

    # ── Final output ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RANKED DIFFERENTIAL DIAGNOSIS")
    print("="*60)

    active = [f for f in fused if not f.is_suppressed]
    for i, f in enumerate(active, 1):
        print(f"\n  {i}. {f.label_name} ({f.label_id})")
        print(f"     Score:      {f.score:.3f}  [{f.confidence_label}]")
        print(f"     Tier:       {f.tier}  — {f.default_action}")
        print(f"     Components: AI={f.pai:.3f}  Sym={f.s_symptom:.3f}  "
              f"Risk={f.s_risk:.3f}  Rule={f.s_rule:.3f}")
        if f.snomed_ct:
            print(f"     SNOMED-CT:  {f.snomed_ct}  |  ICD-10: {f.icd10}")
        if f.supporting:
            print(f"     Supporting: {', '.join(f.supporting)}")
        if f.contradicting:
            print(f"     Against:    {', '.join(f.contradicting)}")

    suppressed = [f for f in fused if f.is_suppressed]
    if suppressed:
        print(f"\n  Suppressed (shown for transparency):")
        for f in suppressed:
            print(f"    {f.label_id:12s} pai={f.pai:.3f}  score={f.score:.3f}")

    if payload.get("critical_alerts"):
        print("\n  *** CRITICAL ALERTS ***")
        for alert in payload["critical_alerts"]:
            print(f"  *** {alert['label_name']} — {alert['action']} ***")

    print("\n" + "="*60)

    # ── Save output ────────────────────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Output saved to: {output_path}")

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ECGenius — end-to-end ECG diagnosis pipeline"
    )
    parser.add_argument("--mock",       action="store_true",
                        help="Use synthetic signal + mock model output")
    parser.add_argument("--ecg",        type=str, default=None,
                        help="Path to ECG file (.mat or .npy)")
    parser.add_argument("--fs",         type=int, default=500,
                        help="Sampling frequency of input ECG (Hz)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Path to save output JSON")
    parser.add_argument("--ontology",   type=str, default="ontology/")
    parser.add_argument("--history",    type=str, default="history_module/")
    parser.add_argument("--rules",      type=str, default="rules_engine/")
    args = parser.parse_args()

    if not args.mock and args.ecg is None:
        print("Error: provide --ecg <path> or use --mock")
        print("Example: python run_pipeline.py --mock")
        sys.exit(1)

    run(
        ecg_path=args.ecg,
        source_fs=args.fs,
        checkpoint=args.checkpoint,
        mock=args.mock,
        ontology_dir=args.ontology,
        history_dir=args.history,
        rules_dir=args.rules,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()