"""
ECGenius — run_pipeline.py
===========================
Final end-to-end pipeline.

Flow:
  model.pt  →  {label: probability}
      ↓
  ontology_mapper.py   (enrich with tier, hierarchy, SNOMED, ICD-10)
      ↓
  rule_executor.py     (apply cardiologist rules from rules.csv)
      ↓
  history_encoder.py   (patient symptoms + vitals → score deltas)
      ↓
  decision_fusion.py   (Score(D) = 0.5×Pai + S_symptom + S_risk + S_rule)
      ↓
  explainability.py    (ranked DDx + XAI JSON for UI)

Usage:
    python run_pipeline.py --mock
    python run_pipeline.py --model models/checkpoints/best_model.pt
    python run_pipeline.py --mock --output results/output.json
    python run_pipeline.py --mock --patient '{"symptoms":{"chest_pain":true},"risk_factors":{"htn":true},"vitals":{"sbp":88,"hr":110,"spo2":94}}'
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ECGenius")

# ── Defaults ──────────────────────────────────────────────────────────────────

MOCK_MODEL_OUTPUT = {
    # ── Ischemia ──────────────────────────────────────────────
   "Sinus_Bradycardia": 0.6667,
    "ST_T_Change":       0.6553,
    "T_Wave_Change":     0.3333,
    "QT_Interval_Ext":   0.3333,
    "ST_Elevation":      0.3333,
    
}

DEFAULT_PATIENT = {
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
        "smoking":   False,
        "prior_mi":  False,
    },
    "vitals": {
        "sbp": 120,
        "hr": 58,      # Bradycardia from ECG
        "spo2": 97,
    },
}

# ── Model loader ──────────────────────────────────────────────────────────────

def load_model_and_predict(model_path: str, label_encoder_path: str) -> dict[str, float]:
    """
    Load model.pt → return {label_id: probability} dict.
    Your model outputs probabilities directly (sigmoid already applied).
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch not installed. Run: pip install torch\nOr use --mock.")

    # Load label encoder
    enc_path = Path(label_encoder_path)
    if not enc_path.exists():
        raise FileNotFoundError(f"label_encoder.json not found: {enc_path}")
    with open(enc_path, encoding="utf-8") as f:
        encoder = json.load(f)
    if isinstance(encoder, list):
        labels = encoder
    elif "labels" in encoder:
        labels = encoder["labels"]
    elif "idx_to_label" in encoder:
        labels = [encoder["idx_to_label"][str(i)] for i in range(len(encoder["idx_to_label"]))]
    else:
        raise ValueError("Unrecognised label_encoder.json format.")

    # Resolve device
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Loading model from %s on %s", model_path, device)

    pt_path = Path(model_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

    checkpoint = torch.load(str(pt_path), map_location=device, weights_only=True)

    # Case 1: checkpoint already contains pre-computed output dict
    if isinstance(checkpoint, dict) and "model_output" in checkpoint:
        raw = checkpoint["model_output"]
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items()}

    # Case 2: checkpoint is a state_dict → load into architecture
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict",
                     checkpoint.get("state_dict", checkpoint))
    try:
        from models.architectures.cnn_transformer import ECGCNNTransformer
        model = ECGCNNTransformer(n_labels=len(labels)).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        dummy = torch.zeros(1, 12, 5000, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model(dummy)).squeeze(0).cpu().tolist()
        return {label: float(p) for label, p in zip(labels, probs)}

    except ImportError:
        # Case 3: try TorchScript
        scripted = torch.jit.load(str(pt_path), map_location=device)
        scripted.eval()
        dummy = torch.zeros(1, 12, 5000, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(scripted(dummy)).squeeze(0).cpu().tolist()
        return {label: float(p) for label, p in zip(labels, probs)}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(
    model_path:    str | None = None,
    mock:          bool       = False,
    patient:       dict       = None,
    ontology_dir:  str        = "ontology/",
    history_dir:   str        = "history_module/",
    rules_dir:     str        = "rules_engine/",
    label_encoder: str        = "data/processed/labels/label_encoder.json",
    output_path:   str | None = None,
    threshold:     float      = 0.10,
) -> dict:

    patient = patient or DEFAULT_PATIENT

    # Resolve all dirs relative to project root so Windows paths work correctly
    ontology_dir  = str(PROJECT_ROOT / ontology_dir)
    history_dir   = str(PROJECT_ROOT / history_dir)
    rules_dir     = str(PROJECT_ROOT / rules_dir)
    label_encoder = str(PROJECT_ROOT / label_encoder)

    _banner("ECGenius — Full Pipeline")

    # ── Step 1: Model probabilities ────────────────────────────────────────
    _header("1/5  Model probabilities")
    if mock:
        logger.info("MOCK MODE — using hardcoded probabilities")
        model_output = MOCK_MODEL_OUTPUT
    else:
        model_output = load_model_and_predict(model_path, label_encoder)

    # Apply threshold
    model_output = {k: v for k, v in model_output.items() if v >= threshold}
    _print_probabilities(model_output)

    # ── Step 2: Ontology mapping ───────────────────────────────────────────
    _header("2/5  Ontology mapping")
    from inference.ontology_mapper import OntologyMapper
    mapper  = OntologyMapper(ontology_dir=ontology_dir, threshold=threshold)
    results = mapper.map(model_output)

    for r in results:
        status = "[SUPPRESSED]" if r.is_suppressed else f"Tier {r.tier}"
        print(f"  {r.label_id:16s}  pai={r.pai:.3f}  {status:12s}  "
              f"{' > '.join(r.hierarchy)}")

    # ── Step 3: Rule engine ────────────────────────────────────────────────
    _header("3/5  Rule engine")
    from rules_engine.rule_executor import RuleExecutor
    executor = RuleExecutor(rules_dir=rules_dir, strict=False)
    results, derived_log = executor.execute(results, patient, mapper)

    print(f"  {len(derived_log)} derived log entries")
    for entry in derived_log:
        print(f"  {entry}")

    # ── Step 4: History encoding ───────────────────────────────────────────
    _header("4/5  Patient history")
    from history_module.history_encoder import HistoryEncoder
    encoder = HistoryEncoder(history_module_dir=history_dir)

    for w in encoder.validate_patient(patient):
        logger.warning("Patient data: %s", w)

    history_deltas = encoder.encode_all([r.label_id for r in results], patient)

    for lid, delta in history_deltas.items():
        if delta.score_delta != 0 or delta.tier_delta != 0:
            print(f"  {lid:16s}  score_delta={delta.score_delta:+.3f}  "
                  f"tier_delta={delta.tier_delta:+d}")
            for ev in delta.evidence:
                print(f"    {ev}")

    # ── Step 5: Decision fusion + explainability ───────────────────────────
    _header("5/5  Decision fusion (Naive Bayes)")
    import pandas as pd
    from inference.decision_fusion import DecisionFusion, _build_patient_evidence

    # Load CPT table (history_rules.csv) — single load per pipeline run
    _cpt_path = Path(history_dir) / "history_rules.csv"
    cpt_df = pd.read_csv(str(_cpt_path), comment="#") if _cpt_path.exists() else None
    if cpt_df is None:
        logger.warning("history_rules.csv not found at %s — NB fusion disabled.", _cpt_path)

    # Build patient_evidence dict from structured patient data
    patient_evidence = {
        "chest_pain":    patient.get("symptoms",     {}).get("chest_pain",   False),
        "diaphoresis":   patient.get("symptoms",     {}).get("diaphoresis",  False),
        "dyspnea":       patient.get("symptoms",     {}).get("dyspnea",      False),
        "syncope":       patient.get("symptoms",     {}).get("syncope",      False),
        "palpitations":  patient.get("symptoms",     {}).get("palpitations", False),
        "dizziness":     patient.get("symptoms",     {}).get("dizziness",    False),
        "cad":           patient.get("risk_factors", {}).get("cad",          False),
        "dm":            patient.get("risk_factors", {}).get("dm",           False),
        "smoking":       patient.get("risk_factors", {}).get("smoking",      False),
        "htn":           patient.get("risk_factors", {}).get("htn",          False),
        "prior_mi":      patient.get("risk_factors", {}).get("prior_mi",     False),
        "young_age":     patient.get("risk_factors", {}).get("young_age",    False),
        "asymptomatic":  not any(bool(v) for v in
                             patient.get("symptoms", {}).values()),
        "sbp_lt_90":     patient.get("vitals", {}).get("sbp", 120) < 90,
        "sbp_gt_140":    patient.get("vitals", {}).get("sbp", 120) > 140,
        "hr_lt_40":      patient.get("vitals", {}).get("hr",  75)  < 40,
        "hr_lt_60":      patient.get("vitals", {}).get("hr",  75)  < 60,
        "hr_gt_100":     patient.get("vitals", {}).get("hr",  75)  > 100,
        "hr_gt_150":     patient.get("vitals", {}).get("hr",  75)  > 150,
        "hr_gt_160":     patient.get("vitals", {}).get("hr",  75)  > 160,
        "hr_gt_200":     patient.get("vitals", {}).get("hr",  75)  > 200,
        "spo2_lt_94":    patient.get("vitals", {}).get("spo2", 98) < 94,
        "no_chest_pain": not patient.get("symptoms", {}).get("chest_pain", False),
    }

    fused_raw = DecisionFusion().fuse(
        ontology_results = results,
        history_deltas   = history_deltas,
        derived_log      = derived_log,
        patient          = patient,
        cpt_table        = cpt_df,
        patient_evidence = patient_evidence,
    )

    # Unwrap FusionOutput object (your local decision_fusion.py returns this)
    if hasattr(fused_raw, 'results'):
        fused       = fused_raw.results
        derived_log = getattr(fused_raw, 'derived_log', derived_log)
    else:
        fused = fused_raw

    # Build payload — works with both FusionResult and FusedResult field names
    def _get(obj, *attrs):
        for a in attrs:
            if hasattr(obj, a):
                v = getattr(obj, a)
                return round(v, 3) if isinstance(v, float) else v
        return None

    payload = {
        "primary_diagnosis": None,
        "differential":      [],
        "suppressed":        [],
        "critical_alerts":   [],
        "total_considered":  len(fused),
    }
    for f in fused:
        entry = {
            "label_id":         f.label_id,
            "label_name":       f.label_name,
            "score":            round(f.score_final, 3),
            "confidence_label": f.confidence_label,
            "tier":             f.tier,
            "default_action":   f.default_action,
            "snomed_ct":        f.snomed_ct,
            "icd10":            f.icd10,
            "aha_guideline":    f.aha_guideline,
            "supporting":       getattr(f, 'supporting', []),
            "contradicting":    getattr(f, 'contradicting', []),
            "hierarchy":        f.hierarchy,
            "is_suppressed":    f.is_suppressed,
            "score_breakdown":  getattr(f, 'score_breakdown', {
                "pai":       _get(f, 'pai'),
                "s_ai":      _get(f, 's_ai'),
                "s_symptom": _get(f, 's_symptom'),
                "s_risk":    _get(f, 's_risk'),
                "s_rule":    _get(f, 's_rule'),
            }),
        }
        if f.is_suppressed:
            payload["suppressed"].append(entry)
        else:
            payload["differential"].append(entry)
            if f.tier == 1 and f.confidence_label in ("CONFIRMED", "PROBABLE"):
                payload["critical_alerts"].append(entry)
    if payload["differential"]:
        payload["primary_diagnosis"] = payload["differential"][0]

    # ── Print results ──────────────────────────────────────────────────────
    _banner("Ranked Differential Diagnosis")

    active     = [f for f in fused if not f.is_suppressed]
    suppressed = [f for f in fused if f.is_suppressed]

    CONF_ICON = {"CONFIRMED": "**", "PROBABLE": "* ", "POSSIBLE": "? ", "INCIDENTAL": "~ "}

    for i, f in enumerate(active, 1):
        print(f"\n  {i}. {CONF_ICON.get(f.confidence_label,'  ')} "
              f"{f.label_name} ({f.label_id})")
        print(f"       Score:      {f.score_final:.3f}  [{f.confidence_label}]")
        print(f"       Tier:       {f.tier}  ->  {f.default_action}")
        if f.nb_breakdown:
            bd = f.nb_breakdown
            print(f"       NB:         prior={bd.get('prior',0):.4f}  "
                  f"log_ll={bd.get('log_likelihood',0):.4f}  "
                  f"posterior={f.score_final:.4f}")
        else:
            print(f"       Components: AI={f.pai:.3f}  "
                  f"Symptoms={f.s_symptom:.3f}  "
                  f"Risk={f.s_risk:.3f}  "
                  f"Rules={f.s_rule:.3f}")
        if f.snomed_ct:
            print(f"       SNOMED-CT:  {f.snomed_ct}  |  ICD-10: {f.icd10}")
        if f.aha_guideline:
            print(f"       Guideline:  {f.aha_guideline}")
        if f.supporting:
            print(f"       Supporting: {', '.join(f.supporting)}")
        if f.contradicting:
            print(f"       Against:    {', '.join(f.contradicting)}")
        print(f"       Hierarchy:  {' > '.join(f.hierarchy)}")

    if suppressed:
        print(f"\n  --- Suppressed (transparency) ---")
        for f in suppressed:
            print(f"  {f.label_id:16s}  pai={f.pai:.3f}  posterior={f.score_final:.4f}")

    # ── NB calculation breakdown table ────────────────────────────────────────
    _print_nb_table(active, suppressed)

    # ── History evidence per label ─────────────────────────────────────────
    has_evidence = any(
        getattr(history_deltas.get(f.label_id, None), 'evidence', None)
        for f in active
    )
    if has_evidence:
        print("\n  --- Patient Evidence (from history_encoder) ------------------")
        for f in active:
            delta = history_deltas.get(f.label_id)
            if not delta or not delta.evidence:
                continue
            print(f"\n  {f.label_id}:")
            for ev in delta.evidence:
                print(f"    {ev}")

    if payload.get("critical_alerts"):
        print()
        for alert in payload["critical_alerts"]:
            print(f"\n  !!! CRITICAL: {alert['label_name']} — {alert['default_action']} !!!")

    print()

    # ── Save output ────────────────────────────────────────────────────────
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved → {output_path}\n")

    return payload


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def _header(text: str) -> None:
    print(f"\n[{text}]")

def _print_probabilities(model_output: dict) -> None:
    for label, prob in sorted(model_output.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 25)
        print(f"  {label:20s}  {prob:.3f}  {bar}")

def _print_nb_table(active: list, suppressed: list) -> None:
    """Print the full Naive Bayes calculation breakdown table."""
    all_with_nb = [r for r in active + suppressed if r.nb_breakdown]
    if not all_with_nb:
        return

    print("\n  --- Naive Bayes Calculation Breakdown ----------------------------")
    header = (f"  {'Label':22s}  {'Prior':>7s}  {'log(LR)':>8s}  "
              f"{'Unnorm':>12s}  {'Posterior':>10s}  Tier")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in sorted(active + suppressed, key=lambda x: -x.score_final):
        bd = r.nb_breakdown
        if not bd:
            continue
        prior      = bd.get("prior", r.pai)
        log_ll     = bd.get("log_likelihood", 0.0)
        unnorm     = bd.get("unnorm_posterior", 0.0)
        posterior  = r.score_final
        tier_label = bd.get("tier", r.confidence_label)
        tag = " [suppressed]" if r.is_suppressed else ""
        print(
            f"  {r.label_id:22s}  {prior:7.4f}  {log_ll:8.4f}  "
            f"{unnorm:12.3e}  {posterior:10.6f}  {tier_label}{tag}"
        )

    # Per-disease evidence terms
    print("\n  --- Evidence Likelihood Ratios -----------------------------------")
    for r in active:
        bd = r.nb_breakdown
        if not bd or not bd.get("evidence_terms"):
            continue
        print(f"\n  {r.label_id}:")
        for (feat, p_d, p_nd, lr_log) in bd.get("evidence_terms", []):
            lr    = p_d / max(p_nd, 1e-9)
            arrow = "+" if lr_log > 0 else "-"
            print(f"    [{arrow}] {feat:20s}  p_d={p_d:.2f}  p_nd={p_nd:.2f}  "
                  f"LR={lr:.2f}  log(LR)={lr_log:+.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ECGenius — ontology-guided ECG differential diagnosis"
    )
    parser.add_argument("--model",         type=str,   default=None,
                        help="Path to model.pt checkpoint")
    parser.add_argument("--mock",          action="store_true",
                        help="Use mock probabilities — no model needed")
    parser.add_argument("--patient",       type=str,   default=None,
                        help="Patient history as JSON string")
    parser.add_argument("--output",        type=str,   default=None,
                        help="Save output JSON to this path")
    parser.add_argument("--threshold",     type=float, default=0.10,
                        help="Min probability to include a label (default 0.10)")
    parser.add_argument("--ontology",      type=str,   default="ontology/")
    parser.add_argument("--history",       type=str,   default="history_module/")
    parser.add_argument("--rules",         type=str,   default="rules_engine/")
    parser.add_argument("--label-encoder", type=str,
                        default="data/processed/labels/label_encoder.json")

    args = parser.parse_args()

    if not args.mock and not args.model:
        print("\nError: provide --model path/to/best_model.pt  or  --mock\n")
        print("Examples:")
        print("  python run_pipeline.py --mock")
        print("  python run_pipeline.py --model models/checkpoints/best_model.pt")
        sys.exit(1)

    patient = None
    if args.patient:
        try:
            patient = json.loads(args.patient)
        except json.JSONDecodeError as e:
            print(f"Error parsing --patient JSON: {e}")
            sys.exit(1)

    run(
        model_path    = args.model,
        mock          = args.mock,
        patient       = patient,
        ontology_dir  = args.ontology,
        history_dir   = args.history,
        rules_dir     = args.rules,
        label_encoder = args.label_encoder,
        output_path   = args.output,
        threshold     = args.threshold,
    )

if __name__ == "__main__":
    main()