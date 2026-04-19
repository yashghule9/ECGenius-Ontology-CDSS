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
    "STEMI":            0.74,
    "NSTEMI":           0.31,
    "Unstable_Angina":  0.18,
    # ── Rhythm ────────────────────────────────────────────────
    "AF":               0.42,
    "NSR":              0.61,
    "VF":               0.19,
    "VT":               0.14,
    "SVT":              0.22,
    "Atrial_Flutter":   0.11,
    "Sinus_Tachycardia":0.28,
    # ── Structural ────────────────────────────────────────────
    "LVH":              0.38,
    "LVH_Strain":       0.29,
    "Pericarditis":     0.12,
    "Long_QT":          0.10,
    # ── Conduction ────────────────────────────────────────────
    "LBBB":             0.24,
    "RBBB":             0.15,
    "Heart_Block_1st":  0.17,
    # ── ECG pattern findings (trigger nodes for derived rules) ─
    "ST_Elevation":     0.80,
    "ST_Depression":    0.35,
    "Irregular_RR":     0.55,
    "Wide_QRS":         0.30,
}

DEFAULT_PATIENT = {
    "symptoms": {
        "chest_pain":   True,
        "palpitations": False,
        "syncope":      True,
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
    _header("5/5  Decision fusion")
    from inference.decision_fusion import DecisionFusion

    # Inject history delta scores directly into patient dict
    # so decision_fusion.EvidenceScorer picks them up even if it
    # recomputes — avoids double-counting by capping contributions
    patient_with_deltas = dict(patient)
    patient_with_deltas['_history_deltas'] = {
        lid: delta.score_delta for lid, delta in history_deltas.items()
    }

    # Try passing history_deltas if fuse() accepts it, else fall back
    import inspect as _i
    _fuse_params = list(_i.signature(DecisionFusion.fuse).parameters.keys())
    if 'history_deltas' in _fuse_params:
        fused_raw = DecisionFusion().fuse(results, history_deltas, derived_log, patient)
    else:
        # Local decision_fusion.py uses EvidenceScorer internally
        # Patch: pre-apply history deltas to result.score so fusion sees them
        _delta_map = {lid: d.score_delta for lid, d in history_deltas.items()}
        for r in results:
            if r.label_id in _delta_map:
                r.score = r.score + _delta_map[r.label_id]
        fused_raw = DecisionFusion().fuse(results, patient, derived_log)

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
            if f.tier == 1:
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
            print(f"  {f.label_id:16s}  s_ai={f.s_ai:.3f}  score={f.score_final:.3f}")

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