"""
ECGenius — evaluation/evaluate_system.py
==========================================
Compares pipeline_outputs.json (what the SYSTEM said)
against annotations.csv (what the CARDIOLOGIST said).

Produces every metric needed for the paper:
  - Per-label AUC-ROC, F1, Precision, Recall
  - Macro / weighted averages
  - Tier-1 Recall  (THE safety metric — must be 1.0000)
  - Top-1 and Top-3 accuracy
  - Ablation table (AI-only vs AI+Rules vs AI+History vs Full)
  - Calibration (ECE) for AI probabilities
  - Per-label confusion matrix
  - Missed critical diagnoses list (names + scores for clinical audit)

Usage:
    python evaluation/evaluate_system.py \\
        --pipeline-output  evaluation/results/pipeline_outputs.json \\
        --annotations      data/raw/aiims/annotations.csv \\
        --probabilities    data/processed/model_probabilities_aiims.csv \\
        --output-dir       evaluation/results/

    # Quick check (no files needed):
    python evaluation/evaluate_system.py --mock
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ECGenius.Evaluate")

LABELS_15 = [
    "AF", "NSR", "STEMI", "NSTEMI", "LVH", "VF", "VT", "SVT",
    "LBBB", "RBBB", "Ischemia", "ST_Elevation", "ST_Depression", "TWI", "PVC",
]
TIER1_LABELS = {"STEMI", "VF", "VT", "NSTEMI", "ST_Elevation"}
CONFIDENCE_RANK = {"CONFIRMED": 4, "PROBABLE": 3, "POSSIBLE": 2, "INCIDENTAL": 1, "NOT_IN_DDx": 0}


def _sep(t: str) -> None:
    print(f"\n{'='*62}\n  {t}\n{'='*62}")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_pipeline_output(path: Path) -> dict[str, dict]:
    """
    Load pipeline_outputs.json.
    Returns {record_id: output_dict}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("records", raw) if isinstance(raw, dict) else raw
    return {r["record_id"]: r for r in records if "record_id" in r}


def load_annotations(path: Path) -> dict[str, dict[str, int]]:
    """
    Load ground truth annotations.csv.
    Returns {record_id: {label: 0_or_1}}
    """
    result: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("record_id", row.get("patient_id", ""))
            if not rid:
                continue
            labels = {}
            for label in LABELS_15:
                val = row.get(label, "0")
                try:
                    labels[label] = int(float(val)) if val != "" else 0
                except (ValueError, TypeError):
                    labels[label] = 0
            result[rid] = labels
    return result


def load_probabilities(path: Optional[Path]) -> dict[str, dict[str, float]]:
    """Load raw AI model probabilities (for AI-only baseline metrics)."""
    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = row.get("record_id", "")
            if not rid:
                continue
            probs = {}
            for label in LABELS_15:
                try:
                    probs[label] = float(row.get(label, 0) or 0)
                except (ValueError, TypeError):
                    probs[label] = 0.0
            result[rid] = probs
    return result


# ── Matrix builders ───────────────────────────────────────────────────────────

def build_score_matrices(
    pipeline_output: dict[str, dict],
    annotations:     dict[str, dict[str, int]],
    ai_probs:        dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build aligned matrices for all metrics.

    Returns:
      y_true         (N, 15) int      ground truth
      y_score_system (N, 15) float    system final score (from pipeline)
      y_score_ai     (N, 15) float    raw AI probability (from model_probs)
      common_ids     list[str]        aligned record IDs
    """
    common_ids = sorted(
        set(pipeline_output.keys()) & set(annotations.keys())
    )

    if not common_ids:
        raise ValueError(
            "No common record IDs between pipeline_outputs and annotations.\n"
            "Check that record_id column matches in both files."
        )

    y_true         = np.zeros((len(common_ids), len(LABELS_15)), dtype=np.float32)
    y_score_system = np.zeros((len(common_ids), len(LABELS_15)), dtype=np.float32)
    y_score_ai     = np.zeros((len(common_ids), len(LABELS_15)), dtype=np.float32)

    for i, rid in enumerate(common_ids):
        # Ground truth
        gt = annotations[rid]
        for j, label in enumerate(LABELS_15):
            y_true[i, j] = gt.get(label, 0)

        # System final scores
        diff = pipeline_output[rid].get("differential", [])
        supp = pipeline_output[rid].get("suppressed", [])
        for entry in diff + supp:
            lid = entry.get("label_id", "")
            if lid in LABELS_15:
                j = LABELS_15.index(lid)
                y_score_system[i, j] = entry.get("score_final", 0.0)

        # AI-only probabilities
        if rid in ai_probs:
            for j, label in enumerate(LABELS_15):
                y_score_ai[i, j] = ai_probs[rid].get(label, 0.0)

    return y_true, y_score_system, y_score_ai, common_ids


# ── Metric computation ────────────────────────────────────────────────────────

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[list[float], float, float]:
    """Per-label AUC-ROC. Returns (per_label_aucs, macro, weighted)."""
    from sklearn.metrics import roc_auc_score
    aucs      = []
    supports  = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        ys = y_score[:, j]
        if yt.sum() == 0 or yt.sum() == len(yt):
            aucs.append(float("nan"))    # cannot compute AUC with single class
        else:
            try:
                aucs.append(float(roc_auc_score(yt, ys)))
            except Exception:
                aucs.append(float("nan"))
        supports.append(int(yt.sum()))

    valid = [(a, s) for a, s in zip(aucs, supports) if not np.isnan(a)]
    if not valid:
        return aucs, float("nan"), float("nan")

    macro    = float(np.nanmean(aucs))
    total_s  = sum(s for _, s in valid)
    weighted = float(sum(a * s for a, s in valid) / total_s) if total_s > 0 else float("nan")
    return aucs, macro, weighted


def compute_f1_precision_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[dict], dict]:
    """Per-label F1, precision, recall at a threshold."""
    from sklearn.metrics import precision_recall_fscore_support
    y_pred = (y_score >= threshold).astype(int)
    per_label = []
    for j, label in enumerate(LABELS_15):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_label.append({
            "label_id": label,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "support":   int(yt.sum()),
        })

    macro_f1  = float(np.mean([r["f1"] for r in per_label]))
    macro_p   = float(np.mean([r["precision"] for r in per_label]))
    macro_r   = float(np.mean([r["recall"] for r in per_label]))
    return per_label, {"f1_macro": round(macro_f1, 4),
                       "precision_macro": round(macro_p, 4),
                       "recall_macro": round(macro_r, 4)}


def compute_tier1_recall(
    pipeline_output: dict[str, dict],
    annotations:     dict[str, dict[str, int]],
    common_ids:      list[str],
) -> dict:
    """
    THE safety metric.
    A Tier-1 label is "caught" if it appears anywhere in the system's
    differential (not suppressed) with confidence ≥ POSSIBLE.

    Returns dict with:
      tier1_recall      — fraction of true Tier-1 cases caught
      tier1_missed      — count of missed critical diagnoses
      tier1_total       — total true Tier-1 cases in test set
      missed_details    — list of {record_id, label, pai, system_score, ...}
    """
    total   = 0
    caught  = 0
    missed_details = []

    for rid in common_ids:
        gt     = annotations.get(rid, {})
        output = pipeline_output.get(rid, {})
        diff   = output.get("differential", [])

        # Build system label → entry map
        sys_label_map = {e["label_id"]: e for e in diff}

        for label in TIER1_LABELS:
            if gt.get(label, 0) != 1:
                continue   # not a true positive for this label
            total += 1

            sys_entry = sys_label_map.get(label)
            if sys_entry is None:
                # Label not in differential at all → MISSED
                missed_details.append({
                    "record_id":         rid,
                    "missed_label":      label,
                    "pai":               None,
                    "system_score":      None,
                    "system_confidence": "NOT_IN_DDx",
                    "reason":            "label not in differential (below threshold or suppressed)",
                })
            elif sys_entry.get("confidence_label") == "INCIDENTAL":
                # In differential but marked INCIDENTAL → MISSED (should never happen per safety rules)
                missed_details.append({
                    "record_id":         rid,
                    "missed_label":      label,
                    "pai":               sys_entry.get("pai"),
                    "system_score":      sys_entry.get("score_final"),
                    "system_confidence": "INCIDENTAL",
                    "reason":            "safety override should have prevented this",
                })
            else:
                caught += 1

    tier1_recall = caught / total if total > 0 else 1.0
    return {
        "tier1_recall":   round(tier1_recall, 4),
        "tier1_caught":   caught,
        "tier1_missed":   total - caught,
        "tier1_total":    total,
        "missed_details": missed_details,
    }


def compute_topk_accuracy(
    pipeline_output: dict[str, dict],
    annotations:     dict[str, dict[str, int]],
    common_ids:      list[str],
    k: int = 3,
) -> dict:
    """
    Top-k accuracy: true primary label appears in system's top-k differential.
    Uses the highest-confidence single ground truth label per record.
    """
    top1_hits = 0
    topk_hits = 0
    total     = 0

    for rid in common_ids:
        gt     = annotations.get(rid, {})
        output = pipeline_output.get(rid, {})
        diff   = [e["label_id"] for e in output.get("differential", [])]

        # Get true positive labels (there may be several in multi-label)
        true_labels = [l for l in LABELS_15 if gt.get(l, 0) == 1]
        if not true_labels:
            continue

        # Use Tier-1 label if present, else first true label
        primary = next((l for l in true_labels if l in TIER1_LABELS), true_labels[0])
        total += 1

        if diff and diff[0] == primary:
            top1_hits += 1
        if primary in diff[:k]:
            topk_hits += 1

    return {
        "top1_accuracy": round(top1_hits / total, 4) if total > 0 else 0.0,
        f"top{k}_accuracy": round(topk_hits / total, 4) if total > 0 else 0.0,
        "n_evaluated": total,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (macro across all labels)."""
    ece_per_label = []
    bins = np.linspace(0, 1, n_bins + 1)

    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        ece_j = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (yp >= lo) & (yp < hi)
            if mask.sum() == 0:
                continue
            acc  = float(yt[mask].mean())
            conf = float(yp[mask].mean())
            ece_j += abs(acc - conf) * mask.sum() / len(yt)
        ece_per_label.append(ece_j)

    return round(float(np.mean(ece_per_label)), 4)


# ── Ablation study ────────────────────────────────────────────────────────────

def ablation_study(
    y_true:         np.ndarray,
    y_score_ai:     np.ndarray,
    y_score_system: np.ndarray,
    pipeline_output: dict[str, dict],
    annotations:    dict[str, dict[str, int]],
    common_ids:     list[str],
) -> list[dict]:
    """
    Four-row ablation table for the paper.
    Compares AI-only vs Full ECGenius system.
    Middle rows (AI+Rules, AI+History) require score components saved separately.
    If not available, those rows are marked as N/A.
    """
    rows = []

    def _row(variant: str, y_score: np.ndarray, use_pipeline: bool = False) -> dict:
        aucs, macro_auc, weighted_auc = compute_auc(y_true, y_score)
        _, f1s = compute_f1_precision_recall(y_true, y_score, threshold=0.5)

        if use_pipeline:
            t1 = compute_tier1_recall(pipeline_output, annotations, common_ids)
            topk = compute_topk_accuracy(pipeline_output, annotations, common_ids)
        else:
            # For AI-only: Tier-1 recall = fraction of Tier-1 positives where AI score > 0.5
            t1_total = 0
            t1_caught = 0
            for i, rid in enumerate(common_ids):
                gt = annotations.get(rid, {})
                for label in TIER1_LABELS:
                    if gt.get(label, 0) == 1:
                        t1_total += 1
                        j = LABELS_15.index(label) if label in LABELS_15 else -1
                        if j >= 0 and y_score[i, j] >= 0.5:
                            t1_caught += 1
            t1 = {
                "tier1_recall": round(t1_caught / t1_total, 4) if t1_total > 0 else 1.0,
                "tier1_missed": t1_total - t1_caught,
                "tier1_total":  t1_total,
            }
            # Top-1: highest system score
            top1 = 0
            total = 0
            for i, rid in enumerate(common_ids):
                gt = annotations.get(rid, {})
                true_labels = [l for l in LABELS_15 if gt.get(l, 0) == 1]
                if not true_labels:
                    continue
                primary = next((l for l in true_labels if l in TIER1_LABELS), true_labels[0])
                pred_top = LABELS_15[int(np.argmax(y_score[i]))]
                total += 1
                if pred_top == primary:
                    top1 += 1
            topk = {
                "top1_accuracy": round(top1/total, 4) if total > 0 else 0.0,
                "top3_accuracy": "N/A",
            }

        return {
            "variant":        variant,
            "macro_auc":      round(macro_auc, 4),
            "f1_macro":       f1s["f1_macro"],
            "top1_accuracy":  topk["top1_accuracy"],
            "top3_accuracy":  topk.get("top3_accuracy", "N/A"),
            "tier1_recall":   t1["tier1_recall"],
            "tier1_missed":   t1.get("tier1_missed", "N/A"),
        }

    rows.append(_row("1. AI only (LightV2)",           y_score_ai,     use_pipeline=False))
    rows.append({"variant": "2. AI + Ontology rules", "note": "Run with --rules-only flag (not yet implemented — see README)", **{k:"N/A" for k in ["macro_auc","f1_macro","top1_accuracy","top3_accuracy","tier1_recall","tier1_missed"]}})
    rows.append({"variant": "3. AI + Patient history", "note": "Run batch with --no-rules flag (not yet implemented)", **{k:"N/A" for k in ["macro_auc","f1_macro","top1_accuracy","top3_accuracy","tier1_recall","tier1_missed"]}})
    rows.append(_row("4. Full ECGenius (AI+Rules+History)", y_score_system, use_pipeline=True))

    return rows


# ── Report printers ───────────────────────────────────────────────────────────

def print_tier1_results(t1: dict) -> None:
    recall  = t1["tier1_recall"]
    missed  = t1["tier1_missed"]
    total   = t1["tier1_total"]

    print(f"\n  Tier-1 Recall:  {recall:.4f}  ({t1['tier1_caught']}/{total} caught)")

    if missed == 0:
        print(f"  ✓  ZERO CRITICAL DIAGNOSES MISSED — safety guarantee holds.")
        print(f"  ✓  This is your key paper result: Tier-1 Recall = 1.0000")
    else:
        print(f"\n  !!! WARNING: {missed} CRITICAL DIAGNOSES MISSED !!!")
        print(f"  These records require clinical review:")
        for m in t1["missed_details"]:
            print(f"\n    Record:      {m['record_id']}")
            print(f"    Missed:      {m['missed_label']}")
            print(f"    AI Pai:      {m['pai']}")
            print(f"    Sys Score:   {m['system_score']}")
            print(f"    Confidence:  {m['system_confidence']}")
            print(f"    Reason:      {m['reason']}")
        print(f"\n  Action: Investigate why safety override failed for these cases.")
        print(f"  Check:  decision_fusion.py Tier-1 safety override condition (Pai >= 0.70)")


def print_ablation_table(rows: list[dict]) -> None:
    print(f"\n  {'Variant':<42} {'AUC':>7} {'F1':>7} {'Top-1':>7} {'Top-3':>7} {'T1-Rcl':>8} {'T1-Miss':>8}")
    print(f"  {'─'*88}")
    for r in rows:
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        print(
            f"  {r['variant']:<42} "
            f"{fmt(r.get('macro_auc','N/A')):>7} "
            f"{fmt(r.get('f1_macro','N/A')):>7} "
            f"{fmt(r.get('top1_accuracy','N/A')):>7} "
            f"{fmt(r.get('top3_accuracy','N/A')):>7} "
            f"{fmt(r.get('tier1_recall','N/A')):>8} "
            f"{str(r.get('tier1_missed','N/A')):>8}"
        )


def print_per_label_auc(aucs: list[float]) -> None:
    print(f"\n  {'Label':<18} {'AUC':>7}  {'Bar':<25}  {'Tier'}")
    print(f"  {'─'*60}")
    rows = sorted(zip(LABELS_15, aucs), key=lambda x: -x[1] if not np.isnan(x[1]) else -999)
    for label, auc in rows:
        bar  = "█" * int(auc * 25) if not np.isnan(auc) else "—"
        tier = " ← TIER-1" if label in TIER1_LABELS else ""
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "  N/A  "
        print(f"  {label:<18} {auc_str:>7}  {bar:<25}  {tier}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    from typing import Optional

    _sep("ECGenius — System Evaluation")

    if args.mock:
        print("  MOCK MODE — generating synthetic data")
        _run_mock_evaluation()
        return

    # Load data
    pipeline_path  = PROJECT_ROOT / args.pipeline_output
    ann_path       = PROJECT_ROOT / args.annotations
    probs_path     = PROJECT_ROOT / args.probabilities if args.probabilities else None
    out_dir        = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pipeline_path.exists():
        print(f"\n  ERROR: pipeline output not found: {pipeline_path}")
        print(f"  Run first:  python scripts/run_full_pipeline_batch.py ...")
        sys.exit(1)

    if not ann_path.exists():
        print(f"\n  ERROR: annotations not found: {ann_path}")
        sys.exit(1)

    print(f"  Pipeline output : {pipeline_path}")
    print(f"  Annotations     : {ann_path}")

    pipeline_output = load_pipeline_output(pipeline_path)
    annotations     = load_annotations(ann_path)
    ai_probs        = load_probabilities(probs_path)

    # Build matrices
    y_true, y_score_system, y_score_ai, common_ids = build_score_matrices(
        pipeline_output, annotations, ai_probs
    )
    print(f"\n  Matched records : {len(common_ids)}")
    print(f"  Labels          : {len(LABELS_15)}")
    print(f"  Positive labels : {int(y_true.sum())} total across all labels")

    if len(common_ids) < 30:
        print(f"\n  WARNING: Only {len(common_ids)} records. Metrics will be noisy.")
        print(f"  Need ≥ 200 test records for stable AUC estimates.")

    # ── Metric 1: AUC-ROC ────────────────────────────────────────────────
    _sep("AUC-ROC")
    aucs_sys, macro_auc_sys, weighted_auc_sys = compute_auc(y_true, y_score_system)
    aucs_ai,  macro_auc_ai,  weighted_auc_ai  = compute_auc(y_true, y_score_ai)

    print(f"\n  System:  Macro AUC = {macro_auc_sys:.4f}  |  Weighted AUC = {weighted_auc_sys:.4f}")
    if not np.isnan(macro_auc_ai):
        print(f"  AI-only: Macro AUC = {macro_auc_ai:.4f}  |  Weighted AUC = {weighted_auc_ai:.4f}")
        delta = macro_auc_sys - macro_auc_ai
        print(f"  Delta (System − AI): {delta:+.4f}")

    print_per_label_auc(aucs_sys)

    # ── Metric 2: F1, Precision, Recall ──────────────────────────────────
    _sep("F1 / Precision / Recall  (threshold = 0.50)")
    per_label_f1, macro_f1 = compute_f1_precision_recall(y_true, y_score_system, threshold=0.5)
    print(f"\n  Macro F1 = {macro_f1['f1_macro']:.4f}   "
          f"Precision = {macro_f1['precision_macro']:.4f}   "
          f"Recall = {macro_f1['recall_macro']:.4f}")

    print(f"\n  {'Label':<18} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>9}")
    print(f"  {'─'*54}")
    for r in sorted(per_label_f1, key=lambda x: -x["f1"]):
        tier = " T1" if r["label_id"] in TIER1_LABELS else ""
        print(f"  {r['label_id']:<18} {r['precision']:>7.4f} {r['recall']:>7.4f} "
              f"{r['f1']:>7.4f} {r['support']:>9}{tier}")

    # ── Metric 3: TIER-1 RECALL (THE SAFETY METRIC) ──────────────────────
    _sep("TIER-1 RECALL — Safety Metric")
    t1 = compute_tier1_recall(pipeline_output, annotations, common_ids)
    print_tier1_results(t1)

    # ── Metric 4: Top-k accuracy ──────────────────────────────────────────
    _sep("Top-1 / Top-3 Accuracy")
    topk = compute_topk_accuracy(pipeline_output, annotations, common_ids, k=3)
    print(f"\n  Top-1 accuracy : {topk['top1_accuracy']:.4f}")
    print(f"  Top-3 accuracy : {topk['top3_accuracy']:.4f}")
    print(f"  N evaluated    : {topk['n_evaluated']}")

    # ── Metric 5: Calibration (ECE) ───────────────────────────────────────
    _sep("Calibration — Expected Calibration Error (ECE)")
    if y_score_ai.sum() > 0:
        ece_ai  = compute_ece(y_true, y_score_ai)
        ece_sys = compute_ece(y_true, y_score_system)
        print(f"\n  ECE (AI-only)  : {ece_ai:.4f}   (target < 0.05 for clinical use)")
        print(f"  ECE (System)   : {ece_sys:.4f}")
        if ece_ai > 0.05:
            print(f"\n  WARNING: ECE > 0.05 — apply Platt scaling or temperature scaling.")
            print(f"  Code: from sklearn.calibration import CalibratedClassifierCV")
    else:
        print("  No AI probabilities available — skipping calibration.")
        ece_ai = ece_sys = None

    # ── Metric 6: Ablation table ──────────────────────────────────────────
    _sep("Ablation Study")
    ablation_rows = ablation_study(
        y_true, y_score_ai, y_score_system,
        pipeline_output, annotations, common_ids
    )
    print_ablation_table(ablation_rows)

    # ── Save all results ──────────────────────────────────────────────────
    _sep("Saving Results")
    summary = {
        "n_records":         len(common_ids),
        "macro_auc_system":  round(macro_auc_sys, 4),
        "macro_auc_ai":      round(macro_auc_ai, 4) if not np.isnan(macro_auc_ai) else None,
        "weighted_auc_system": round(weighted_auc_sys, 4),
        "f1_macro":          macro_f1["f1_macro"],
        "precision_macro":   macro_f1["precision_macro"],
        "recall_macro":      macro_f1["recall_macro"],
        "tier1_recall":      t1["tier1_recall"],
        "tier1_missed":      t1["tier1_missed"],
        "tier1_total":       t1["tier1_total"],
        "top1_accuracy":     topk["top1_accuracy"],
        "top3_accuracy":     topk["top3_accuracy"],
        "ece_ai":            ece_ai,
        "ece_system":        ece_sys,
        "per_label_auc":     {l: round(a, 4) if not np.isnan(a) else None
                              for l, a in zip(LABELS_15, aucs_sys)},
        "per_label_f1":      per_label_f1,
        "missed_tier1_details": t1["missed_details"],
        "ablation":          ablation_rows,
    }

    out_path = out_dir / "system_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  system_metrics.json → {out_path}")

    # Per-label CSV (for Excel table in paper)
    csv_path = out_dir / "per_label_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["label_id", "auc_roc", "precision", "recall", "f1", "support", "tier1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        auc_map = dict(zip(LABELS_15, aucs_sys))
        for r in per_label_f1:
            w.writerow({
                "label_id":  r["label_id"],
                "auc_roc":   round(auc_map.get(r["label_id"], float("nan")), 4),
                "precision": r["precision"],
                "recall":    r["recall"],
                "f1":        r["f1"],
                "support":   r["support"],
                "tier1":     "YES" if r["label_id"] in TIER1_LABELS else "",
            })
    print(f"  per_label_metrics.csv → {csv_path}")

    # Ablation CSV
    abl_path = out_dir / "ablation_table.csv"
    with open(abl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        w.writeheader()
        w.writerows(ablation_rows)
    print(f"  ablation_table.csv → {abl_path}")

    print(f"\n  EVALUATION COMPLETE")
    print(f"  Key results:")
    print(f"    Macro AUC      : {macro_auc_sys:.4f}")
    print(f"    Tier-1 Recall  : {t1['tier1_recall']:.4f}  ({'SAFE' if t1['tier1_missed']==0 else 'UNSAFE'})")
    print(f"    Top-1 Accuracy : {topk['top1_accuracy']:.4f}")
    print()


def _run_mock_evaluation():
    """Quick self-test with synthetic data — no files needed."""
    import numpy as np
    rng = np.random.default_rng(42)
    n   = 200

    y_true = np.zeros((n, len(LABELS_15)))
    for i in range(n):
        n_pos = rng.choice([1, 2], p=[0.7, 0.3])
        y_true[i, rng.choice(len(LABELS_15), n_pos, replace=False)] = 1

    y_score_ai  = np.clip(0.7 * y_true + 0.15 * rng.random((n, len(LABELS_15))), 0, 1)
    y_score_sys = np.clip(y_score_ai + 0.05 * rng.random((n, len(LABELS_15))) * y_true, 0, 1)

    print("\n  Macro AUC (AI-only):", round(compute_auc(y_true, y_score_ai)[1], 4))
    print("  Macro AUC (System): ", round(compute_auc(y_true, y_score_sys)[1], 4))
    print("  ECE (AI-only):      ", compute_ece(y_true, y_score_ai))
    print("\n  Mock evaluation OK — real evaluation needs pipeline_outputs.json + annotations.csv")


def main():
    parser = argparse.ArgumentParser(description="ECGenius System Evaluation")
    parser.add_argument("--pipeline-output", type=str,
                        default="evaluation/results/pipeline_outputs.json")
    parser.add_argument("--annotations", type=str,
                        default="data/raw/aiims/annotations.csv")
    parser.add_argument("--probabilities", type=str,
                        default="data/processed/model_probabilities_aiims.csv",
                        help="AI-only probabilities CSV (for baseline comparison)")
    parser.add_argument("--output-dir", type=str,
                        default="evaluation/results/")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
