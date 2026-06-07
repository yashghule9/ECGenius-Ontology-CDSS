"""
Standalone test for the Naive Bayes evidence fusion engine.

Scenario: 80% AI-prior STEMI patient with multiple confirming features.
Expected: STEMI posterior ≈ 0.9995+, tier=Confirmed, Tier-1 override logged.

Run:
    python tests/test_naive_bayes_fusion.py
    # or
    pytest tests/test_naive_bayes_fusion.py -v
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from inference.decision_fusion import naive_bayes_fusion

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Test scenario ─────────────────────────────────────────────────────────────

AI_PRIORS = {
    "STEMI":            0.80,
    "NSTEMI":           0.10,
    "Unstable_Angina":  0.05,
    "AF":               0.03,
    "Pericarditis":     0.02,
}

PATIENT_EVIDENCE = {
    "chest_pain":   True,
    "diaphoresis":  True,
    "dyspnea":      False,
    "syncope":      False,
    "cad":          True,
    "dm":           False,
    "smoking":      True,
    "prior_mi":     False,
    "sbp_lt_90":    True,
    "spo2_lt_94":   True,
    "hr_gt_100":    True,
}

CPT_PATH = PROJECT_ROOT / "history_module" / "history_rules.csv"


def _load_cpt() -> pd.DataFrame:
    if not CPT_PATH.exists():
        raise FileNotFoundError(f"history_rules.csv not found at {CPT_PATH}")
    return pd.read_csv(str(CPT_PATH), comment="#")


def _print_breakdown_table(results: dict) -> None:
    header = f"{'Label':20s}  {'Prior':>8s}  {'log_LL':>8s}  {'Unnorm':>12s}  {'Posterior':>10s}  {'Tier'}"
    print("\n" + header)
    print("-" * len(header))
    for label, entry in results.items():
        bd = entry["breakdown"]
        print(
            f"  {label:18s}  {bd.get('prior', 0):8.5f}  "
            f"{bd.get('log_likelihood', 0):8.4f}  "
            f"{bd.get('unnorm_posterior', 0):12.6e}  "
            f"{entry['posterior']:10.6f}  "
            f"{entry['tier']}"
        )


def test_stemi_dominant():
    cpt_df  = _load_cpt()
    results = naive_bayes_fusion(
        ai_priors        = AI_PRIORS,
        patient_evidence = PATIENT_EVIDENCE,
        cpt_table        = cpt_df,
        triage_table     = pd.DataFrame(),
        rules_table      = pd.DataFrame(),
        tier1_labels     = ["STEMI", "VF", "VT"],
    )

    print("\n" + "=" * 60)
    print("  Naive Bayes Fusion — STEMI Scenario")
    print("=" * 60)
    _print_breakdown_table(results)

    stemi = results.get("STEMI")
    assert stemi is not None, "STEMI must be present in results"

    # STEMI should dominate given the confirming evidence profile
    assert stemi["posterior"] > 0.99, (
        f"Expected STEMI posterior > 0.99, got {stemi['posterior']:.6f}"
    )
    assert list(results.keys())[0] == "STEMI", "STEMI must rank first"
    assert stemi["tier"] == "Confirmed", (
        f"Expected Tier-1 override → Confirmed, got {stemi['tier']}"
    )

    # Verify log-space computation is consistent
    bd = stemi["breakdown"]
    expected_log_score = bd["log_prior"] + bd["log_likelihood"]
    assert math.isclose(bd["log_score"], expected_log_score, rel_tol=1e-6), (
        f"log_score mismatch: {bd['log_score']} vs {expected_log_score}"
    )

    print(f"\n  STEMI posterior:  {stemi['posterior']:.6f}  [{stemi['tier']}]")
    print(f"  Tier-1 override:  {'PASS' if stemi['tier'] == 'Confirmed' else 'FAIL'}")
    print(f"\n  All assertions passed.\n")


def test_normalization_sums_to_one():
    cpt_df  = _load_cpt()
    results = naive_bayes_fusion(
        ai_priors        = AI_PRIORS,
        patient_evidence = PATIENT_EVIDENCE,
        cpt_table        = cpt_df,
        triage_table     = pd.DataFrame(),
        rules_table      = pd.DataFrame(),
        tier1_labels     = ["STEMI", "VF", "VT"],
    )
    total = sum(entry["posterior"] for entry in results.values())
    assert math.isclose(total, 1.0, abs_tol=1e-5), (
        f"Posteriors must sum to 1.0, got {total:.8f}"
    )
    print(f"  Posterior sum = {total:.8f}  [PASS]")


def test_no_evidence_preserves_prior_order():
    """With no active evidence, order should follow prior."""
    cpt_df = _load_cpt()
    empty_evidence = {k: False for k in PATIENT_EVIDENCE}
    results = naive_bayes_fusion(
        ai_priors        = AI_PRIORS,
        patient_evidence = empty_evidence,
        cpt_table        = cpt_df,
        triage_table     = pd.DataFrame(),
        rules_table      = pd.DataFrame(),
        tier1_labels     = ["STEMI", "VF", "VT"],
    )
    labels_by_posterior = list(results.keys())
    prior_order = sorted(AI_PRIORS, key=lambda d: -AI_PRIORS[d])
    assert labels_by_posterior == prior_order, (
        f"With no evidence, order should match prior.\n"
        f"Got: {labels_by_posterior}\nExpected: {prior_order}"
    )
    print(f"  No-evidence order matches prior:  PASS")


def test_invalid_priors_raises():
    cpt_df = _load_cpt()
    try:
        naive_bayes_fusion(
            ai_priors        = {},
            patient_evidence = {},
            cpt_table        = cpt_df,
            triage_table     = pd.DataFrame(),
            rules_table      = pd.DataFrame(),
            tier1_labels     = [],
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Empty ai_priors raises ValueError:  PASS")


if __name__ == "__main__":
    print("\nRunning Naive Bayes fusion tests...\n")
    test_stemi_dominant()
    test_normalization_sums_to_one()
    test_no_evidence_preserves_prior_order()
    test_invalid_priors_raises()
    print("All tests passed.\n")
