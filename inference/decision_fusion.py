"""
ECGenius — inference/decision_fusion.py
========================================
Multi-class Naive Bayes evidence fusion engine.

Bayesian formulation:
    P(D | E) ∝ P(D) · ∏_i  P(E_i | D) / P(E_i | ¬D)

All computation is in LOG-SPACE to prevent floating-point underflow:
    logL(D) = log P_AI(D)  +  Σ_i  log[ P(E_i|D) / P(E_i|¬D) ]
    posterior(D) = exp(logL(D)) / Σ_{D'} exp(logL(D'))

P_AI(D) is the raw sigmoid output from LightECGNet v2, normalised to a
proper distribution before use as the log-prior.
P(E_i|D) / P(E_i|¬D) are likelihood ratios from the CPT table
(history_rules.csv, columns p_given_disease / p_given_not_disease).

Backward compatibility:
    If history_rules.csv still has the legacy `action` / `delta` columns
    (migration period), DecisionFusion.fuse() falls back to the old
    weighted additive formula with a DeprecationWarning.

CPT values initialized from clinical literature.
To be recalibrated via MLE on AIIMS Nagpur Indian cohort (Phase 3, Month 18).
Use bnlearn or pgmpy for parametric learning:
    fit(data, estimator=MaximumLikelihoodEstimator)
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONFIDENCE_BANDS = [
    (0.80, "CONFIRMED"),
    (0.60, "PROBABLE"),
    (0.30, "POSSIBLE"),
    (0.00, "INCIDENTAL"),
]

TIER_LABELS = {1: "life-threatening", 2: "urgent", 3: "routine"}

# Legacy additive-score weights (used only in fallback path)
_LEGACY_W_AI        = 0.50
_LEGACY_CAP_SYMPTOM = 0.30
_LEGACY_CAP_RISK    = 0.10
_LEGACY_CAP_RULE    = 0.10

LAPLACE_EPS: float = 1e-6   # avoids log(0)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    """Final ranked diagnosis — everything the UI needs in one object."""
    rank:             int
    label_id:         str
    label_name:       str
    category:         str
    description:      str
    hierarchy:        list[str]

    # Score breakdown (kept for XAI transparency)
    pai:              float   # raw AI probability (sigmoid output)
    s_ai:             float   # normalised prior used in NB (or 0.5×pai in legacy)
    s_symptom:        float   # legacy field — 0.0 in NB mode
    s_risk:           float   # legacy field — 0.0 in NB mode
    s_rule:           float   # legacy field — 0.0 in NB mode
    score_final:      float   # posterior probability (NB) or clamped score (legacy)
    confidence_label: str

    # Triage
    tier:             int
    tier_label:       str
    default_action:   str
    allow_downgrade:  bool

    # Terminology
    snomed_ct:        str
    icd10:            str
    aha_guideline:    str
    clinical_notes:   str

    # Flags
    is_suppressed:    bool
    is_derived:       bool = False

    # XAI evidence (from history_encoder / NB breakdown)
    supporting:       list[str] = field(default_factory=list)
    contradicting:    list[str] = field(default_factory=list)
    evidence_log:     list[str] = field(default_factory=list)

    # Naive Bayes per-disease breakdown for SHAP/explainability
    nb_breakdown:     dict = field(default_factory=dict)

    # Filled by explainability.py later
    explanation:      str = ""

    def to_dict(self) -> dict:
        return {
            "rank":             self.rank,
            "label_id":         self.label_id,
            "label_name":       self.label_name,
            "category":         self.category,
            "hierarchy":        self.hierarchy,
            "score_breakdown": {
                "pai":        round(self.pai, 4),
                "s_ai":       round(self.s_ai, 4),
                "s_symptom":  round(self.s_symptom, 4),
                "s_risk":     round(self.s_risk, 4),
                "s_rule":     round(self.s_rule, 4),
                "total":      round(self.score_final, 4),
            },
            "nb_breakdown":     self.nb_breakdown,
            "confidence_label": self.confidence_label,
            "tier":             self.tier,
            "tier_label":       self.tier_label,
            "default_action":   self.default_action,
            "snomed_ct":        self.snomed_ct,
            "icd10":            self.icd10,
            "aha_guideline":    self.aha_guideline,
            "clinical_notes":   self.clinical_notes,
            "is_suppressed":    self.is_suppressed,
            "is_derived":       self.is_derived,
            "supporting":       self.supporting,
            "contradicting":    self.contradicting,
            "evidence_log":     self.evidence_log,
            "explanation":      self.explanation,
        }


@dataclass
class FusionOutput:
    """Top-level output handed to UI and explainability layer."""
    patient_id:      str
    results:         list[FusionResult]
    active_results:  list[FusionResult]
    top_diagnosis:   Optional[FusionResult]
    critical_alerts: list[FusionResult]
    derived_log:     list[str]
    metadata:        dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "patient_id":      self.patient_id,
            "top_diagnosis":   self.top_diagnosis.to_dict() if self.top_diagnosis else None,
            "critical_alerts": [r.to_dict() for r in self.critical_alerts],
            "active_results":  [r.to_dict() for r in self.active_results],
            "all_results":     [r.to_dict() for r in self.results],
            "derived_log":     self.derived_log,
            "metadata":        self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── Standalone Naive Bayes fusion function ────────────────────────────────────

def naive_bayes_fusion(
    ai_priors: dict,
    patient_evidence: dict,
    cpt_table: pd.DataFrame,
    triage_table: pd.DataFrame,
    rules_table: pd.DataFrame,
    tier1_labels: list,
) -> dict:
    """
    Multi-class Naive Bayes evidence fusion in log-space.

    Parameters
    ----------
    ai_priors : dict
        {disease_label: float} — raw sigmoid outputs from LightECGNet v2.
        Used as the prior P(D) after normalisation.
    patient_evidence : dict
        {feature_name: bool} — present/absent flags derived from patient data.
        Only features where value is True contribute log-LR updates.
    cpt_table : pd.DataFrame
        Loaded from history_rules.csv.  Must have columns:
          label_id, trigger, p_given_disease, p_given_not_disease
        Falls back to legacy weighted-sum if `action` / `delta` columns are
        detected instead (DeprecationWarning is raised).
    triage_table : pd.DataFrame
        Loaded from ontology/triage.csv — reserved for future tier lookup.
    rules_table : pd.DataFrame
        Loaded from ontology/rules.csv — reserved for future integration.
    tier1_labels : list
        Labels that must never be suppressed (e.g. ["STEMI", "VF", "VT"]).

    Returns
    -------
    dict
        Sorted by posterior descending.  Each key is a disease label; each
        value is a dict with keys:
          posterior, tier, breakdown
        where breakdown contains:
          prior, log_likelihood, unnorm_posterior, final_posterior, tier
    """
    if not ai_priors:
        raise ValueError("ai_priors must be a non-empty dict of {label: probability}.")

    # ── Detect legacy CSV and fall back ──────────────────────────────────────
    if "action" in cpt_table.columns and "delta" in cpt_table.columns:
        warnings.warn(
            "history_rules.csv still uses legacy `action`/`delta` columns. "
            "Falling back to weighted additive scoring. "
            "Migrate to p_given_disease / p_given_not_disease columns.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _legacy_weighted_sum(ai_priors, patient_evidence, cpt_table, tier1_labels)

    # ── Step 1: Normalise AI priors to a proper distribution ─────────────────
    total_prior = sum(ai_priors.values())
    if total_prior <= 0.0:
        raise ValueError("ai_priors values must sum to a positive number.")
    normalised_prior: dict[str, float] = {
        d: float(v) / total_prior for d, v in ai_priors.items()
    }

    # ── Step 2: Build a fast lookup index from the CPT table ─────────────────
    # CPT values initialized from clinical literature.
    # To be recalibrated via MLE on AIIMS Nagpur Indian cohort (Phase 3, Month 18).
    # Use bnlearn or pgmpy for parametric learning: fit(data, estimator=MaximumLikelihoodEstimator)
    # triage_table and rules_table are reserved for Phase 3 tier/rule integration
    _ = triage_table
    _ = rules_table

    cpt_index: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in cpt_table.iterrows():
        lid     = str(row["label_id"]).strip()
        trigger = str(row["trigger"]).strip()
        p_pos   = float(row["p_given_disease"])
        p_neg   = float(row["p_given_not_disease"])
        cpt_index[(lid, trigger)] = (p_pos, p_neg)

    # Active evidence features (only True flags contribute)
    active_features = [f for f, v in patient_evidence.items() if bool(v)]

    log_scores: dict[str, float] = {}
    breakdowns: dict[str, dict] = {}

    for disease, prior in normalised_prior.items():
        # Start with log prior
        log_prior = math.log(max(prior, LAPLACE_EPS))
        log_ll    = 0.0
        ev_terms  = []

        for feature in active_features:
            key = (disease, feature)
            if key not in cpt_index:
                # No CPT row for this (disease, feature) pair — skip, do not penalise
                continue

            p_pos, p_neg = cpt_index[key]

            # Laplace smoothing prevents log(0)
            p_pos = max(p_pos, LAPLACE_EPS)
            p_neg = max(p_neg, LAPLACE_EPS)

            # Log likelihood ratio: log[ P(E|D) / P(E|¬D) ]
            lr_log = math.log(p_pos / p_neg)
            log_ll += lr_log
            ev_terms.append((feature, round(p_pos, 4), round(p_neg, 4), round(lr_log, 4)))

        log_scores[disease] = log_prior + log_ll
        breakdowns[disease] = {
            "prior":                round(prior, 6),
            "log_prior":            round(log_prior, 6),
            "log_likelihood":       round(log_ll, 6),
            "log_score":            round(log_prior + log_ll, 6),
            "evidence_terms":       ev_terms,
        }

    # ── Step 3: Exponentiate — shift by max for numerical stability ───────────
    max_log = max(log_scores.values())
    unnorm: dict[str, float] = {
        d: math.exp(ls - max_log) for d, ls in log_scores.items()
    }

    # ── Step 4: Normalise to obtain posterior distribution ───────────────────
    total_unnorm = sum(unnorm.values())
    if total_unnorm <= 0.0 or not math.isfinite(total_unnorm):
        # Numerical fallback: use normalised priors directly
        logger.warning("Numerical instability in NB posteriors — falling back to normalised priors.")
        posterior: dict[str, float] = dict(normalised_prior)
    else:
        posterior = {d: float64(v / total_unnorm) for d, v in unnorm.items()}

    # ── Step 5: Assign confidence tiers ──────────────────────────────────────
    def _confidence_tier(p: float, label: str) -> str:
        # tier1_labels captured from enclosing scope via closure
        if label in tier1_labels and p > 0.15:  # Tier-1 safety override
            logger.info(
                "Tier-1 safety override: '%s' posterior=%.4f → Confirmed", label, p
            )
            return "Confirmed"
        if p >= 0.80: return "Confirmed"
        if p >= 0.60: return "Probable"
        if p >= 0.30: return "Possible"
        return "Incidental"

    # ── Step 6: Finalise breakdown and sort ──────────────────────────────────
    result: dict[str, dict] = {}
    for disease, post in posterior.items():
        bd = breakdowns[disease]
        bd["unnorm_posterior"] = round(unnorm.get(disease, 0.0), 8)
        bd["final_posterior"]  = round(post, 6)
        bd["tier"]             = _confidence_tier(post, disease)
        result[disease] = {
            "posterior":   round(post, 6),
            "tier":        bd["tier"],
            "breakdown":   bd,
        }

    # Sort by posterior descending
    result = dict(sorted(result.items(), key=lambda x: -x[1]["posterior"]))
    return result


def float64(x: float) -> float:
    """Ensure float64 precision (no-op at runtime; signals intent for numpy paths)."""
    return float(np.float64(x))


# ── Legacy fallback ───────────────────────────────────────────────────────────

def _legacy_weighted_sum(
    ai_priors: dict,
    patient_evidence: dict,
    cpt_table: pd.DataFrame,
    tier1_labels: list,
) -> dict:
    """Weighted additive scoring — used only when legacy CSV columns are detected."""
    total = sum(ai_priors.values()) or 1.0
    result = {}
    for disease, pai in ai_priors.items():
        score = 0.5 * (pai / total)
        mask  = cpt_table["label_id"] == disease
        for _, row in cpt_table[mask].iterrows():
            trigger = str(row.get("trigger", "")).strip()
            if patient_evidence.get(trigger):
                action = str(row.get("action", "")).strip()
                delta  = float(row.get("delta", 0) or 0)
                if action == "add_score":
                    score += delta
                elif action == "reduce_score":
                    score -= abs(delta)
        score = max(0.0, min(1.0, score))
        tier  = "Confirmed" if score >= 0.80 else \
                "Probable"  if score >= 0.60 else \
                "Possible"  if score >= 0.30 else "Incidental"
        result[disease] = {"posterior": round(score, 6), "tier": tier, "breakdown": {}}
    return dict(sorted(result.items(), key=lambda x: -x[1]["posterior"]))


# ── Main class ────────────────────────────────────────────────────────────────

class DecisionFusion:
    """
    Combines AI output + ontology + patient evidence into ranked DDx.

    In NB mode (default): calls naive_bayes_fusion() internally.
    Falls back to legacy weighted-sum if the CPT table uses old schema.
    """

    def fuse(
        self,
        ontology_results: list,
        history_deltas:   dict,
        derived_log:      list[str],
        patient:          dict = None,
        patient_id:       str  = "unknown",
        cpt_table:        Optional[pd.DataFrame] = None,
        patient_evidence: Optional[dict]         = None,
    ) -> FusionOutput:
        """
        Parameters
        ----------
        ontology_results : list[OntologyResult]
        history_deltas : dict[str, HistoryDelta]
        derived_log : list[str]
        patient : dict
        patient_id : str
        cpt_table : pd.DataFrame, optional
            If provided, enables Naive Bayes mode.  If omitted, legacy mode.
        patient_evidence : dict, optional
            Pre-built {feature: bool} dict.  If omitted, built from patient.
        """
        patient     = patient or {}
        results_out = []

        # ── Build AI priors — exclude ECG pattern nodes (they are rule triggers,
        #    not standalone diagnoses; their CPT entries do not exist) ────────
        _PATTERN_CATEGORY = "Pattern"
        ai_priors = {
            r.label_id: float(r.pai)
            for r in ontology_results
            if getattr(r, "category", "") != _PATTERN_CATEGORY
        }

        # ── Run NB fusion if CPT table is available ───────────────────────────
        nb_posteriors: dict[str, dict] = {}
        if cpt_table is not None:
            if patient_evidence is None:
                patient_evidence = _build_patient_evidence(patient)
            try:
                nb_posteriors = naive_bayes_fusion(
                    ai_priors        = ai_priors,
                    patient_evidence = patient_evidence,
                    cpt_table        = cpt_table,
                    triage_table     = pd.DataFrame(),
                    rules_table      = pd.DataFrame(),
                    tier1_labels     = ["STEMI", "VF", "VT"],
                )
            except Exception as exc:
                logger.warning("NB fusion failed (%s) — falling back to legacy mode.", exc)

        for result in ontology_results:
            delta = history_deltas.get(result.label_id)

            # ── Score and confidence ──────────────────────────────────────────
            if nb_posteriors and result.label_id in nb_posteriors:
                nb_entry = nb_posteriors[result.label_id]
                score       = round(nb_entry["posterior"], 4)
                confidence  = nb_entry["tier"].upper()   # Confirmed→CONFIRMED etc.
                nb_bd       = nb_entry["breakdown"]
                s_ai        = round(nb_bd.get("prior", result.pai), 4)
                s_symptom   = 0.0
                s_risk      = 0.0
                s_rule      = 0.0
            else:
                # Legacy additive path
                s_ai      = round(_LEGACY_W_AI * result.pai, 4)
                s_symptom, s_risk = self._split_delta(delta, patient)
                s_rule    = round(min(_LEGACY_CAP_RULE, max(-_LEGACY_CAP_RULE, result.score)), 4)
                raw        = s_ai + s_symptom + s_risk + s_rule
                score      = round(min(max(raw, 0.0), 1.0), 4)
                confidence = self._confidence_label(score)
                nb_bd      = {}

            # ── Tier adjustment from history ──────────────────────────────────
            tier = result.tier
            if delta and delta.tier_delta != 0:
                if result.allow_downgrade or delta.tier_delta < 0:
                    tier = max(1, min(3, result.tier + delta.tier_delta))

            # Safety: Tier-1 labels — posterior > 0.15 → CONFIRMED (NB mode)
            # or legacy: Tier-1 INCIDENTAL elevated to POSSIBLE
            if tier == 1:
                if nb_posteriors and score > 0.15:
                    confidence = "CONFIRMED"
                elif confidence == "INCIDENTAL":
                    logger.warning(
                        "Safety override: Tier-1 '%s' score=%.3f elevated to POSSIBLE.",
                        result.label_id, score,
                    )
                    confidence = "POSSIBLE"
                elif not nb_posteriors and result.pai >= 0.70 and confidence == "POSSIBLE":
                    logger.warning(
                        "Safety override: Tier-1 '%s' Pai=%.3f elevated POSSIBLE→PROBABLE.",
                        result.label_id, result.pai,
                    )
                    confidence = "PROBABLE"

            # ── XAI evidence ──────────────────────────────────────────────────
            supporting    = delta.supporting    if delta else []
            contradicting = delta.contradicting if delta else []
            evidence_log  = delta.evidence      if delta else []
            is_derived    = any(result.label_id in e for e in derived_log)

            results_out.append(FusionResult(
                rank=0,
                label_id=result.label_id,
                label_name=result.label_name,
                category=result.category,
                description=getattr(result, 'description', ''),
                hierarchy=result.hierarchy,
                pai=result.pai,
                s_ai=s_ai,
                s_symptom=s_symptom,
                s_risk=s_risk,
                s_rule=s_rule,
                score_final=score,
                confidence_label=confidence,
                tier=tier,
                tier_label=TIER_LABELS.get(tier, "unknown"),
                default_action=result.default_action,
                allow_downgrade=result.allow_downgrade,
                snomed_ct=result.snomed_ct,
                icd10=result.icd10,
                aha_guideline=result.aha_guideline,
                clinical_notes=result.clinical_notes,
                is_suppressed=result.is_suppressed,
                is_derived=is_derived,
                supporting=supporting,
                contradicting=contradicting,
                evidence_log=evidence_log,
                nb_breakdown=nb_bd,
            ))

        # ── Sort and rank ─────────────────────────────────────────────────────
        results_out.sort(key=lambda r: (
            r.is_suppressed,
            r.tier if not r.is_suppressed else 99,
            -r.score_final,
        ))
        rank = 1
        for r in results_out:
            if not r.is_suppressed:
                r.rank = rank
                rank += 1

        active   = [r for r in results_out if not r.is_suppressed]
        critical = [r for r in active
                    if r.tier == 1 and r.confidence_label in ("CONFIRMED", "PROBABLE")]
        top      = active[0] if active else None

        logger.info(
            "[Fusion] Complete — %d results (%d active, %d suppressed, %d critical)",
            len(results_out), len(active),
            len(results_out) - len(active), len(critical),
        )

        mode = "NaiveBayes" if nb_posteriors else "legacy-weighted-sum"
        return FusionOutput(
            patient_id=patient_id,
            results=results_out,
            active_results=active,
            top_diagnosis=top,
            critical_alerts=critical,
            derived_log=derived_log,
            metadata={
                "fusion_mode":      mode,
                "total_candidates": len(results_out),
                "active_count":     len(active),
                "critical_count":   len(critical),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split_delta(self, delta, patient: dict) -> tuple[float, float]:
        """Legacy: split HistoryDelta.score_delta into (s_symptom, s_risk)."""
        if delta is None or delta.score_delta == 0:
            return 0.0, 0.0
        symptoms     = patient.get("symptoms",     {})
        risk_factors = patient.get("risk_factors", {})
        sym_hits  = sum(1 for t in delta.supporting if t.replace(" ", "_") in symptoms)
        risk_hits = sum(1 for t in delta.supporting if t.replace(" ", "_") in risk_factors)
        total     = sym_hits + risk_hits
        sym_ratio = (sym_hits / total) if total > 0 else 0.70
        raw_sym   = delta.score_delta * sym_ratio
        raw_risk  = delta.score_delta * (1 - sym_ratio)
        s_symptom = round(max(-_LEGACY_CAP_SYMPTOM, min(_LEGACY_CAP_SYMPTOM, raw_sym)),  4)
        s_risk    = round(max(-_LEGACY_CAP_RISK,    min(_LEGACY_CAP_RISK,    raw_risk)), 4)
        return s_symptom, s_risk

    @staticmethod
    def _confidence_label(score: float) -> str:
        for threshold, label in CONFIDENCE_BANDS:
            if score >= threshold:
                return label
        return "INCIDENTAL"


# ── Patient evidence builder ──────────────────────────────────────────────────

def _build_patient_evidence(patient: dict) -> dict:
    """Build {feature: bool} dict from structured patient dict."""
    symptoms     = patient.get("symptoms",     {})
    risk_factors = patient.get("risk_factors", {})
    vitals       = patient.get("vitals",       {})
    sbp = float(vitals.get("sbp",  120))
    hr  = float(vitals.get("hr",   75))
    spo2= float(vitals.get("spo2", 98))
    sym_present  = list(symptoms.values())
    return {
        "chest_pain":   bool(symptoms.get("chest_pain",   False)),
        "diaphoresis":  bool(symptoms.get("diaphoresis",  False)),
        "dyspnea":      bool(symptoms.get("dyspnea",      False)),
        "syncope":      bool(symptoms.get("syncope",      False)),
        "palpitations": bool(symptoms.get("palpitations", False)),
        "dizziness":    bool(symptoms.get("dizziness",    False)),
        "cad":          bool(risk_factors.get("cad",       False)),
        "dm":           bool(risk_factors.get("dm",        False)),
        "smoking":      bool(risk_factors.get("smoking",   False)),
        "htn":          bool(risk_factors.get("htn",       False)),
        "prior_mi":     bool(risk_factors.get("prior_mi",  False)),
        "young_age":    float(vitals.get("age", patient.get("age", 99))) < 40
                        or bool(risk_factors.get("young_age", False)),
        "asymptomatic": not any(bool(v) for v in sym_present),
        "sbp_lt_90":    sbp < 90,
        "sbp_gt_140":   sbp > 140,
        "hr_lt_40":     hr < 40,
        "hr_lt_60":     hr < 60,
        "hr_gt_100":    hr > 100,
        "hr_gt_150":    hr > 150,
        "hr_gt_160":    hr > 160,
        "hr_gt_200":    hr > 200,
        "spo2_lt_94":   spo2 < 94,
        "no_chest_pain": not bool(symptoms.get("chest_pain", False)),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    from dataclasses import dataclass as dc
    from history_module.history_encoder import HistoryDelta

    @dc
    class StubResult:
        label_id: str; label_name: str; category: str; description: str
        hierarchy: list; pai: float; score: float; tier: int
        default_action: str; allow_downgrade: bool
        snomed_ct: str = ""; icd10: str = ""
        aha_guideline: str = ""; clinical_notes: str = ""
        is_suppressed: bool = False

    stubs = [
        StubResult("STEMI", "ST-Elevation MI", "Ischemia", "Acute MI",
                   ["ROOT", "Ischemia", "STEMI"], 0.74, 0.0, 1,
                   "Immediate PCI", False, "57054005", "I21",
                   "2013_AHA_STEMI", "Time-critical"),
        StubResult("AF", "Atrial Fibrillation", "Rhythm", "Irregular rhythm",
                   ["ROOT", "Rhythm", "AF"], 0.42, 0.0, 2,
                   "Rate control", True, "49436004", "I48"),
        StubResult("LVH", "Left Ventricular Hypertrophy", "Structural", "LVH",
                   ["ROOT", "LVH"], 0.38, 0.0, 3,
                   "Outpatient eval", True, "164873001", "I51.7"),
    ]

    deltas = {
        "STEMI": HistoryDelta("STEMI", +0.0, 0, [], ["chest pain", "cad"], []),
        "AF":    HistoryDelta("AF",    +0.0, 0, [], ["htn"], []),
        "LVH":   HistoryDelta("LVH",   +0.0, 0, [], ["htn", "dyspnea"], []),
    }

    patient = {
        "symptoms":     {"chest_pain": True, "dyspnea": True},
        "risk_factors": {"htn": True, "cad": True},
        "vitals":       {"sbp": 88, "hr": 102},
    }

    cpt_df = pd.read_csv("history_module/history_rules.csv", comment="#")
    output = DecisionFusion().fuse(stubs, deltas, [], patient, cpt_table=cpt_df)

    print("\n=== Ranked DDx (Naive Bayes mode) ===\n")
    for r in output.active_results:
        print(f"  Rank {r.rank}  {r.label_id:8s}  {r.confidence_label:12s}"
              f"  posterior={r.score_final:.4f}")
        if r.nb_breakdown:
            bd = r.nb_breakdown
            print(f"    prior={bd.get('prior',0):.4f}  "
                  f"log_ll={bd.get('log_likelihood',0):.4f}  "
                  f"tier={bd.get('tier','?')}")

    print(f"\n  Top: {output.top_diagnosis.label_id}")
    print(f"  Fusion mode: {output.metadata['fusion_mode']}")
    print(f"  Critical alerts: {[r.label_id for r in output.critical_alerts]}")
