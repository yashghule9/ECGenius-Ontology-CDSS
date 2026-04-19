"""
ECGenius — inference/decision_fusion.py
========================================
Final scoring engine.

Scoring formula (from your PPT research):
  Score(D) = (0.5 × Pai(D)) + S_symptom + S_risk + S_rule

  Component         Source                          Cap
  ─────────────────────────────────────────────────────
  AI model          0.5 × Pai(D) from model.pt      0.50
  Symptoms          history_encoder HistoryDelta     0.30
  Risk factors      history_encoder HistoryDelta     0.10
  Clinical rules    rule_executor score delta        0.10
  ─────────────────────────────────────────────────────
  Total max                                          1.00

Single source of truth for symptom/risk weights = history_rules.csv
No EvidenceScorer, no symptom_weights.csv needed.

Confidence thresholds:
  >= 0.80  →  CONFIRMED
  0.60-0.79 →  PROBABLE
  0.30-0.59 →  POSSIBLE
  < 0.30   →  INCIDENTAL
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Score caps ────────────────────────────────────────────────────────────────
W_AI        = 0.50
CAP_SYMPTOM = 0.30
CAP_RISK    = 0.10
CAP_RULE    = 0.10

CONFIDENCE_BANDS = [
    (0.80, "CONFIRMED"),
    (0.60, "PROBABLE"),
    (0.30, "POSSIBLE"),
    (0.00, "INCIDENTAL"),
]

TIER_LABELS = {1: "life-threatening", 2: "urgent", 3: "routine"}


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

    # Score breakdown (kept separate for XAI transparency)
    pai:              float   # raw AI probability
    s_ai:             float   # 0.5 × pai
    s_symptom:        float   # from history_encoder (symptom triggers)
    s_risk:           float   # from history_encoder (risk factor triggers)
    s_rule:           float   # from rule_executor delta
    score_final:      float   # clamped final Score(D)
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

    # XAI evidence (from history_encoder)
    supporting:       list[str] = field(default_factory=list)
    contradicting:    list[str] = field(default_factory=list)
    evidence_log:     list[str] = field(default_factory=list)

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


# ── Main class ────────────────────────────────────────────────────────────────

class DecisionFusion:
    """
    Combines AI output + ontology + history deltas into ranked DDx.

    history_rules.csv is the ONLY source of symptom/risk weights.
    history_encoder.py computes HistoryDelta per label.
    This class just splits that delta into s_symptom and s_risk.
    No separate weight file needed.
    """

    def fuse(
        self,
        ontology_results: list,             # list[OntologyResult] from rule_executor
        history_deltas:   dict,             # {label_id: HistoryDelta} from history_encoder
        derived_log:      list[str],        # from rule_executor
        patient:          dict = None,      # raw patient dict (for context only)
        patient_id:       str  = "unknown",
    ) -> FusionOutput:
        """
        Parameters
        ----------
        ontology_results : list[OntologyResult]
            After ontology_mapper + rule_executor. result.score holds
            any rule delta already applied by rule_executor.
        history_deltas : dict[str, HistoryDelta]
            From history_encoder.encode_all(). Contains score_delta,
            tier_delta, supporting, contradicting, evidence lists.
        derived_log : list[str]
            XAI log from rule_executor.
        patient : dict
            Raw patient dict — used for context only, not for scoring.
        patient_id : str
            For traceability in output JSON.

        Returns
        -------
        FusionOutput
        """
        patient    = patient or {}
        results_out = []

        for result in ontology_results:
            delta = history_deltas.get(result.label_id)

            # ── Component 1: AI model ──────────────────────────────────
            s_ai = round(W_AI * result.pai, 4)

            # ── Component 2 & 3: Symptoms + Risk from history_encoder ──
            # Split the single score_delta into symptom vs risk portions
            # based on what triggers fired (supporting list tells us which)
            s_symptom, s_risk = self._split_delta(delta, patient)

            # ── Component 4: Rule engine delta ─────────────────────────
            # rule_executor wrote its delta into result.score
            s_rule = round(
                min(CAP_RULE, max(-CAP_RULE, result.score)),
                4,
            )

            # ── Final score ────────────────────────────────────────────
            raw   = s_ai + s_symptom + s_risk + s_rule
            score = round(min(max(raw, 0.0), 1.0), 4)

            confidence = self._confidence_label(score)

            # ── Tier adjustment from history ───────────────────────────
            tier = result.tier
            if delta and delta.tier_delta != 0:
                if result.allow_downgrade or delta.tier_delta < 0:
                    tier = max(1, min(3, result.tier + delta.tier_delta))

            # Safety: Tier-1 labels can never be INCIDENTAL
            if tier == 1 and confidence == "INCIDENTAL":
                logger.warning(
                    "Safety override: Tier-1 '%s' score=%.3f elevated to POSSIBLE.",
                    result.label_id, score,
                )
                confidence = "POSSIBLE"

            # Safety: Tier-1 labels with high model confidence cannot be POSSIBLE.
            # When history is absent or sparse, the AI contribution alone (capped at 0.5)
            # can prevent a high-confidence Tier-1 prediction from reaching PROBABLE.
            # This override prevents a 95%-confident STEMI from being labelled "POSSIBLE".
            if tier == 1 and result.pai >= 0.70 and confidence == "POSSIBLE":
                logger.warning(
                    "Safety override: Tier-1 '%s' Pai=%.3f elevated from POSSIBLE to PROBABLE "
                    "(insufficient history data suppressed total score).",
                    result.label_id, result.pai,
                )
                confidence = "PROBABLE"

            # ── XAI evidence ───────────────────────────────────────────
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
            ))

        # ── Sort: non-suppressed first, Tier-1 first, then score desc ─
        results_out.sort(key=lambda r: (
            r.is_suppressed,
            r.tier if not r.is_suppressed else 99,
            -r.score_final,
        ))

        # ── Assign ranks ───────────────────────────────────────────────
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

        return FusionOutput(
            patient_id=patient_id,
            results=results_out,
            active_results=active,
            top_diagnosis=top,
            critical_alerts=critical,
            derived_log=derived_log,
            metadata={
                "w_ai":             W_AI,
                "cap_symptom":      CAP_SYMPTOM,
                "cap_risk":         CAP_RISK,
                "cap_rule":         CAP_RULE,
                "total_candidates": len(results_out),
                "active_count":     len(active),
                "critical_count":   len(critical),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _split_delta(self, delta, patient: dict) -> tuple[float, float]:
        """
        Split HistoryDelta.score_delta into (s_symptom, s_risk).

        Strategy: count triggers that came from symptoms vs risk_factors
        and weight proportionally. Falls back to 70/30 split if unknown.
        """
        if delta is None or delta.score_delta == 0:
            return 0.0, 0.0

        symptoms     = patient.get("symptoms",     {})
        risk_factors = patient.get("risk_factors", {})

        sym_hits  = sum(
            1 for t in delta.supporting
            if t.replace(" ", "_") in symptoms
        )
        risk_hits = sum(
            1 for t in delta.supporting
            if t.replace(" ", "_") in risk_factors
        )
        total = sym_hits + risk_hits
        sym_ratio = (sym_hits / total) if total > 0 else 0.70

        raw_sym  = delta.score_delta * sym_ratio
        raw_risk = delta.score_delta * (1 - sym_ratio)

        s_symptom = round(max(-CAP_SYMPTOM, min(CAP_SYMPTOM, raw_sym)),  4)
        s_risk    = round(max(-CAP_RISK,    min(CAP_RISK,    raw_risk)), 4)

        return s_symptom, s_risk

    @staticmethod
    def _confidence_label(score: float) -> str:
        for threshold, label in CONFIDENCE_BANDS:
            if score >= threshold:
                return label
        return "INCIDENTAL"


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
                   ["ROOT","Ischemia","STEMI"], 0.74, 0.40, 1,
                   "Immediate PCI", False, "57054005", "I21",
                   "2013_AHA_STEMI", "Time-critical"),
        StubResult("AF", "Atrial Fibrillation", "Rhythm", "Irregular rhythm",
                   ["ROOT","Rhythm","AF"], 0.42, 0.0, 2,
                   "Rate control", True, "49436004", "I48"),
        StubResult("LVH", "Left Ventricular Hypertrophy", "Structural", "LVH",
                   ["ROOT","LVH"], 0.38, 0.0, 3,
                   "Outpatient eval", True, "164873001", "I51.7"),
    ]

    deltas = {
        "STEMI": HistoryDelta("STEMI", +0.40, 0,
                              ["[H1] chest pain → +0.30", "[H7] cad → +0.10"],
                              ["chest pain", "cad"], []),
        "AF":    HistoryDelta("AF",    +0.09, 0,
                              ["[H16] htn → +0.09"],
                              ["htn"], []),
        "LVH":   HistoryDelta("LVH",   +0.25, 0,
                              ["[H19] htn → +0.18", "[H20] dyspnea → +0.07"],
                              ["htn", "dyspnea"], []),
    }

    patient = {
        "symptoms":     {"chest_pain": True, "dyspnea": True},
        "risk_factors": {"htn": True, "cad": True},
        "vitals":       {"sbp": 88, "hr": 102},
    }

    output = DecisionFusion().fuse(stubs, deltas, ["R3: STEMI boosted"], patient)

    print("\n=== Ranked DDx ===\n")
    for r in output.active_results:
        print(f"  Rank {r.rank}  {r.label_id:8s}  {r.confidence_label:12s}"
              f"  score={r.score_final:.3f}"
              f"  (ai={r.s_ai:.3f} sym={r.s_symptom:.3f}"
              f" risk={r.s_risk:.3f} rule={r.s_rule:.3f})")

    print(f"\n  Top: {output.top_diagnosis.label_id}")
    print(f"  Critical alerts: {[r.label_id for r in output.critical_alerts]}")