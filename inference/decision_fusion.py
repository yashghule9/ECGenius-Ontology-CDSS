"""
ECGenius — inference/decision_fusion.py
========================================
Final scoring engine. Takes fully enriched + rule-processed
OntologyResult objects and produces the ranked differential
diagnosis list with confidence labels and clinical actions.

Scoring formula (from your research):
  Score(D) = (0.5 × Pai(D)) + S_symptom + S_risk + S_rule

  Component         Max contribution
  ─────────────────────────────────
  AI model          0.50  (0.5 × Pai(D) ∈ [0,1])
  Symptoms          0.30
  Risk factors      0.10
  Clinical rules    0.10  (already applied by rule_executor)
  ─────────────────────────────────
  Total max         1.00

Confidence thresholds:
  ≥ 0.80  →  CONFIRMED
  0.60–0.79  →  PROBABLE
  0.30–0.59  →  POSSIBLE
  < 0.30  →  INCIDENTAL

Output: FusionResult — the final DDx object handed to
explainability.py and the clinician UI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score weight constants — change here, nowhere else
# ---------------------------------------------------------------------------

W_AI       = 0.50   # weight on Pai(D)
MAX_SYMPTOM = 0.30
MAX_RISK    = 0.10
# Rule/history deltas already baked into result.score by rule_executor

CONFIDENCE_BANDS = [
    (0.80, "CONFIRMED"),
    (0.60, "PROBABLE"),
    (0.30, "POSSIBLE"),
    (0.00, "INCIDENTAL"),
]

TIER_LABELS = {1: "life-threatening", 2: "urgent", 3: "routine"}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """
    Final ranked diagnosis — everything the UI and explainability
    layer needs in one object.
    """
    rank: int                        # 1 = most likely
    label_id: str
    label_name: str
    category: str
    hierarchy: list[str]

    # Score breakdown (for XAI)
    pai: float                       # raw AI probability
    s_ai: float                      # 0.5 × pai
    s_symptom: float                 # symptom contribution
    s_risk: float                    # risk factor contribution
    s_rule: float                    # rule/history delta already applied
    score_final: float               # clamped sum
    confidence_label: str            # CONFIRMED / PROBABLE / POSSIBLE / INCIDENTAL

    # Clinical action
    tier: int
    tier_label: str
    default_action: str
    allow_downgrade: bool

    # Terminology
    snomed_ct: str
    icd10: str
    aha_guideline: str
    clinical_notes: str

    # Flags
    is_suppressed: bool
    derived: bool = False            # True if injected by derived rule

    # XAI narrative (populated by explainability.py)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "category": self.category,
            "hierarchy": self.hierarchy,
            "score_breakdown": {
                "pai":       round(self.pai, 4),
                "s_ai":      round(self.s_ai, 4),
                "s_symptom": round(self.s_symptom, 4),
                "s_risk":    round(self.s_risk, 4),
                "s_rule":    round(self.s_rule, 4),
                "total":     round(self.score_final, 4),
            },
            "confidence_label": self.confidence_label,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "default_action": self.default_action,
            "allow_downgrade": self.allow_downgrade,
            "snomed_ct": self.snomed_ct,
            "icd10": self.icd10,
            "aha_guideline": self.aha_guideline,
            "clinical_notes": self.clinical_notes,
            "is_suppressed": self.is_suppressed,
            "derived": self.derived,
            "explanation": self.explanation,
        }


@dataclass
class FusionOutput:
    """Top-level output handed to the UI / explainability layer."""
    patient_id: str
    results: list[FusionResult]          # all labels, sorted
    active_results: list[FusionResult]   # non-suppressed only
    top_diagnosis: Optional[FusionResult]
    critical_alerts: list[FusionResult]  # Tier-1 CONFIRMED/PROBABLE labels
    derived_log: list[str]               # from rule_executor
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "top_diagnosis": self.top_diagnosis.to_dict() if self.top_diagnosis else None,
            "critical_alerts": [r.to_dict() for r in self.critical_alerts],
            "active_results": [r.to_dict() for r in self.active_results],
            "all_results": [r.to_dict() for r in self.results],
            "derived_log": self.derived_log,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Symptom / risk score helper
# ---------------------------------------------------------------------------

class EvidenceScorer:
    """
    Converts patient history into per-label score contributions.

    Uses symptom_weights.csv (from history_module/) if available,
    otherwise falls back to uniform weights.

    symptom_weights.csv schema:
        label_id, symptom_id, weight
        STEMI, chest_pain, 0.20
        STEMI, dyspnea,    0.08
        AF,    palpitations, 0.15
        ...
    """

    def __init__(self, weights_path: Optional[Path] = None):
        self._symptom_weights: dict[str, dict[str, float]] = {}
        self._risk_weights: dict[str, dict[str, float]] = {}

        if weights_path and Path(weights_path).exists():
            self._load_weights(Path(weights_path))
        else:
            logger.info(
                "EvidenceScorer: no symptom_weights.csv found — "
                "using uniform fallback weights."
            )

    def _load_weights(self, path: Path) -> None:
        import csv
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label   = row["label_id"].strip()
                symptom = row["symptom_id"].strip()
                weight  = float(row.get("weight", 0.05))
                bucket  = row.get("bucket", "symptom").strip().lower()

                if bucket == "risk":
                    self._risk_weights.setdefault(label, {})[symptom] = weight
                else:
                    self._symptom_weights.setdefault(label, {})[symptom] = weight

        logger.info(
            "EvidenceScorer: loaded weights for %d labels",
            len(self._symptom_weights) + len(self._risk_weights),
        )

    def symptom_score(self, label_id: str, patient: dict) -> float:
        """
        Returns S_symptom ∈ [0, MAX_SYMPTOM].
        Sums weights of present symptoms for this label.
        Falls back to 0.05 per present symptom if no weights loaded.
        """
        symptoms = patient.get("symptoms", {})
        weights  = self._symptom_weights.get(label_id, {})
        total    = 0.0

        if weights:
            for sym_id, w in weights.items():
                if symptoms.get(sym_id):
                    total += w
        else:
            # Uniform fallback: 0.05 per present symptom, capped
            present_count = sum(1 for v in symptoms.values() if v)
            total = min(present_count * 0.05, MAX_SYMPTOM)

        return min(total, MAX_SYMPTOM)

    def risk_score(self, label_id: str, patient: dict) -> float:
        """
        Returns S_risk ∈ [0, MAX_RISK].
        """
        risk_factors = patient.get("risk_factors", {})
        weights      = self._risk_weights.get(label_id, {})
        total        = 0.0

        if weights:
            for rf_id, w in weights.items():
                if risk_factors.get(rf_id):
                    total += w
        else:
            present_count = sum(1 for v in risk_factors.values() if v)
            total = min(present_count * 0.02, MAX_RISK)

        return min(total, MAX_RISK)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DecisionFusion:
    """
    Combines AI model output + ontology + history + rule deltas
    into a final ranked differential diagnosis.

    Parameters
    ----------
    weights_path : Path | None
        Path to history_module/symptom_weights.csv
    min_score_to_include : float
        FusionResults below this score are still returned but
        labelled INCIDENTAL. Set to 0.0 to include all.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        min_score_to_include: float = 0.0,
    ):
        self.scorer = EvidenceScorer(weights_path)
        self.min_score = min_score_to_include

    def fuse(
        self,
        ontology_results: list,           # list[OntologyResult] from rule_executor
        patient: dict,                    # from history_encoder
        derived_log: list[str],           # from rule_executor
        patient_id: str = "unknown",
    ) -> FusionOutput:
        """
        Main entry point.

        Parameters
        ----------
        ontology_results : list[OntologyResult]
            After ontology_mapper + rule_executor have run.
            result.score may already have rule deltas baked in.

        patient : dict
            {"symptoms": {...}, "risk_factors": {...}, "vitals": {...}}

        derived_log : list[str]
            Human-readable log from rule_executor for XAI.

        patient_id : str
            For traceability in the output JSON.

        Returns
        -------
        FusionOutput
        """
        fusion_results = []

        for result in ontology_results:
            # ── Score components ──────────────────────────────────────
            s_ai      = W_AI * result.pai
            s_symptom = self.scorer.symptom_score(result.label_id, patient)
            s_risk    = self.scorer.risk_score(result.label_id, patient)

            # rule_executor already wrote deltas into result.score
            # result.score starts at 0 and gets rule/history deltas
            s_rule = result.score   # whatever rule_executor accumulated

            raw_total = s_ai + s_symptom + s_risk + s_rule
            score_final = round(min(max(raw_total, 0.0), 1.0), 4)

            confidence = self._confidence_label(score_final)

            fr = FusionResult(
                rank=0,   # filled in after sort
                label_id=result.label_id,
                label_name=result.label_name,
                category=result.category,
                hierarchy=result.hierarchy,
                pai=result.pai,
                s_ai=round(s_ai, 4),
                s_symptom=round(s_symptom, 4),
                s_risk=round(s_risk, 4),
                s_rule=round(s_rule, 4),
                score_final=score_final,
                confidence_label=confidence,
                tier=result.tier,
                tier_label=TIER_LABELS.get(result.tier, "unknown"),
                default_action=result.default_action,
                allow_downgrade=result.allow_downgrade,
                snomed_ct=result.snomed_ct,
                icd10=result.icd10,
                aha_guideline=result.aha_guideline,
                clinical_notes=result.clinical_notes,
                is_suppressed=result.is_suppressed,
            )
            fusion_results.append(fr)

        # ── Sort: non-suppressed first, then by score desc ────────────
        fusion_results.sort(key=lambda r: (r.is_suppressed, -r.score_final))

        # ── Assign ranks (non-suppressed only) ────────────────────────
        rank = 1
        for fr in fusion_results:
            if not fr.is_suppressed:
                fr.rank = rank
                rank += 1

        # ── Derived subsets ───────────────────────────────────────────
        active   = [r for r in fusion_results if not r.is_suppressed]
        critical = [
            r for r in active
            if r.tier == 1 and r.confidence_label in ("CONFIRMED", "PROBABLE")
        ]
        top = active[0] if active else None

        # ── Safety check: if ANY Tier-1 label is in results,
        #    ensure it is never buried below INCIDENTAL regardless
        #    of score (belt-and-suspenders on top of rule_executor) ───
        for r in active:
            if r.tier == 1 and r.confidence_label == "INCIDENTAL":
                logger.warning(
                    "Safety override: Tier-1 label '%s' had INCIDENTAL confidence "
                    "(score=%.3f). Elevating to POSSIBLE for clinician review.",
                    r.label_id, r.score_final,
                )
                r.confidence_label = "POSSIBLE"

        output = FusionOutput(
            patient_id=patient_id,
            results=fusion_results,
            active_results=active,
            top_diagnosis=top,
            critical_alerts=critical,
            derived_log=derived_log,
            metadata={
                "weight_ai":        W_AI,
                "max_symptom":      MAX_SYMPTOM,
                "max_risk":         MAX_RISK,
                "total_candidates": len(fusion_results),
                "active_count":     len(active),
                "critical_count":   len(critical),
            },
        )

        logger.info(
            "DecisionFusion complete — patient=%s  top=%s (%s, %.3f)  "
            "critical_alerts=%d",
            patient_id,
            top.label_id if top else "none",
            top.confidence_label if top else "—",
            top.score_final if top else 0.0,
            len(critical),
        )

        return output

    @staticmethod
    def _confidence_label(score: float) -> str:
        for threshold, label in CONFIDENCE_BANDS:
            if score >= threshold:
                return label
        return "INCIDENTAL"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    from dataclasses import dataclass as dc

    @dc
    class StubResult:
        label_id: str; label_name: str; category: str
        hierarchy: list; pai: float; score: float
        tier: int; default_action: str; allow_downgrade: bool
        snomed_ct: str = ""; icd10: str = ""; aha_guideline: str = ""
        clinical_notes: str = ""; is_suppressed: bool = False

    stubs = [
        StubResult("STEMI","ST-Elevation MI","Ischemia",
                   ["ROOT","Ischemia","STEMI"],0.55,0.30,1,"Immediate PCI",False),
        StubResult("AF","Atrial Fibrillation","Rhythm",
                   ["ROOT","Rhythm","AF"],0.71,0.0,2,"Rate control",True),
        StubResult("VF","Ventricular Fibrillation","Rhythm",
                   ["ROOT","Rhythm","VF"],0.78,0.0,1,"Start ACLS",False),
        StubResult("LVH","Left Ventricular Hypertrophy","Structural",
                   ["ROOT","Structural","LVH"],0.42,0.0,3,"Outpatient eval",True),
    ]

    patient = {
        "symptoms":     {"chest_pain": True, "palpitations": True},
        "risk_factors": {"htn": True, "dm": False, "cad": True},
        "vitals":       {"sbp": 88, "hr": 140, "spo2": 94},
    }

    fusion = DecisionFusion()
    output = fusion.fuse(stubs, patient, derived_log=["R3: STEMI boosted +0.30"], patient_id="PT001")

    print("\n=== Fusion output ===\n")
    for r in output.active_results:
        print(f"  Rank {r.rank}  {r.label_id:10s}  {r.confidence_label:12s}  "
              f"score={r.score_final:.3f}  "
              f"(ai={r.s_ai:.2f} sym={r.s_symptom:.2f} risk={r.s_risk:.2f} rule={r.s_rule:.2f})")

    print(f"\n  Critical alerts: {[r.label_id for r in output.critical_alerts]}")
    print(f"  Top diagnosis:   {output.top_diagnosis.label_id if output.top_diagnosis else 'none'}")