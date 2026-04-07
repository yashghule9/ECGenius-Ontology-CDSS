"""
ECGenius — history_module/history_encoder.py
=============================================
Converts raw patient history (symptoms, risk factors, vitals)
into a validated, structured dict ready for:
  1. rule_executor.py  — RuleContext.symptom_present()
  2. decision_fusion.py — EvidenceScorer

Also applies history_rules.csv to compute per-label score
deltas BEFORE decision_fusion runs, so the fusion engine
sees pre-adjusted scores from rule_executor.

Responsibilities:
  - Load and validate history_schema.json
      (defines what to collect per diagnosis)
  - Load history_rules.csv
      (defines how history maps to score/tier changes)
  - Validate incoming patient_history.json against schema
  - Encode vitals into threshold flags (sbp_lt_90 etc.)
  - Expose encode() → structured patient dict
  - Expose apply_history_rules() → updates OntologyResult scores
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HistoryRule:
    """One row from history_rules.csv."""
    rule_id: str
    label_id: str
    trigger: str       # symptom_id, risk_factor_id, or vital threshold key
    action: str        # add_score | reduce_score | reduce_tier | add_tier
    delta: float


@dataclass
class VitalThreshold:
    """
    Encodes a vital sign reading into named boolean flags.
    e.g. sbp=88  →  sbp_lt_90=True, sbp_lt_100=True
    """
    vital_id: str
    value: float
    flags: dict[str, bool]   # {"sbp_lt_90": True, "sbp_gt_140": False, ...}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class HistoryEncoder:
    """
    Validates and structures patient history for downstream use.

    Parameters
    ----------
    history_module_dir : str | Path
        Directory containing history_schema.json, history_rules.csv,
        symptoms.csv, risk_factors.csv, vitals.csv
    strict : bool
        If True, raise on unknown/missing required fields.
        If False (default), warn and fill with None.
    """

    # Vital sign thresholds automatically generated as boolean flags.
    # Extend this dict as your cardiologists define new thresholds.
    VITAL_THRESHOLDS: dict[str, list[tuple[str, str, float]]] = {
        "sbp": [
            ("sbp_lt_70",  "lt", 70),
            ("sbp_lt_90",  "lt", 90),
            ("sbp_lt_100", "lt", 100),
            ("sbp_gt_140", "gt", 140),
            ("sbp_gt_160", "gt", 160),
        ],
        "hr": [
            ("hr_lt_40",   "lt", 40),
            ("hr_lt_60",   "lt", 60),
            ("hr_gt_100",  "gt", 100),
            ("hr_gt_150",  "gt", 150),
        ],
        "spo2": [
            ("spo2_lt_88", "lt", 88),
            ("spo2_lt_92", "lt", 92),
            ("spo2_lt_95", "lt", 95),
        ],
        "rr": [
            ("rr_gt_20",   "gt", 20),
            ("rr_gt_30",   "gt", 30),
        ],
        "temp": [
            ("temp_gt_38", "gt", 38.0),
            ("temp_lt_36", "lt", 36.0),
        ],
    }

    def __init__(
        self,
        history_module_dir: str | Path = "history_module/",
        strict: bool = False,
    ):
        self.dir = Path(history_module_dir)
        self.strict = strict

        self._schema: dict[str, dict]   = {}   # label_id → {symptoms, risk_factors, vitals}
        self._h_rules: list[HistoryRule] = []
        self._known_symptoms: set[str]   = set()
        self._known_risks: set[str]      = set()
        self._known_vitals: set[str]     = set()

        self._load_all()
        logger.info(
            "HistoryEncoder ready — schema for %d labels, %d history rules",
            len(self._schema), len(self._h_rules),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, raw_history: dict) -> dict:
        """
        Validate and structure a raw patient history dict.

        Parameters
        ----------
        raw_history : dict
            As loaded from patient_history.json:
            {
              "symptoms":     {"chest_pain": true, ...},
              "risk_factors": {"htn": true, ...},
              "vitals":       {"sbp": 88, "hr": 140, ...}
            }

        Returns
        -------
        dict
            Validated patient dict with:
            - "symptoms"     : {symptom_id: bool}
            - "risk_factors" : {rf_id: bool}
            - "vitals"       : {vital_id: float}
            - "vital_flags"  : {threshold_key: bool}  ← auto-generated
            - "raw"          : original input (for audit trail)
        """
        symptoms     = self._validate_symptoms(raw_history.get("symptoms", {}))
        risk_factors = self._validate_risks(raw_history.get("risk_factors", {}))
        vitals, flags = self._validate_vitals(raw_history.get("vitals", {}))

        # Merge vital_flags into the top-level so symptom_present()
        # in RuleContext can resolve "sbp_lt_90" directly
        merged_symptoms = {**symptoms, **flags}

        encoded = {
            "symptoms":     merged_symptoms,
            "risk_factors": risk_factors,
            "vitals":       vitals,
            "vital_flags":  flags,
            "raw":          raw_history,
        }

        logger.debug(
            "Encoded history — symptoms_present=%d  risks_present=%d  vitals=%d  flags=%d",
            sum(1 for v in symptoms.values() if v),
            sum(1 for v in risk_factors.values() if v),
            len(vitals),
            sum(1 for v in flags.values() if v),
        )

        return encoded

    def apply_history_rules(self, results: list, patient: dict) -> list:
        """
        Apply history_rules.csv deltas directly to OntologyResult.score
        and OntologyResult.tier before decision_fusion runs.

        This is a lighter version of what rule_executor does for
        ECG-level rules — history rules are specifically about
        patient context modifying diagnosis scores.

        Parameters
        ----------
        results : list[OntologyResult]
            From rule_executor (may already have some rule deltas).
        patient : dict
            From encode() above.

        Returns
        -------
        list[OntologyResult]
            Same list, mutated in-place.
        """
        result_map = {r.label_id: r for r in results}

        for rule in self._h_rules:
            target = result_map.get(rule.label_id)
            if target is None or target.is_suppressed:
                continue

            # Check trigger
            if not self._trigger_fires(rule.trigger, patient):
                continue

            if rule.action == "add_score":
                target.score = min(1.0, target.score + rule.delta)
                logger.debug("[%s] History rule: %s score += %.2f",
                             rule.rule_id, rule.label_id, rule.delta)

            elif rule.action == "reduce_score":
                if not target.allow_downgrade:
                    logger.info("[%s] History rule: reduce_score blocked "
                                "(allow_downgrade=False) for %s",
                                rule.rule_id, rule.label_id)
                    continue
                target.score = max(0.0, target.score - abs(rule.delta))
                logger.debug("[%s] History rule: %s score -= %.2f",
                             rule.rule_id, rule.label_id, abs(rule.delta))

            elif rule.action == "reduce_tier":
                if target.tier == 1:
                    logger.warning("[%s] History rule: refusing to reduce tier "
                                   "for Tier-1 label %s", rule.rule_id, rule.label_id)
                    continue
                if not target.allow_downgrade:
                    continue
                old = target.tier
                target.tier = min(3, target.tier + int(abs(rule.delta)))
                logger.info("[%s] History rule: %s tier %d→%d",
                            rule.rule_id, rule.label_id, old, target.tier)

            elif rule.action == "add_tier":
                old = target.tier
                target.tier = max(1, target.tier - int(abs(rule.delta)))
                logger.info("[%s] History rule: %s tier %d→%d (upgraded)",
                            rule.rule_id, rule.label_id, old, target.tier)

            else:
                logger.warning("[%s] History rule: unknown action '%s' — skipping.",
                               rule.rule_id, rule.action)

        return results

    def questions_for_labels(self, label_ids: list[str]) -> dict:
        """
        Return the minimal set of questions to ask for a given
        list of candidate diagnoses — used by the UI to build
        the dynamic history form.

        Returns
        -------
        dict
            {
              "symptoms":     [symptom_id, ...],
              "risk_factors": [rf_id, ...],
              "vitals":       [vital_id, ...]
            }
        """
        needed_symptoms: set[str] = set()
        needed_risks:    set[str] = set()
        needed_vitals:   set[str] = set()

        for label_id in label_ids:
            schema = self._schema.get(label_id, {})
            needed_symptoms.update(schema.get("symptoms", []))
            needed_risks.update(schema.get("risk_factors", []))
            needed_vitals.update(schema.get("vitals", []))

        return {
            "symptoms":     sorted(needed_symptoms),
            "risk_factors": sorted(needed_risks),
            "vitals":       sorted(needed_vitals),
        }

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_schema()
        self._load_history_rules()
        self._load_known_fields()

    def _load_schema(self) -> None:
        path = self.dir / "history_schema.json"
        if not path.exists():
            logger.warning("history_schema.json not found at %s — no schema validation.", path)
            return
        with open(path, encoding="utf-8") as f:
            self._schema = json.load(f)

    def _load_history_rules(self) -> None:
        path = self.dir / "history_rules.csv"
        if not path.exists():
            logger.warning("history_rules.csv not found at %s.", path)
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._h_rules.append(HistoryRule(
                    rule_id=row.get("rule_id", "?"),
                    label_id=row.get("label_id", "").strip(),
                    trigger=row.get("trigger", "").strip(),
                    action=row.get("action", "").strip(),
                    delta=float(row.get("delta", 0) or 0),
                ))

    def _load_known_fields(self) -> None:
        """Load valid symptom/risk/vital IDs from CSVs for validation."""
        for fname, target in [
            ("symptoms.csv",     self._known_symptoms),
            ("risk_factors.csv", self._known_risks),
            ("vitals.csv",       self._known_vitals),
        ]:
            path = self.dir / fname
            if not path.exists():
                continue
            import csv as _csv
            id_col = fname.replace(".csv", "").rstrip("s") + "_id"
            # symptoms.csv → symptom_id, risk_factors.csv → risk_factor_id
            # vitals.csv → vital_id
            with open(path, newline="", encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    # try common id column names
                    for col in [id_col, "symptom_id", "risk_id", "vital_id", "id"]:
                        if col in row:
                            target.add(row[col].strip())
                            break

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_symptoms(self, raw: dict) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for k, v in raw.items():
            if self._known_symptoms and k not in self._known_symptoms:
                msg = f"Unknown symptom_id '{k}' in patient history."
                if self.strict:
                    raise ValueError(msg)
                logger.warning(msg)
            out[k] = bool(v)
        return out

    def _validate_risks(self, raw: dict) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for k, v in raw.items():
            if self._known_risks and k not in self._known_risks:
                msg = f"Unknown risk_factor_id '{k}' in patient history."
                if self.strict:
                    raise ValueError(msg)
                logger.warning(msg)
            out[k] = bool(v)
        return out

    def _validate_vitals(self, raw: dict) -> tuple[dict[str, float], dict[str, bool]]:
        vitals: dict[str, float] = {}
        flags:  dict[str, bool]  = {}

        for k, v in raw.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                logger.warning("Vital '%s' has non-numeric value '%s' — skipping.", k, v)
                continue

            vitals[k] = val

            # Generate threshold flags
            for flag_key, op, threshold in self.VITAL_THRESHOLDS.get(k, []):
                if op == "lt":
                    flags[flag_key] = val < threshold
                elif op == "gt":
                    flags[flag_key] = val > threshold

        return vitals, flags

    def _trigger_fires(self, trigger: str, patient: dict) -> bool:
        """Check if a history rule trigger is satisfied."""
        symptoms     = patient.get("symptoms", {})
        risk_factors = patient.get("risk_factors", {})
        flags        = patient.get("vital_flags", {})

        # Negated: "no_chest_pain"
        if trigger.startswith("no_"):
            base = trigger[3:]
            val  = symptoms.get(base, risk_factors.get(base, None))
            if val is not None:
                return not bool(val)
            return False

        # Direct symptom / risk
        if trigger in symptoms:
            return bool(symptoms[trigger])
        if trigger in risk_factors:
            return bool(risk_factors[trigger])

        # Vital flag
        if trigger in flags:
            return bool(flags[trigger])

        # Special: "asymptomatic" — no symptoms present at all
        if trigger == "asymptomatic":
            return not any(symptoms.values())

        logger.debug("History trigger '%s' not resolved — treating as False.", trigger)
        return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    encoder = HistoryEncoder(history_module_dir="history_module/")

    raw = {
        "symptoms":     {"chest_pain": True, "palpitations": False, "syncope": False},
        "risk_factors": {"htn": True, "dm": False, "cad": True, "young_age": False},
        "vitals":       {"sbp": 88, "hr": 140, "spo2": 94},
    }

    encoded = encoder.encode(raw)

    print("\n=== Encoded patient history ===\n")
    print("Symptoms:", encoded["symptoms"])
    print("Risks:   ", encoded["risk_factors"])
    print("Vitals:  ", encoded["vitals"])
    print("Flags:   ", {k: v for k, v in encoded["vital_flags"].items() if v})

    print("\n=== Questions needed for STEMI + AF ===\n")
    qs = encoder.questions_for_labels(["STEMI", "AF"])
    print(json.dumps(qs, indent=2))