"""
ECGenius — history_module/history_encoder.py
=============================================
Converts raw patient history (symptoms, risk factors, vitals)
into a validated, structured dict ready for:
  1. rule_executor.py  — RuleContext.symptom_present()
  2. decision_fusion.py — score delta computation

Responsibilities:
  - Load and validate history_schema.json
  - Load history_rules.csv
  - Validate incoming patient history dict
  - Encode vitals into threshold flags (sbp_lt_90 etc.)
  - Expose encode() → structured patient dict
  - Expose encode_all() → {label_id: HistoryDelta}
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HistoryRule:
    rule_id:  str
    label_id: str
    trigger:  str
    # Legacy additive schema
    action:   str   = ""
    delta:    float = 0.0
    # NB CPT schema (p_given_disease / p_given_not_disease)
    p_given_disease:     float = 0.0
    p_given_not_disease: float = 0.0


@dataclass
class HistoryDelta:
    label_id:     str
    score_delta:  float
    tier_delta:   int
    evidence:     list[str]
    supporting:   list[str]
    contradicting: list[str]

    def to_dict(self) -> dict:
        return {
            "label_id":     self.label_id,
            "score_delta":  round(self.score_delta, 4),
            "tier_delta":   self.tier_delta,
            "evidence":     self.evidence,
            "supporting":   self.supporting,
            "contradicting": self.contradicting,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HistoryEncoder:
    """
    Encodes patient history into score deltas for each predicted label.

    Parameters
    ----------
    history_module_dir : str | Path
        Folder containing history_schema.json, history_rules.csv,
        symptom_weights.csv, symptoms.csv, risk_factors.csv, vitals.csv
    strict : bool
        If True raise on unknown fields. If False warn and continue.
    """

    MAX_SCORE_DELTA = 0.40

    VITAL_THRESHOLDS: dict[str, list[tuple[str, str, float]]] = {
        "sbp": [
            ("sbp_lt_70",  "lt", 70),
            ("sbp_lt_90",  "lt", 90),
            ("sbp_lt_100", "lt", 100),
            ("sbp_gt_140", "gt", 140),
            ("sbp_gt_160", "gt", 160),
        ],
        "hr": [
            ("hr_lt_40",  "lt", 40),
            ("hr_lt_60",  "lt", 60),
            ("hr_gt_100", "gt", 100),
            ("hr_gt_150", "gt", 150),
            ("hr_gt_160", "gt", 160),
            ("hr_gt_200", "gt", 200),
        ],
        "spo2": [
            ("spo2_lt_88", "lt", 88),
            ("spo2_lt_92", "lt", 92),
            ("spo2_lt_94", "lt", 94),
            ("spo2_lt_95", "lt", 95),
        ],
        "rr": [
            ("rr_gt_20", "gt", 20),
            ("rr_gt_30", "gt", 30),
        ],
    }

    def __init__(
        self,
        history_module_dir: str | Path = "history_module/",
        strict: bool = False,
    ):
        self.dir    = Path(history_module_dir)
        self.strict = strict

        self._schema:   dict[str, dict]    = {}
        self._h_rules:  list[HistoryRule]  = []
        self._weights:  dict[str, float]   = {}
        self._nb_mode:  bool               = False   # True when CPT schema detected

        self._load_all()
        logger.info(
            "HistoryEncoder ready — schema for %d labels, %d rules, %d weights",
            len(self._schema), len(self._h_rules), len(self._weights),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, raw_history: dict) -> dict:
        """
        Validate and structure a raw patient history dict.
        Adds vital_flags automatically from VITAL_THRESHOLDS.

        Returns enriched patient dict with keys:
          symptoms, risk_factors, vitals, vital_flags, raw
        """
        symptoms     = {k: bool(v) for k, v in raw_history.get("symptoms", {}).items()}
        risk_factors = {k: bool(v) for k, v in raw_history.get("risk_factors", {}).items()}
        vitals, flags = self._process_vitals(raw_history.get("vitals", {}))

        # Merge vital flags into symptoms so trigger evaluation works
        merged_symptoms = {**symptoms, **flags}

        return {
            "symptoms":     merged_symptoms,
            "risk_factors": risk_factors,
            "vitals":       vitals,
            "vital_flags":  flags,
            "raw":          raw_history,
        }

    def compute_delta(self, label_id: str, patient: dict) -> HistoryDelta:
        """
        Compute history score delta for one label against patient dict.
        patient should already be the output of encode() — or the raw
        dict (both work, encode() just adds vital_flags).
        """
        # Auto-encode if raw dict passed without vital_flags
        if "vital_flags" not in patient:
            patient = self.encode(patient)

        applicable = [r for r in self._h_rules if r.label_id == label_id]

        score_delta   = 0.0
        tier_delta    = 0
        evidence      = []
        supporting    = []
        contradicting = []

        for rule in applicable:
            triggered, desc = self._evaluate_trigger(rule.trigger, patient)
            if not triggered:
                continue

            if self._nb_mode:
                # NB schema: score is computed by naive_bayes_fusion.
                # Here we only track which features fired for XAI/explainability.
                p_d  = max(rule.p_given_disease,     1e-6)
                p_nd = max(rule.p_given_not_disease,  1e-6)
                lr   = p_d / p_nd
                if lr >= 1.0:
                    supporting.append(desc)
                    evidence.append(
                        f"[{rule.rule_id}] {desc}: "
                        f"LR={lr:.2f}  (p_d={p_d:.2f} / p_nd={p_nd:.2f})"
                    )
                else:
                    contradicting.append(desc)
                    evidence.append(
                        f"[{rule.rule_id}] {desc}: "
                        f"LR={lr:.2f}  (reduces posterior)"
                    )
            else:
                # Legacy additive schema
                weight = self._weights.get(rule.trigger, 1.0)

                if rule.action == "add_score":
                    adj = rule.delta * weight
                    score_delta += adj
                    supporting.append(desc)
                    evidence.append(f"[{rule.rule_id}] {desc} -> score +{adj:.2f}")

                elif rule.action == "reduce_score":
                    adj = abs(rule.delta) * weight
                    score_delta -= adj
                    contradicting.append(desc)
                    evidence.append(f"[{rule.rule_id}] {desc} -> score -{adj:.2f}")

                elif rule.action == "reduce_tier":
                    steps = int(abs(rule.delta))
                    tier_delta += steps
                    contradicting.append(desc)
                    evidence.append(f"[{rule.rule_id}] {desc} -> tier +{steps}")

                elif rule.action == "add_tier":
                    steps = int(abs(rule.delta))
                    tier_delta -= steps
                    supporting.append(desc)
                    evidence.append(f"[{rule.rule_id}] {desc} -> tier -{steps}")

                else:
                    logger.warning("[%s] Unknown action '%s' — skipping.", rule.rule_id, rule.action)

        # Clamp
        score_delta = max(-self.MAX_SCORE_DELTA, min(self.MAX_SCORE_DELTA, score_delta))

        return HistoryDelta(
            label_id=label_id,
            score_delta=round(score_delta, 4),
            tier_delta=tier_delta,
            evidence=evidence,
            supporting=supporting,
            contradicting=contradicting,
        )

    def encode_all(self, label_ids: list[str], patient: dict) -> dict[str, HistoryDelta]:
        """Compute deltas for multiple labels. Returns {label_id: HistoryDelta}."""
        if "vital_flags" not in patient:
            patient = self.encode(patient)
        return {lid: self.compute_delta(lid, patient) for lid in label_ids}

    def validate_patient(self, patient: dict) -> list[str]:
        """Basic sanity checks on patient dict. Returns list of warnings."""
        warnings = []
        for section in ("symptoms", "risk_factors", "vitals"):
            if section not in patient:
                warnings.append(f"Missing section '{section}' in patient history.")
        vitals = patient.get("vitals", {})
        checks = [("sbp", 0, 300), ("hr", 0, 400), ("spo2", 50, 100)]
        for key, lo, hi in checks:
            if key in vitals:
                val = vitals[key]
                if not isinstance(val, (int, float)) or not (lo <= val <= hi):
                    warnings.append(f"Suspicious {key} value: {val}")
        return warnings

    def questions_for_labels(self, label_ids: list[str]) -> dict:
        """Return minimal question set for a list of candidate diagnoses."""
        syms, risks, vits = set(), set(), set()
        for lid in label_ids:
            schema = self._schema.get(lid, {})
            syms.update(schema.get("symptoms", []))
            risks.update(schema.get("risk_factors", []))
            vits.update(schema.get("vitals", []))
        return {
            "symptoms":     sorted(syms),
            "risk_factors": sorted(risks),
            "vitals":       sorted(vits),
        }

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def _evaluate_trigger(self, trigger: str, patient: dict) -> tuple[bool, str]:
        symptoms     = patient.get("symptoms", {})
        risk_factors = patient.get("risk_factors", {})
        vital_flags  = patient.get("vital_flags", {})

        # Special: asymptomatic
        if trigger == "asymptomatic":
            raw_syms = patient.get("raw", {}).get("symptoms", symptoms)
            return not any(raw_syms.values()), "patient is asymptomatic"

        # Negated
        if trigger.startswith("no_"):
            base = trigger[3:]
            val  = symptoms.get(base, risk_factors.get(base, None))
            if val is not None:
                return not bool(val), f"absence of {base.replace('_', ' ')}"
            return False, f"{base} not recorded"

        # Vital flag (e.g. sbp_lt_90)
        if trigger in vital_flags:
            return bool(vital_flags[trigger]), trigger.replace("_", " ")

        # Boolean symptom
        if trigger in symptoms:
            return bool(symptoms[trigger]), trigger.replace("_", " ")

        # Boolean risk factor
        if trigger in risk_factors:
            return bool(risk_factors[trigger]), trigger.replace("_", " ")

        logger.debug("Trigger '%s' not found — treating as absent.", trigger)
        return False, f"{trigger} not recorded"

    # ------------------------------------------------------------------
    # Vitals processing
    # ------------------------------------------------------------------

    def _process_vitals(self, raw: dict) -> tuple[dict[str, float], dict[str, bool]]:
        vitals: dict[str, float] = {}
        flags:  dict[str, bool]  = {}
        for k, v in raw.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                logger.warning("Vital '%s' non-numeric '%s' — skipping.", k, v)
                continue
            vitals[k] = val
            for flag_key, op, threshold in self.VITAL_THRESHOLDS.get(k, []):
                flags[flag_key] = val < threshold if op == "lt" else val > threshold
        return vitals, flags

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_schema()
        self._load_history_rules()
        self._load_weights()

    def _load_schema(self) -> None:
        path = self.dir / "history_schema.json"
        if not path.exists():
            logger.warning("history_schema.json not found at %s", path)
            return
        with open(path, encoding="utf-8") as f:
            self._schema = json.load(f)

    def _load_history_rules(self) -> None:
        path = self.dir / "history_rules.csv"
        if not path.exists():
            logger.warning("history_rules.csv not found at %s", path)
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader  = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            # Detect schema by column names
            is_nb = "p_given_disease" in fieldnames and "p_given_not_disease" in fieldnames
            self._nb_mode = is_nb
            if is_nb:
                logger.info("history_rules.csv: NB CPT schema detected.")
            else:
                logger.info("history_rules.csv: legacy additive schema detected.")
            for row in reader:
                rule_id_raw = (row.get("rule_id") or "").strip()
                if not rule_id_raw or rule_id_raw.startswith("#"):
                    continue
                if is_nb:
                    self._h_rules.append(HistoryRule(
                        rule_id  = rule_id_raw,
                        label_id = (row.get("label_id") or "").strip(),
                        trigger  = (row.get("trigger")  or "").strip(),
                        p_given_disease     = float(row.get("p_given_disease")     or 0.5),
                        p_given_not_disease = float(row.get("p_given_not_disease") or 0.5),
                    ))
                else:
                    self._h_rules.append(HistoryRule(
                        rule_id  = rule_id_raw,
                        label_id = (row.get("label_id") or "").strip(),
                        trigger  = (row.get("trigger")  or "").strip(),
                        action   = (row.get("action")   or "").strip(),
                        delta    = float(row.get("delta") or 0),
                    ))

    def _load_weights(self) -> None:
        path = self.dir / "symptom_weights.csv"
        if not path.exists():
            return
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("symptom_id", "").strip()
                if sid:
                    self._weights[sid] = float(row.get("weight", 1.0) or 1.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                        format="%(levelname)s | %(message)s")

    encoder = HistoryEncoder(history_module_dir="history_module/")

    raw = {
        "symptoms":     {"chest_pain": True, "palpitations": False, "dyspnea": True},
        "risk_factors": {"htn": True, "dm": False, "cad": True},
        "vitals":       {"sbp": 88, "hr": 140, "spo2": 94},
    }

    encoded = encoder.encode(raw)
    print("Active vital flags:", {k: v for k, v in encoded["vital_flags"].items() if v})

    for label in ["STEMI", "AF", "LVH"]:
        delta = encoder.compute_delta(label, encoded)
        print(f"\n{label}: score_delta={delta.score_delta:+.3f}  tier_delta={delta.tier_delta}")
        for e in delta.evidence:
            print(f"  {e}")