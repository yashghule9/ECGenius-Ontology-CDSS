"""
ECGenius — rules_engine/rule_executor.py
=========================================
A plugin-based, data-driven rule engine.

Design philosophy
-----------------
Medical experts define rules purely in rules.csv — NO code changes needed
for new rule instances. When experts invent a brand-new *rule_type* that has
never been seen before, the engine logs a warning and skips it safely.
A developer then registers a handler for that type and it starts working.

Rule types currently supported
-------------------------------
  mutual_exclusion  — only the highest-Pai(D) label in a group survives
  precedence        — one label forcibly overrides / suppresses others
  derived           — a new label is inferred when ECG pattern + optional
                      symptom conditions are both satisfied
  downgrade         — reduce a label's score or tier based on missing context

Adding a new rule type (for developers)
----------------------------------------
  1. Write a function:
        def _handle_mytype(ctx: RuleContext, rule: RuleRow) -> None: ...
  2. Register it:
        @RuleExecutor.register("mytype")
        def _handle_mytype(...): ...
  That's it. No other files need to change.

Rule CSV columns used
----------------------
  rule_id, rule_type, primary_label, related_labels (pipe-separated),
  required_symptoms (pipe-separated), action, delta
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RuleRow:
    """One parsed row from rules.csv."""
    rule_id: str
    rule_type: str
    primary_label: str
    related_labels: list[str]        # pipe-separated → list
    required_symptoms: list[str]     # pipe-separated → list (may be empty)
    action: str
    delta: float                     # score or tier change; 0 if not applicable


@dataclass
class RuleContext:
    """
    Everything the rule engine needs to make decisions.
    Passed to every handler function.

    results     — list of OntologyResult objects from ontology_mapper.py
                  Handlers mutate these in-place (score, tier, is_suppressed).

    patient     — structured patient history dict from history_encoder.py
                  Shape: {"symptoms": {...}, "risk_factors": {...}, "vitals": {...}}

    derived_log — list of strings describing any labels derived during execution.
                  Used for explainability output.
    """
    results: list          # list[OntologyResult] — avoid circular import
    patient: dict
    derived_log: list[str] = field(default_factory=list)

    # Internal lookup rebuilt at execution time for O(1) access
    _result_map: dict = field(default_factory=dict, repr=False)

    def build_map(self) -> None:
        self._result_map = {r.label_id: r for r in self.results}

    def get(self, label_id: str):
        return self._result_map.get(label_id)

    def symptom_present(self, symptom_id: str) -> bool:
        """Check if a symptom / risk_factor / vital threshold is satisfied."""
        symptoms = self.patient.get("symptoms", {})
        risk_factors = self.patient.get("risk_factors", {})
        vitals = self.patient.get("vitals", {})

        # Boolean symptoms and risk factors
        if symptom_id in symptoms:
            return bool(symptoms[symptom_id])
        if symptom_id in risk_factors:
            return bool(risk_factors[symptom_id])

        # Negated form: "no_chest_pain" → chest_pain is False
        if symptom_id.startswith("no_"):
            base = symptom_id[3:]
            val = symptoms.get(base, risk_factors.get(base, None))
            if val is not None:
                return not bool(val)

        # Vital thresholds encoded as e.g. "sbp_lt_90"
        if "_lt_" in symptom_id:
            vital_key, threshold = symptom_id.split("_lt_")
            val = vitals.get(vital_key)
            if val is not None:
                return float(val) < float(threshold)
        if "_gt_" in symptom_id:
            vital_key, threshold = symptom_id.split("_gt_")
            val = vitals.get(vital_key)
            if val is not None:
                return float(val) > float(threshold)

        # Unknown symptom id — treat as absent, warn once
        logger.debug("symptom_present: unknown id '%s' — treating as absent.", symptom_id)
        return False

    def all_symptoms_present(self, symptom_ids: list[str]) -> bool:
        return all(self.symptom_present(s) for s in symptom_ids)

    def any_symptom_present(self, symptom_ids: list[str]) -> bool:
        return any(self.symptom_present(s) for s in symptom_ids)


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HandlerFn = Callable[[RuleContext, RuleRow], None]
_HANDLER_REGISTRY: dict[str, HandlerFn] = {}


def register(rule_type: str):
    """Decorator to register a rule handler."""
    def decorator(fn: HandlerFn) -> HandlerFn:
        _HANDLER_REGISTRY[rule_type] = fn
        logger.debug("Registered handler for rule_type='%s'", rule_type)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------

@register("mutual_exclusion")
def _handle_mutual_exclusion(ctx: RuleContext, rule: RuleRow) -> None:
    """
    Keep only the highest-Pai(D) label from a group.
    All others are marked is_suppressed=True.

    Safety: Tier-1 labels are NEVER suppressed regardless of Pai(D).
    """
    all_ids = list(dict.fromkeys([rule.primary_label] + rule.related_labels))
    present = [ctx.get(lid) for lid in all_ids if ctx.get(lid) and not ctx.get(lid).is_suppressed]

    if len(present) <= 1:
        return

    present.sort(key=lambda r: -r.pai)
    winner = present[0]

    for loser in present[1:]:
        if loser.tier == 1:
            logger.warning(
                "[%s] Refusing to suppress Tier-1 label '%s' in ME group. "
                "Both '%s' (winner) and '%s' will be shown.",
                rule.rule_id, loser.label_id, winner.label_id, loser.label_id,
            )
            continue
        loser.is_suppressed = True
        logger.info("[%s] ME: suppressed '%s' (pai=%.3f) → winner '%s' (pai=%.3f)",
                    rule.rule_id, loser.label_id, loser.pai, winner.label_id, winner.pai)


@register("precedence")
def _handle_precedence(ctx: RuleContext, rule: RuleRow) -> None:
    """
    primary_label overrides / suppresses related_labels entirely.
    Only fires if primary_label is present and not already suppressed.

    Example: VF overrides NSR and AF — if VF is detected, rhythm
    alternatives are irrelevant.

    action='override' — suppress all related labels.
    """
    primary = ctx.get(rule.primary_label)
    if primary is None or primary.is_suppressed:
        return

    if rule.action == "override":
        for lid in rule.related_labels:
            target = ctx.get(lid)
            if target is None:
                continue
            if target.tier == 1:
                logger.warning(
                    "[%s] Precedence: refusing to override Tier-1 '%s'.",
                    rule.rule_id, target.label_id,
                )
                continue
            target.is_suppressed = True
            logger.info("[%s] Precedence: '%s' overrides '%s'",
                        rule.rule_id, rule.primary_label, lid)
    else:
        logger.warning("[%s] Precedence: unknown action '%s' — skipping.",
                       rule.rule_id, rule.action)


@register("derived")
def _handle_derived(ctx: RuleContext, rule: RuleRow) -> None:
    """
    Infer a new label when:
      1. primary_label (ECG pattern) is present with sufficient Pai(D), AND
      2. All required_symptoms are satisfied in patient history.

    action='add_label'  — inject a new OntologyResult into ctx.results.
    delta               — Pai(D) value assigned to the derived label.

    The derived label needs to exist in labels.csv (ontology_mapper should
    have loaded its metadata). If it's already present from the model, its
    score gets a delta boost instead of creating a duplicate.

    Example R3 from your rules.csv:
      primary=ST_Elevation, required=chest_pain, action=add_label, delta=0.4
      → if model saw ST elevation AND patient has chest pain → derive STEMI
    """
    trigger = ctx.get(rule.primary_label)
    if trigger is None or trigger.is_suppressed:
        return

    # Check symptom conditions
    if rule.required_symptoms and not ctx.all_symptoms_present(rule.required_symptoms):
        return

    if rule.action == "add_label":
        derived_id = rule.related_labels[0] if rule.related_labels else rule.primary_label
        existing = ctx.get(derived_id)

        if existing:
            # Already predicted by the model — NB fusion handles its posterior.
            # No additive score boost: the ECG pattern evidence is already encoded
            # in the model prior and the CPT likelihood ratios.
            ctx.derived_log.append(
                f"[{rule.rule_id}] '{derived_id}' confirmed by derived rule "
                f"(trigger: '{rule.primary_label}' + symptoms {rule.required_symptoms})"
            )
            logger.info("[%s] Derived rule confirms existing '%s' — no score delta applied.",
                        rule.rule_id, derived_id)
        else:
            # Not predicted by model — inject a synthetic OntologyResult via mapper
            mapper = getattr(ctx, '_mapper', None)
            if mapper is None:
                logger.warning(
                    "[%s] Derived label '%s' not in model output and no mapper "
                    "injected into context — skipping. "
                    "Pass mapper=mapper_instance to RuleExecutor.",
                    rule.rule_id, derived_id,
                )
                ctx.derived_log.append(
                    f"[{rule.rule_id}] Cannot derive '{derived_id}': mapper not available."
                )
                return

            if not mapper.label_exists(derived_id):
                logger.warning(
                    "[%s] Derived label '%s' not found in ontology — skipping.",
                    rule.rule_id, derived_id,
                )
                ctx.derived_log.append(
                    f"[{rule.rule_id}] Cannot derive '{derived_id}': label not in ontology."
                )
                return

            synthetic = mapper._enrich(derived_id, rule.delta)
            synthetic.score = rule.delta
            ctx.results.append(synthetic)
            ctx.build_map()  # rebuild so downstream rules can see the new label
            ctx.derived_log.append(
                f"[{rule.rule_id}] Derived '{derived_id}' (pai={rule.delta:.2f}) "
                f"from '{rule.primary_label}' + symptoms {rule.required_symptoms}"
            )
            logger.info(
                "[%s] Derived: injected synthetic '%s' (pai=%.2f)",
                rule.rule_id, derived_id, rule.delta,
            )
    else:
        logger.warning("[%s] Derived: unknown action '%s' — skipping.", rule.rule_id, rule.action)


@register("downgrade")
def _handle_downgrade(ctx: RuleContext, rule: RuleRow) -> None:
    """
    Reduce a label's score or tier based on absence of expected context.

    action='reduce_score'  — subtract abs(delta) from label's score.
    action='reduce_tier'   — increase tier number by int(delta) (tier 1→2 = less urgent).

    Safety: allow_downgrade flag on the OntologyResult (from triage.csv)
    must be True. Tier-1 labels cannot have their tier downgraded.

    Example R4: STEMI + no_chest_pain → reduce_score -0.3
    Example R5: AF + asymptomatic    → reduce_tier 1
    """
    target = ctx.get(rule.primary_label)
    if target is None or target.is_suppressed:
        return

    # Check downgrade is permitted
    if not target.allow_downgrade:
        logger.info("[%s] Downgrade blocked — allow_downgrade=False for '%s'.",
                    rule.rule_id, rule.primary_label)
        return

    # Check trigger conditions
    if rule.required_symptoms and not ctx.all_symptoms_present(rule.required_symptoms):
        return

    if rule.action == "reduce_score":
        reduction = abs(rule.delta)
        target.score = max(0.0, target.score - reduction)
        logger.info("[%s] Downgrade: '%s' score -= %.2f (now %.3f)",
                    rule.rule_id, rule.primary_label, reduction, target.score)

    elif rule.action == "reduce_tier":
        if target.tier == 1:
            logger.warning(
                "[%s] Downgrade: refusing to reduce tier for Tier-1 label '%s'.",
                rule.rule_id, rule.primary_label,
            )
            return
        steps = int(abs(rule.delta))
        old_tier = target.tier
        target.tier = min(3, target.tier + steps)
        logger.info("[%s] Downgrade: '%s' tier %d → %d",
                    rule.rule_id, rule.primary_label, old_tier, target.tier)

    else:
        logger.warning("[%s] Downgrade: unknown action '%s' — skipping.",
                       rule.rule_id, rule.action)


# ---------------------------------------------------------------------------
# Main executor class
# ---------------------------------------------------------------------------

class RuleExecutor:
    """
    Loads rules.csv once, executes all rules against a RuleContext.

    Parameters
    ----------
    rules_dir : str | Path
        Directory containing rules.csv (and future rule files).
    strict : bool
        If True, raise ValueError on unknown rule_type.
        If False (default), warn and skip.
    """

    # Expose registry so callers can check or extend it
    registry = _HANDLER_REGISTRY

    @staticmethod
    def register(rule_type: str):
        """Allow external code to register new handlers at runtime."""
        return register(rule_type)

    def __init__(self, rules_dir: str | Path = "rules_engine/", strict: bool = False):
        self.rules_dir = Path(rules_dir)
        self.strict = strict
        self._rules: list[RuleRow] = []
        self._load_rules()
        logger.info(
            "RuleExecutor ready — %d rules loaded, %d handler types registered",
            len(self._rules), len(_HANDLER_REGISTRY),
        )

    def execute(
        self,
        results: list,
        patient: dict,
        mapper=None,
    ) -> tuple[list, list[str]]:
        """
        Run all rules against the current patient context.

        Parameters
        ----------
        results : list[OntologyResult]
            From OntologyMapper.map() — will be mutated in-place.
        patient : dict
            From history_encoder.py — symptoms, risk_factors, vitals.
        mapper : OntologyMapper | None
            Optional — needed for derived rules that inject new labels.

        Returns
        -------
        results : list[OntologyResult]
            Same list, mutated.
        derived_log : list[str]
            Human-readable log of derived/boosted labels for XAI output.
        """
        ctx = RuleContext(results=results, patient=patient)
        ctx.build_map()

        # Inject mapper for derived rules if provided
        if mapper is not None:
            ctx._mapper = mapper  # handlers can access via ctx._mapper

        unknown_types: set[str] = set()

        for rule in self._rules:
            handler = _HANDLER_REGISTRY.get(rule.rule_type)

            if handler is None:
                if rule.rule_type not in unknown_types:
                    msg = (
                        f"No handler registered for rule_type='{rule.rule_type}' "
                        f"(first seen in rule '{rule.rule_id}'). "
                        f"Register a handler with @RuleExecutor.register('{rule.rule_type}')."
                    )
                    if self.strict:
                        raise ValueError(msg)
                    logger.warning(msg)
                    unknown_types.add(rule.rule_type)
                continue

            try:
                handler(ctx, rule)
            except Exception as exc:
                # Never let a single rule crash the whole inference pipeline
                logger.error(
                    "Rule '%s' (type='%s') raised an exception: %s — skipping.",
                    rule.rule_id, rule.rule_type, exc, exc_info=True,
                )

        # Re-sort: non-suppressed first, then by score desc (score may have changed)
        ctx.results.sort(key=lambda r: (r.is_suppressed, -(r.score or r.pai)))

        return ctx.results, ctx.derived_log

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        """
        Load rules from rules_engine/rule_definitions/*.py files (auto-import)
        then parse rules.csv.

        Auto-import lets experts drop a new file like
        `rules_engine/rule_definitions/conduction_rules.py` and have its
        @register decorators activate without touching any other file.
        """
        self._auto_import_rule_definitions()

        rules_csv = self.rules_dir / "rules.csv"
        if not rules_csv.exists():
            # Also check ontology/ folder (rules.csv may live there)
            rules_csv = Path("ontology") / "rules.csv"

        if not rules_csv.exists():
            logger.warning("rules.csv not found — no rules will be applied.")
            return

        with open(rules_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rule_id_raw = (row.get("rule_id") or "").strip()
                if not rule_id_raw or rule_id_raw.startswith("#"):
                    continue
                related_raw = row.get("related_labels", "") or ""
                symptom_raw = row.get("required_symptoms", "") or ""

                self._rules.append(RuleRow(
                    rule_id=row.get("rule_id", "?"),
                    rule_type=row.get("rule_type", "").strip().lower(),
                    primary_label=row.get("primary_label", "").strip(),
                    related_labels=[r.strip() for r in related_raw.split("|") if r.strip()],
                    required_symptoms=[s.strip() for s in symptom_raw.split("|") if s.strip()],
                    action=row.get("action", "").strip(),
                    delta=float(row.get("delta", 0) or 0),
                ))

        logger.info("Loaded %d rules from %s", len(self._rules), rules_csv)

    def _auto_import_rule_definitions(self) -> None:
        """
        Auto-import all .py files in rules_engine/rule_definitions/
        so their @register decorators fire without manual imports.
        """
        import importlib.util, sys

        defs_dir = self.rules_dir / "rule_definitions"
        if not defs_dir.exists():
            return

        for py_file in sorted(defs_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_name = f"rule_definitions.{py_file.stem}"
            if module_name in sys.modules:
                continue
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[module_name] = mod
                logger.info("Auto-imported rule definitions from '%s'", py_file.name)
            except Exception as exc:
                logger.error("Failed to import rule definitions '%s': %s", py_file, exc)


# ---------------------------------------------------------------------------
# Smoke test — python rule_executor.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(levelname)s | %(name)s | %(message)s")

    from dataclasses import dataclass

    # Minimal stub so we can test without a full OntologyResult import
    @dataclass
    class StubResult:
        label_id: str
        pai: float
        tier: int = 2
        score: float = 0.0
        is_suppressed: bool = False
        allow_downgrade: bool = True
        confidence_label: str = "UNSCORED"

    fake_results = [
        StubResult("AF",    pai=0.82),
        StubResult("NSR",   pai=0.61),
        StubResult("STEMI", pai=0.74, tier=1, allow_downgrade=False),
        StubResult("LVH",   pai=0.45),
    ]

    fake_patient = {
        "symptoms":     {"chest_pain": True,  "palpitations": True},
        "risk_factors": {"htn": True, "dm": False},
        "vitals":       {"sbp": 88, "hr": 110},
    }

    executor = RuleExecutor(rules_dir="rules_engine/", strict=False)
    results, log = executor.execute(fake_results, fake_patient)

    print("\n=== Rule executor output ===\n")
    for r in results:
        flag = "[SUPPRESSED]" if r.is_suppressed else f"[Tier {r.tier}]"
        print(f"  {flag:14s}  {r.label_id:10s}  pai={r.pai:.3f}  score={r.score:.3f}")

    if log:
        print("\n=== Derived log ===")
        for entry in log:
            print(" ", entry)