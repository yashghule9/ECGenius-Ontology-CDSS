"""
ECGenius — rules_engine/rule_validator.py
==========================================
Validates rules.csv against the ontology before any deployment.
Run this whenever your cardiologists update rules.csv.

Checks performed:
  1. All label_ids in rules.csv exist in labels.csv
  2. All related_labels exist in labels.csv
  3. All required_symptoms exist in symptoms.csv + risk_factors.csv
  4. Delta values are within safe bounds per action type
  5. No circular derived rules (A derives B derives A)
  6. Tier-1 labels are never in a downgrade rule without allow_downgrade check
  7. Mutual exclusion groups have ≥ 2 members
  8. No duplicate rule_ids

Usage:
    python rule_validator.py
    python rule_validator.py --rules ontology/rules.csv --labels ontology/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class RuleValidator:

    SAFE_DELTA_BOUNDS = {
        "add_score":    (0.0,  0.5),
        "reduce_score": (-0.5, 0.0),
        "reduce_tier":  (1,    2),
        "add_tier":     (1,    1),
        "keep_highest": (0,    0),
        "override":     (0,    0),
        "add_label":    (0.0,  0.5),
    }

    def __init__(
        self,
        rules_path:   str = "ontology/rules.csv",
        labels_path:  str = "ontology/labels.csv",
        triage_path:  str = "ontology/triage.csv",
        symptoms_path: str = "history_module/symptoms.csv",
        risks_path:   str = "history_module/risk_factors.csv",
    ):
        self.rules_path   = Path(rules_path)
        self.labels_path  = Path(labels_path)
        self.triage_path  = Path(triage_path)
        self.symptoms_path = Path(symptoms_path)
        self.risks_path   = Path(risks_path)

        self._errors:   list[str] = []
        self._warnings: list[str] = []

    def validate(self) -> bool:
        """
        Run all checks. Returns True if no errors (warnings OK).
        Prints a full report to stdout.
        """
        labels   = self._load_labels()
        triage   = self._load_triage()
        symptoms = self._load_symptoms()
        rules    = self._load_rules()

        self._check_duplicate_rule_ids(rules)
        self._check_label_ids(rules, labels)
        self._check_related_labels(rules, labels)
        self._check_symptoms(rules, symptoms)
        self._check_delta_bounds(rules)
        self._check_tier1_safety(rules, triage)
        self._check_me_group_size(rules)
        self._check_circular_derived(rules)

        self._print_report()
        return len(self._errors) == 0

    # ------------------------------------------------------------------

    def _check_duplicate_rule_ids(self, rules: list[dict]) -> None:
        seen: set[str] = set()
        for r in rules:
            rid = r.get("rule_id", "")
            if rid in seen:
                self._errors.append(f"Duplicate rule_id: '{rid}'")
            seen.add(rid)

    def _check_label_ids(self, rules: list[dict], labels: set[str]) -> None:
        for r in rules:
            primary = r.get("primary_label", "").strip()
            if primary and primary not in labels:
                self._errors.append(
                    f"[{r['rule_id']}] primary_label '{primary}' not in labels.csv"
                )

    def _check_related_labels(self, rules: list[dict], labels: set[str]) -> None:
        for r in rules:
            related_raw = r.get("related_labels", "")
            for lid in [x.strip() for x in related_raw.split("|") if x.strip()]:
                if lid not in labels:
                    self._errors.append(
                        f"[{r['rule_id']}] related_label '{lid}' not in labels.csv"
                    )

    def _check_symptoms(self, rules: list[dict], symptoms: set[str]) -> None:
        for r in rules:
            sym_raw = r.get("required_symptoms", "")
            for sym in [x.strip() for x in sym_raw.split("|") if x.strip()]:
                # Strip negated prefix
                base = sym[3:] if sym.startswith("no_") else sym
                # Vital threshold flags are auto-generated — skip
                if "_lt_" in base or "_gt_" in base:
                    continue
                if base not in symptoms and base != "asymptomatic":
                    self._warnings.append(
                        f"[{r['rule_id']}] required_symptom '{sym}' not in "
                        f"symptoms.csv or risk_factors.csv — may be intentional."
                    )

    def _check_delta_bounds(self, rules: list[dict]) -> None:
        for r in rules:
            action = r.get("action", "").strip()
            try:
                delta = float(r.get("delta", 0) or 0)
            except ValueError:
                self._errors.append(
                    f"[{r['rule_id']}] Non-numeric delta: '{r.get('delta')}'"
                )
                continue

            bounds = self.SAFE_DELTA_BOUNDS.get(action)
            if bounds is None:
                self._warnings.append(
                    f"[{r['rule_id']}] Unknown action '{action}' — "
                    f"no delta bound check performed."
                )
                continue

            lo, hi = bounds
            if not (lo <= abs(delta) <= hi) and delta != 0:
                self._warnings.append(
                    f"[{r['rule_id']}] Delta {delta} for action '{action}' "
                    f"outside recommended bounds [{lo}, {hi}]."
                )

    def _check_tier1_safety(self, rules: list[dict], triage: dict[str, int]) -> None:
        downgrade_actions = {"reduce_score", "reduce_tier"}
        for r in rules:
            if r.get("action", "") not in downgrade_actions:
                continue
            primary = r.get("primary_label", "").strip()
            tier = triage.get(primary, 3)
            if tier == 1:
                self._errors.append(
                    f"[{r['rule_id']}] Downgrade rule targets Tier-1 label "
                    f"'{primary}'. This is blocked at runtime but should be "
                    f"removed or reviewed by a cardiologist."
                )

    def _check_me_group_size(self, rules: list[dict]) -> None:
        for r in rules:
            if r.get("rule_type", "").strip() != "mutual_exclusion":
                continue
            related_raw = r.get("related_labels", "")
            members = [x.strip() for x in related_raw.split("|") if x.strip()]
            primary = r.get("primary_label", "").strip()
            all_members = list(dict.fromkeys([primary] + members))
            if len(all_members) < 2:
                self._errors.append(
                    f"[{r['rule_id']}] Mutual exclusion group has fewer than 2 members."
                )

    def _check_circular_derived(self, rules: list[dict]) -> None:
        """Detect A→B→A cycles in derived rules using DFS."""
        graph: dict[str, list[str]] = {}
        for r in rules:
            if r.get("rule_type", "").strip() != "derived":
                continue
            src  = r.get("primary_label", "").strip()
            dsts = [x.strip() for x in r.get("related_labels", "").split("|") if x.strip()]
            graph.setdefault(src, []).extend(dsts)

        def has_cycle(node: str, visited: set, stack: set) -> bool:
            visited.add(node)
            stack.add(node)
            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    if has_cycle(neighbour, visited, stack):
                        return True
                elif neighbour in stack:
                    return True
            stack.discard(node)
            return False

        visited: set = set()
        for node in list(graph.keys()):
            if node not in visited:
                if has_cycle(node, visited, set()):
                    self._errors.append(
                        f"Circular derived rule detected involving '{node}'."
                    )

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_labels(self) -> set[str]:
        if not self.labels_path.exists():
            self._warnings.append(f"labels.csv not found at {self.labels_path}")
            return set()
        with open(self.labels_path, newline="") as f:
            return {row["label_id"].strip() for row in csv.DictReader(f)}

    def _load_triage(self) -> dict[str, int]:
        if not self.triage_path.exists():
            return {}
        with open(self.triage_path, newline="") as f:
            return {
                row["label_id"].strip(): int(row.get("tier", 3))
                for row in csv.DictReader(f)
            }

    def _load_symptoms(self) -> set[str]:
        known: set[str] = set()
        for path in [self.symptoms_path, self.risks_path]:
            if not path.exists():
                continue
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    for col in ["symptom_id", "risk_id", "id"]:
                        if col in row:
                            known.add(row[col].strip())
                            break
        return known

    def _load_rules(self) -> list[dict]:
        if not self.rules_path.exists():
            self._errors.append(f"rules.csv not found at {self.rules_path}")
            return []
        with open(self.rules_path, newline="") as f:
            return list(csv.DictReader(f))

    # ------------------------------------------------------------------

    def _print_report(self) -> None:
        print("\n" + "=" * 55)
        print("ECGenius Rule Validator Report")
        print("=" * 55)

        if self._errors:
            print(f"\n  ERRORS ({len(self._errors)}) — must fix before deployment:")
            for e in self._errors:
                print(f"    [ERROR] {e}")
        else:
            print("\n  No errors found.")

        if self._warnings:
            print(f"\n  WARNINGS ({len(self._warnings)}) — review recommended:")
            for w in self._warnings:
                print(f"    [WARN]  {w}")
        else:
            print("  No warnings.")

        status = "PASS" if not self._errors else "FAIL"
        print(f"\n  Overall: {status}")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

    parser = argparse.ArgumentParser(description="ECGenius rule validator")
    parser.add_argument("--rules",    default="ontology/rules.csv")
    parser.add_argument("--labels",   default="ontology/labels.csv")
    parser.add_argument("--triage",   default="ontology/triage.csv")
    parser.add_argument("--symptoms", default="history_module/symptoms.csv")
    parser.add_argument("--risks",    default="history_module/risk_factors.csv")
    args = parser.parse_args()

    validator = RuleValidator(
        rules_path=args.rules,
        labels_path=args.labels,
        triage_path=args.triage,
        symptoms_path=args.symptoms,
        risks_path=args.risks,
    )
    ok = validator.validate()
    sys.exit(0 if ok else 1)