"""
ECGenius — inference/ontology_mapper.py
========================================
Bridges raw AI model output (Pai(D) probabilities) to the full
ECGenius ontology layer.

Responsibilities:
  1. Load ontology CSVs (labels, triage, mappings)
  2. Accept model output dict  {label_id: float probability}
  3. Filter to leaf nodes only (is_leaf == TRUE)
  4. Enrich each prediction with:
       - parent hierarchy chain
       - triage tier + default action + allow_downgrade flag
       - SNOMED-CT, ICD-10, AHA guideline, clinical notes
  5. Apply ontology-level mutual exclusion rules (e.g. only one
     primary rhythm label survives)
  6. Return a list of OntologyResult dataclasses, sorted by Pai(D)

This output is consumed by decision_fusion.py, which adds
symptom / risk / rule score deltas on top.

Usage:
    mapper = OntologyMapper(ontology_dir="ontology/")
    results = mapper.map(model_output={"AF": 0.82, "NSR": 0.61, ...})
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LabelMeta:
    """One row from labels.csv."""
    label_id: str
    label_name: str
    parent_label: str          # empty string for ROOT
    category: str
    description: str
    is_leaf: bool


@dataclass
class TriageMeta:
    """One row from triage.csv."""
    label_id: str
    tier: int                  # 1 = life-threatening, 2 = urgent, 3 = routine
    description: str
    default_action: str
    allow_downgrade: bool


@dataclass
class MappingMeta:
    """One row from mappings.csv."""
    label_id: str
    snomed_ct: str
    icd10: str
    aha_guideline: str
    notes: str


@dataclass
class OntologyResult:
    """
    Fully enriched prediction for one diagnosis.
    Passed to decision_fusion.py for final scoring.
    """
    label_id: str
    label_name: str
    category: str
    description: str
    hierarchy: list[str]        # [ROOT, ..., parent, label_id]

    # AI model probability (Pai(D))
    pai: float

    # Triage
    tier: int
    default_action: str
    allow_downgrade: bool

    # Terminologies
    snomed_ct: str
    icd10: str
    aha_guideline: str
    clinical_notes: str

    # Populated later by decision_fusion.py
    score: float = 0.0
    confidence_label: str = "UNSCORED"   # CONFIRMED/PROBABLE/POSSIBLE/INCIDENTAL
    is_suppressed: bool = False           # True if mutual exclusion removed it

    def to_dict(self) -> dict:
        return {
            "label_id": self.label_id,
            "label_name": self.label_name,
            "category": self.category,
            "description": self.description,
            "hierarchy": self.hierarchy,
            "pai": round(self.pai, 4),
            "tier": self.tier,
            "default_action": self.default_action,
            "allow_downgrade": self.allow_downgrade,
            "snomed_ct": self.snomed_ct,
            "icd10": self.icd10,
            "aha_guideline": self.aha_guideline,
            "clinical_notes": self.clinical_notes,
            "score": round(self.score, 4),
            "confidence_label": self.confidence_label,
            "is_suppressed": self.is_suppressed,
        }


# ---------------------------------------------------------------------------
# Mutual-exclusion groups
# Loaded from rules.csv (rule_type == "mutual_exclusion").
# The mapper handles these early so fusion only sees clean candidates.
# ---------------------------------------------------------------------------

@dataclass
class MutualExclusionGroup:
    primary_label: str
    related_labels: list[str]   # all labels in the group (including primary)
    action: str                 # typically "keep_highest"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OntologyMapper:
    """
    Load ontology files once at startup, map model outputs on every call.

    Parameters
    ----------
    ontology_dir : str | Path
        Directory containing labels.csv, triage.csv, mappings.csv, rules.csv
    threshold : float
        Pai(D) below this value is ignored entirely (default 0.1)
    """

    # Score thresholds for confidence label (used AFTER fusion, but stored here
    # so the UI can render preliminary labels even before history is collected).
    CONFIDENCE_THRESHOLDS = {
        "CONFIRMED":   0.80,
        "PROBABLE":    0.60,
        "POSSIBLE":    0.30,
        "INCIDENTAL":  0.00,
    }

    def __init__(self, ontology_dir: str | Path = "ontology/", threshold: float = 0.10):
        self.ontology_dir = Path(ontology_dir)
        self.threshold = threshold

        # Internal lookup tables
        self._labels:   dict[str, LabelMeta]   = {}
        self._triage:   dict[str, TriageMeta]  = {}
        self._mappings: dict[str, MappingMeta] = {}
        self._me_groups: list[MutualExclusionGroup] = []

        self._load_all()
        logger.info(
            "OntologyMapper ready — %d labels, %d triage entries, "
            "%d mappings, %d ME groups",
            len(self._labels), len(self._triage),
            len(self._mappings), len(self._me_groups),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(self, model_output: dict[str, float]) -> list[OntologyResult]:
        """
        Main entry point.

        Parameters
        ----------
        model_output : dict
            {label_id: Pai(D)}  — raw sigmoid/softmax outputs from the model.
            Keys must match label_id values in labels.csv.

        Returns
        -------
        list[OntologyResult]
            Sorted by Pai(D) descending. Suppressed labels are included but
            flagged with is_suppressed=True so the UI can optionally show them.
        """
        # 1. Filter: threshold + leaf-only
        candidates = self._filter_candidates(model_output)

        # 2. Enrich each candidate with ontology metadata
        results = [self._enrich(label_id, pai) for label_id, pai in candidates]

        # 3. Apply mutual exclusion rules
        results = self._apply_mutual_exclusion(results)

        # 4. Sort: non-suppressed first, then by pai desc
        results.sort(key=lambda r: (r.is_suppressed, -r.pai))

        return results

    def get_hierarchy(self, label_id: str) -> list[str]:
        """
        Walk the parent chain upward and return the full path from ROOT.
        Example: ["ROOT", "Ischemia", "STEMI"]
        """
        chain = []
        current = label_id
        visited = set()

        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            meta = self._labels.get(current)
            if meta is None or not meta.parent_label:
                break
            current = meta.parent_label

        return list(reversed(chain))

    def label_exists(self, label_id: str) -> bool:
        return label_id in self._labels

    def get_label_meta(self, label_id: str) -> Optional[LabelMeta]:
        return self._labels.get(label_id)

    def get_triage(self, label_id: str) -> Optional[TriageMeta]:
        return self._triage.get(label_id)

    def get_mapping(self, label_id: str) -> Optional[MappingMeta]:
        return self._mappings.get(label_id)

    def results_to_json(self, results: list[OntologyResult]) -> str:
        return json.dumps([r.to_dict() for r in results], indent=2)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_labels()
        self._load_triage()
        self._load_mappings()
        self._load_me_rules()

    def _load_labels(self) -> None:
        path = self.ontology_dir / "labels.csv"
        self._assert_file(path)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._labels[row["label_id"]] = LabelMeta(
                    label_id=row["label_id"],
                    label_name=row["label_name"],
                    parent_label=row.get("parent_label", ""),
                    category=row.get("category", ""),
                    description=row.get("description", ""),
                    is_leaf=row.get("is_leaf", "FALSE").strip().upper() == "TRUE",
                )

    def _load_triage(self) -> None:
        path = self.ontology_dir / "triage.csv"
        self._assert_file(path)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._triage[row["label_id"]] = TriageMeta(
                    label_id=row["label_id"],
                    tier=int(row.get("tier", 3)),
                    description=row.get("description", ""),
                    default_action=row.get("default_action", ""),
                    allow_downgrade=row.get("allow_downgrade", "TRUE").strip().upper() == "TRUE",
                )

    def _load_mappings(self) -> None:
        path = self.ontology_dir / "mappings.csv"
        self._assert_file(path)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._mappings[row["label_id"]] = MappingMeta(
                    label_id=row["label_id"],
                    snomed_ct=row.get("snomed_ct", ""),
                    icd10=row.get("icd10", ""),
                    aha_guideline=row.get("aha_guideline", ""),
                    notes=row.get("notes", ""),
                )

    def _load_me_rules(self) -> None:
        """
        Load mutual_exclusion rows from rules.csv.
        Other rule types (precedence, derived, downgrade) are handled by rule_executor.py.
        """
        path = self.ontology_dir / "rules.csv"
        if not path.exists():
            logger.warning("rules.csv not found — no mutual exclusion groups loaded.")
            return

        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rule_id = (row.get("rule_id") or "").strip()
                if not rule_id or rule_id.startswith("#"):
                    continue
                if (row.get("rule_type") or "").strip() != "mutual_exclusion":
                    continue
                related_raw = row.get("related_labels", "")
                related = [r.strip() for r in related_raw.split("|") if r.strip()]
                primary = row.get("primary_label", "").strip()

                # Include primary in the group for keep_highest logic
                all_in_group = list(dict.fromkeys([primary] + related))

                self._me_groups.append(MutualExclusionGroup(
                    primary_label=primary,
                    related_labels=all_in_group,
                    action=row.get("action", "keep_highest").strip(),
                ))

    # ------------------------------------------------------------------
    # Core mapping steps
    # ------------------------------------------------------------------

    def _filter_candidates(self, model_output: dict[str, float]) -> list[tuple[str, float]]:
        """
        Keep only leaf labels above the probability threshold.
        Warn about unknown label IDs.
        """
        candidates = []
        for label_id, pai in model_output.items():
            if pai < self.threshold:
                continue

            meta = self._labels.get(label_id)
            if meta is None:
                logger.warning("Model output label '%s' not found in labels.csv — skipping.", label_id)
                continue
            if not meta.is_leaf:
                logger.debug("Label '%s' is not a leaf node — skipping.", label_id)
                continue

            candidates.append((label_id, float(pai)))

        return candidates

    def _enrich(self, label_id: str, pai: float) -> OntologyResult:
        """Build a fully enriched OntologyResult for one label."""
        label  = self._labels.get(label_id)
        triage = self._triage.get(label_id)
        mapping = self._mappings.get(label_id)

        # Fallback gracefully if triage or mapping rows are missing
        if triage is None:
            logger.warning("No triage entry for '%s' — defaulting to tier 3.", label_id)
            triage = TriageMeta(
                label_id=label_id, tier=3,
                description="", default_action="Clinical review",
                allow_downgrade=True,
            )

        if mapping is None:
            logger.debug("No mapping entry for '%s'.", label_id)
            mapping = MappingMeta(
                label_id=label_id, snomed_ct="",
                icd10="", aha_guideline="", notes="",
            )

        return OntologyResult(
            label_id=label_id,
            label_name=label.label_name,
            category=label.category,
            description=label.description,
            hierarchy=self.get_hierarchy(label_id),
            pai=pai,
            tier=triage.tier,
            default_action=triage.default_action,
            allow_downgrade=triage.allow_downgrade,
            snomed_ct=mapping.snomed_ct,
            icd10=mapping.icd10,
            aha_guideline=mapping.aha_guideline,
            clinical_notes=mapping.notes,
        )

    def _apply_mutual_exclusion(self, results: list[OntologyResult]) -> list[OntologyResult]:
        """
        For each mutual exclusion group, keep only the highest-Pai(D) member.
        All others in the group are marked is_suppressed=True.

        A Tier-1 label (life-threatening) can never be suppressed —
        safety guardrail to ensure critical findings are never silently dropped.
        """
        # Build a lookup: label_id -> result for fast access
        result_map: dict[str, OntologyResult] = {r.label_id: r for r in results}

        for group in self._me_groups:
            # Find which group members are present in this patient's results
            present = [
                result_map[lid]
                for lid in group.related_labels
                if lid in result_map and not result_map[lid].is_suppressed
            ]

            if len(present) <= 1:
                continue  # no conflict in this group

            if group.action == "keep_highest":
                # Sort by pai desc; keep the winner
                present.sort(key=lambda r: -r.pai)
                winner = present[0]

                for loser in present[1:]:
                    # SAFETY: never suppress a Tier-1 label
                    if loser.tier == 1:
                        logger.warning(
                            "Mutual exclusion: refusing to suppress Tier-1 label '%s'. "
                            "Both '%s' and '%s' will be shown.",
                            loser.label_id, winner.label_id, loser.label_id,
                        )
                        continue
                    loser.is_suppressed = True
                    logger.debug(
                        "ME group '%s': suppressed '%s' (pai=%.3f) in favour of '%s' (pai=%.3f)",
                        group.primary_label, loser.label_id, loser.pai,
                        winner.label_id, winner.pai,
                    )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"OntologyMapper: required file not found: {path}\n"
                f"Check that ontology_dir points to the correct folder."
            )

    @classmethod
    def confidence_label(cls, score: float) -> str:
        """Map a final fusion score to a human-readable confidence label."""
        for label, threshold in cls.CONFIDENCE_THRESHOLDS.items():
            if score >= threshold:
                return label
        return "INCIDENTAL"


# ---------------------------------------------------------------------------
# Quick smoke test — run directly: python ontology_mapper.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    # Simulated model output — replace with real inference output
    fake_model_output = {
        "AF":    0.82,
        "NSR":   0.61,   # should be suppressed (same rhythm ME group as AF)
        "STEMI": 0.74,
        "LVH":   0.45,
        "VF":    0.19,
    }

    mapper = OntologyMapper(ontology_dir="ontology/")
    results = mapper.map(fake_model_output)

    print("\n=== OntologyMapper output ===\n")
    for r in results:
        status = "[SUPPRESSED]" if r.is_suppressed else f"[Tier {r.tier}]"
        print(
            f"{status:15s}  {r.label_id:10s}  Pai={r.pai:.3f}  "
            f"Hierarchy={' > '.join(r.hierarchy)}"
        )
        if r.snomed_ct:
            print(f"               SNOMED: {r.snomed_ct}  ICD-10: {r.icd10}")
        print()

    print("\n=== JSON output (first result) ===\n")
    if results:
        print(json.dumps(results[0].to_dict(), indent=2))
        print(json.dumps(results[1].to_dict(), indent=2))