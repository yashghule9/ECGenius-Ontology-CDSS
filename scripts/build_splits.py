"""
ECGenius — scripts/build_splits.py
====================================
Builds patient-level train/val/test splits and multilabel_targets.csv.
Must be run ONCE on real data. Results go to data/processed/.

Split strategy: 70% train / 15% val / 15% test
  - Split on PATIENT ID (not record ID) to prevent leakage
  - Stratify by most common label to preserve class balance
  - Rare Tier-1 labels guaranteed in test set

Usage:
    python scripts/build_splits.py \\
        --annotations data/raw/physionet/annotations.csv \\
        --metadata    data/raw/physionet/metadata.csv \\
        --source      physionet

    # AIIMS:
    python scripts/build_splits.py \\
        --annotations data/raw/aiims/annotations.csv \\
        --metadata    data/raw/aiims/metadata.csv \\
        --source      aiims

    # Combined (recommended for paper):
    python scripts/build_splits.py --combined
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LABELS_15 = [
    "AF", "NSR", "STEMI", "NSTEMI", "LVH", "VF", "VT", "SVT",
    "LBBB", "RBBB", "Ischemia", "ST_Elevation", "ST_Depression", "TWI", "PVC",
]
TIER1_LABELS = {"STEMI", "VF", "VT", "NSTEMI"}


def load_annotations(path: Path, source_tag: str = "") -> list[dict]:
    """Load annotations CSV. Each row = one ECG record."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if source_tag:
                row["_source"] = source_tag
            rows.append(row)
    return rows


def load_metadata(path: Path) -> dict[str, dict]:
    """patient_id → metadata dict."""
    meta = {}
    if not path.exists():
        return meta
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("patient_id", row.get("record_id", ""))
            meta[pid] = row
    return meta


def get_primary_label(row: dict) -> str:
    """Get the highest-confidence label for stratification."""
    for label in TIER1_LABELS:    # prioritise Tier-1
        if row.get(label, "0") == "1":
            return label
    for label in LABELS_15:
        if row.get(label, "0") == "1":
            return label
    return "UNKNOWN"


def patient_level_split(
    records:   list[dict],
    metadata:  dict[str, dict],
    val_frac:  float = 0.15,
    test_frac: float = 0.15,
    seed:      int   = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split record IDs at the patient level.
    Returns (train_ids, val_ids, test_ids).
    """
    rng = np.random.default_rng(seed)

    # Map: patient_id → list of record_ids
    # If metadata has patient_id different from record_id, use that.
    # Otherwise, treat record_id as patient_id (1 record per patient).
    patient_to_records: dict[str, list[str]] = defaultdict(list)
    record_to_label:    dict[str, str]       = {}

    for row in records:
        rid = row.get("record_id", row.get("patient_id", ""))
        pid = metadata.get(rid, {}).get("patient_id", rid)
        patient_to_records[pid].append(rid)
        record_to_label[rid] = get_primary_label(row)

    patient_ids = list(patient_to_records.keys())
    rng.shuffle(patient_ids)

    # Primary label for each patient (for stratified split)
    patient_label = {
        pid: record_to_label.get(patient_to_records[pid][0], "NSR")
        for pid in patient_ids
    }

    # Group by label
    label_groups: dict[str, list[str]] = defaultdict(list)
    for pid in patient_ids:
        label_groups[patient_label[pid]].append(pid)

    train_pids: list[str] = []
    val_pids:   list[str] = []
    test_pids:  list[str] = []

    # Stratified split per label group
    for label, pids in label_groups.items():
        rng.shuffle(pids)
        n      = len(pids)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        # Tier-1 must appear in test set — guarantee at least 1
        if label in TIER1_LABELS and n >= 3:
            test_pids.extend(pids[:n_test])
            val_pids.extend(pids[n_test: n_test + n_val])
            train_pids.extend(pids[n_test + n_val:])
        else:
            test_pids.extend(pids[:n_test])
            val_pids.extend(pids[n_test: n_test + n_val])
            train_pids.extend(pids[n_test + n_val:])

    # Convert back to record IDs
    def pids_to_rids(pids: list[str]) -> list[str]:
        return [rid for pid in pids for rid in patient_to_records[pid]]

    return pids_to_rids(train_pids), pids_to_rids(val_pids), pids_to_rids(test_pids)


def build_multilabel_targets(records: list[dict], split_ids: list[str]) -> list[dict]:
    """Build multilabel_targets.csv rows for a given split."""
    record_dict = {r.get("record_id", r.get("patient_id", "")): r for r in records}
    targets = []
    for rid in split_ids:
        if rid not in record_dict:
            continue
        row = record_dict[rid]
        target = {"patient_id": rid}
        for label in LABELS_15:
            target[label] = row.get(label, "0")
        targets.append(target)
    return targets


def write_split_file(ids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"  Saved {len(ids):5d} IDs → {path}")


def write_targets_csv(targets: list[dict], path: Path) -> None:
    if not targets:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
        writer.writeheader()
        writer.writerows(targets)
    print(f"  Saved {len(targets):5d} rows → {path}")


def print_split_stats(
    train_ids: list[str],
    val_ids:   list[str],
    test_ids:  list[str],
    record_to_label: dict[str, str],
) -> None:
    from collections import Counter
    print(f"\n  Split sizes: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")
    print(f"\n  {'Label':<16} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"  {'-'*40}")

    all_splits = [
        ("train", train_ids),
        ("val",   val_ids),
        ("test",  test_ids),
    ]
    label_counts = {}
    for split_name, ids in all_splits:
        label_counts[split_name] = Counter(record_to_label.get(i, "?") for i in ids)

    for label in LABELS_15:
        t = label_counts["train"].get(label, 0)
        v = label_counts["val"].get(label, 0)
        ts = label_counts["test"].get(label, 0)
        tier = " TIER-1" if label in TIER1_LABELS else ""
        print(f"  {label:<16} {t:>7} {v:>7} {ts:>7}{tier}")


def main():
    parser = argparse.ArgumentParser(description="Build patient-level splits")
    parser.add_argument("--annotations", type=str,
                        default="data/raw/physionet/annotations.csv")
    parser.add_argument("--metadata",    type=str,
                        default="data/raw/physionet/metadata.csv")
    parser.add_argument("--source",      type=str, default="physionet",
                        choices=["physionet", "aiims"])
    parser.add_argument("--combined",    action="store_true",
                        help="Merge PhysioNet + AIIMS annotations for combined split")
    parser.add_argument("--val-frac",    type=float, default=0.15)
    parser.add_argument("--test-frac",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data/processed"

    if args.combined:
        pn_ann  = PROJECT_ROOT / "data/raw/physionet/annotations.csv"
        ai_ann  = PROJECT_ROOT / "data/raw/aiims/annotations.csv"
        pn_meta = PROJECT_ROOT / "data/raw/physionet/metadata.csv"
        ai_meta = PROJECT_ROOT / "data/raw/aiims/metadata.csv"

        records  = []
        metadata = {}
        if pn_ann.exists():
            records  += load_annotations(pn_ann, "physionet")
            metadata.update(load_metadata(pn_meta))
        if ai_ann.exists():
            records  += load_annotations(ai_ann, "aiims")
            metadata.update(load_metadata(ai_meta))
    else:
        ann_path  = PROJECT_ROOT / args.annotations
        meta_path = PROJECT_ROOT / args.metadata
        records   = load_annotations(ann_path, args.source)
        metadata  = load_metadata(meta_path)

    print(f"\n  Total records loaded: {len(records)}")

    record_to_label = {
        r.get("record_id", r.get("patient_id", "")): get_primary_label(r)
        for r in records
    }

    train_ids, val_ids, test_ids = patient_level_split(
        records, metadata, args.val_frac, args.test_frac, args.seed
    )

    print_split_stats(train_ids, val_ids, test_ids, record_to_label)

    # Save split files
    print("\n  Writing split files...")
    write_split_file(train_ids, out_dir / "splits/train_ids.txt")
    write_split_file(val_ids,   out_dir / "splits/val_ids.txt")
    write_split_file(test_ids,  out_dir / "splits/test_ids.txt")

    # Build and save targets CSV
    print("\n  Building multilabel_targets.csv...")
    all_targets = build_multilabel_targets(records, train_ids + val_ids + test_ids)
    write_targets_csv(all_targets, out_dir / "labels/multilabel_targets.csv")

    print("\n  DONE. Verify:")
    print("  1. No patient appears in both train and test (check patient_level_split)")
    print("  2. Tier-1 labels present in test set")
    print("  3. Run: python scripts/validate_data.py to re-check splits")


if __name__ == "__main__":
    main()
