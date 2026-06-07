"""
ECGenius — scripts/validate_data.py
=====================================
Run BEFORE any model inference or training.
Checks signal quality, label distribution, patient split safety,
and file integrity for both PhysioNet and AIIMS data.

Usage:
    python scripts/validate_data.py --source physionet
    python scripts/validate_data.py --source aiims
    python scripts/validate_data.py --source both
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ECGenius.Validate")

LABELS_15 = [
    "AF", "NSR", "STEMI", "NSTEMI", "LVH", "VF", "VT", "SVT",
    "LBBB", "RBBB", "Ischemia", "ST_Elevation", "ST_Depression", "TWI", "PVC",
]
TIER1_LABELS = {"STEMI", "VF", "VT", "NSTEMI", "ST_Elevation"}


def _sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Signal checks ─────────────────────────────────────────────────────────────

def check_mat_file(mat_path: Path) -> dict:
    """Validate one .mat ECG file. Returns dict of findings."""
    try:
        from scipy.io import loadmat
    except ImportError:
        return {"error": "scipy not installed"}

    result = {"file": mat_path.name, "ok": True, "warnings": [], "errors": []}

    try:
        mat = loadmat(str(mat_path))
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"Cannot load: {e}")
        return result

    # Find signal array
    signal = None
    for key in ("val", "data", "signal", "ecg"):
        if key in mat:
            signal = mat[key]
            result["key"] = key
            break

    if signal is None:
        candidates = [k for k in mat if not k.startswith("_")]
        if candidates:
            signal = mat[candidates[0]]
            result["key"] = candidates[0]
            result["warnings"].append(f"Non-standard key used: '{candidates[0]}'")
        else:
            result["ok"] = False
            result["errors"].append("No signal array found in .mat")
            return result

    signal = np.array(signal, dtype=np.float32)

    # Shape check
    result["raw_shape"] = list(signal.shape)
    if signal.ndim == 1:
        result["warnings"].append("Single-lead signal (expected 12-lead)")
    elif signal.ndim == 2:
        r, c = signal.shape
        if r == 12 and c > 100:
            result["shape"] = [r, c]
            result["n_leads"] = r
            result["n_samples"] = c
        elif c == 12 and r > 100:
            result["shape"] = [c, r]
            result["n_leads"] = c
            result["n_samples"] = r
            result["warnings"].append("Transposed — will auto-fix (n_samples, 12) → (12, n_samples)")
        else:
            result["warnings"].append(f"Unexpected shape: {signal.shape}")
    else:
        result["errors"].append(f"Unexpected ndim={signal.ndim}")
        result["ok"] = False
        return result

    n_samples = result.get("n_samples", signal.shape[-1])

    # Sampling frequency from .hea file
    hea = mat_path.with_suffix(".hea")
    if hea.exists():
        try:
            line = hea.read_text(encoding="utf-8").splitlines()[0]
            fs   = int(line.split()[2])
            result["fs_hz"] = fs
            duration = n_samples / fs
            result["duration_sec"] = round(duration, 1)
            if duration < 8.0:
                result["warnings"].append(f"Short recording: {duration:.1f}s (need ≥10s)")
            if fs not in (250, 500, 1000):
                result["warnings"].append(f"Unusual sampling rate: {fs} Hz")
        except Exception:
            result["warnings"].append("Could not parse .hea — assuming 500 Hz")
            result["fs_hz"] = 500
    else:
        result["warnings"].append("No .hea file — assuming 500 Hz")
        result["fs_hz"] = 500

    # Signal quality
    if result.get("n_leads"):
        arr = signal if signal.shape[0] == 12 else signal.T
        flat_leads   = [i for i in range(12) if np.ptp(arr[i]) < 0.01]
        nan_leads    = [i for i in range(12) if np.any(np.isnan(arr[i]))]
        inf_leads    = [i for i in range(12) if np.any(np.isinf(arr[i]))]
        clipped_leads= [i for i in range(12) if np.ptp(arr[i]) > 50000]

        if flat_leads:
            result["warnings"].append(f"Flat leads: {flat_leads}")
        if nan_leads:
            result["errors"].append(f"NaN values in leads: {nan_leads}")
            result["ok"] = False
        if inf_leads:
            result["errors"].append(f"Inf values in leads: {inf_leads}")
            result["ok"] = False
        if clipped_leads:
            result["warnings"].append(f"Possibly clipped (huge amplitude) leads: {clipped_leads}")

        result["amplitude_range"] = [
            round(float(arr.min()), 3),
            round(float(arr.max()), 3),
        ]

    return result


def validate_ecg_directory(ecg_dir: Path, max_check: int = 50) -> None:
    """Scan directory and validate up to max_check .mat files."""
    mat_files = sorted(ecg_dir.glob("*.mat"))
    if not mat_files:
        print(f"  WARNING: No .mat files found in {ecg_dir}")
        return

    print(f"  Found {len(mat_files)} .mat files in {ecg_dir}")
    print(f"  Checking first {min(max_check, len(mat_files))} files...\n")

    ok_count  = 0
    warn_count = 0
    err_count  = 0
    all_fs     = []
    all_durations = []

    for mat_path in mat_files[:max_check]:
        r = check_mat_file(mat_path)
        if r.get("errors"):
            print(f"  ERROR  {r['file']:30s}  {r['errors']}")
            err_count += 1
        elif r.get("warnings"):
            print(f"  WARN   {r['file']:30s}  {r['warnings']}")
            warn_count += 1
        else:
            ok_count += 1
        if "fs_hz" in r:
            all_fs.append(r["fs_hz"])
        if "duration_sec" in r:
            all_durations.append(r["duration_sec"])

    print(f"\n  Summary: {ok_count} OK  |  {warn_count} warnings  |  {err_count} errors")
    if all_fs:
        unique_fs = set(all_fs)
        print(f"  Sampling rates found: {unique_fs}")
        if len(unique_fs) > 1:
            print(f"  WARNING: Mixed sampling rates! Preprocessor handles this but verify.")
    if all_durations:
        print(f"  Duration: min={min(all_durations):.1f}s  max={max(all_durations):.1f}s  "
              f"mean={sum(all_durations)/len(all_durations):.1f}s")


# ── Label checks ──────────────────────────────────────────────────────────────

def check_annotations(annotations_path: Path) -> dict:
    """Validate annotations.csv for label distribution and completeness."""
    if not annotations_path.exists():
        print(f"  MISSING: {annotations_path}")
        return {}

    rows = []
    with open(annotations_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"  EMPTY: {annotations_path}")
        return {}

    print(f"  Records in annotations: {len(rows)}")

    # Which label columns exist?
    header = list(rows[0].keys())
    found_labels   = [l for l in LABELS_15 if l in header]
    missing_labels = [l for l in LABELS_15 if l not in header]

    print(f"  Labels found ({len(found_labels)}): {found_labels}")
    if missing_labels:
        print(f"  Labels MISSING from CSV: {missing_labels}")
        print(f"  --> Add these columns with 0/1 values")

    # Label distribution
    label_counts = {l: 0 for l in found_labels}
    multi_label  = 0
    no_label     = 0

    for row in rows:
        positives = [l for l in found_labels if row.get(l, "0") == "1"]
        for l in positives:
            label_counts[l] += 1
        if len(positives) > 1:
            multi_label += 1
        if len(positives) == 0:
            no_label += 1

    print(f"\n  Label distribution:")
    total = len(rows)
    for label in LABELS_15:
        if label in label_counts:
            count = label_counts[label]
            pct   = 100 * count / total
            bar   = "█" * int(pct / 2)
            tier1 = " ← TIER-1" if label in TIER1_LABELS else ""
            print(f"    {label:16s}  {count:5d}  ({pct:5.1f}%)  {bar}{tier1}")

    print(f"\n  Multi-label records: {multi_label} ({100*multi_label/total:.1f}%)")
    print(f"  Unlabelled records:  {no_label}")

    # Minimum count check for rare labels
    for label in TIER1_LABELS:
        if label in label_counts and label_counts[label] < 30:
            print(f"  WARNING: {label} has only {label_counts[label]} examples.")
            print(f"    --> AUC for this label will be unreliable. Need ≥30 positives.")

    return label_counts


# ── Patient split check ───────────────────────────────────────────────────────

def check_patient_leakage(
    annotations_path: Path,
    metadata_path:    Path,
    split_dir:        Path,
) -> None:
    """
    Verify that no patient appears in both train and test splits.
    This is the most critical check for publication credibility.
    """
    if not metadata_path.exists():
        print(f"  SKIP: metadata.csv not found at {metadata_path}")
        return

    # Load patient → record mapping
    patient_to_records: dict[str, list[str]] = defaultdict(list)
    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("patient_id", row.get("record_id", ""))
            rid = row.get("record_id", pid)
            patient_to_records[pid].append(rid)

    print(f"  Unique patients: {len(patient_to_records)}")
    multi = {p: r for p, r in patient_to_records.items() if len(r) > 1}
    if multi:
        print(f"  Patients with multiple recordings: {len(multi)}")
        print(f"  --> These must be kept together in one split (no leakage)")
    else:
        print(f"  All patients have 1 recording (no multi-record leakage risk)")

    # Check split files
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}_ids.txt"
        if path.exists():
            ids = set(path.read_text().strip().splitlines())
            print(f"  {split}_ids.txt: {len(ids)} records")

    # Check for overlap between train and test
    train_path = split_dir / "train_ids.txt"
    test_path  = split_dir / "test_ids.txt"
    if train_path.exists() and test_path.exists():
        train_ids = set(train_path.read_text().strip().splitlines())
        test_ids  = set(test_path.read_text().strip().splitlines())
        overlap   = train_ids & test_ids
        if overlap:
            print(f"\n  CRITICAL LEAKAGE DETECTED: {len(overlap)} records in both train+test!")
            print(f"  Samples: {list(overlap)[:5]}")
            print(f"  --> Rebuild splits using patient_id-level stratification.")
        else:
            print(f"  No record-level leakage (train ∩ test = ∅)  ✓")

        # Check patient-level leakage
        record_to_patient = {r: p for p, records in patient_to_records.items()
                             for r in records}
        train_patients = {record_to_patient.get(r, r) for r in train_ids}
        test_patients  = {record_to_patient.get(r, r) for r in test_ids}
        patient_overlap = train_patients & test_patients
        if patient_overlap:
            print(f"  PATIENT-LEVEL LEAKAGE: {len(patient_overlap)} patients in both splits!")
            print(f"  --> This inflates AUC. Rebuild with patient_id-level split.")
        else:
            print(f"  No patient-level leakage  ✓")


# ── History CSV check ─────────────────────────────────────────────────────────

def check_patient_history(history_path: Path) -> None:
    """Validate patient_history.csv format for the history encoder."""
    if not history_path.exists():
        print(f"  INFO: No patient_history.csv yet at {history_path}")
        print(f"  --> Run scripts/generate_mock_aiims_data.py for testing")
        print(f"  --> Or fill in with real AIIMS symptom data when available")
        return

    rows = []
    with open(history_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"  Patient history records: {len(rows)}")
    header = list(rows[0].keys()) if rows else []

    sym_cols  = [k for k in header if k.startswith("sym_")]
    rf_cols   = [k for k in header if k.startswith("rf_")]
    vital_cols = [k for k in header if k in ("sbp","dbp","hr","spo2","rr")]

    print(f"  Symptom columns ({len(sym_cols)}): {sym_cols}")
    print(f"  Risk factor columns ({len(rf_cols)}): {rf_cols}")
    print(f"  Vital columns ({len(vital_cols)}): {vital_cols}")

    # Missing value rates
    print(f"\n  Missing value rates:")
    for col in sym_cols[:5] + vital_cols:
        missing = sum(1 for r in rows if r.get(col, "") == "") / max(len(rows), 1)
        flag = " ← HIGH" if missing > 0.30 else ""
        print(f"    {col:20s}  {missing*100:5.1f}% missing{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Validate ECGenius data files")
    parser.add_argument("--source", choices=["physionet", "aiims", "both"],
                        default="both")
    parser.add_argument("--max-check", type=int, default=50,
                        help="Max number of .mat files to individually inspect")
    args = parser.parse_args()

    sources = ["physionet", "aiims"] if args.source == "both" else [args.source]

    for source in sources:
        _sep(f"Validating {source.upper()} Data")

        ecg_dir      = PROJECT_ROOT / "data/raw" / source / "ecg_waveforms"
        annotations  = PROJECT_ROOT / "data/raw" / source / "annotations.csv"
        metadata     = PROJECT_ROOT / "data/raw" / source / "metadata.csv"
        history      = PROJECT_ROOT / "data/raw" / "aiims" / "patient_history.csv"
        split_dir    = PROJECT_ROOT / "data/processed/splits"

        print(f"\n  [1] ECG Signal Files")
        validate_ecg_directory(ecg_dir, max_check=args.max_check)

        print(f"\n  [2] Label Annotations")
        check_annotations(annotations)

        print(f"\n  [3] Patient Split Integrity")
        check_patient_leakage(annotations, metadata, split_dir)

        if source == "aiims":
            print(f"\n  [4] Patient History (Symptoms + Vitals)")
            check_patient_history(history)

    _sep("Validation Complete")
    print("  Fix all ERROR items before proceeding.")
    print("  WARNING items should be investigated but may be acceptable.")
    print()


if __name__ == "__main__":
    main()
