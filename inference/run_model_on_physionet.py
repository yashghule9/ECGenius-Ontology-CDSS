"""
ECGenius — inference/run_model_on_physionet.py
===============================================
Loads the LightV2 checkpoint, processes PhysioNet .mat/.csv ECG files,
and writes model_probabilities.csv ready for the downstream pipeline.

Checkpoint format handled:
  • Folder checkpoint (unzipped .pt): data.pkl + data/0 ... data/143
  • Standard .pt file (torch.save state_dict or full model)
  • TorchScript (.pt saved via torch.jit.save)

ECG formats handled:
  • .mat  — PhysioNet/CinC 2020-2021 format  (key 'val', shape (leads, samples))
  • .csv  — fallback  (rows = leads, cols = samples  OR  rows = samples, cols = leads)

Usage:
    # Real model:
    python inference/run_model_on_physionet.py \\
        --checkpoint models/checkpoints/lightv2_p1_fold2 \\
        --ecg_dir    data/raw/physionet/ecg_waveforms \\
        --output     data/processed/model_probabilities.csv

    # Mock mode (no model, no ECG files needed):
    python inference/run_model_on_physionet.py --mock

    # Single file:
    python inference/run_model_on_physionet.py \\
        --checkpoint models/checkpoints/lightv2_p1_fold2 \\
        --ecg_file   data/raw/physionet/ecg_waveforms/JS00001.mat
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ECGenius.physionet")

# ── Label list (must match label_encoder.json) ────────────────────────────────
LABEL_ENCODER_PATH = PROJECT_ROOT / "data/processed/labels/label_encoder.json"
DEFAULT_LABELS = [
    "AF", "NSR", "STEMI", "NSTEMI", "LVH",
    "VF", "VT", "SVT", "LBBB", "RBBB",
    "Ischemia", "ST_Elevation", "ST_Depression", "TWI", "PVC",
]

# ── Default config ────────────────────────────────────────────────────────────
TARGET_FS      = 500    # Hz — LightV2 trained at 500 Hz
SEGMENT_SEC    = 10.0   # seconds
N_LEADS        = 12
SEGMENT_SAMPLES = int(TARGET_FS * SEGMENT_SEC)  # 5000

# ── LightV2 Architecture ──────────────────────────────────────────────────────

def _build_lightv2(n_labels: int, n_leads: int = 12, base_ch: int = 160):
    """
    Construct LightV2 architecture.

    Design derived from checkpoint metadata:
      - 144 tensor files  →  6 TCN blocks (3 stages × 2 blocks)
      - 1.45 MB           →  ~362K params  →  base_ch ≈ 160
      - Description: multi-scale stem + depthwise-sep TCN + SE + lead attention

    If strict state_dict loading fails, inspect the printed key mismatch
    and adjust the class names to match. The architecture shape is fixed by
    the checkpoint file sizes.
    """
    import torch
    import torch.nn as nn

    class _DSConv(nn.Module):
        """Depthwise-separable conv → BN → GELU."""
        def __init__(self, ch: int, k: int, d: int = 1):
            super().__init__()
            pad = (k - 1) * d // 2
            self.dw  = nn.Conv1d(ch, ch, k, padding=pad, dilation=d, groups=ch, bias=False)
            self.pw  = nn.Conv1d(ch, ch, 1, bias=False)
            self.bn  = nn.BatchNorm1d(ch)
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(self.bn(self.pw(self.dw(x))))

    class _SE(nn.Module):
        """Squeeze-and-Excitation channel attention."""
        def __init__(self, ch: int, r: int = 8):
            super().__init__()
            mid = max(ch // r, 4)
            self.gap  = nn.AdaptiveAvgPool1d(1)
            self.fc1  = nn.Linear(ch, mid)
            self.relu = nn.ReLU()
            self.fc2  = nn.Linear(mid, ch)
            self.sig  = nn.Sigmoid()
        def forward(self, x):
            w = self.gap(x).squeeze(-1)
            w = self.sig(self.fc2(self.relu(self.fc1(w)))).unsqueeze(-1)
            return x * w

    class _TCNBlock(nn.Module):
        """Two DSConv layers with growing dilation + residual + SE."""
        def __init__(self, ch: int, k: int = 7, d: int = 1):
            super().__init__()
            self.conv1 = _DSConv(ch, k, d)
            self.conv2 = _DSConv(ch, k, d * 2)
            self.se    = _SE(ch)
            self.bn    = nn.BatchNorm1d(ch)
        def forward(self, x):
            return x + self.se(self.bn(self.conv2(self.conv1(x))))

    class _LeadAttention(nn.Module):
        """Per-lead importance weighting."""
        def __init__(self, n: int):
            super().__init__()
            self.att = nn.Sequential(
                nn.Conv1d(n, n, 1, groups=n),
                nn.Sigmoid(),
            )
        def forward(self, x):
            return x * self.att(x)

    class _MultiScaleStem(nn.Module):
        """
        Three parallel branches (k=7, 15, 31) capturing short, medium,
        and long temporal patterns, merged by a pointwise conv.
        """
        _BRANCH_CH = 64
        _KERNELS   = (7, 15, 31)

        def __init__(self, ic: int, oc: int):
            super().__init__()
            bch = self._BRANCH_CH
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(ic, bch, k, padding=k // 2, bias=False),
                    nn.BatchNorm1d(bch),
                    nn.GELU(),
                )
                for k in self._KERNELS
            ])
            self.pw = nn.Sequential(
                nn.Conv1d(bch * len(self._KERNELS), oc, 1, bias=False),
                nn.BatchNorm1d(oc),
                nn.GELU(),
            )
        def forward(self, x):
            return self.pw(torch.cat([b(x) for b in self.branches], dim=1))

    class LightV2(nn.Module):
        """
        Full LightV2 classifier.

        Stages with dilations:
          Stage 1: blocks (d=1), (d=2)
          Stage 2: blocks (d=4), (d=8)
          Stage 3: blocks (d=16), (d=32)
        """
        def __init__(self):
            super().__init__()
            ch = base_ch
            self.lead_att = _LeadAttention(n_leads)
            self.stem     = _MultiScaleStem(n_leads, ch)
            # 3 stages × 2 blocks = 6 TCN blocks (matches 144 tensor files)
            self.stage1_b1 = _TCNBlock(ch, 7, 1)
            self.stage1_b2 = _TCNBlock(ch, 7, 2)
            self.stage2_b1 = _TCNBlock(ch, 7, 4)
            self.stage2_b2 = _TCNBlock(ch, 7, 8)
            self.stage3_b1 = _TCNBlock(ch, 7, 16)
            self.stage3_b2 = _TCNBlock(ch, 7, 32)
            self.gap        = nn.AdaptiveAvgPool1d(1)
            self.dropout    = nn.Dropout(p=0.30)
            self.classifier = nn.Linear(ch, n_labels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.lead_att(x)
            x = self.stem(x)
            x = self.stage1_b1(x)
            x = self.stage1_b2(x)
            x = self.stage2_b1(x)
            x = self.stage2_b2(x)
            x = self.stage3_b1(x)
            x = self.stage3_b2(x)
            x = self.gap(x).squeeze(-1)
            x = self.dropout(x)
            return self.classifier(x)

    return LightV2()


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():    return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _rezip_folder(folder: Path) -> str:
    """
    Re-pack an unzipped PyTorch checkpoint folder back into a .pt ZIP.
    PyTorch uses the folder name as the archive prefix inside the ZIP.

    PyTorch ZIP structure:
        {name}/data.pkl
        {name}/data/0, /1, ...  (raw tensor bytes)

    Returns path to the temporary .pt file.
    """
    name = folder.name
    tmp  = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()

    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
        data_pkl = folder / "data.pkl"
        if data_pkl.exists():
            zf.write(data_pkl, f"{name}/data.pkl")

        data_dir = folder / "data"
        if data_dir.is_dir():
            for tensor_file in sorted(data_dir.iterdir()):
                if tensor_file.is_file():
                    zf.write(tensor_file, f"{name}/data/{tensor_file.name}")

        # Fallback: walk everything
        if not data_pkl.exists():
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    zf.write(f, f"{name}/{f.relative_to(folder)}")

    logger.info("Re-zipped folder checkpoint → %s", tmp.name)
    return tmp.name


def load_lightv2(
    checkpoint_path: str | Path,
    n_labels:        int = 15,
    device:          Optional[str] = None,
) -> object:
    """
    Load LightV2 from checkpoint.

    Tries (in order):
      1. Folder checkpoint → re-zip → torch.load (state dict)
      2. .pt file → torch.load as nn.Module (saved with torch.save(model))
      3. .pt file → torch.load as state dict → LightV2(strict=True)
      4. .pt file → torch.load as state dict → LightV2(strict=False, partial)
      5. torch.jit.load (TorchScript)

    Returns nn.Module in eval() mode.
    """
    import torch

    if device is None:
        device = _auto_device()
    ckpt  = Path(checkpoint_path)

    tmp_path = None
    try:
        load_path = str(ckpt)

        # Strategy 0: folder checkpoint — re-zip first
        if ckpt.is_dir():
            tmp_path = _rezip_folder(ckpt)
            load_path = tmp_path

        # Try torch.load
        try:
            obj = torch.load(load_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning("torch.load failed: %s. Trying weights_only=True.", e)
            obj = torch.load(load_path, map_location=device, weights_only=True)

        # Already a Module?
        import torch.nn as nn
        if isinstance(obj, nn.Module):
            logger.info("Loaded as full model object (Strategy 2).")
            obj.eval()
            return obj.to(device)

        # Extract state dict
        if isinstance(obj, dict):
            state_dict = (
                obj.get("model_state_dict") or
                obj.get("state_dict")        or
                obj
            )
            if not isinstance(state_dict, dict):
                raise ValueError("Cannot extract state_dict from checkpoint dict.")

            model = _build_lightv2(n_labels=n_labels).to(device)

            # Strict load
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("State dict loaded strict=True (Strategy 3).")
                model.eval()
                return model
            except RuntimeError as e_strict:
                logger.warning("Strict load failed: %s", e_strict)

            # Non-strict load — log mismatches
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(
                    "Missing keys (%d): %s%s",
                    len(missing), missing[:5],
                    " ..." if len(missing) > 5 else ""
                )
            if unexpected:
                logger.warning(
                    "Unexpected keys (%d): %s%s",
                    len(unexpected), unexpected[:5],
                    " ..." if len(unexpected) > 5 else ""
                )
            logger.info("State dict loaded strict=False (Strategy 4).")
            model.eval()
            return model

    except Exception as e_load:
        logger.warning("torch.load strategies failed: %s. Trying TorchScript.", e_load)

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    # TorchScript fallback
    try:
        import torch
        model = torch.jit.load(str(ckpt), map_location=device)
        model.eval()
        logger.info("Loaded as TorchScript (Strategy 5).")
        return model
    except Exception as e_jit:
        raise RuntimeError(
            f"\n\nAll checkpoint loading strategies failed for: {ckpt}\n"
            f"  TorchScript error: {e_jit}\n\n"
            f"If the model was saved with torch.save(model) (not state_dict),\n"
            f"copy the original model class file into this project and import it\n"
            f"before calling load_lightv2(). Then Strategy 2 will succeed.\n\n"
            f"Alternatively, run with --mock to use simulated probabilities."
        ) from e_jit


# ── ECG loader abstraction ────────────────────────────────────────────────────

class ECGLoader:
    """
    Flexible ECG loader supporting .mat and .csv.

    PhysioNet 2020/2021 .mat format:
      scipy.io.loadmat(path)['val']  →  (n_leads, n_samples)
      Sampling frequency from paired .hea file or assumed 500 Hz.

    CSV fallback:
      rows × cols matrix. Auto-transposes if needed to get (n_leads, n_samples).
    """

    def __init__(self, source_fs: int = 500):
        self.source_fs = source_fs

    def load(self, ecg_path: Path) -> tuple[np.ndarray, int]:
        """
        Returns (signal, fs) where signal.shape = (n_leads, n_samples).
        fs is the detected or assumed sampling frequency.
        """
        ext = ecg_path.suffix.lower()
        if ext == ".mat":
            return self._load_mat(ecg_path)
        elif ext in (".csv", ".txt"):
            return self._load_csv(ecg_path)
        else:
            raise ValueError(f"Unsupported ECG format: {ext}. Use .mat or .csv.")

    def _load_mat(self, path: Path) -> tuple[np.ndarray, int]:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy required for .mat loading: pip install scipy")

        mat  = loadmat(str(path))
        # PhysioNet standard key is 'val'; also try 'data', 'signal', 'ecg'
        data = None
        for key in ("val", "data", "signal", "ecg"):
            if key in mat:
                data = mat[key].astype(np.float32)
                break
        if data is None:
            candidates = [k for k in mat if not k.startswith("_")]
            data = mat[candidates[0]].astype(np.float32)
            logger.warning("No standard key found in .mat; using first array: '%s'", candidates[0])

        # Ensure (n_leads, n_samples)
        if data.ndim == 1:
            data = data[np.newaxis, :]   # single lead
        if data.shape[0] > data.shape[1] and data.shape[0] > 12:
            data = data.T                # was (n_samples, n_leads)

        # Try to read fs from .hea file
        fs = self._parse_hea_fs(path) or self.source_fs
        return data, fs

    def _load_csv(self, path: Path) -> tuple[np.ndarray, int]:
        data = np.genfromtxt(str(path), delimiter=",", dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        # Auto-transpose: assume more columns than rows means (n_leads, n_samples)
        if data.shape[0] > data.shape[1]:
            data = data.T
        return data, self.source_fs

    def _parse_hea_fs(self, mat_path: Path) -> Optional[int]:
        """Read sampling frequency from paired .hea header file."""
        hea = mat_path.with_suffix(".hea")
        if not hea.exists():
            return None
        try:
            first_line = hea.read_text(encoding="utf-8").splitlines()[0]
            parts = first_line.split()
            # WFDB format: recordname n_leads fs n_samples
            if len(parts) >= 3:
                return int(parts[2])
        except Exception:
            pass
        return None


# ── Preprocessing (wraps existing inference/preprocess.py) ───────────────────

def preprocess_ecg(signal: np.ndarray, source_fs: int) -> np.ndarray:
    """
    Run ECGenius preprocessing pipeline on a raw signal.
    Delegates to inference/preprocess.py (Preprocessor class).
    Returns float32 array of shape (12, 5000).
    """
    try:
        from inference.preprocess import Preprocessor
        prep = Preprocessor(target_fs=TARGET_FS, segment_length_sec=SEGMENT_SEC,
                            leads=N_LEADS, source_fs=source_fs)
        return prep.process(signal)
    except ImportError:
        # Inline minimal fallback if preprocess.py unavailable
        logger.warning("inference.preprocess not found — using inline fallback.")
        return _minimal_preprocess(signal, source_fs)


def _minimal_preprocess(signal: np.ndarray, source_fs: int) -> np.ndarray:
    """Minimal fallback: resample + segment + z-score (no filter)."""
    from scipy.signal import resample

    if signal.shape[0] > N_LEADS:
        signal = signal.T
    if signal.shape[0] > N_LEADS:
        signal = signal[:N_LEADS]

    # Resample
    if source_fs != TARGET_FS:
        target_n = int(signal.shape[1] * TARGET_FS / source_fs)
        signal   = resample(signal, target_n, axis=1)

    # Segment
    n = signal.shape[1]
    if n >= SEGMENT_SAMPLES:
        start  = (n - SEGMENT_SAMPLES) // 2
        signal = signal[:, start: start + SEGMENT_SAMPLES]
    else:
        pad    = SEGMENT_SAMPLES - n
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="edge")

    # Z-score per lead
    mean = signal.mean(axis=1, keepdims=True)
    std  = signal.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return ((signal - mean) / std).astype(np.float32)


# ── Model inference ───────────────────────────────────────────────────────────

def run_model(
    model,
    ecg_array: np.ndarray,
    labels:    list[str],
) -> dict[str, float]:
    """
    Forward pass: (12, 5000) numpy → {label: sigmoid_probability}.
    """
    import torch
    tensor = torch.from_numpy(ecg_array).unsqueeze(0)   # (1, 12, 5000)
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    return {label: float(p) for label, p in zip(labels, probs)}


# ── Mock mode ─────────────────────────────────────────────────────────────────

MOCK_PROBS_TEMPLATE = {
    "AF": 0.42, "NSR": 0.61, "STEMI": 0.74, "NSTEMI": 0.31,
    "LVH": 0.38, "VF": 0.19, "VT": 0.14, "SVT": 0.22,
    "LBBB": 0.24, "RBBB": 0.15, "Ischemia": 0.28,
    "ST_Elevation": 0.80, "ST_Depression": 0.35,
    "TWI": 0.18, "PVC": 0.22,
}

def _mock_predict(record_id: str, labels: list[str]) -> dict[str, float]:
    """Generate deterministic mock probabilities (seed = hash of record_id)."""
    rng = np.random.default_rng(abs(hash(record_id)) % (2**32))
    return {
        label: float(np.clip(
            MOCK_PROBS_TEMPLATE.get(label, 0.15) + rng.normal(0, 0.05), 0.01, 0.99
        ))
        for label in labels
    }


# ── Batch processor ───────────────────────────────────────────────────────────

def process_directory(
    ecg_dir:    Path,
    model,
    labels:     list[str],
    mock:       bool       = False,
    source_fs:  int        = 500,
    max_files:  int        = 0,    # 0 = no limit
) -> list[dict]:
    """
    Scan ecg_dir for .mat/.csv files, run inference on each.
    Returns list of result dicts (one per ECG file).
    """
    patterns = list(ecg_dir.glob("*.mat")) + list(ecg_dir.glob("*.csv"))
    if not patterns:
        logger.warning("No .mat or .csv files found in %s", ecg_dir)
        return []

    if max_files > 0:
        patterns = patterns[:max_files]

    loader  = ECGLoader(source_fs=source_fs)
    results = []
    errors  = []

    for idx, ecg_path in enumerate(patterns, 1):
        record_id = ecg_path.stem
        logger.info("[%d/%d]  %s", idx, len(patterns), record_id)

        try:
            if mock:
                probs = _mock_predict(record_id, labels)
            else:
                signal, fs  = loader.load(ecg_path)
                preprocessed = preprocess_ecg(signal, fs)
                probs        = run_model(model, preprocessed, labels)

            row = {"record_id": record_id, "ecg_file": str(ecg_path)}
            row.update(probs)
            results.append(row)

        except Exception as e:
            logger.error("Failed on %s: %s", record_id, e)
            errors.append(record_id)

    logger.info(
        "Processed %d/%d files successfully. %d errors.",
        len(results), len(patterns), len(errors),
    )
    if errors:
        logger.warning("Failed records: %s", errors)

    return results


# ── Output writer ─────────────────────────────────────────────────────────────

def write_probabilities_csv(results: list[dict], output_path: Path) -> None:
    """
    Write model_probabilities.csv.

    Columns: record_id, ecg_file, {label_0}, ..., {label_N}
    One row per ECG recording.
    This CSV is the input for the downstream pipeline.
    """
    if not results:
        logger.warning("No results to write.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Round probabilities to 4 decimal places
            rounded = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in row.items()
            }
            writer.writerow(rounded)

    logger.info("Saved model probabilities → %s  (%d records)", output_path, len(results))


# ── Label loader ──────────────────────────────────────────────────────────────

def _load_labels() -> list[str]:
    if LABEL_ENCODER_PATH.exists():
        with open(LABEL_ENCODER_PATH, encoding="utf-8") as f:
            enc = json.load(f)
        if "labels" in enc:
            return enc["labels"]
        if "idx_to_label" in enc:
            return [enc["idx_to_label"][str(i)] for i in range(len(enc["idx_to_label"]))]
    logger.warning("label_encoder.json not found — using default label list.")
    return DEFAULT_LABELS


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LightV2 on PhysioNet ECG files → model_probabilities.csv"
    )
    parser.add_argument("--checkpoint", type=str,
                        default="models/checkpoints/lightv2_p1_fold2",
                        help="Path to LightV2 checkpoint (file or unzipped folder)")
    parser.add_argument("--ecg_dir", type=str,
                        default="data/raw/physionet/ecg_waveforms",
                        help="Directory containing .mat or .csv ECG files")
    parser.add_argument("--ecg_file", type=str, default=None,
                        help="Single ECG file (overrides --ecg_dir)")
    parser.add_argument("--output", type=str,
                        default="data/processed/model_probabilities.csv")
    parser.add_argument("--source_fs", type=int, default=500,
                        help="Source sampling frequency of input ECGs (default 500)")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Limit number of files processed (0 = all)")
    parser.add_argument("--mock", action="store_true",
                        help="Mock mode: generate synthetic probabilities (no model needed)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu / cuda / mps (auto-detected if omitted)")

    args = parser.parse_args()
    labels = _load_labels()

    # ── Load model ────────────────────────────────────────────────────────
    model = None
    if not args.mock:
        ckpt = PROJECT_ROOT / args.checkpoint
        if not ckpt.exists():
            logger.error(
                "Checkpoint not found: %s\n"
                "  Place the unzipped LightV2 folder at that path,\n"
                "  or run with --mock for synthetic output.", ckpt
            )
            sys.exit(1)
        try:
            model = load_lightv2(str(ckpt), n_labels=len(labels), device=args.device)
            logger.info("LightV2 loaded. Labels: %d", len(labels))
        except RuntimeError as e:
            logger.error("%s", e)
            logger.info("Falling back to mock mode automatically.")
            args.mock = True

    # ── Process ECGs ──────────────────────────────────────────────────────
    if args.ecg_file:
        ecg_path  = PROJECT_ROOT / args.ecg_file
        record_id = ecg_path.stem
        if args.mock:
            probs = _mock_predict(record_id, labels)
        else:
            loader = ECGLoader(args.source_fs)
            signal, fs = loader.load(ecg_path)
            preprocessed = preprocess_ecg(signal, fs)
            probs = run_model(model, preprocessed, labels)
        results = [{"record_id": record_id, "ecg_file": str(ecg_path), **probs}]
    else:
        ecg_dir = PROJECT_ROOT / args.ecg_dir
        results = process_directory(
            ecg_dir    = ecg_dir,
            model      = model,
            labels     = labels,
            mock       = args.mock,
            source_fs  = args.source_fs,
            max_files  = args.max_files,
        )

    # ── Write output ──────────────────────────────────────────────────────
    output_path = PROJECT_ROOT / args.output
    write_probabilities_csv(results, output_path)

    # ── Summary ───────────────────────────────────────────────────────────
    if results:
        print(f"\n{'='*55}")
        print(f"  ECGenius — PhysioNet Model Inference Complete")
        print(f"{'='*55}")
        print(f"  Records processed : {len(results)}")
        print(f"  Output            : {output_path}")
        print(f"  Mode              : {'MOCK' if args.mock else 'REAL MODEL'}")
        print(f"\n  Next step:")
        print(f"  python run_pipeline.py --probabilities {output_path}")
        print()


if __name__ == "__main__":
    main()
