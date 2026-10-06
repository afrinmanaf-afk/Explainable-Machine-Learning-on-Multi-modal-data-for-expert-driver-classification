# =============================================================================
# ADAPTIVE COGNITIVE PRESSURE FIELD (A-CPF) + KOOPMAN + DFCG
# Local VS Code Edition —  v7.0  (5-Fold StratifiedGroupKFold CV)
# =============================================================================
#
# This version runs entirely locally.  Kaggle-specific auto-cloning and
# auto-install have been removed.  Set RAW_DATA_PATH and OUTPUT_DIR below.
#
# CHANGES vs v6.0.1 (kept):
#   [FIX-01]  DataSynchronizer: global feature schema built before segment concat
#   [FIX-02]  CognitiveDataset: multi-subject concat fills missing columns with 0.0
#   [FIX-03]  fold_cfg deep copy to avoid global CONFIG mutation
#   [FIX-04]  NaN/Inf guards at all stages
#   [FIX-05]  Koopman spectral-radius clipping
#   [FIX-06]  CombinedLoss initialised on correct device, p=2 for Frobenius
#   [FIX-07]  DFCG num_nodes tied to latent_dim
#   [FIX-08]  _autocast safe on PyTorch 1.x / CPU
#   [FIX-09]  DataLoader persistent_workers/pin_memory guards
#   [FIX-10]  _norm() NaN guard
#   [FIX-11]  scheduler phase-aligned with warmup
#   [FIX-12]  CPF inference batched with explicit memory cleanup
#   [FIX-13]  DataLoader worker cleanup before trainer.free()
#   [FIX-14]  finite-loss abort per batch
#   [FIX-15]  gradient validation: zero non-finite grads
#   [FIX-16]  _load_csv duplicate index rows dropped
#   [FIX-17]  resample tolerance relative to median spacing
#   [FIX-18]  make_subject_folds uses integer groups
#   [FIX-19]  extended diagnostics
#   [FIX-20]  CognitiveDataset skips corrupted samples
#   [FIX-21]  reproducibility seeds per fold
# =============================================================================

import os, sys, subprocess, importlib, importlib.util, warnings, contextlib, gc
import copy, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# =============================================================================
# LOCAL ENVIRONMENT SETUP (VS Code) – REPLACE Kaggle logic
# =============================================================================

# Disable Kaggle detection entirely
IN_KAGGLE = False

# -------- SET YOUR DATASET PATH HERE -----------------------------------------
RAW_DATA_PATH = str(Path("./data/raw/ExpertDrivingDataset/Dataset").resolve())
# -----------------------------------------------------------------------------

OUTPUT_DIR = "./outputs"

# Create output subdirectories
for sub in ["checkpoints", "figures", "logs", "processed", "results", "synchronized"]:
    Path(OUTPUT_DIR, sub).mkdir(parents=True, exist_ok=True)

# -------- NO AUTO-INSTALL (you install dependencies manually) ----------------

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.signal import welch
from scipy.stats import entropy, mannwhitneyu

# ── reproducibility seed ────────────────────────────────────────────────────
_GLOBAL_SEED = 42

def _set_seed(seed: int = _GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_set_seed()

# ── torch version ────────────────────────────────────────────────────────────
_TORCH_MAJOR = int(torch.__version__.split(".")[0])
_TORCH_MINOR = int(torch.__version__.split(".")[1].split("+")[0]
                   if len(torch.__version__.split(".")) > 1 else "0")

# ── AMP utilities ────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _autocast(enabled: bool):
    """
    Safe autocast that works on PyTorch 1.x and 2.x, CPU and CUDA.
    [FIX-08]
    """
    if not enabled or not torch.cuda.is_available():
        yield
        return
    if _TORCH_MAJOR >= 2:
        with torch.amp.autocast("cuda", enabled=True):
            yield
    else:
        with torch.cuda.amp.autocast(enabled=True):
            yield


def _make_scaler(use_amp: bool):
    if not use_amp:
        return None
    if _TORCH_MAJOR >= 2:
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def _safe_load(path, device):
    if _TORCH_MAJOR >= 2:
        return torch.load(path, map_location=device, weights_only=True)
    return torch.load(path, map_location=device)


try:
    import mne;          MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
try:
    import neurokit2 as nk; NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False   # benchmark=True requires this

_PIN_MEMORY = torch.cuda.is_available()

# =============================================================================
# 2. ENVIRONMENT & PATH DETECTION – LOCAL ONLY (no cloning)
# =============================================================================
# We no longer auto-clone.  If the dataset is not at RAW_DATA_PATH, raise an error.

def find_dataset_path() -> str:
    if os.path.exists(RAW_DATA_PATH):
        return RAW_DATA_PATH
    # Optionally allow user override via environment variable
    env_path = os.environ.get("EXPERT_DRIVING_DATA_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    raise FileNotFoundError(
        f"Dataset not found at {RAW_DATA_PATH}. "
        f"Please place the 'Dataset' folder there or set EXPERT_DRIVING_DATA_PATH."
    )

RAW_DATA_PATH = find_dataset_path()
print(f"Using dataset at: {RAW_DATA_PATH}")

# =============================================================================
# 3. NaN / Inf GUARD UTILITIES   [FIX-04]
# =============================================================================

def _check_finite_df(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Replace Inf with NaN, then report and forward-fill NaN columns."""
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_counts = df.isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        print(f"  [NaN-WARN] {tag}: {len(bad)} cols with NaN "
              f"(total {int(bad.sum())} cells) — forward/back filling.")
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df


def _assert_finite_tensor(t: torch.Tensor, tag: str, abort: bool = True) -> bool:
    """Return True if finite, False otherwise.  If abort=True, raise RuntimeError."""
    if torch.isfinite(t).all():
        return True
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    msg = (f"[NON-FINITE] {tag}: NaN={n_nan}, Inf={n_inf}, "
           f"shape={tuple(t.shape)}")
    if abort:
        raise RuntimeError(msg)
    print(f"  {msg}")
    return False


def _safe_tensor(t: torch.Tensor, tag: str = "") -> torch.Tensor:
    """Replace NaN/Inf with 0 in-place (non-aborting)."""
    if not torch.isfinite(t).all():
        n_bad = (~torch.isfinite(t)).sum().item()
        print(f"  [CLAMP] {tag}: replacing {n_bad} non-finite values with 0.")
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def _gpu_mem_info() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        res   = torch.cuda.memory_reserved()  / 1e9
        return f"GPU mem: alloc={alloc:.2f}GB  reserved={res:.2f}GB"
    return "GPU: not available"

# =============================================================================
# 4. SCALE / LABEL UTILITIES
# =============================================================================

def diagnose_scale_csv(raw_path: str):
    scale_dir = Path(raw_path) / "3-Driver" / "5-Scale"
    for fname in ["NASA-TLX.csv", "Stress.csv", "DES.csv", "SAM.csv"]:
        f = scale_dir / fname
        if f.exists():
            df = pd.read_csv(f)
            print(f"  [{fname}]  shape={df.shape}  cols={df.columns.tolist()}")
            print(df.head(3).to_string(index=False))
        else:
            print(f"  [{fname}]  NOT FOUND at {f}")


def _load_scale_csv(filepath: Path) -> Optional[pd.DataFrame]:
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        print(f"  [WARN] Could not read {filepath.name}: {exc}")
        return None

    df.columns = [str(c).strip() for c in df.columns]

    sid_col = next(
        (c for c in df.columns
         if any(k in c.lower() for k in ["subject", "participant", "id", "driver"])),
        df.columns[0])

    scen_col = next(
        (c for c in df.columns
         if any(k in c.lower() for k in ["scenario", "condition", "task", "scene"])),
        None)

    val_col = next(
        (c for c in df.columns
         if c not in (sid_col, scen_col)
         and pd.api.types.is_numeric_dtype(df[c])),
        None)

    if scen_col and val_col:
        out = df[[sid_col, scen_col, val_col]].copy()
        out.columns = ["subject_id", "scenario", "value"]
    else:
        value_cols = [c for c in df.columns if c != sid_col]
        out = df.melt(id_vars=[sid_col], value_vars=value_cols,
                      var_name="scenario", value_name="value")
        out.columns = ["subject_id", "scenario", "value"]

    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    out["scenario"]   = out["scenario"].astype(str).str.strip()
    out["value"]      = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


def load_all_scale_labels(raw_path: str) -> Dict[str, pd.DataFrame]:
    scale_dir = Path(raw_path) / "3-Driver" / "5-Scale"
    files = {
        "NASA_TLX": scale_dir / "NASA-TLX.csv",
        "Stress":   scale_dir / "Stress.csv",
        "DES":      scale_dir / "DES.csv",
        "SAM":      scale_dir / "SAM.csv",
    }
    result = {}
    for name, path in files.items():
        df = _load_scale_csv(path)
        if df is not None and not df.empty:
            result[name] = df
            print(f"  [SCALE] Loaded '{name}' — {len(df)} rows")
        else:
            print(f"  [SCALE] '{name}' not available ({path.name})")
    return result

# =============================================================================
# 5. CONFIGURATION
# =============================================================================
TRAFFIC_COL_MAP: Dict[str, List[str]] = {
    "density":                 ["density", "VehicleDensity", "traffic_density",
                                "Density", "DENSITY", "trafficDensity"],
    "vehicle_count":           ["vehicle_count", "VehicleCount", "num_vehicles",
                                "NumVehicles", "count"],
    "road_complexity":         ["road_complexity", "RoadComplexity", "RoadType",
                                "road_type", "complexity", "Complexity"],
    "intersection_complexity": ["intersection_complexity", "IntersectionComplexity",
                                "IntersectionType", "intersection_type"],
}

ALL_SCENARIOS = [
    "1_EnterAuxiliaryStreet", "2_EnterMainStreet", "3_U-Turn",
    "4_StraightDrive",        "5_RightTurn-P1",    "6_RightTurn-P2",
    "7_LeftTurn-1",           "8_LeftTurn-2",      "9_RightTurn-1",
    "10_RightTurn-2",         "11_RightTurn-3",    "12_RightTurn-4",
    "13_LeftTurn-3",
]
BASELINE_SCENARIO = "0_Baseline"

# Known label column prefixes — used to exclude them from feature detection
_LABEL_PREFIXES = frozenset(["NASA_TLX", "Stress", "DES", "SAM",
                              "nasa_tlx", "stress", "des", "sam"])

CONFIG: Dict = {
    "data": {
        "raw_path":          RAW_DATA_PATH,
        "synchronized_path": str(Path(OUTPUT_DIR) / "synchronized"),
        "processed_path":    str(Path(OUTPUT_DIR) / "processed"),
        "subjects": (
            [f"E{i:02d}" for i in range(1, 11)] +
            [f"N{i:02d}" for i in range(1, 11)]
        ),
        "scenarios":        ALL_SCENARIOS,
        "include_baseline": True,
        "common_fs":        100,
        "label_tlx_keywords":    ["nasa_tlx", "tlx", "workload", "mental",
                                   "rating", "load", "nasa"],
        "label_stress_keywords": ["stress", "arousal", "anxiety", "pressure"],
    },
    "feature_extraction": {
        "eeg":  {"bands": {"theta":[4,8],"alpha":[8,13],
                           "beta":[13,30],"gamma":[30,45]},
                 "window": 2.0, "step": 1.0},
        "eda":  {"window": 2.0, "step": 1.0},
        "bvp":  {"window": 2.0, "step": 1.0},
        "can":  {"features": ["speed","acceleration","brake",
                              "steering_angle","jerk","lane_deviation"]},
        "traffic": {"density_window": 10},
        "eyetrack": {"features": ["pupil_left","pupil_right","gaze_x","gaze_y",
                                   "fixation_duration","saccade_amplitude"]},
    },
    "models": {
        "acpf":    {"input_dim": None, "hidden_dim": 256,
                    "output_dim": 1, "fusion": "gated", "dropout": 0.15},
        "koopman": {"input_dim": None, "latent_dim": 24,
                    "encoder_hidden": [128, 64], "decoder_hidden": [64, 128],
                    "forecast_steps": [5, 10, 15]},
        "dfcg":    {"num_nodes": 24},   # must equal koopman.latent_dim
    },
    "training": {
        "batch_size":              64,
        "epochs":                  60,
        "cv_epochs":                30,
        "learning_rate":           3e-4,
        "weight_decay":            1e-4,
        "mixed_precision":         True,
        "early_stopping_patience": 10,
        "cv_early_stopping_patience": 6,
        "grad_accum_steps":        4,
        "warmup_epochs":           3,
        "noise_std":               0.02,
        "scheduler_T0":            15,
        "checkpoint_dir": str(Path(OUTPUT_DIR) / "checkpoints"),
        "log_dir":        str(Path(OUTPUT_DIR) / "logs"),
        "sequence_length": 30,
        "n_folds":          5,
        "cv_seed":          42,
        # Safety: abort if this fraction of batches in an epoch are non-finite
        "nan_batch_abort_frac": 0.10,
    },
    "loss": {
        "lambda_koopman":  0.1,
        "lambda_graph":    0.05,
        "lambda_koop_reg": 0.01,
    },
    "evaluation": {"horizons": [5, 10, 15]},
}

# Enforce consistency: dfcg.num_nodes must equal koopman.latent_dim
assert CONFIG["models"]["dfcg"]["num_nodes"] == CONFIG["models"]["koopman"]["latent_dim"], (
    "dfcg.num_nodes must equal koopman.latent_dim for DFCG to operate on Koopman latents.")

for _d in [
    CONFIG["data"]["synchronized_path"],
    CONFIG["data"]["processed_path"],
    CONFIG["training"]["checkpoint_dir"],
    CONFIG["training"]["log_dir"],
    str(Path(OUTPUT_DIR) / "figures"),
    str(Path(OUTPUT_DIR) / "results"),
]:
    Path(_d).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 6. DATA SYNCHRONISATION
# =============================================================================

class DataSynchronizer:
    def __init__(self, config: Dict, scale_labels: Dict[str, pd.DataFrame]):
        self.raw_path    = Path(config["data"]["raw_path"])
        self.sync_path   = Path(config["data"]["synchronized_path"])
        self.common_fs   = config["data"]["common_fs"]
        self.scenarios   = config["data"]["scenarios"]
        self.include_bl  = config["data"].get("include_baseline", False)
        self.scale_labels = scale_labels
        self.sync_path.mkdir(parents=True, exist_ok=True)

    def synchronize_subject(self, subject_id: str) -> pd.DataFrame:
        print(f"  Synchronising {subject_id} ...")
        driver_type = "ExpertDriver" if subject_id.startswith("E") else "NoviceDriver"

        all_scen = (([BASELINE_SCENARIO] if self.include_bl else []) +
                    self.scenarios)
        all_segments: List[pd.DataFrame] = []

        eda_timediv = self._load_timedivision(
            self.raw_path / "3-Driver" / "3-EDA" / "Timestamp"
            / f"{subject_id}-timedivision.csv")
        bvp_timediv = self._load_timedivision(
            self.raw_path / "3-Driver" / "4-BVP" / "Timestamp"
            / f"{subject_id}-timedivision.csv")
        timediv = eda_timediv if eda_timediv is not None else bvp_timediv

        eda_full = self._load_csv(
            self.raw_path / "3-Driver" / "3-EDA" / driver_type / f"{subject_id}.csv")
        bvp_full = self._load_csv(
            self.raw_path / "3-Driver" / "4-BVP" / driver_type / f"{subject_id}.csv")

        for scenario in all_scen:
            start_time, end_time = self._get_time_window(timediv, scenario)
            data_dict: Dict[str, pd.DataFrame] = {}

            traf_f = self.raw_path/"1-TrafficRecorder"/scenario/driver_type/f"{subject_id}.csv"
            if traf_f.exists():
                data_dict["Traffic"] = self._load_csv(traf_f, start_time, end_time)

            can_f = self.raw_path/"2-CANBus"/scenario/driver_type/f"{subject_id}.csv"
            if can_f.exists():
                data_dict["CAN"] = self._load_csv(can_f, start_time, end_time)

            if eda_full is not None and not eda_full.empty:
                data_dict["EDA"] = self._slice_df(eda_full, start_time, end_time)
            if bvp_full is not None and not bvp_full.empty:
                data_dict["BVP"] = self._slice_df(bvp_full, start_time, end_time)

            eeg_f = (self.raw_path/"3-Driver"/"1-EEG"/scenario
                     /driver_type/f"{subject_id}.set")
            if eeg_f.exists() and MNE_AVAILABLE:
                data_dict["EEG"] = self._load_eeg(
                    eeg_f, start_time, end_time, subject_id, scenario)

            et_f = (self.raw_path/"3-Driver"/"2-EyeTracking"/scenario
                    /driver_type/f"{subject_id}.csv")
            if et_f.exists():
                data_dict["EyeTrack"] = self._load_csv(et_f, start_time, end_time)

            if not data_dict:
                continue

            ref_ts = None
            for key in ("EEG", "Traffic", "CAN", "EDA"):
                if key in data_dict and not data_dict[key].empty:
                    ref_ts = data_dict[key].index; break
            if ref_ts is None:
                first = next((v for v in data_dict.values() if not v.empty), None)
                if first is None: continue
                ref_ts = first.index

            aligned = pd.DataFrame(index=ref_ts)
            for name, df in data_dict.items():
                if df.empty: continue
                resampled = self._resample_to_common(df, ref_ts)
                for col in resampled.columns:
                    # Sanitise column name: strip whitespace, avoid duplicates
                    safe_col = f"{name}_{col}".strip().replace(" ", "_")
                    if safe_col in aligned.columns:
                        safe_col = safe_col + "_dup"
                    aligned[safe_col] = resampled[col]

            aligned["scenario"] = scenario
            for label_name, label_df in self.scale_labels.items():
                val = self._lookup_label(label_df, subject_id, scenario)
                if val is not None:
                    aligned[label_name] = float(val)

            # Guard: replace Inf before saving
            num_cols = aligned.select_dtypes(include=[np.number]).columns
            aligned[num_cols] = aligned[num_cols].replace(
                [np.inf, -np.inf], np.nan)
            all_segments.append(aligned)

        if not all_segments:
            print(f"    [WARN] {subject_id}: no data segments found.")
            return pd.DataFrame()

        # [FIX-01] Align all segments to the union of their columns BEFORE concat
        # so that missing columns are 0.0, never NaN.
        all_cols: List[str] = []
        seen = set()
        for seg in all_segments:
            for c in seg.columns:
                if c not in seen:
                    all_cols.append(c)
                    seen.add(c)

        aligned_segments = []
        for seg in all_segments:
            missing_in_seg = [c for c in all_cols if c not in seg.columns]
            if missing_in_seg:
                seg = seg.copy()
                for c in missing_in_seg:
                    seg[c] = 0.0
            aligned_segments.append(seg[all_cols])

        combined = pd.concat(aligned_segments, axis=0).sort_index()

        # Final NaN/Inf cleanup
        combined = _check_finite_df(combined, f"synchronize_subject({subject_id})")

        out_path = self.sync_path / f"{subject_id}.parquet"
        combined.to_parquet(out_path)
        print(f"    {subject_id}: {len(combined)} rows, "
              f"{len(combined.columns)} cols → {out_path.name}")
        return combined

    # ---- helpers ------------------------------------------------------------

    def _load_timedivision(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists(): return None
        try:    return pd.read_csv(path)
        except: return None

    def _get_time_window(
        self, timediv: Optional[pd.DataFrame], scenario: str,
    ) -> Tuple[float, Optional[float]]:
        if timediv is None: return 0.0, None
        if "scenario" in timediv.columns:
            row = timediv[timediv["scenario"].astype(str).str.strip() == scenario]
            if not row.empty:
                return (float(row["start"].values[0]) if "start" in row.columns else 0.0,
                        float(row["end"].values[0])   if "end"   in row.columns else None)
        scen_num = scenario.split("_")[0]
        if "scenario" in timediv.columns:
            row = timediv[timediv["scenario"].astype(str).str.strip() == scen_num]
            if not row.empty:
                return (float(row["start"].values[0]) if "start" in row.columns else 0.0,
                        float(row["end"].values[0])   if "end"   in row.columns else None)
        start = float(timediv["start"].iloc[0]) if "start" in timediv.columns else 0.0
        end   = float(timediv["end"].iloc[0])   if "end"   in timediv.columns else None
        return start, end

    def _lookup_label(
        self, label_df: pd.DataFrame, subject_id: str, scenario: str,
    ) -> Optional[float]:
        sid  = subject_id.strip()
        rows = label_df[label_df["subject_id"] == sid]
        if rows.empty: return None
        exact = rows[rows["scenario"] == scenario]
        if not exact.empty: return float(exact["value"].iloc[0])
        scen_num = scenario.split("_")[0]
        fuzzy = rows[rows["scenario"].str.startswith(scen_num)]
        if not fuzzy.empty: return float(fuzzy["value"].iloc[0])
        return float(rows["value"].mean())

    def _load_csv(
        self, filepath: Path, start: float = 0.0, end: Optional[float] = None,
    ) -> pd.DataFrame:
        if not filepath.exists(): return pd.DataFrame()
        try:   df = pd.read_csv(filepath)
        except: return pd.DataFrame()
        if df.shape[1] < 2: return pd.DataFrame()
        # Sanitise column names
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        time_col = df.columns[0]
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df[df[time_col] >= start]
        if end is not None: df = df[df[time_col] <= end]
        df = df.set_index(time_col)
        # [FIX-16] Drop duplicate index rows, ensure monotonic before interpolate
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        df = (df.apply(pd.to_numeric, errors="coerce")
                .interpolate(method="index", limit_direction="both"))
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    def _slice_df(self, df: pd.DataFrame,
                  start: float, end: Optional[float]) -> pd.DataFrame:
        if df.empty: return df
        result = df[df.index >= start]
        if end is not None: result = result[result.index <= end]
        return result

    def _load_eeg(
        self, filepath: Path, start: float, end: Optional[float],
        subject_id: str, scenario: str,
    ) -> pd.DataFrame:
        try:
            raw  = mne.io.read_raw_eeglab(str(filepath), preload=True, verbose=False)
            tmax = end if end is not None else raw.times[-1]
            raw.crop(tmin=float(start), tmax=float(tmax))
            raw.resample(self.common_fs)
            df = pd.DataFrame(raw.get_data().T, columns=raw.ch_names,
                              index=raw.times)
            df.index.name = "time"
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return df
        except Exception as exc:
            print(f"    [EEG WARN] {subject_id}/{scenario} — "
                  f"{filepath.name}: {exc}")
            return pd.DataFrame()

    def _resample_to_common(
        self, data: pd.DataFrame, ref_index: pd.Index,
    ) -> pd.DataFrame:
        if data.empty: return pd.DataFrame(index=ref_index)
        data = data.copy()
        data.index = pd.to_numeric(data.index, errors="coerce")
        data = data[~data.index.isna()]
        if data.index.duplicated().any():
            data = data[~data.index.duplicated(keep="first")]
        data = data.sort_index()
        # [FIX-17] Relative tolerance: 5 % of median sample spacing
        if len(ref_index) > 1:
            median_spacing = float(np.median(np.diff(ref_index.values.astype(float))))
            tol = max(0.0, median_spacing * 0.5)
        else:
            tol = 0.05
        resampled = data.reindex(ref_index, method="nearest", tolerance=tol)
        resampled = resampled.interpolate(method="index", limit_direction="both")
        resampled = resampled.fillna(0.0)
        return resampled

# =============================================================================
# 7. FEATURE EXTRACTORS  (reference / offline use, unchanged logic)
# =============================================================================

class EEGFeatureExtractor:
    def __init__(self, config):
        cfg = config["feature_extraction"]["eeg"]
        self.bands  = cfg["bands"]; self.window = cfg["window"]; self.step = cfg["step"]

    def extract(self, eeg_df: pd.DataFrame) -> pd.DataFrame:
        if eeg_df.empty: return pd.DataFrame()
        t = eeg_df.index.values.astype(float)
        data = eeg_df.values.T.astype(float)
        fs   = float(1.0/np.median(np.diff(t))) if len(t)>1 else 100.0
        win  = int(self.window*fs); step = int(self.step*fs)
        feat_list, times_out = [], []
        for s in range(0, data.shape[1]-win+1, step):
            seg = data[:, s:s+win]; ch_feats = []
            for ch in range(data.shape[0]):
                sig = seg[ch]
                if len(sig)<4: continue
                freqs, psd  = welch(sig, fs=fs, nperseg=min(256,len(sig)))
                total_power = float(np.sum(psd))+1e-12
                bp = {b: float(np.sum(psd[(freqs>=lo)&(freqs<=hi)]))/total_power
                      for b,(lo,hi) in self.bands.items()}
                pn = psd/(np.sum(psd)+1e-12)
                ch_feats.append([bp["theta"],bp["alpha"],bp["beta"],bp["gamma"],
                                  bp["theta"]/(bp["alpha"]+1e-6),
                                  float(entropy(pn)), float(np.log(total_power))])
            if ch_feats:
                feat_list.append(np.mean(ch_feats, axis=0))
                times_out.append(t[s+win//2])
        if not feat_list: return pd.DataFrame()
        return pd.DataFrame(feat_list, index=times_out,
                            columns=["Theta","Alpha","Beta","Gamma",
                                     "ThetaAlphaRatio","SpectralEntropy",
                                     "DifferentialEntropy"])


class CANFeatureExtractor:
    def __init__(self, config):
        self.features = config["feature_extraction"]["can"]["features"]

    def extract(self, can_df: pd.DataFrame) -> pd.DataFrame:
        if can_df.empty: return pd.DataFrame()
        can_df = can_df.copy()
        if "jerk" not in can_df.columns and "acceleration" in can_df.columns:
            dt = can_df.index.to_series().diff().abs().clip(1e-6)
            can_df["jerk"] = can_df["acceleration"].diff() / dt
        avail = [f for f in self.features if f in can_df.columns]
        return can_df[avail] if avail else pd.DataFrame()


class TrafficFeatureExtractor:
    def __init__(self, config):
        self.col_map = TRAFFIC_COL_MAP

    def extract(self, traffic_df: pd.DataFrame) -> pd.DataFrame:
        if traffic_df.empty: return pd.DataFrame()
        traffic_df = traffic_df.copy()
        missing = []
        for canonical, aliases in self.col_map.items():
            if canonical in traffic_df.columns: continue
            matched = next((a for a in aliases if a in traffic_df.columns), None)
            if matched: traffic_df[canonical] = traffic_df[matched]
            else:       missing.append(canonical); traffic_df[canonical] = 0.5
        if missing:
            print(f"    [WARN] TrafficFeatureExtractor: placeholders for {missing}")
        return traffic_df[list(self.col_map.keys())]

# =============================================================================
# 8. GLOBAL FEATURE SCHEMA BUILDER   [FIX-01, FIX-02]
# =============================================================================

def build_global_feature_schema(
    sync_dir: str,
    subject_ids: List[str],
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Scan every parquet file and collect the union of all numeric feature columns.
    Returns a deterministically ordered list (sorted) so every dataset uses the
    same column ordering regardless of which subjects are loaded.
    """
    exclude = set(exclude_cols or [])
    # Always exclude known non-feature columns
    exclude.update(["scenario"])
    exclude.update(_LABEL_PREFIXES)
    # Also exclude any column that starts with a known label prefix
    all_feature_cols: set = set()
    sync_path = Path(sync_dir)

    for sid in subject_ids:
        pq = sync_path / f"{sid}.parquet"
        if not pq.exists():
            continue
        try:
            df = pd.read_parquet(pq)
        except Exception as exc:
            print(f"  [WARN] schema scan: cannot read {pq.name}: {exc}")
            continue

        for c in df.columns:
            if c in exclude:
                continue
            # Skip columns whose base name matches a label prefix
            c_lower = c.lower()
            if any(c_lower.startswith(lp.lower()) for lp in _LABEL_PREFIXES):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                all_feature_cols.add(c)

    schema = sorted(all_feature_cols)
    print(f"  [SCHEMA] Global feature schema: {len(schema)} columns "
          f"(union over {len(subject_ids)} subjects)")
    return schema

# =============================================================================
# 9. MODELS  (architecture unchanged; numerical fixes applied)
# =============================================================================

class ACPF(nn.Module):
    """Adaptive Cognitive Pressure Field v2."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int = 1, fusion: str = "gated",
                 num_modalities: int = 5, dropout: float = 0.15):
        super().__init__()
        self.pad_dim       = (-input_dim) % num_modalities
        self.total_dim     = input_dim + self.pad_dim
        self.mod_size      = self.total_dim // num_modalities
        self.num_modalities = num_modalities

        inner = max(32, self.mod_size)
        self.gate_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.mod_size, inner), nn.GELU(),
                nn.Linear(inner, 1),
            )
            for _ in range(num_modalities)
        ])

        self.ln_in  = nn.LayerNorm(self.total_dim)
        self.proj   = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru    = nn.GRU(hidden_dim, hidden_dim, num_layers=2,
                             batch_first=True, dropout=dropout)
        self.ln_gru = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        squeeze = x.dim() == 2
        if squeeze: x = x.unsqueeze(1)
        B, T, _ = x.shape

        # [FIX-04] Clamp input to finite values before any computation
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.pad_dim > 0:
            x = F.pad(x, (0, self.pad_dim))

        x = self.ln_in(x)

        gates = torch.cat(
            [g(x[..., m * self.mod_size:(m + 1) * self.mod_size])
             for m, g in enumerate(self.gate_nets)],
            dim=-1)
        gates = F.softmax(gates, dim=-1)

        weighted = x.clone()
        for m in range(self.num_modalities):
            sl = slice(m * self.mod_size, (m + 1) * self.mod_size)
            weighted[..., sl] = x[..., sl] * gates[..., m:m + 1]

        h = self.proj(weighted)
        g, _ = self.gru(h)
        h = self.ln_gru(h + g)

        out = self.head(h)

        if squeeze:
            out   = out.squeeze(1)
            gates = gates.squeeze(1)
        return out, gates


class KoopmanModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int,
                 encoder_hidden: List[int], decoder_hidden: List[int],
                 forecast_steps: List[int]):
        super().__init__()
        self.latent_dim     = latent_dim
        self.forecast_steps = forecast_steps

        enc, prev = [], input_dim
        for h in encoder_hidden:
            enc += [nn.Linear(prev, h), nn.GELU()]; prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], latent_dim
        for h in decoder_hidden:
            dec += [nn.Linear(prev, h), nn.GELU()]; prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

        self.K = nn.Parameter(
            torch.randn(latent_dim, latent_dim) / (latent_dim ** 0.5))

    def _clipped_K(self) -> torch.Tensor:
        """
        [FIX-05] Return K with spectral radius ≤ 1 via SVD clamp.
        Prevents exponential blow-up in multi-step forecasting.
        Applied only during forward (no gradient through SVD needed for stability).
        """
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(self.K, full_matrices=False)
            S_clipped = S.clamp(max=1.0)
            K_stable  = U @ torch.diag(S_clipped) @ Vh
        # Straight-through: use K for grad, K_stable for value
        return self.K + (K_stable - self.K).detach()

    def forward(self, x: torch.Tensor,
                ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:
        squeeze = x.dim() == 2
        if squeeze: x = x.unsqueeze(1)
        B, T, feat_dim = x.shape

        # [FIX-04] Input guard
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        z_flat  = self.encoder(x.reshape(B * T, feat_dim))
        z       = z_flat.view(B, T, self.latent_dim)
        x_recon = self.decoder(z_flat).view(B, T, feat_dim)

        K_stable = self._clipped_K()
        z_last   = z[:, -1, :]
        forecasts: Dict[int, torch.Tensor] = {}
        for k in self.forecast_steps:
            z_f = z_last
            for _ in range(k):
                z_f = torch.matmul(z_f, K_stable.t())
            forecasts[k] = self.decoder(z_f)

        if squeeze:
            x_recon = x_recon.squeeze(1)
            z       = z.squeeze(1)
        return x_recon, forecasts, z


class DFCG(nn.Module):
    """
    Dynamic Feature Coupling Graph.
    [FIX-07] num_nodes is tied to the actual latent_dim passed at construction.
    """
    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_mlp  = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, node_states: torch.Tensor) -> torch.Tensor:
        squeeze = node_states.dim() == 2
        if squeeze: node_states = node_states.unsqueeze(1)
        B, T, N = node_states.shape

        # [FIX-07] Runtime assertion to catch any future latent_dim mismatch
        if N != self.num_nodes:
            raise ValueError(
                f"DFCG expected num_nodes={self.num_nodes} but got N={N}. "
                f"Ensure dfcg.num_nodes == koopman.latent_dim.")

        outs = []
        for t in range(T):
            s   = node_states[:, t, :]
            adj = self._adj(s)
            outs.append(torch.sigmoid(
                s + torch.bmm(adj, s.unsqueeze(-1)).squeeze(-1)))
        out = torch.stack(outs, dim=1)
        return out.squeeze(1) if squeeze else out

    def _adj(self, state: torch.Tensor) -> torch.Tensor:
        B, N  = state.shape
        si    = state.unsqueeze(2).expand(B, N, N)
        sj    = state.unsqueeze(1).expand(B, N, N)
        pairs = torch.stack([si, sj], dim=-1).reshape(B * N * N, 2)
        return self.edge_mlp(pairs).view(B, N, N)

# =============================================================================
# 10. LOSS FUNCTION   [FIX-06, FIX-13, FIX-14]
# =============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, lambda_koopman=0.1, lambda_graph=0.05,
                 lambda_koop_reg=0.01):
        super().__init__()
        self.lk   = lambda_koopman
        self.lg   = lambda_graph
        self.lreg = lambda_koop_reg
        self.mse  = nn.MSELoss()

    def forward(self, pred_pressure, target_pressure,
                koopman_recon, koopman_input,
                koopman_forecasts, target_forecasts,
                graph_loss, K_matrix):

        L_cpf   = self.mse(pred_pressure, target_pressure)
        L_recon = self.mse(koopman_recon, koopman_input)

        # [FIX-13] Initialise accumulator on the correct device
        L_koop = torch.zeros(1, device=pred_pressure.device,
                             dtype=pred_pressure.dtype).squeeze()
        n = 0
        for k, pred in koopman_forecasts.items():
            if int(k) in target_forecasts:
                L_koop = L_koop + self.mse(pred, target_forecasts[int(k)])
                n += 1
        if n > 0: L_koop = L_koop / n

        if not isinstance(graph_loss, torch.Tensor):
            graph_loss = torch.tensor(float(graph_loss),
                                      device=pred_pressure.device,
                                      dtype=pred_pressure.dtype)

        # [FIX-06] Use p=2 (Frobenius) — works on PyTorch 1.x and 2.x
        reg = torch.norm(K_matrix, p=2)

        total = (L_cpf
                 + self.lk * (L_recon + L_koop)
                 + self.lg * graph_loss
                 + self.lreg * reg)

        return total, {
            "L_cpf":  L_cpf.item(),  "L_recon": L_recon.item(),
            "L_koop": L_koop.item(), "L_graph": graph_loss.item(),
            "L_reg":  reg.item(),
        }

# =============================================================================
# 11. DATASET   [FIX-02, FIX-10, FIX-20]
# =============================================================================

def _find_col(df: pd.DataFrame, keywords: List[str],
              exclude_prefixes: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the first column matching any keyword.
    [FIX-02/FIX-24] Excludes columns whose names start with a known sensor prefix,
    so sensor data named e.g. 'Traffic_stress' is not picked up as the stress label.
    """
    exclude_pref = exclude_prefixes or []
    for col in df.columns:
        c_lower = col.lower()
        # Skip columns that look like sensor data (prefixed by modality)
        if any(c_lower.startswith(ep.lower()) for ep in exclude_pref):
            continue
        if any(kw in c_lower for kw in keywords):
            return col
    return None

# Known sensor modality prefixes that cannot be label columns
_SENSOR_PREFIXES = ["Traffic_", "CAN_", "EDA_", "BVP_", "EEG_", "EyeTrack_"]


class CognitiveDataset(Dataset):
    def __init__(self, sync_dir: str, subject_ids: List[str],
                 config: Dict, mode: str = "train",
                 feat_cols: Optional[List[str]] = None,
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None):
        self.sync_dir        = Path(sync_dir)
        self.sequence_length = config["training"]["sequence_length"]
        self.horizons        = config["evaluation"]["horizons"]
        tlx_kws    = config["data"]["label_tlx_keywords"]
        stress_kws = config["data"]["label_stress_keywords"]

        frames = []
        for sid in subject_ids:
            f = self.sync_dir / f"{sid}.parquet"
            if f.exists():
                try:
                    frames.append(pd.read_parquet(f))
                except Exception as exc:
                    print(f"  [WARN] Cannot read parquet for {sid}: {exc}")
            else:
                print(f"  [WARN] Missing parquet for {sid}")
        if not frames:
            raise ValueError(
                f"No parquet files found in {sync_dir} for {subject_ids}.")

        # [FIX-02] Multi-subject concat: align columns to union BEFORE concat
        # so missing columns become 0.0, not NaN.
        all_cols: List[str] = []
        seen_cols: set = set()
        for fr in frames:
            for c in fr.columns:
                if c not in seen_cols:
                    all_cols.append(c)
                    seen_cols.add(c)

        aligned_frames = []
        for fr in frames:
            missing = [c for c in all_cols if c not in fr.columns]
            if missing:
                fr = fr.copy()
                for c in missing:
                    fr[c] = 0.0
            aligned_frames.append(fr[all_cols])

        df = pd.concat(aligned_frames, axis=0, ignore_index=False).sort_index()

        print(f"  [DIAG] mode={mode} — {len(df)} rows, "
              f"{len(df.columns)} cols | "
              f"cols[:20]: {df.columns.tolist()[:20]}")

        # Identify label and non-feature columns
        # [FIX-02/FIX-24] Use sensor-prefix exclusion in _find_col
        tlx_col    = _find_col(df, tlx_kws,    exclude_prefixes=_SENSOR_PREFIXES)
        stress_col = _find_col(df, stress_kws, exclude_prefixes=_SENSOR_PREFIXES)

        exclude = {"scenario"}
        if tlx_col:    exclude.add(tlx_col)
        if stress_col: exclude.add(stress_col)
        # Also exclude any column starting with a known label prefix
        for c in df.columns:
            c_lower = c.lower()
            if any(c_lower.startswith(lp.lower()) for lp in _LABEL_PREFIXES):
                if c not in ("scenario",):
                    exclude.add(c)

        # ---------- Enforce consistent feature set ----------
        if feat_cols is not None:
            # Add any missing columns as zeros
            added = 0
            for c in feat_cols:
                if c not in df.columns:
                    df[c] = 0.0
                    added += 1
            if added:
                print(f"  [DIAG] mode={mode}: added {added} missing feature cols as 0.0")
            self.feature_cols = list(feat_cols)
        else:
            self.feature_cols = [
                c for c in df.columns
                if c not in exclude
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        # ----------------------------------------------------

        # Ensure all feature columns are float32 and finite
        feats_df = df[self.feature_cols].copy()
        feats_df = feats_df.apply(pd.to_numeric, errors="coerce")
        feats_df = feats_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feats = feats_df.values.astype(np.float32)

        print(f"  [DIAG] mode={mode}: feature_dim={feats.shape[1]}, "
              f"rows={feats.shape[0]}")

        # Normalization
        if mean is not None and std is not None:
            self.mean_ = mean.astype(np.float32)
            self.std_  = std.astype(np.float32)
        else:
            self.mean_ = feats.mean(axis=0)
            raw_std    = feats.std(axis=0)
            # [FIX-10] Prevent zero-std division
            raw_std[raw_std < 1e-8] = 1.0
            self.std_  = raw_std.astype(np.float32)

        print(f"  [DIAG] mode={mode}: norm mean range "
              f"[{self.mean_.min():.4f}, {self.mean_.max():.4f}], "
              f"std range [{self.std_.min():.4f}, {self.std_.max():.4f}]")

        # Normalize and guard
        self.features = (feats - self.mean_) / self.std_
        # [FIX-10] Final NaN/Inf clamp after normalisation
        self.features = np.nan_to_num(
            self.features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        nan_count = np.isnan(self.features).sum()
        inf_count = np.isinf(self.features).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  [ERROR] mode={mode}: features still contain "
                  f"NaN={nan_count}, Inf={inf_count} after clamp — check data.")

        # Save normalizer for diagnostics
        np.savez(
            Path(config["data"]["processed_path"]) / f"normalizer_{mode}.npz",
            mean=self.mean_, std=self.std_,
            feat_cols=np.array(self.feature_cols, dtype=object))

        # Label handling
        if tlx_col is None:
            print(f"  [WARN] mode={mode}: NASA_TLX column not found — "
                  f"SYNTHETIC random labels.")
            rng = np.random.default_rng(0)
            df["_tlx"] = rng.random(len(df)).astype(np.float32)
            tlx_col = "_tlx"
        else:
            print(f"  [INFO] mode={mode}: TLX label = '{tlx_col}'")

        if stress_col is None:
            df["_stress"] = df[tlx_col]
            stress_col = "_stress"
        else:
            print(f"  [INFO] mode={mode}: Stress label = '{stress_col}'")

        def _norm(a: np.ndarray) -> np.ndarray:
            # [FIX-10] NaN guard before normalisation
            a = np.nan_to_num(a.astype(np.float32), nan=0.0,
                              posinf=0.0, neginf=0.0)
            lo, hi = float(a.min()), float(a.max())
            normed = (a - lo) / (hi - lo + 1e-8)
            return np.nan_to_num(normed, nan=0.0).astype(np.float32)

        tlx    = df[tlx_col].values.astype(np.float32)
        stress = df[stress_col].values.astype(np.float32)
        pressure_raw = (_norm(tlx) + _norm(stress)) / 2.0
        self.pressure = np.nan_to_num(pressure_raw, nan=0.0).astype(np.float32)

        # [FIX-20] Build valid-index list: skip samples whose feature window
        # contains any NaN or Inf (should be 0 after above, but belt-and-suspenders)
        max_h = max(self.horizons or [0])
        n_total = len(self.features)
        self._valid_indices: List[int] = []
        for idx in range(n_total - self.sequence_length - max_h):
            seg = self.features[idx: idx + self.sequence_length]
            if np.isfinite(seg).all():
                self._valid_indices.append(idx)

        skipped = (n_total - self.sequence_length - max_h) - len(self._valid_indices)
        if skipped > 0:
            print(f"  [DIAG] mode={mode}: skipped {skipped} corrupted samples")
        print(f"  [DIAG] mode={mode}: valid samples = {len(self._valid_indices)}")

    @property
    def feature_dim(self) -> int:
        return len(self.feature_cols)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Dict:
        real_idx = self._valid_indices[idx]
        sl    = slice(real_idx, real_idx + self.sequence_length)
        f_seq = torch.from_numpy(self.features[sl])
        p_seq = torch.from_numpy(self.pressure[sl]).unsqueeze(-1)
        future = {}
        for h in self.horizons:
            fi = real_idx + self.sequence_length + h - 1
            future[h] = (torch.from_numpy(self.features[fi])
                         if fi < len(self.features)
                         else torch.zeros(self.feature_dim))
        return {"features": f_seq, "pressure": p_seq, "future_targets": future}

# =============================================================================
# 12. TRAINER   [FIX-09, FIX-11, FIX-14, FIX-15]
# =============================================================================

class CognitiveTrainer:
    def __init__(self, config: Dict, train_loader: DataLoader,
                 val_loader: DataLoader, feature_dim: int,
                 epochs_override: Optional[int] = None,
                 patience_override: Optional[int] = None):
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")
        print(f"  {_gpu_mem_info()}")

        kp = config["models"]["koopman"]
        ld = kp["latent_dim"]

        self.acpf = ACPF(
            feature_dim,
            config["models"]["acpf"]["hidden_dim"],
            config["models"]["acpf"]["output_dim"],
            config["models"]["acpf"]["fusion"],
            dropout=config["models"]["acpf"]["dropout"],
        ).to(self.device)

        self.koopman = KoopmanModel(
            feature_dim, ld,
            kp["encoder_hidden"], kp["decoder_hidden"],
            kp["forecast_steps"],
        ).to(self.device)

        # [FIX-07] DFCG num_nodes must equal koopman latent_dim
        assert config["models"]["dfcg"]["num_nodes"] == ld, (
            "dfcg.num_nodes must equal koopman.latent_dim")
        self.dfcg = DFCG(config["models"]["dfcg"]["num_nodes"]).to(self.device)

        total_params = sum(p.numel() for p in
                           (list(self.acpf.parameters()) +
                            list(self.koopman.parameters()) +
                            list(self.dfcg.parameters())) if p.requires_grad)
        print(f"  Total trainable parameters: {total_params:,}")

        self.optimizer = torch.optim.Adam(
            list(self.acpf.parameters()) +
            list(self.koopman.parameters()) +
            list(self.dfcg.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.criterion = CombinedLoss(
            config["loss"]["lambda_koopman"],
            config["loss"]["lambda_graph"],
            config["loss"]["lambda_koop_reg"],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config["training"]["scheduler_T0"],
            T_mult=2,
            eta_min=1e-6,
        )

        self.use_amp       = (config["training"]["mixed_precision"]
                              and self.device.type == "cuda")
        self.scaler        = _make_scaler(self.use_amp)
        self.patience      = (patience_override
                              if patience_override is not None
                              else config["training"]["early_stopping_patience"])
        self.accum_steps   = config["training"]["grad_accum_steps"]
        self.warmup_epochs = config["training"]["warmup_epochs"]
        self.noise_std     = config["training"]["noise_std"]
        self.base_lr       = config["training"]["learning_rate"]
        self.nan_abort_frac = config["training"].get("nan_batch_abort_frac", 0.10)
        self.best_val      = float("inf")
        self.counter       = 0
        self.ckpt_dir      = Path(config["training"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _params(self):
        return (list(self.acpf.parameters()) +
                list(self.koopman.parameters()) +
                list(self.dfcg.parameters()))

    def _run_batch(self, batch):
        feats = batch["features"].to(self.device, non_blocking=True)
        p_tgt = batch["pressure"].to(self.device, non_blocking=True)
        f_tgt = {int(k): v.to(self.device, non_blocking=True)
                 for k, v in batch["future_targets"].items()}

        # [FIX-04] Clamp batch inputs before forward
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        p_tgt = torch.nan_to_num(p_tgt, nan=0.0, posinf=0.0, neginf=0.0)
        f_tgt = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                 for k, v in f_tgt.items()}

        with _autocast(self.use_amp):
            p_pred, _             = self.acpf(feats)
            x_recon, forecasts, z = self.koopman(feats)
            z_graph               = self.dfcg(z)
            g_loss = ((z_graph[:, 1:] - z_graph[:, :-1]) ** 2).mean()
            loss, info = self.criterion(
                p_pred, p_tgt, x_recon, feats,
                forecasts, f_tgt, g_loss, self.koopman.K)
        return loss, info

    def train_epoch(self) -> float:
        self.acpf.train(); self.koopman.train(); self.dfcg.train()
        total = 0.0
        self.optimizer.zero_grad()
        n_batches = len(self.train_loader)
        if n_batches == 0:
            return 0.0

        nan_batches = 0

        for step, batch in enumerate(
                tqdm(self.train_loader, desc="  Train", leave=False)):
            if self.noise_std > 0:
                batch["features"] = (batch["features"] +
                                     torch.randn_like(batch["features"])
                                     * self.noise_std)

            try:
                loss, _ = self._run_batch(batch)
            except Exception as exc:
                print(f"  [WARN] batch {step} forward failed: {exc}")
                nan_batches += 1
                continue

            # [FIX-14] Detect non-finite loss; skip batch
            if not torch.isfinite(loss):
                print(f"  [WARN] Non-finite loss at batch {step}: {loss.item()}")
                nan_batches += 1
                if nan_batches / n_batches > self.nan_abort_frac:
                    raise RuntimeError(
                        f"Aborting: {nan_batches}/{n_batches} batches had "
                        f"non-finite loss (>{self.nan_abort_frac*100:.0f}%). "
                        f"Check feature data and normalization.")
                self.optimizer.zero_grad()
                continue

            loss = loss / self.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((step + 1) % self.accum_steps == 0 or
                    (step + 1) == n_batches):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    # [FIX-15] Zero any non-finite gradients before step
                    for p in self._params():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.zero_()
                    nn.utils.clip_grad_norm_(self._params(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # [FIX-15] Zero non-finite gradients on CPU too
                    for p in self._params():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            p.grad.zero_()
                    nn.utils.clip_grad_norm_(self._params(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total += loss.item() * self.accum_steps

        return total / max(n_batches, 1)

    def validate(self) -> float:
        self.acpf.eval(); self.koopman.eval(); self.dfcg.eval()
        total = 0.0
        if len(self.val_loader) == 0:
            return float("inf")
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Val  ", leave=False):
                feats = batch["features"].to(self.device, non_blocking=True)
                p_tgt = batch["pressure"].to(self.device, non_blocking=True)
                f_tgt = {int(k): v.to(self.device, non_blocking=True)
                         for k, v in batch["future_targets"].items()}
                # [FIX-04] Clamp
                feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                p_tgt = torch.nan_to_num(p_tgt, nan=0.0, posinf=0.0, neginf=0.0)
                f_tgt = {k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                         for k, v in f_tgt.items()}
                with _autocast(self.use_amp):
                    p_pred, _             = self.acpf(feats)
                    x_recon, forecasts, _ = self.koopman(feats)
                    loss, _ = self.criterion(
                        p_pred, p_tgt, x_recon, feats,
                        forecasts, f_tgt,
                        torch.zeros(1, device=self.device),
                        self.koopman.K)
                if torch.isfinite(loss):
                    total += loss.item()
        return total / max(len(self.val_loader), 1)

    def train(self, epochs: int) -> List[Dict]:
        history = []
        for epoch in range(1, epochs + 1):
            # [FIX-11] Warmup: set LR directly; do NOT step scheduler
            if epoch <= self.warmup_epochs:
                warmup_factor = epoch / max(self.warmup_epochs, 1)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.base_lr * warmup_factor
            else:
                # Step scheduler with epoch offset so T=0 aligns with post-warmup
                self.scheduler.step(epoch - self.warmup_epochs - 1)

            tl = self.train_epoch()
            vl = self.validate()

            cur_lr = self.optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train": tl,
                             "val": vl, "lr": cur_lr})
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train {tl:.4f} | Val {vl:.4f} | "
                  f"LR {cur_lr:.2e} | {_gpu_mem_info()}")

            if vl < self.best_val:
                self.best_val = vl; self.counter = 0
                self._save_checkpoint(epoch, vl)
                print(f"  ✓ checkpoint saved (val={vl:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break
        return history

    def _save_checkpoint(self, epoch: int, vl: float):
        """Safe checkpoint save with temp-file swap. [FIX-safe]"""
        ckpt_path = self.ckpt_dir / "best_model.pt"
        tmp_path  = self.ckpt_dir / "best_model_tmp.pt"
        torch.save({
            "epoch":           epoch,
            "acpf_state":      self.acpf.state_dict(),
            "koopman_state":   self.koopman.state_dict(),
            "dfcg_state":      self.dfcg.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss":   vl,
        }, tmp_path)
        tmp_path.replace(ckpt_path)   # atomic on POSIX

    def load_best(self):
        ckpt = self.ckpt_dir / "best_model.pt"
        if ckpt.exists():
            s = _safe_load(ckpt, self.device)
            self.acpf.load_state_dict(s["acpf_state"])
            self.koopman.load_state_dict(s["koopman_state"])
            self.dfcg.load_state_dict(s["dfcg_state"])
            print(f"  Loaded best checkpoint "
                  f"(epoch {s['epoch']}, val={s['best_val_loss']:.4f})")
        else:
            print("  [WARN] No checkpoint found — keeping current weights.")

    def free(self):
        """[FIX-13, FIX-20] Clean release of all GPU/CPU resources."""
        del self.acpf, self.koopman, self.dfcg, self.optimizer, self.scheduler
        if self.scaler is not None:
            del self.scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# =============================================================================
# 13. DATALOADER FACTORY   [FIX-09, FIX-21]
# =============================================================================

def _build_loaders(
    train_ds: CognitiveDataset,
    val_ds: CognitiveDataset,
    config: Dict,
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    BS = config["training"]["batch_size"]
    # [FIX-21] Use 0 workers on CPU or if dataset is small to avoid /dev/shm issues
    # On Kaggle T4, use 2 workers; guard with persistent_workers only when nw>0
    if torch.cuda.is_available() and len(train_ds) > BS * 4:
        nw = 2
    else:
        nw = 0

    if len(train_ds) == 0:
        print("  [WARN] Empty training set for this split — skipping.")
        return None, None

    eff_bs_train = min(BS, max(1, len(train_ds)))
    train_loader = DataLoader(
        train_ds,
        batch_size=eff_bs_train,
        shuffle=True,
        num_workers=nw,
        pin_memory=(_PIN_MEMORY and nw > 0),
        persistent_workers=(nw > 0),
        drop_last=(len(train_ds) >= eff_bs_train * 2),
    )

    if len(val_ds) == 0:
        print("  [WARN] Empty validation set — training without early stopping.")
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                num_workers=0, pin_memory=False,
                                persistent_workers=False)
    else:
        eff_bs_val = min(BS, max(1, len(val_ds)))
        val_loader = DataLoader(
            val_ds,
            batch_size=eff_bs_val,
            shuffle=False,
            num_workers=nw,
            pin_memory=(_PIN_MEMORY and nw > 0),
            persistent_workers=(nw > 0),
        )

    return train_loader, val_loader


def _cleanup_loaders(*loaders):
    """[FIX-13] Explicitly shut down DataLoader worker processes."""
    for loader in loaders:
        if loader is None:
            continue
        try:
            loader._iterator = None   # stop iterator if active
        except Exception:
            pass
    gc.collect()

# =============================================================================
# 14. 5-FOLD STRATIFIED GROUP K-FOLD   [FIX-18, FIX-03]
# =============================================================================

def make_subject_folds(
    subjects: List[str], n_folds: int = 5, seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    subjects = list(subjects)
    n = len(subjects)
    # [FIX-18] Use integer indices as groups (compatible with all sklearn versions)
    X      = np.arange(n).reshape(-1, 1)
    y      = np.array([0 if s.startswith("E") else 1 for s in subjects])
    groups = np.arange(n)   # each subject is its own group

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Need at least one Expert and one Novice subject for "
            "StratifiedGroupKFold.")

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups):
        train_subj = [subjects[i] for i in train_idx]
        val_subj   = [subjects[i] for i in val_idx]
        folds.append((train_subj, val_subj))
    return folds


def run_cross_validation(config: Dict, all_sids: List[str]) -> pd.DataFrame:
    # [FIX-03] Always work on a deep copy so the global CONFIG is never mutated
    config = copy.deepcopy(config)

    n_folds   = config["training"]["n_folds"]
    seed      = config["training"]["cv_seed"]
    cv_epochs = config["training"]["cv_epochs"]
    cv_pat    = config["training"]["cv_early_stopping_patience"]
    base_ckpt = Path(config["training"]["checkpoint_dir"])
    res_dir   = Path(OUTPUT_DIR) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    folds = make_subject_folds(all_sids, n_folds=n_folds, seed=seed)

    print(f"\n{'='*70}\n  {n_folds}-FOLD STRATIFIED GROUP K-FOLD CV\n{'='*70}")
    for i, (tr, va) in enumerate(folds, start=1):
        n_e_tr = sum(1 for s in tr if s.startswith("E"))
        n_n_tr = sum(1 for s in tr if s.startswith("N"))
        n_e_va = sum(1 for s in va if s.startswith("E"))
        n_n_va = sum(1 for s in va if s.startswith("N"))
        print(f"  Fold {i}: train={len(tr)} (E={n_e_tr},N={n_n_tr})  "
              f"val={len(va)} (E={n_e_va},N={n_n_va})")

    # Build global feature schema once, before any fold
    global_feat_cols = build_global_feature_schema(
        config["data"]["synchronized_path"], all_sids)

    fold_rows: List[Dict] = []
    all_histories: List[pd.DataFrame] = []

    for fold_idx, (train_subjects, val_subjects) in enumerate(folds, start=1):
        print(f"\n{'#'*70}\n  FOLD {fold_idx}/{n_folds}\n{'#'*70}")
        print(f"  Train ({len(train_subjects)}): {train_subjects}")
        print(f"  Val   ({len(val_subjects)}):   {val_subjects}")

        # [FIX-03] Deep-copy config per fold so mutations don't accumulate
        fold_cfg = copy.deepcopy(config)
        fold_cfg["training"]["checkpoint_dir"] = str(base_ckpt / f"fold_{fold_idx}")
        Path(fold_cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

        # [FIX-21] Set seed per fold for reproducibility
        _set_seed(seed + fold_idx)

        try:
            train_ds = CognitiveDataset(
                fold_cfg["data"]["synchronized_path"], train_subjects,
                fold_cfg, mode=f"cv_train_fold{fold_idx}",
                feat_cols=global_feat_cols)   # use global schema
            # Capture training normalisation stats
            feat_cols  = train_ds.feature_cols
            train_mean = train_ds.mean_
            train_std  = train_ds.std_

            val_ds = CognitiveDataset(
                fold_cfg["data"]["synchronized_path"], val_subjects,
                fold_cfg, mode=f"cv_val_fold{fold_idx}",
                feat_cols=feat_cols,
                mean=train_mean,
                std=train_std)
        except ValueError as exc:
            print(f"  [WARN] Fold {fold_idx} skipped — {exc}")
            continue

        feature_dim = train_ds.feature_dim
        fold_cfg["models"]["acpf"]["input_dim"]    = feature_dim
        fold_cfg["models"]["koopman"]["input_dim"] = feature_dim
        print(f"  [DIAG] Fold {fold_idx}: feature_dim={feature_dim}")

        train_loader, val_loader = _build_loaders(train_ds, val_ds, fold_cfg)
        if train_loader is None:
            print(f"  [WARN] Fold {fold_idx} skipped — no usable training data.")
            continue

        trainer = CognitiveTrainer(
            fold_cfg, train_loader, val_loader, feature_dim,
            patience_override=cv_pat)
        history = trainer.train(cv_epochs)
        trainer.load_best()

        # ---- fold evaluation -----------------------------------------------
        trainer.acpf.eval(); trainer.koopman.eval(); trainer.dfcg.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Fold {fold_idx} eval",
                              leave=False):
                feats  = batch["features"].to(trainer.device)
                feats  = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                p_true = batch["pressure"].cpu().numpy().flatten()
                with _autocast(trainer.use_amp):
                    p_pred, _ = trainer.acpf(feats)
                all_true.extend(p_true.tolist())
                all_pred.extend(p_pred.cpu().numpy().flatten().tolist())

        if len(all_true) > 0:
            true = np.array(all_true, dtype=np.float32)
            pred = np.array(all_pred, dtype=np.float32)
            rmse = float(np.sqrt(mean_squared_error(true, pred)))
            mae  = float(mean_absolute_error(true, pred))
            r2   = float(r2_score(true, pred)) if len(true) > 1 else float("nan")
        else:
            rmse = mae = r2 = float("nan")
            print(f"  [WARN] Fold {fold_idx}: validation loader produced no samples.")

        print(f"  Fold {fold_idx} → RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

        fold_rows.append({
            "fold": fold_idx,
            "n_train_subjects": len(train_subjects),
            "n_val_subjects":    len(val_subjects),
            "train_subjects":    ";".join(train_subjects),
            "val_subjects":      ";".join(val_subjects),
            "best_val_loss":     trainer.best_val,
            "rmse": rmse, "mae": mae, "r2": r2,
        })
        hdf = pd.DataFrame(history); hdf["fold"] = fold_idx
        all_histories.append(hdf)

        # [FIX-13] Clean up loaders before freeing trainer
        _cleanup_loaders(train_loader, val_loader)
        trainer.free()
        del trainer, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"  {_gpu_mem_info()}")

    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(res_dir / "cv_fold_metrics.csv", index=False)

    if all_histories:
        pd.concat(all_histories, axis=0, ignore_index=True).to_csv(
            res_dir / "cv_training_history.csv", index=False)

    if not metrics_df.empty:
        summary = metrics_df[["rmse", "mae", "r2"]].agg(["mean", "std"])
        summary.to_csv(res_dir / "cv_summary.csv")
        print(f"\n{'='*70}\n  CV SUMMARY ({len(metrics_df)} folds)\n{'='*70}")
        print(metrics_df[["fold", "rmse", "mae", "r2"]].to_string(index=False))
        print("\n  Mean ± Std:")
        for col in ["rmse", "mae", "r2"]:
            print(f"    {col.upper():5s}: "
                  f"{summary.loc['mean', col]:.4f} ± {summary.loc['std', col]:.4f}")
    else:
        print("  [WARN] No folds produced metrics — check dataset/parquet availability.")

    return metrics_df

# =============================================================================
# 15. CPF EXPERT vs NOVICE VISUALISATION   [FIX-12]
# =============================================================================

def evaluate_and_plot_cpf(
    trainer, all_subjects: List[str], config: Dict,
    output_dir: str, train_ds: "CognitiveDataset",
) -> Optional[pd.DataFrame]:
    """
    Run CPF inference on ALL subjects using consistent training-set
    normalisation, then produce 6 publication-quality figures.
    """
    fig_dir = Path(output_dir) / "figures"
    res_dir = Path(output_dir) / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    train_mean = train_ds.mean_
    train_std  = train_ds.std_
    feat_cols  = train_ds.feature_cols
    seq_len    = config["training"]["sequence_length"]
    bs         = config["training"]["batch_size"]

    trainer.acpf.eval()
    device = trainer.device

    print(f"\n  Running CPF inference on {len(all_subjects)} subjects …")
    expert_count = sum(1 for s in all_subjects if s.startswith("E"))
    novice_count = sum(1 for s in all_subjects if s.startswith("N"))
    print(f"  Expert: {expert_count}  |  Novice: {novice_count}")

    all_records: List[Dict] = []
    sync_dir = Path(config["data"]["synchronized_path"])

    with torch.no_grad():
        for sid in tqdm(all_subjects, desc="  CPF inference"):
            pq = sync_dir / f"{sid}.parquet"
            if not pq.exists():
                print(f"    [WARN] {sid}: parquet not found, skipping."); continue

            df_raw      = pd.read_parquet(pq)
            driver_type = "Expert" if sid.startswith("E") else "Novice"

            scen_arr = (df_raw["scenario"].values
                        if "scenario" in df_raw.columns
                        else np.full(len(df_raw), "unknown", dtype=object))

            # Use the exact same columns and normalisation as training
            feats_raw = (df_raw
                         .reindex(columns=feat_cols)
                         .fillna(0.0)
                         .values
                         .astype(np.float32))
            feats_norm = np.nan_to_num(
                (feats_raw - train_mean) / train_std,
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            n = len(feats_norm)
            if n < seq_len:
                print(f"    [WARN] {sid}: only {n} rows < seq_len={seq_len}."); continue

            n_seqs = n - seq_len + 1
            cpf_preds: List[float] = []

            # [FIX-12] Process in batches; explicit del to cap VRAM use
            for b0 in range(0, n_seqs, bs):
                b1    = min(b0 + bs, n_seqs)
                batch = np.stack([feats_norm[i:i + seq_len]
                                  for i in range(b0, b1)], axis=0)
                x = torch.from_numpy(batch).to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                with _autocast(trainer.use_amp):
                    p_pred, _ = trainer.acpf(x)
                mid = seq_len // 2
                cpf_preds.extend(p_pred[:, mid, 0].cpu().numpy().tolist())
                del x, p_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for i, cpf_val in enumerate(cpf_preds):
                mid_row  = i + seq_len // 2
                scenario = str(scen_arr[min(mid_row, n - 1)])
                all_records.append({
                    "subject_id":  sid,
                    "driver_type": driver_type,
                    "scenario":    scenario,
                    "time_idx":    i,
                    "cpf":         float(cpf_val),
                })

    if not all_records:
        print("  [WARN] No records collected for CPF analysis.")
        return None

    df_cpf = pd.DataFrame(all_records)
    expert_df = df_cpf[df_cpf["driver_type"] == "Expert"]
    novice_df = df_cpf[df_cpf["driver_type"] == "Novice"]

    if expert_df.empty or novice_df.empty:
        print(f"  [WARN] One group is empty: "
              f"expert={len(expert_df)} novice={len(novice_df)}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False,   "figure.dpi": 120,
    })
    E_COL, N_COL = "#2E86AB", "#E94F37"

    # ------------------------------------------------------------------
    # FIGURE 1: Time-series overlay
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

    for ax, grp_df, color, label in [
        (axes[0], expert_df, E_COL, "Expert"),
        (axes[1], novice_df, N_COL, "Novice"),
    ]:
        if grp_df.empty:
            ax.set_title(f"{label}: no data"); continue

        subj_means = []
        for sid, sdf in grp_df.groupby("subject_id"):
            w = min(50, max(5, len(sdf) // 20))
            rm = (sdf.set_index("time_idx")["cpf"]
                  .rolling(window=w, center=True, min_periods=1).mean())
            ax.plot(rm.index, rm.values,
                    color=color, alpha=0.3, linewidth=0.6)
            subj_means.append(rm.values)

        if subj_means:
            min_len  = min(len(v) for v in subj_means)
            arr      = np.stack([v[:min_len] for v in subj_means], axis=0)
            mu       = arr.mean(axis=0)
            sigma    = arr.std(axis=0)
            xs       = np.arange(min_len)
            ax.plot(xs, mu, color=color, linewidth=2.5,
                    label=f"Group mean (n={grp_df['subject_id'].nunique()})")
            ax.fill_between(xs, mu - sigma, mu + sigma,
                            color=color, alpha=0.18, label="±1 σ")
            ax.axhline(float(np.nanmean(mu)), color=color,
                       linestyle="--", linewidth=1.5,
                       label=f"Overall mean={np.nanmean(mu):.3f}")

        ax.set_ylim(0, 1)
        ax.set_ylabel("CPF", fontsize=11)
        ax.set_title(f"{label} Drivers — Adaptive Cognitive Pressure Field",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Sequence index (stride-1 windows)", fontsize=10)

    plt.tight_layout(h_pad=2.5)
    fig.savefig(fig_dir / "cpf_timeseries_expert_vs_novice.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: cpf_timeseries_expert_vs_novice.png")

    # ------------------------------------------------------------------
    # FIGURE 2: Distribution comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 60)
    for grp_df, color, lbl in [
        (expert_df, E_COL, "Expert"),
        (novice_df, N_COL, "Novice"),
    ]:
        if grp_df.empty: continue
        vals = grp_df["cpf"].values
        ax.hist(vals, bins=bins, color=color, alpha=0.45, density=True,
                edgecolor="white", linewidth=0.4,
                label=f"{lbl}  (n={len(vals):,}  "
                      f"μ={vals.mean():.3f}  σ={vals.std():.3f})")
        from scipy.stats import gaussian_kde
        kde  = gaussian_kde(vals, bw_method="silverman")
        xs   = np.linspace(0, 1, 300)
        ax.plot(xs, kde(xs), color=color, linewidth=2.5)

    ax.set_xlabel("CPF", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.set_title("CPF Distribution: Expert vs Novice", fontsize=12,
                 fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(fig_dir / "cpf_distribution_expert_vs_novice.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: cpf_distribution_expert_vs_novice.png")

    # ------------------------------------------------------------------
    # FIGURE 3: Box plot + Mann-Whitney U significance
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    groups, labels, colors = [], [], []
    for grp_df, color, lbl in [
        (expert_df, E_COL, "Expert"),
        (novice_df, N_COL, "Novice"),
    ]:
        if grp_df.empty: continue
        groups.append(grp_df["cpf"].values)
        labels.append(f"{lbl}\n(n={grp_df['subject_id'].nunique()} subj)")
        colors.append(color)

    if groups:
        bp = ax.boxplot(groups, labels=labels, patch_artist=True, notch=True,
                        showmeans=True, meanline=True,
                        medianprops=dict(color="black", linewidth=2),
                        meanprops=dict(color="gold", linewidth=2, linestyle="--"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)

        if len(groups) == 2:
            stat, pval = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            sig = ("***" if pval < 0.001 else "**" if pval < 0.01
                   else "*" if pval < 0.05 else "n.s.")
            ax.text(1.5, 0.96, f"Mann-Whitney U\np={pval:.3e}  {sig}",
                    ha="center", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="gray", alpha=0.8))

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CPF", fontsize=11)
    ax.set_title("CPF Box Plot: Expert vs Novice", fontsize=12,
                 fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(fig_dir / "cpf_boxplot_significance.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Saved: cpf_boxplot_significance.png")

    # ------------------------------------------------------------------
    # FIGURE 4: Per-scenario heatmap
    # ------------------------------------------------------------------
    scen_stats = (df_cpf
                  .groupby(["scenario", "driver_type"])["cpf"]
                  .mean()
                  .reset_index()
                  .pivot(index="scenario", columns="driver_type", values="cpf"))
    scen_stats = scen_stats.reindex(columns=["Expert", "Novice"])
    scen_stats = scen_stats.dropna(how="all")

    if not scen_stats.empty and len(scen_stats) > 1:
        fig, ax = plt.subplots(figsize=(9, max(4, len(scen_stats) * 0.5)))
        im = ax.imshow(scen_stats.values, cmap="RdYlBu_r",
                       aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(scen_stats.columns)))
        ax.set_xticklabels(scen_stats.columns, fontsize=11)
        ax.set_yticks(range(len(scen_stats.index)))
        ax.set_yticklabels(scen_stats.index, fontsize=9)
        for i in range(len(scen_stats.index)):
            for j in range(len(scen_stats.columns)):
                val = scen_stats.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if (val < 0.35 or val > 0.65) else "black")
        ax.set_title("Mean CPF by Scenario: Expert vs Novice",
                     fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Mean CPF", shrink=0.8)
        plt.tight_layout()
        fig.savefig(fig_dir / "cpf_scenario_heatmap.png", bbox_inches="tight")
        plt.close(fig)
        print("  ✓ Saved: cpf_scenario_heatmap.png")

    # ------------------------------------------------------------------
    # FIGURE 5: Per-subject CPF trajectories (small multiples)
    # ------------------------------------------------------------------
    all_sids2 = sorted(df_cpf["subject_id"].unique())
    n_subs    = len(all_sids2)
    if n_subs > 0:
        cols = min(4, n_subs)
        rows = (n_subs + cols - 1) // cols
        fig, axes2 = plt.subplots(rows, cols,
                                  figsize=(cols * 4, rows * 2.5),
                                  sharex=False, sharey=True)
        axes2 = np.array(axes2).flatten()

        for ax, sid in zip(axes2, all_sids2):
            sdf     = df_cpf[df_cpf["subject_id"] == sid].sort_values("time_idx")
            dtype   = sdf["driver_type"].iloc[0]
            color   = E_COL if dtype == "Expert" else N_COL
            w       = min(50, max(5, len(sdf) // 20))
            rm      = sdf["cpf"].rolling(window=w, center=True, min_periods=1).mean()
            ax.plot(sdf["time_idx"].values, rm.values,
                    color=color, linewidth=1.0)
            ax.fill_between(sdf["time_idx"].values,
                            rm.values - sdf["cpf"].rolling(w, min_periods=1).std().values,
                            rm.values + sdf["cpf"].rolling(w, min_periods=1).std().values,
                            color=color, alpha=0.15)
            ax.set_title(f"{sid} ({dtype[0]})", fontsize=9)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
            ax.axhline(sdf["cpf"].mean(), color=color, linestyle=":",
                       linewidth=1.0, alpha=0.8)

        for ax in axes2[n_subs:]:
            ax.set_visible(False)

        fig.suptitle("CPF Trajectories — All Subjects", fontsize=13,
                     fontweight="bold", y=1.01)
        expert_patch = mpatches.Patch(color=E_COL, label="Expert")
        novice_patch = mpatches.Patch(color=N_COL, label="Novice")
        fig.legend(handles=[expert_patch, novice_patch],
                   loc="lower center", ncol=2, fontsize=10, y=-0.01)
        plt.tight_layout()
        fig.savefig(fig_dir / "cpf_per_subject_trajectories.png",
                    bbox_inches="tight")
        plt.close(fig)
        print("  ✓ Saved: cpf_per_subject_trajectories.png")

    # ------------------------------------------------------------------
    # FIGURE 6: Scenario-level bar chart
    # ------------------------------------------------------------------
    if not scen_stats.empty and len(scen_stats) > 1:
        fig, ax = plt.subplots(figsize=(14, 5))
        x    = np.arange(len(scen_stats))
        w    = 0.35
        for offset, col, color in [(-w/2, "Expert", E_COL),
                                    (+w/2, "Novice", N_COL)]:
            vals = scen_stats[col].values if col in scen_stats.columns else np.zeros(len(x))
            ax.bar(x + offset, np.nan_to_num(vals, nan=0.0), w,
                   color=color, alpha=0.7, label=col, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(scen_stats.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean CPF", fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
        ax.set_title("Mean CPF per Scenario: Expert vs Novice",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y")
        plt.tight_layout()
        fig.savefig(fig_dir / "cpf_scenario_barchart.png", bbox_inches="tight")
        plt.close(fig)
        print("  ✓ Saved: cpf_scenario_barchart.png")

    # ---- data exports -------------------------------------------------------
    df_cpf.to_csv(res_dir / "cpf_all_subjects.csv", index=False)

    summary = (df_cpf.groupby("driver_type")["cpf"]
               .agg(count="count", mean="mean", std="std",
                    min="min", median="median", max="max")
               .round(4))
    summary.to_csv(res_dir / "cpf_group_summary.csv")

    print(f"\n{'='*50}")
    print("  CPF Summary Statistics")
    print(f"{'='*50}")
    print(summary.to_string())
    if not expert_df.empty and not novice_df.empty:
        stat, pval = mannwhitneyu(
            expert_df["cpf"].values, novice_df["cpf"].values,
            alternative="two-sided")
        print(f"\n  Mann-Whitney U: stat={stat:.1f}  p={pval:.4e}")
    print(f"{'='*50}")
    print(f"  ✓ Saved: cpf_all_subjects.csv, cpf_group_summary.csv")
    print(f"  All figures → {fig_dir}")

    return df_cpf

# =============================================================================
# 16. MAIN PIPELINE   [FIX-03, FIX-01, FIX-02]
# =============================================================================

def run_pipeline(config: Dict):
    # Work on a deep copy so the global CONFIG is never mutated
    config = copy.deepcopy(config)

    # Step 0: Diagnostics
    print("\n=== Scale file diagnostic ===")
    diagnose_scale_csv(config["data"]["raw_path"])

    # Step 1: Load scale labels
    print("\n=== Loading scale labels ===")
    scale_labels = load_all_scale_labels(config["data"]["raw_path"])
    if not scale_labels:
        print("  [WARN] No scale label files — will use synthetic labels.")

    # Step 2: Synchronise subjects
    print("\n=== Synchronising subjects ===")
    syncer   = DataSynchronizer(config, scale_labels)
    all_sids = config["data"]["subjects"]

    for sid in all_sids:
        out = Path(config["data"]["synchronized_path"]) / f"{sid}.parquet"
        if not out.exists():
            syncer.synchronize_subject(sid)
        else:
            print(f"  {sid}: already synchronised.")

    sync_dir = Path(config["data"]["synchronized_path"])
    usable_sids = [s for s in all_sids if (sync_dir / f"{s}.parquet").exists()]
    if len(usable_sids) < len(all_sids):
        missing = sorted(set(all_sids) - set(usable_sids))
        print(f"  [WARN] {len(missing)} subjects had no synchronised data "
              f"and will be excluded: {missing}")
    all_sids = usable_sids

    base_ckpt_dir = config["training"]["checkpoint_dir"]

    # Step 3: 5-fold StratifiedGroupKFold cross-validation
    print("\n=== 5-Fold StratifiedGroupKFold Cross-Validation ===")
    cv_metrics = run_cross_validation(config, all_sids)

    # Step 4: Train ONE final production model on ALL subjects
    print("\n=== Training final production model on ALL subjects ===")
    _set_seed(config["training"]["cv_seed"])

    # [FIX-03] Use a fresh deep copy for the final model config
    final_cfg = copy.deepcopy(config)
    final_cfg["training"]["checkpoint_dir"] = str(Path(base_ckpt_dir) / "final")
    Path(final_cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    folds_for_holdout = make_subject_folds(
        all_sids,
        n_folds=final_cfg["training"]["n_folds"],
        seed=final_cfg["training"]["cv_seed"])
    _, final_val_subjects = folds_for_holdout[0]
    final_train_subjects  = all_sids

    # Build global schema for consistent feature columns
    global_feat_cols = build_global_feature_schema(
        final_cfg["data"]["synchronized_path"], all_sids)

    train_ds = CognitiveDataset(
        final_cfg["data"]["synchronized_path"], final_train_subjects,
        final_cfg, mode="final_train",
        feat_cols=global_feat_cols)

    feat_cols_final  = train_ds.feature_cols
    train_mean_final = train_ds.mean_
    train_std_final  = train_ds.std_

    val_ds = CognitiveDataset(
        final_cfg["data"]["synchronized_path"], final_val_subjects,
        final_cfg, mode="final_val",
        feat_cols=feat_cols_final,
        mean=train_mean_final,
        std=train_std_final)

    feature_dim = train_ds.feature_dim
    print(f"\nFeature dimension: {feature_dim}")
    final_cfg["models"]["acpf"]["input_dim"]    = feature_dim
    final_cfg["models"]["koopman"]["input_dim"] = feature_dim

    train_loader, val_loader = _build_loaders(train_ds, val_ds, final_cfg)

    trainer = CognitiveTrainer(final_cfg, train_loader, val_loader, feature_dim)
    history = trainer.train(final_cfg["training"]["epochs"])
    trainer.load_best()

    # Step 5: Evaluation on final holdout
    print("\n=== Evaluation (final model holdout) ===")
    trainer.acpf.eval(); trainer.koopman.eval(); trainer.dfcg.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            feats  = batch["features"].to(trainer.device)
            feats  = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            p_true = batch["pressure"].cpu().numpy().flatten()
            with _autocast(trainer.use_amp):
                p_pred, _ = trainer.acpf(feats)
            all_true.extend(p_true.tolist())
            all_pred.extend(p_pred.cpu().numpy().flatten().tolist())

    res_dir = Path(OUTPUT_DIR) / "results"
    if len(all_true) > 0:
        true = np.array(all_true,  dtype=np.float32)
        pred = np.array(all_pred,  dtype=np.float32)
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        mae  = float(mean_absolute_error(true, pred))
        r2   = float(r2_score(true, pred)) if len(true) > 1 else float("nan")
        print(f"\n  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
        pd.DataFrame({"true": true, "pred": pred}).to_csv(
            res_dir / "pressure_predictions.csv", index=False)
        pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "R2": [r2]}).to_csv(
            res_dir / "metrics_final_model.csv", index=False)
    else:
        print("  [WARN] No validation samples available for the final model.")

    pd.DataFrame(history).to_csv(res_dir / "training_history_final.csv", index=False)

    # Standard plots
    try:
        if len(all_true) > 0:
            n = min(500, len(true))
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(true[:n],  label="True Pressure",      alpha=0.85, linewidth=1.0)
            ax.plot(pred[:n],  label="Predicted Pressure", alpha=0.85, linewidth=1.0)
            ax.set_title("Cognitive Pressure Prediction (final model holdout sample)")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Pressure [0, 1]")
            ax.legend(); fig.tight_layout()
            fig.savefig(res_dir / "pressure_sample.png", dpi=120)
            plt.close(fig)
    except Exception as exc:
        print(f"  pressure plot failed: {exc}")

    try:
        hd = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(hd["epoch"], hd["train"], label="Train")
        axes[0].plot(hd["epoch"], hd["val"],   label="Val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("Train / Val Loss"); axes[0].legend()
        axes[1].semilogy(hd["epoch"], hd["lr"])
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("LR (log)")
        axes[1].set_title("Learning Rate Schedule")
        fig.tight_layout()
        fig.savefig(res_dir / "training_curves.png", dpi=120)
        plt.close(fig)
    except Exception as exc:
        print(f"  training curve plot failed: {exc}")

    # Step 6: CPF Expert vs Novice analysis on ALL subjects
    print("\n=== CPF Expert vs Novice Analysis (ALL subjects) ===")
    df_cpf = evaluate_and_plot_cpf(
        trainer, all_sids, final_cfg, OUTPUT_DIR, train_ds)

    _cleanup_loaders(train_loader, val_loader)
    trainer.free()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.  All outputs → {OUTPUT_DIR}")
    print(f"  5-fold CV metrics  → {res_dir/'cv_fold_metrics.csv'}")
    print(f"  CV summary         → {res_dir/'cv_summary.csv'}")
    print(f"{'='*60}")
    return {"cv_metrics": cv_metrics, "cpf": df_cpf}

# =============================================================================
# 17. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_pipeline(CONFIG)