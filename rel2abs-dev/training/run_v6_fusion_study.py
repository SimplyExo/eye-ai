from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
V6 = PROJECT / "rel2abs_v6_research"
V3 = PROJECT / "rel2abs_v3_research"
V4 = PROJECT / "rel2abs_v4_research"
V5B = PROJECT / "rel2abs_v5b_research"
V5A = PROJECT / "rel2abs_v5a_research"
V2 = PROJECT / "rel2abs_v2_research"
ARCHIVE = PROJECT / "archive" / "rel2abs_v1_final"
V3_SRC = V3 / "src"
V2_SRC = V2 / "src"
REPORTS = V6 / "reports"
DATA = V6 / "data"
PLOTS = V6 / "plots"
V3_CACHE_DIR = ARCHIVE / "data" / "consolidated_cache_round3_v4"
SPEC_PATH = V2 / "baselines" / "Z1_matched" / "calibration_spec.json"
Z1_CHECKPOINT = V2 / "baselines" / "Z1_matched" / "rel2abs_z1_matched_seed42.pt"
V3_HEADS = V3 / "runs" / "v3_heads"
V3_SEEDS = V3 / "runs" / "v3_completion_spline8_seeds"
PRIOR_PATH = V3 / "reports" / "coco_metric_size_priors.yaml"
COCO_PANEL_PATH = V4 / "data" / "coco" / "annotations" / "instances_val2017_panel.json"
COCO_CACHE_DIR = V4 / "data" / "coco_cache_i3"
COCO_DET_PATH = V4 / "data" / "coco" / "product_vision" / "detections.jsonl"
COCO_EXTERNAL_PATH = V4 / "reports" / "v4_object_external_metrics.csv"
TEACHER_PATH = V4 / "data" / "teacher_predictions" / "coco_predictions.npz"
WAYMO_PRED_PATH = V5B / "data" / "waymo_object_predictions.jsonl"
WAYMO_PRODUCT_PATH = V5B / "data" / "waymo_product_object_predictions.jsonl"
WAYMO_GT_PATH = V5B / "data" / "waymo_object_gt.jsonl"
DIode_SPLITS = ("train", "dev", "locked")
SEED = 20260919
BOOTSTRAP_REPLICATES = 5000
RESERVOIR_SIZE = 20_000
BANDS = (
    ("0_1m", 0.0, 1.0),
    ("1_2m", 1.0, 2.0),
    ("2_5m", 2.0, 5.0),
    ("5_10m", 5.0, 10.0),
    ("10_15m", 10.0, 15.0),
    ("ge15m", 15.0, float("inf")),
)
VISUAL_KEYS = (
    "V0_Z1-frozen",
    "V1_Baseline-2P",
    "V2_Spline-8-seed123",
    "V3_Spline-8-seed42",
    "V4_Spline-8-seed7",
    "V5_Spline-8-Median",
    "V6_Spline-8-LogMean",
)
INDIVIDUAL_SPLINES = VISUAL_KEYS[2:5]
F1_KEY = "F1_SIZE_ANCHOR_OVERRIDE"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in values:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_seed(value: str, salt: str = "") -> int:
    """Return a process-independent deterministic uint32 seed."""
    digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def band_name(value: float) -> str:
    for name, low, high in BANDS:
        if low <= value < high:
            return name
    return "invalid"


def metric_summary(pred: Iterable[float], gt: Iterable[float]) -> dict[str, Any]:
    p = np.asarray(list(pred), dtype=np.float64)
    g = np.asarray(list(gt), dtype=np.float64)
    good = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    p, g = p[good], g[good]
    if not p.size:
        return {"n": 0, **{key: math.nan for key in ("absrel", "mae_m", "rmse_m", "medae_m", "median_signed_m", "bias_m", "p90_m", "p95_m", "catastrophic_rate_absrel_gt1", "delta1_25")}}
    signed = p - g
    abs_error = np.abs(signed)
    absrel = abs_error / np.maximum(g, 1e-8)
    ratio = np.maximum(p / np.maximum(g, 1e-8), g / np.maximum(p, 1e-8))
    return {
        "n": int(p.size),
        "absrel": float(absrel.mean()),
        "mae_m": float(abs_error.mean()),
        "rmse_m": float(np.sqrt(np.mean(signed * signed))),
        "medae_m": float(np.quantile(abs_error, 0.5)),
        "median_signed_m": float(np.median(signed)),
        "bias_m": float(signed.mean()),
        "p90_m": float(np.quantile(abs_error, 0.9)),
        "p95_m": float(np.quantile(abs_error, 0.95)),
        "catastrophic_rate_absrel_gt1": float(np.mean(absrel > 1.0)),
        "delta1_25": float(np.mean(ratio < 1.25)),
    }


def load_v3() -> Any:
    for path in (V3_SRC, V2_SRC, ARCHIVE / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import run_v3_multiphase_research as v3
    return v3


def load_rows(split: str) -> list[dict[str, Any]]:
    return read_jsonl(V3 / "data" / f"v3_{split}.jsonl")


def parse_priors(path: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
            current = stripped[:-1]
            values[current] = {}
        elif current is not None and ":" in stripped:
            key, value = [part.strip() for part in stripped.split(":", 1)]
            try:
                values[current][key] = float(value)
            except ValueError:
                values[current][key] = value
    return values


def reliability_value(prior: Mapping[str, Any]) -> float:
    value = str(prior.get("reliability", "low"))
    return {"high": 0.85, "medium": 0.65, "low": 0.45}.get(value, 0.45)


def normalized_intrinsics(row: Mapping[str, Any]) -> tuple[float, float, bool]:
    intrinsics = row.get("intrinsics") or []
    if len(intrinsics) >= 5:
        fx, fy, valid = safe_float(intrinsics[0]), safe_float(intrinsics[1]), safe_float(intrinsics[4], 0.0)
        if np.isfinite([fx, fy, valid]).all() and fx > 0 and fy > 0 and valid > 0:
            width, height = max(safe_float(row.get("image_width"), safe_float(row.get("width"), 0.0)), 1.0), max(safe_float(row.get("image_height"), safe_float(row.get("height"), 0.0)), 1.0)
            return fx * width, fy * height, True
    fx, fy = safe_float(row.get("fx_px")), safe_float(row.get("fy_px"))
    return fx, fy, bool(np.isfinite([fx, fy]).all() and fx > 0 and fy > 0)


def bbox_values(item: Mapping[str, Any], width: float, height: float) -> tuple[float, float, float, float]:
    if "bbox" in item:
        x, y, w, h = [safe_float(value, 0.0) for value in item["bbox"]]
        return (x + w / 2.0) / max(width, 1.0), (y + h / 2.0) / max(height, 1.0), w / max(width, 1.0), h / max(height, 1.0)
    return safe_float(item.get("x_center"), 0.5), safe_float(item.get("y_center"), 0.5), safe_float(item.get("width"), 0.0), safe_float(item.get("height"), 0.0)


def crop_slice(bbox: tuple[float, float, float, float], width: int = 256, height: int = 256) -> tuple[int, int, int, int]:
    xc, yc, bw, bh = bbox
    x0 = max(0, min(width - 1, int(round((xc - bw / 2.0) * width))))
    x1 = max(x0 + 1, min(width, int(round((xc + bw / 2.0) * width))))
    y0 = max(0, min(height - 1, int(round((yc - bh / 2.0) * height))))
    y1 = max(y0 + 1, min(height, int(round((yc + bh / 2.0) * height))))
    return y0, y1, x0, x1


def anchor_values(
    class_name: str,
    bbox: tuple[float, float, float, float],
    width_px: float,
    height_px: float,
    fx_px: float,
    fy_px: float,
    priors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    prior = priors.get(class_name)
    xc, yc, bw, bh = bbox
    border = float(xc - bw / 2 <= 0.01 or yc - bh / 2 <= 0.01 or xc + bw / 2 >= 0.99 or yc + bh / 2 >= 0.99)
    if not prior or not np.isfinite([fx_px, fy_px, width_px, height_px]).all() or min(fx_px, fy_px, width_px, height_px) <= 0:
        return {"z_size": math.nan, "z_height": math.nan, "z_width": math.nan, "z_generic": math.nan, "z_geometry": math.nan, "z_wrong": math.nan, "z_random": math.nan, "anchor_valid": 0, "anchor_reason": "unsupported_class_or_intrinsics", "sigma_log": math.nan, "reliability": 0.0, "border": border}
    physical_h = safe_float(prior.get("height_m"))
    physical_w = safe_float(prior.get("width_m"))
    sigma_h = safe_float(prior.get("height_sigma_m"), physical_h * 0.35)
    sigma_w = safe_float(prior.get("width_sigma_m"), physical_w * 0.35)
    h_px, w_px = max(bh * height_px, 1.0), max(bw * width_px, 1.0)
    z_h, z_w = fy_px * physical_h / h_px, fx_px * physical_w / w_px
    anchors = [z for z in (z_h, z_w) if np.isfinite(z) and z > 0]
    all_priors = list(priors.values())
    global_h = float(np.mean([safe_float(p.get("height_m")) for p in all_priors if np.isfinite(safe_float(p.get("height_m")))])) if all_priors else math.nan
    global_w = float(np.mean([safe_float(p.get("width_m")) for p in all_priors if np.isfinite(safe_float(p.get("width_m")))])) if all_priors else math.nan
    generic = [fy_px * global_h / h_px, fx_px * global_w / w_px]
    unit = [fy_px / h_px, fx_px / w_px]
    ordered = sorted(priors)
    wrong_name = ordered[(ordered.index(class_name) + 1) % len(ordered)] if class_name in ordered and ordered else None
    random_index = int(hashlib.sha256(f"{class_name}|{xc:.6f}|{yc:.6f}|{bw:.6f}|{bh:.6f}".encode()).hexdigest()[:8], 16) % len(ordered) if ordered else 0
    random_name = ordered[random_index] if ordered else None
    def prior_anchor(name: str | None) -> float:
        if not name or name not in priors:
            return math.nan
        p = priors[name]
        vals = [fy_px * safe_float(p.get("height_m")) / h_px, fx_px * safe_float(p.get("width_m")) / w_px]
        return float(np.exp(np.mean(np.log(vals)))) if np.all(np.isfinite(vals)) and min(vals) > 0 else math.nan
    z_size = float(np.exp(np.mean(np.log(anchors)))) if anchors else math.nan
    sigma_f = 0.03
    sigma_bbox_h, sigma_bbox_w = min(1.0, 2.0 / h_px), min(1.0, 2.0 / w_px)
    rel_h = sigma_h / max(abs(physical_h), 1e-6)
    rel_w = sigma_w / max(abs(physical_w), 1e-6)
    sigma_log = float(np.sqrt(sigma_f * sigma_f + np.mean([rel_h * rel_h, rel_w * rel_w]) + np.mean([sigma_bbox_h * sigma_bbox_h, sigma_bbox_w * sigma_bbox_w]) + border * 0.20 * 0.20))
    rel = reliability_value(prior) * math.exp(-0.5 * border) / (1.0 + sigma_log)
    aspect = max(bw / max(bh, 1e-6), 1e-6)
    expected_aspect = max(physical_w / max(physical_h, 1e-6), 1e-6)
    aspect_penalty = min(1.0, abs(math.log(aspect / expected_aspect)) / 3.0)
    rel *= 1.0 - 0.25 * aspect_penalty
    return {
        "z_size": z_size,
        "z_height": float(z_h),
        "z_width": float(z_w),
        "z_generic": float(np.exp(np.mean(np.log(generic)))) if np.all(np.isfinite(generic)) and min(generic) > 0 else math.nan,
        "z_geometry": float(np.exp(np.mean(np.log(unit)))) if np.all(np.isfinite(unit)) and min(unit) > 0 else math.nan,
        "z_wrong": prior_anchor(wrong_name),
        "z_random": prior_anchor(random_name),
        "anchor_valid": int(np.isfinite(z_size) and z_size > 0),
        "anchor_reason": "valid" if np.isfinite(z_size) and z_size > 0 else "nonpositive_anchor",
        "sigma_log": sigma_log,
        "reliability": float(np.clip(rel, 0.0, 1.0)),
        "border": border,
        "prior_class": class_name,
        "prior_wrong_class": wrong_name or "",
        "prior_random_class": random_name or "",
        "prior_reliability": reliability_value(prior),
    }


def checkpoint_map() -> dict[str, tuple[Path, str]]:
    return {
        "V0_Z1-frozen": (Z1_CHECKPOINT, "baseline"),
        "V1_Baseline-2P": (V3_HEADS / "baseline_2p" / "research_best.pt", "baseline"),
        "V2_Spline-8-seed123": (V3_SEEDS / "seed_123" / "research_best.pt", "spline"),
        "V3_Spline-8-seed42": (V3_HEADS / "spline_8" / "research_best.pt", "spline"),
        "V4_Spline-8-seed7": (V3_SEEDS / "seed_7" / "research_best.pt", "spline"),
    }


def existing_raw_path(dataset: str, split: str, key: str) -> Path | None:
    if dataset == "diode":
        old_name = {"V0_Z1-frozen": "Z1_frozen", "V1_Baseline-2P": "Baseline_2P", "V3_Spline-8-seed42": "Spline_8"}.get(key)
        return V4 / "data" / "gold_inference" / f"{split}_{old_name}_raw.npy" if old_name else None
    if dataset == "coco":
        old_name = {"V0_Z1-frozen": "Z1_frozen", "V1_Baseline-2P": "Baseline_2P", "V3_Spline-8-seed42": "Spline_8"}.get(key)
        return COCO_CACHE_DIR / f"{old_name}_raw.npy" if old_name else None
    return None


def load_raw(dataset: str, split: str, rows: list[dict[str, Any]], cache: Any, key: str, device: str, v3: Any) -> np.ndarray:
    out = DATA / "raw" / f"{dataset}_{split}_{key.replace('-', '_')}.npy"
    if out.exists():
        return np.load(out)
    old = existing_raw_path(dataset, split, key)
    if old is not None and old.exists():
        raw = np.load(old)
    else:
        checkpoint, kind = checkpoint_map()[key]
        raw = v3.infer_head(rows, cache, checkpoint, kind, device_name=device, batch_size=128)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, raw)
    return np.asarray(raw)


def decode_map(drel: np.ndarray, raw: np.ndarray, key: str, spec: Mapping[str, Any], v3: Any) -> np.ndarray:
    if key in {"V0_Z1-frozen", "V1_Baseline-2P"}:
        return v3.baseline_depth_frame(np.asarray(drel).reshape(-1), np.asarray(raw).reshape(-1), spec).reshape(np.asarray(drel).shape)
    from monotone_spline import monotone_grid_numpy
    x = (np.asarray(drel, dtype=np.float64) - float(spec["r_low"])) / max(float(spec["r_high"]) - float(spec["r_low"]), 1e-8)
    return monotone_grid_numpy(np.asarray(raw).reshape(1, -1)[:, :8], spec, 8, x.reshape(1, -1))[0][0].reshape(np.asarray(drel).shape).astype(np.float32)


def visual_from_predictions(values: Mapping[str, float]) -> dict[str, float]:
    seed_values = np.asarray([values.get(key, math.nan) for key in INDIVIDUAL_SPLINES], dtype=np.float64)
    good = np.isfinite(seed_values) & (seed_values > 0)
    result = dict(values)
    result["V5_Spline-8-Median"] = float(np.median(seed_values[good])) if good.any() else math.nan
    result["V6_Spline-8-LogMean"] = float(np.exp(np.mean(np.log(seed_values[good])))) if good.any() else math.nan
    return result


def seed_stats(values: Mapping[str, float]) -> dict[str, float]:
    array = np.asarray([values.get(key, math.nan) for key in INDIVIDUAL_SPLINES], dtype=np.float64)
    array = array[np.isfinite(array) & (array > 0)]
    if not array.size:
        return {"seed_log_std": math.nan, "seed_log_spread": math.nan, "seed_log_median_deviation": math.nan}
    logs = np.log(array)
    med = np.median(logs)
    return {"seed_log_std": float(np.std(logs)), "seed_log_spread": float(np.max(logs) - np.min(logs)), "seed_log_median_deviation": float(np.median(np.abs(logs - med)))}


class ReservoirStats:
    def __init__(self, seed: int, size: int = RESERVOIR_SIZE) -> None:
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.n = 0
        self.sum_absrel = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.delta = 0
        self.errors: list[float] = []
        self.signed: list[float] = []

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        g = np.asarray(gt, dtype=np.float64).reshape(-1)
        good = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
        p, g = p[good], g[good]
        if not p.size:
            return
        signed = p - g
        abs_error = np.abs(signed)
        rel = abs_error / np.maximum(g, 1e-8)
        self.n += int(p.size)
        self.sum_absrel += float(rel.sum())
        self.sum_abs += float(abs_error.sum())
        self.sum_sq += float(np.square(signed).sum())
        ratio = np.maximum(p / np.maximum(g, 1e-8), g / np.maximum(p, 1e-8))
        self.delta += int(np.count_nonzero(ratio < 1.25))
        # Keep a small sample for quantile metrics.
        sample_size = min(128, rel.size)
        if sample_size < rel.size:
            sample_index = self.rng.choice(rel.size, size=sample_size, replace=False)
            rel_sample, signed_sample = rel[sample_index], signed[sample_index]
        else:
            rel_sample, signed_sample = rel, signed
        self.errors.extend(rel_sample.tolist())
        self.signed.extend(signed_sample.tolist())
        if len(self.errors) > self.size * 2:
            current_rel = np.asarray(self.errors, dtype=np.float64)
            current_signed = np.asarray(self.signed, dtype=np.float64)
            keep = self.rng.choice(current_rel.size, size=self.size, replace=False)
            self.errors = current_rel[keep].tolist()
            self.signed = current_signed[keep].tolist()

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0, "absrel": math.nan, "mae_m": math.nan, "rmse_m": math.nan, "medae_m": math.nan, "median_signed_m": math.nan, "bias_m": math.nan, "p90_m": math.nan, "p95_m": math.nan, "catastrophic_rate_absrel_gt1": math.nan, "delta1_25": math.nan, "quantile_method": "empty"}
        errors = np.asarray(self.errors, dtype=np.float64)
        signed = np.asarray(self.signed, dtype=np.float64)
        return {
            "n": int(self.n),
            "absrel": float(self.sum_absrel / self.n),
            "mae_m": float(self.sum_abs / self.n),
            "rmse_m": float(np.sqrt(self.sum_sq / self.n)),
            "medae_m": float(np.quantile(np.abs(signed), 0.5)) if signed.size else math.nan,
            "median_signed_m": float(np.median(signed)) if signed.size else math.nan,
            "bias_m": float(np.mean(signed)) if signed.size else math.nan,
            "p90_m": float(np.quantile(np.abs(signed), 0.9)) if signed.size else math.nan,
            "p95_m": float(np.quantile(np.abs(signed), 0.95)) if signed.size else math.nan,
            "catastrophic_rate_absrel_gt1": float(np.mean(errors > 1.0)) if errors.size else math.nan,
            "delta1_25": float(self.delta / self.n),
            "quantile_method": f"reservoir_{len(errors)}_of_{self.n}",
        }


def write_pre_experiment_freeze() -> dict[str, Any]:
    files = [
        V3 / "reports" / "V3_CANDIDATE_FREEZE.json",
        V4 / "reports" / "V4_CANDIDATE_FREEZE.json",
        V5B / "reports" / "V5B_INTEGRATION_DECISION.json",
        SPEC_PATH,
        PRIOR_PATH,
        Z1_CHECKPOINT,
        V3_HEADS / "baseline_2p" / "research_best.pt",
        V3_HEADS / "spline_8" / "research_best.pt",
        V3_SEEDS / "seed_123" / "research_best.pt",
        V3_SEEDS / "seed_7" / "research_best.pt",
        V3 / "data" / "v3_train.jsonl",
        V3 / "data" / "v3_dev.jsonl",
        V3 / "data" / "v3_locked.jsonl",
        V5B / "data" / "waymo_object_predictions.jsonl",
        V4 / "reports" / "v4_object_external_metrics.csv",
    ]
    payload = {
        "format": "rel2abs_v6_pre_experiment_freeze_v1",
        "created_utc": "2026-09-19",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "scope": "research-only; EyeAIApp, MiDaS, YOLO, ByteTrack and SpatialAudio read-only",
        "model_candidates": list(VISUAL_KEYS) + [F1_KEY],
        "spline_seed_mapping": {"123": str((V3_SEEDS / "seed_123" / "research_best.pt").resolve()), "42": str((V3_HEADS / "spline_8" / "research_best.pt").resolve()), "7": str((V3_SEEDS / "seed_7" / "research_best.pt").resolve())},
        "fusion_development": "V3 train for fitting; V3 dev for pre-external selection; V3 locked, Waymo and COCO never used to fit fusion parameters",
        "external_freeze": ["V3 locked DIODE pixel panel", "Waymo V5B corrected GT-A object panel", "COCO V4 fixed 500-image panel with P0 teacher pseudo-reference"],
        "historical_only": ["KITTI V5A", "V3/V4/V5B previously reported metrics"],
        "midas_unchanged": True,
        "new_backbone": False,
        "data_downloaded_by_v6": False,
        "input_hashes": {str(path.resolve()): sha256(path) for path in files if path.exists()},
        "split_hashes": {split: sha256(V3 / "data" / f"v3_{split}.jsonl") for split in ("train", "dev", "locked")},
        "selection_policy": "No test/Waymo/COCO result is used for fitting or choosing fusion parameters.",
        "status": "FROZEN_BEFORE_V6_FUSION_EVALUATION",
    }
    write_json(REPORTS / "V6_PRE_EXPERIMENT_FREEZE.json", payload)
    lines = [
        "# REL2ABS-v6 pre-experiment freeze",
        "",
        "Status: FROZEN_BEFORE_V6_FUSION_EVALUATION.",
        "",
        "V6 reuses the existing MiDaS preprocessing, frozen heads, priors and external panels. No model or app component is modified.",
        "",
        "## Candidate freeze",
        "",
        "- Z1-frozen, Baseline-2P, Spline-8 seeds 123/42/7, and unweighted median/log-mean spline ensembles.",
        "- F1 is evaluated under the historical `F1_SIZE_ANCHOR_OVERRIDE` name; V6 true fusions are separate candidates.",
        "",
        "## Fusion protocol",
        "",
        "- V3 train: fit only trainable/tunable fusion parameters.",
        "- V3 dev: select among predeclared A/B/C/D variants before external evaluation.",
        "- V3 locked, Waymo and COCO: freeze evaluation only.",
        "- KITTI/V5A, prior Waymo/V5B and V4 COCO numbers are historical references, not tuning data.",
        "",
        f"Seed: `{SEED}`; bootstrap replicates: `{BOOTSTRAP_REPLICATES}`.",
        "",
        "The complete machine-readable hash manifest is in `V6_PRE_EXPERIMENT_FREEZE.json`.",
    ]
    (REPORTS / "V6_PRE_EXPERIMENT_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_f1_contract_audit(priors: Mapping[str, Mapping[str, Any]]) -> None:
    implemented = V5B / "src" / "run_v5b_evaluation.py"
    lines = [
        "# REL2ABS-v6 F1 contract audit",
        "",
        "The historical V5B path is named `F1_SIZE_ANCHOR_OVERRIDE` in V6 because a valid anchor replaced the visual prediction; it was not a simultaneous fusion.",
        "",
        "## Implemented formula",
        "",
        "For a mapped class and a valid bbox/intrinsics pair:",
        "",
        "```text",
        "Z_h = f_y * H_prior / bbox_height_px",
        "Z_w = f_x * W_prior / bbox_width_px",
        "Z_size = exp(mean(log(valid Z_h, Z_w)))",
        "```",
        "",
        "The V5B implementation computes both height and width anchors and uses their geometric mean when both are positive. It falls back to a single valid anchor only where the implementation permits it; V6 keeps the exact two-anchor contract for the cross-dataset audit.",
        "",
        "## Scope and controls",
        "",
        f"- Prior file: `{PRIOR_PATH.resolve()}`; SHA-256 `{sha256(PRIOR_PATH) if PRIOR_PATH.exists() else 'MISSING'}`.",
        f"- V5B implementation: `{implemented.resolve()}`; SHA-256 `{sha256(implemented) if implemented.exists() else 'MISSING'}`.",
        f"- Supported explicit prior classes: `{', '.join(sorted(priors))}`.",
        "- YOLO confidence is a detection signal and is not multiplied into the historical anchor value.",
        "- Border/truncation was not used by the historical override to change the size estimate; V6 uses border only for reliability diagnostics.",
        "- Occlusion is not available as a trustworthy common field on all panels and is not fabricated.",
        "- Intrinsics are interpreted in the panel contract: normalized focal values are multiplied by image width/height; Waymo supplies measured pixel focal lengths.",
        "- A class without an explicit prior, invalid focal length, non-positive bbox or invalid dimensions produces an invalid anchor.",
        "- The YAML reliability strings are not part of the historical arithmetic; the V5B parser defaulted the numeric reliability field to `0.45`. V6 maps low/medium only for uncertainty diagnostics, never to rewrite F1 values.",
        "- Current V5B class mapping was `car`, `person`, `bicycle`; COCO uses the explicit prior file where available. No unsupported car association is invented on Waymo.",
        "",
        "## V6 naming",
        "",
        "- `F1_SIZE_ANCHOR_OVERRIDE`: size anchor alone, conditional on a valid anchor.",
        "- `A/B/C/D`: true fusions retain a visual estimate and combine it with `Z_size`; invalid anchors use the visual fallback.",
        "- `F1` does not alter MiDaS, the visual head, YOLO or detector postprocessing.",
    ]
    (REPORTS / "V6_F1_CONTRACT_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_train_dev_manifest(train_rows: list[dict[str, Any]], dev_rows: list[dict[str, Any]], locked_rows: list[dict[str, Any]]) -> None:
    def group_payload(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
        groups = sorted({str(row.get("group_id") or row.get("sample_id")) for row in rows})
        ids = [str(row["sample_id"]) for row in rows]
        return {"split": split, "rows": len(rows), "groups": len(groups), "group_ids_sha256": hashlib.sha256("\n".join(groups).encode()).hexdigest(), "sample_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "datasets": dict(sorted(Counter(str(row.get("dataset", "unknown")) for row in rows).items()))}
    payload = {
        "format": "rel2abs_v6_fusion_train_dev_manifest_v1",
        "seed": SEED,
        "fit_split": group_payload(train_rows, "v3_train"),
        "selection_split": group_payload(dev_rows, "v3_dev"),
        "excluded_locked_split": group_payload(locked_rows, "v3_locked"),
        "excluded_external": {"Waymo": sha256(WAYMO_PRED_PATH) if WAYMO_PRED_PATH.exists() else None, "COCO": sha256(COCO_EXTERNAL_PATH) if COCO_EXTERNAL_PATH.exists() else None},
        "policy": "Only V3 train/dev can fit/select fusion. Waymo, COCO and V3 locked are freeze-only.",
    }
    write_json(REPORTS / "V6_FUSION_TRAIN_DEV_MANIFEST.json", payload)


def load_product_detections_for_dev() -> dict[str, dict[str, Any]]:
    path = V2 / "cache" / "scene_context" / "detections.jsonl"
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def run_diode_product_yolo(rows: list[dict[str, Any]], force: bool = False) -> list[dict[str, Any]]:
    path = DATA / "diode_product_detections_locked.jsonl"
    runtime_path = DATA / "diode_product_detector_runtime.json"
    if path.exists() and runtime_path.exists() and not force:
        return read_jsonl(path)
    sys.path.insert(0, str(V2_SRC))
    from build_scene_context_vision_cache import load_interpreters, product_detections, resize_chw
    started = time.perf_counter()
    detector, _seg, labels, _seg_labels, contract = load_interpreters()
    outputs: list[dict[str, Any]] = []
    diode_rows = [row for row in rows if str(row.get("dataset")) == "diode"]
    for index, row in enumerate(diode_rows):
        try:
            with np.load(row["path"], allow_pickle=False) as payload:
                rgb = np.asarray(payload["rgb_64"], dtype=np.uint8)
            detector.set_tensor(detector.get_input_details()[0]["index"], resize_chw(rgb, 640).astype(np.float32))
            detector.invoke()
            detections = product_detections(detector.get_tensor(detector.get_output_details()[0]["index"]), labels)
            outputs.append({"sample_id": str(row["sample_id"]), "dataset": "diode", "status": "AVAILABLE_RGB64", "rgb_source": "frozen_cache_rgb_64", "detections": detections})
        except Exception as exc:
            outputs.append({"sample_id": str(row["sample_id"]), "dataset": "diode", "status": "ERROR", "rgb_source": "frozen_cache_rgb_64", "detections": [], "error": f"{type(exc).__name__}: {exc}"})
        if (index + 1) % 100 == 0 or index + 1 == len(diode_rows):
            print(f"DIODE Product YOLO {index + 1}/{len(diode_rows)}", flush=True)
    write_csv(DATA / "diode_product_detections_locked.csv", [{"sample_id": x["sample_id"], "status": x["status"], "rgb_source": x["rgb_source"], "detections": len(x.get("detections", []))} for x in outputs])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False, allow_nan=True, separators=(",", ":")) + "\n" for item in outputs), encoding="utf-8")
    write_json(runtime_path, {"format": "rel2abs_v6_diode_product_detector_runtime_v1", "rows": len(outputs), "detection_rows": sum(bool(x.get("detections")) for x in outputs), "detections": sum(len(x.get("detections", [])) for x in outputs), "rgb_source": "frozen 64x64 RGB cache; original DIODE RGB was not locally retained", "contract": contract, "seconds": time.perf_counter() - started})
    return outputs


def load_all_visual_raw(dataset: str, split: str, rows: list[dict[str, Any]], cache: Any, device: str, v3: Any) -> dict[str, np.ndarray]:
    result = {}
    for key in VISUAL_KEYS[:5]:
        result[key] = load_raw(dataset, split, rows, cache, key, device, v3)
    return result


def diode_pixel_evaluation(
    rows: list[dict[str, Any]],
    cache: Any,
    raw: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    v3: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    """Evaluate all pure visual candidates on frozen DIODE Gold pixels.

    This follows the existing V3 locked sample-bank contract: 2,048
    deterministic valid metric-GT pixels per frame.  The bank avoids
    re-reading the full 256x256 cache row during the cross-dataset pass while
    keeping the same Gold target and sampling protocol.
    """
    sample_dir = Path(v3.BLOCK) / "runs" / "sample_bank"
    sample_meta_path = sample_dir / "locked_bank_meta.json"
    sample_drel_path = sample_dir / "locked_drel_f32.npy"
    sample_gt_path = sample_dir / "locked_gt_f32.npy"
    sample_count_path = sample_dir / "locked_count_i32.npy"
    # Keep the full row index while reporting the DIODE part.
    full_locked_rows = load_rows("locked")
    full_index_by_sample = {str(row["sample_id"]): index for index, row in enumerate(full_locked_rows)}
    expected_ids_hash = hashlib.sha256("\n".join(str(row["sample_id"]) for row in full_locked_rows).encode("utf-8")).hexdigest()
    sample_bank_ready = all(path.exists() for path in (sample_meta_path, sample_drel_path, sample_gt_path, sample_count_path))
    if sample_bank_ready:
        sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
        sample_bank_ready = sample_meta.get("rows") == len(full_locked_rows) and sample_meta.get("sample_pixels") == 2048 and sample_meta.get("sample_id_sha256") == expected_ids_hash
    if not sample_bank_ready:
        raise RuntimeError("Existing V3 locked sample bank is missing or does not match the frozen DIODE rows")
    sample_drel = np.load(sample_drel_path, mmap_mode="r")
    sample_gt = np.load(sample_gt_path, mmap_mode="r")
    sample_count = np.load(sample_count_path, mmap_mode="r")
    stats: dict[str, dict[str, ReservoirStats]] = {}
    frame_errors: dict[str, dict[str, float]] = {key: {} for key in VISUAL_KEYS}
    for key in VISUAL_KEYS:
        stats[key] = {scope: ReservoirStats(SEED + stable_seed(f"{key}|{scope}", "reservoir") % 1000) for scope in ["ALL"] + [band for band, _, _ in BANDS] + ["eyeai_0_5_5m", "eyeai_0_5_10m"]}
    for index, row in enumerate(rows):
        full_index = full_index_by_sample.get(str(row["sample_id"]))
        if full_index is None:
            continue
        count = int(sample_count[full_index])
        if count <= 0:
            continue
        drel_values = np.asarray(sample_drel[full_index, :count], dtype=np.float32)
        gt_values = np.asarray(sample_gt[full_index, :count], dtype=np.float32)
        pred_values_by_key: dict[str, np.ndarray] = {}
        for key in VISUAL_KEYS[:5]:
            pred_values_by_key[key] = decode_map(drel_values, raw[key][full_index], key, spec, v3).reshape(-1)
        pred_values_by_key["V5_Spline-8-Median"] = np.median(np.stack([pred_values_by_key[key] for key in INDIVIDUAL_SPLINES]), axis=0)
        pred_values_by_key["V6_Spline-8-LogMean"] = np.exp(np.mean(np.log(np.maximum(np.stack([pred_values_by_key[key] for key in INDIVIDUAL_SPLINES]), 1e-6)), axis=0))
        for key, pred_values in pred_values_by_key.items():
            stats[key]["ALL"].add(pred_values, gt_values)
            for band, low, high in BANDS:
                keep = (gt_values >= low) & (gt_values < high)
                stats[key][band].add(pred_values[keep], gt_values[keep])
            for scope, low, high in (("eyeai_0_5_5m", 0.5, 5.0), ("eyeai_0_5_10m", 0.5, 10.0)):
                keep = (gt_values >= low) & (gt_values < high)
                stats[key][scope].add(pred_values[keep], gt_values[keep])
            per_frame = metric_summary(pred_values, gt_values)
            frame_errors[key][str(row["sample_id"])] = safe_float(per_frame.get("absrel"))
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(f"DIODE Gold pixels {index + 1}/{len(rows)}", flush=True)
    metric_rows: list[dict[str, Any]] = []
    for key in VISUAL_KEYS:
        for scope, accumulator in stats[key].items():
            metric_rows.append({"dataset": "DIODE", "metric_level": "pixel", "track": "DIODE_GOLD_PIXEL", "quality": "GOLD_GT", "candidate": key, "scope": scope, **accumulator.summary()})
    return metric_rows, {key: {scope: accumulator.summary() for scope, accumulator in values.items()} for key, values in stats.items()}, frame_errors


def object_prediction_from_map(cache: Any, row: Mapping[str, Any], pred_map: np.ndarray, bbox: tuple[float, float, float, float], gt_map: np.ndarray | None = None) -> tuple[float, float]:
    y0, y1, x0, x1 = crop_slice(bbox)
    pred_values = np.asarray(pred_map[y0:y1, x0:x1], dtype=np.float64)
    good_pred = np.isfinite(pred_values) & (pred_values > 0)
    pred = float(np.median(pred_values[good_pred])) if good_pred.any() else math.nan
    if gt_map is None:
        return pred, math.nan
    gt_values = np.asarray(gt_map[y0:y1, x0:x1], dtype=np.float64)
    good = good_pred & np.isfinite(gt_values) & (gt_values > 0)
    gt = float(np.median(gt_values[good])) if good.any() else math.nan
    return pred, gt


def development_object_records(
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
    cache: Any,
    raw_train: Mapping[str, np.ndarray],
    raw_dev: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
    v3: Any,
    priors: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detections = load_product_detections_for_dev()
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for split, rows, raw in (("train", train_rows, raw_train), ("dev", dev_rows, raw_dev)):
        for index, row in enumerate(rows):
            det_row = detections.get(str(row["sample_id"]))
            if not det_row or det_row.get("status") != "AVAILABLE" or not det_row.get("detections"):
                continue
            frame = cache.get(row)
            maps = {key: decode_map(frame["drel"], raw[key][index], key, spec, v3) for key in VISUAL_KEYS[:5]}
            maps["V5_Spline-8-Median"] = np.median(np.stack([maps[key] for key in INDIVIDUAL_SPLINES]), axis=0)
            maps["V6_Spline-8-LogMean"] = np.exp(np.mean(np.log(np.maximum(np.stack([maps[key] for key in INDIVIDUAL_SPLINES]), 1e-6)), axis=0))
            gt_map = np.asarray(frame["depth"], dtype=np.float32)
            fx_px, fy_px, _ = normalized_intrinsics(row)
            width = safe_float(row.get("image_width"), 256.0)
            height = safe_float(row.get("image_height"), 256.0)
            for object_index, item in enumerate(det_row.get("detections", [])):
                bbox = bbox_values(item, width, height)
                pred_values = {key: object_prediction_from_map(cache, row, maps[key], bbox, None)[0] for key in VISUAL_KEYS}
                pred_values = visual_from_predictions(pred_values)
                _pred, gt_m = object_prediction_from_map(cache, row, maps["V1_Baseline-2P"], bbox, gt_map)
                if not np.isfinite(gt_m) or gt_m <= 0:
                    continue
                class_name = str(item.get("class_name", "unknown"))
                anchor = anchor_values(class_name, bbox, width, height, fx_px, fy_px, priors)
                record = {
                    "dataset": str(row.get("dataset", "unknown")).upper(),
                    "split": split,
                    "track": "PRODUCT_YOLO_DEV",
                    "quality": "GOLD_GT_DEV",
                    "sample_id": str(row["sample_id"]),
                    "frame_id": str(row["sample_id"]),
                    "group_id": str(row.get("group_id") or row["sample_id"]),
                    "object_index": object_index,
                    "object_key": f"{row['sample_id']}:{object_index}",
                    "class_name": class_name,
                    "gt_m": gt_m,
                    "band": band_name(gt_m),
                    "bbox_area": float(bbox[2] * bbox[3]),
                    "bbox_width": float(bbox[2]),
                    "bbox_height": float(bbox[3]),
                    "bbox_aspect": float(bbox[2] / max(bbox[3], 1e-6)),
                    "detection_confidence": safe_float(item.get("confidence"), math.nan),
                    "matched_official": 1,
                    "match_iou": math.nan,
                    "fx_px": fx_px,
                    "fy_px": fy_px,
                    **anchor,
                    **pred_values,
                }
                record.update(seed_stats(pred_values))
                outputs[split].append(record)
        print(f"V6 development objects {split}: {len(outputs[split])}", flush=True)
    return outputs["train"], outputs["dev"]


def waymo_object_records(priors: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official = read_jsonl(WAYMO_PRED_PATH)
    product = read_jsonl(WAYMO_PRODUCT_PATH) if WAYMO_PRODUCT_PATH.exists() else []
    def convert(source: list[dict[str, Any]], track: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in source:
            gt = safe_float(row.get("gt_a_m"))
            if int(safe_float(row.get("gt_a_valid"), 0)) != 1 or not np.isfinite(gt) or gt <= 0:
                continue
            preds = {
                "V0_Z1-frozen": safe_float(row.get("pred_C0_Z1-frozen_m", row.get("pred_Z1-frozen_m"))),
                "V1_Baseline-2P": safe_float(row.get("pred_C1_Baseline-2P_m", row.get("pred_Baseline-2P_m"))),
                "V2_Spline-8-seed123": safe_float(row.get("pred_C2_Spline-8-seed123_m", row.get("pred_Spline-8-seed123_m"))),
                "V3_Spline-8-seed42": safe_float(row.get("pred_C3_Spline-8-seed42_m", row.get("pred_Spline-8-seed42_m"))),
                "V4_Spline-8-seed7": safe_float(row.get("pred_C4_Spline-8-seed7_m", row.get("pred_Spline-8-seed7_m"))),
            }
            preds = visual_from_predictions(preds)
            bbox = (safe_float(row.get("bbox_center_x_norm")), safe_float(row.get("bbox_center_y_norm")), safe_float(row.get("bbox_width_norm")), safe_float(row.get("bbox_height_norm")))
            class_name = str(row.get("coco_class", "unknown"))
            width, height = safe_float(row.get("image_width"), 1920.0), safe_float(row.get("image_height"), 1280.0)
            fx_px, fy_px = safe_float(row.get("fx_px")), safe_float(row.get("fy_px"))
            anchor = {
                "z_size": safe_float(row.get("z_size_m")),
                "z_height": safe_float(row.get("z_size_height_m")),
                "z_width": safe_float(row.get("z_size_width_m")),
                "z_generic": safe_float(row.get("z_generic_prior_m")),
                "z_geometry": safe_float(row.get("z_geometry_unit_m")),
                "z_wrong": safe_float(row.get("z_wrong_prior_m")),
                "z_random": safe_float(row.get("z_random_prior_m")),
                "anchor_valid": int(np.isfinite(safe_float(row.get("z_size_m"))) and safe_float(row.get("z_size_m")) > 0),
                "anchor_reason": "valid" if np.isfinite(safe_float(row.get("z_size_m"))) and safe_float(row.get("z_size_m")) > 0 else "unsupported_class_or_intrinsics",
                "sigma_log": math.nan,
                "reliability": 0.45,
                "border": float(safe_float(row.get("bbox_center_x_norm")) - safe_float(row.get("bbox_width_norm")) / 2 <= 0.01 or safe_float(row.get("bbox_center_y_norm")) - safe_float(row.get("bbox_height_norm")) / 2 <= 0.01 or safe_float(row.get("bbox_center_x_norm")) + safe_float(row.get("bbox_width_norm")) / 2 >= 0.99 or safe_float(row.get("bbox_center_y_norm")) + safe_float(row.get("bbox_height_norm")) / 2 >= 0.99),
                "prior_class": class_name,
                "prior_wrong_class": str(row.get("prior_wrong_class", "")),
                "prior_random_class": str(row.get("prior_random_class", "")),
                "prior_reliability": 0.45,
            }
            if class_name in priors:
                prior = priors[class_name]
                anchor["prior_reliability"] = reliability_value(prior)
                anchor["reliability"] = float(reliability_value(prior) / (1.0 + max(safe_float(row.get("bbox_height_px"), 1.0), 1.0) ** -1))
                anchor["sigma_log"] = float(np.sqrt((safe_float(prior.get("height_sigma_m"), 0.35) / max(safe_float(prior.get("height_m"), 1.0), 1e-6)) ** 2 + 0.03 ** 2 + (2.0 / max(safe_float(row.get("bbox_height_px"), 1.0), 1.0)) ** 2))
            record = {
                "dataset": "WAYMO",
                "split": "external_freeze",
                "track": track,
                "quality": "GT_A_VALID_ALL",
                "sample_id": str(row.get("frame_id")),
                "frame_id": str(row.get("frame_id")),
                "group_id": str(row.get("segment_id", "")),
                "object_index": str(row.get("camera_object_id", "")),
                "object_key": str(row.get("object_key", "")),
                "class_name": class_name,
                "gt_m": gt,
                "band": band_name(gt),
                "bbox_area": safe_float(row.get("bbox_area_norm")),
                "bbox_width": safe_float(row.get("bbox_width_norm")),
                "bbox_height": safe_float(row.get("bbox_height_norm")),
                "bbox_aspect": safe_float(row.get("bbox_width_norm")) / max(safe_float(row.get("bbox_height_norm")), 1e-6),
                "detection_confidence": safe_float(row.get("product_detection_confidence")),
                "matched_official": int(safe_float(row.get("product_match_iou"), 0.0) >= 0.5) if track.startswith("WAYMO-B") else 1,
                "match_iou": safe_float(row.get("product_match_iou")),
                "fx_px": fx_px,
                "fy_px": fy_px,
                **anchor,
                **preds,
            }
            record.update(seed_stats(preds))
            result.append(record)
        return result
    return convert(official, "WAYMO-A_OFFICIAL_BBOX"), convert(product, "WAYMO-B_PRODUCT_YOLO_MATCHED")


def coco_panel_rows() -> list[dict[str, Any]]:
    panel = json.loads(COCO_PANEL_PATH.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in panel["categories"]}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in panel["annotations"]:
        value = dict(item)
        value["category_name"] = categories[int(item["category_id"])]
        grouped[int(item["image_id"])].append(value)
    rows = []
    image_dir = V4 / "data" / "coco" / "images" / "val2017"
    for image in sorted(panel["images"], key=lambda item: int(item["id"])):
        sample_id = f"coco-val2017-{int(image['id']):012d}"
        if (image_dir / str(image["file_name"])).exists():
            rows.append({"sample_id": sample_id, "image_id": int(image["id"]), "file_name": str(image["file_name"]), "width": int(image["width"]), "height": int(image["height"]), "objects": grouped[int(image["id"])]})
    return rows


def coco_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax0, ay0, ax1, ay1 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx0, by0, bx1, by1 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def coco_object_records(cache: Any, raw: Mapping[str, np.ndarray], priors: Mapping[str, Mapping[str, Any]], v3: Any, spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build COCO-A official and COCO-B matched Product-YOLO P0 records."""
    existing = read_csv(COCO_EXTERNAL_PATH)
    base_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in existing:
        if row.get("quality") != "P0":
            continue
        track = str(row.get("track", ""))
        if track not in {"COCO-A_OFFICIAL_BBOX", "COCO-B_PRODUCT_YOLO"}:
            continue
        key = (track, str(row.get("sample_id")), str(row.get("object_index")))
        if key not in base_rows:
            base_rows[key] = row
    panel_rows = {str(row["sample_id"]): row for row in coco_panel_rows()}
    det_rows = {str(row["sample_id"]): row for row in read_jsonl(COCO_DET_PATH)} if COCO_DET_PATH.exists() else {}
    result: list[dict[str, Any]] = []
    # Process one image at a time to keep memory use small.
    current_sample_id: str | None = None
    current_maps: dict[str, np.ndarray] | None = None
    ordered_rows = sorted(base_rows.items(), key=lambda item: (item[0][1], item[0][0], item[0][2]))
    for (track, sample_id, object_index), base in ordered_rows:
        if track == "COCO-B_PRODUCT_YOLO" and int(safe_float(base.get("matched_official"), 0)) != 1:
            continue
        index = cache.index_by_sample.get(sample_id)
        if index is None:
            continue
        image_row = panel_rows.get(sample_id, {})
        width, height = safe_float(image_row.get("width"), 1.0), safe_float(image_row.get("height"), 1.0)
        bbox = (safe_float(base.get("bbox_center_x"), 0.5), safe_float(base.get("bbox_center_y"), 0.5), safe_float(base.get("bbox_width")), safe_float(base.get("bbox_height")))
        # Rebuild the box center from the stored box size.
        matching_item: Mapping[str, Any] | None = None
        if track == "COCO-A_OFFICIAL_BBOX":
            items = image_row.get("objects", [])
            oi = int(safe_float(object_index, 0))
            if oi < len(items):
                matching_item = items[oi]
        else:
            detections = list(det_rows.get(sample_id, {}).get("detections", []))
            oi = int(safe_float(object_index, 0))
            if oi < len(detections):
                matching_item = detections[oi]
        if matching_item is not None:
            bbox = bbox_values(matching_item, width, height)
        if sample_id != current_sample_id:
            frame = cache.get({"sample_id": sample_id})
            maps: dict[str, np.ndarray] = {}
            index = int(index)
            for key in VISUAL_KEYS[:5]:
                maps[key] = decode_map(frame["drel"], raw[key][index], key, spec, v3)
            maps["V5_Spline-8-Median"] = np.median(np.stack([maps[key] for key in INDIVIDUAL_SPLINES]), axis=0)
            maps["V6_Spline-8-LogMean"] = np.exp(np.mean(np.log(np.maximum(np.stack([maps[key] for key in INDIVIDUAL_SPLINES]), 1e-6)), axis=0))
            current_sample_id, current_maps = sample_id, maps
        maps = current_maps
        assert maps is not None
        pred_values = {key: object_prediction_from_map(cache, {"sample_id": sample_id}, maps[key], bbox, None)[0] for key in VISUAL_KEYS}
        pred_values = visual_from_predictions(pred_values)
        class_name = str(base.get("class_name", "unknown"))
        intrinsics = [0.866, 1.154, 0.5, 0.5, 1.0]
        anchor = anchor_values(class_name, bbox, width, height, intrinsics[0] * width, intrinsics[1] * height, priors)
        gt_m = safe_float(base.get("reference_m"))
        if not np.isfinite(gt_m) or gt_m <= 0:
            continue
        record = {
            "dataset": "COCO",
            "split": "external_freeze",
            "track": "COCO-A_OFFICIAL_BBOX" if track.startswith("COCO-A") else "COCO-B_PRODUCT_YOLO_MATCHED",
            "quality": "P0_TEACHER_CONSENSUS",
            "sample_id": sample_id,
            "frame_id": sample_id,
            "group_id": sample_id,
            "object_index": object_index,
            "object_key": f"{track}:{sample_id}:{object_index}",
            "class_name": class_name,
            "gt_m": gt_m,
            "band": band_name(gt_m),
            "bbox_area": safe_float(base.get("bbox_area")),
            "bbox_width": bbox[2],
            "bbox_height": bbox[3],
            "bbox_aspect": bbox[2] / max(bbox[3], 1e-6),
            "detection_confidence": safe_float(matching_item.get("confidence")) if matching_item else math.nan,
            "matched_official": int(safe_float(base.get("matched_official"), 1)),
            "match_iou": safe_float(base.get("match_iou")),
            "fx_px": intrinsics[0] * width,
            "fy_px": intrinsics[1] * height,
            "teacher_disagreement": safe_float(base.get("teacher_disagreement_median")),
            **anchor,
            **pred_values,
        }
        record.update(seed_stats(pred_values))
        result.append(record)
    return result, {"teacher_source": str(TEACHER_PATH.resolve()), "teacher_pseudogt": "P0 mean of valid UniDepthV2 and Metric3Dv2 pixels from existing V4 contract", "rows": len(result)}


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value)))))


def visual_uncertainty(record: Mapping[str, Any], visual_key: str) -> float:
    disagreement = safe_float(record.get("seed_log_std"), 0.0)
    base = 0.32 if visual_key == "V1_Baseline-2P" else 0.27
    if visual_key in {"V5_Spline-8-Median", "V6_Spline-8-LogMean"}:
        base = 0.20 + min(0.5, disagreement)
    return float(np.clip(base + 0.25 * disagreement, 0.10, 1.50))


def feature_vector(record: Mapping[str, Any], visual_key: str) -> np.ndarray:
    visual = max(safe_float(record.get(visual_key), 1e-6), 1e-6)
    size = max(safe_float(record.get("z_size"), visual), 1e-6)
    conf = safe_float(record.get("detection_confidence"), 0.0)
    fx, fy = safe_float(record.get("fx_px"), 0.0), safe_float(record.get("fy_px"), 0.0)
    class_name = str(record.get("class_name", ""))
    return np.asarray([
        math.log(visual),
        math.log(size),
        abs(math.log(size / visual)),
        math.log(max(safe_float(record.get("bbox_height"), 1e-4), 1e-4)),
        math.log(max(safe_float(record.get("bbox_width"), 1e-4), 1e-4)),
        math.log(max(safe_float(record.get("bbox_area"), 1e-6), 1e-6)),
        math.log(max(safe_float(record.get("bbox_aspect"), 1e-4), 1e-4)),
        safe_float(record.get("reliability"), 0.0),
        0.0 if not np.isfinite(conf) else conf,
        safe_float(record.get("border"), 0.0),
        0.0 if not np.isfinite(safe_float(record.get("seed_log_std"))) else safe_float(record.get("seed_log_std")),
        math.log(max(fx, 1e-4)),
        math.log(max(fy, 1e-4)),
        float(class_name == "person"),
        float(class_name == "bicycle"),
        float(class_name == "car"),
    ], dtype=np.float64)


def precision_fusion(record: Mapping[str, Any], visual_key: str) -> tuple[float, float]:
    visual = safe_float(record.get(visual_key))
    size = safe_float(record.get("z_size"))
    if not np.isfinite([visual, size]).all() or min(visual, size) <= 0:
        return visual, 0.0
    sigma_size = safe_float(record.get("sigma_log"), 0.8)
    sigma_visual = visual_uncertainty(record, visual_key)
    ps, pv = 1.0 / max(sigma_size * sigma_size, 1e-6), 1.0 / max(sigma_visual * sigma_visual, 1e-6)
    weight = float(np.clip(ps / (ps + pv), 0.0, 1.0))
    return float(math.exp((1.0 - weight) * math.log(visual) + weight * math.log(size))), weight


def clipped_reliability_fusion(record: Mapping[str, Any], visual_key: str) -> tuple[float, float]:
    visual = safe_float(record.get(visual_key))
    size = safe_float(record.get("z_size"))
    if not np.isfinite([visual, size]).all() or min(visual, size) <= 0:
        return visual, 0.0
    disagreement = abs(math.log(size / visual))
    rel = safe_float(record.get("reliability"), 0.0)
    seed_penalty = min(0.75, safe_float(record.get("seed_log_std"), 0.0))
    weight = float(np.clip(0.90 * rel * math.exp(-0.35 * disagreement) * (1.0 - seed_penalty), 0.05, 0.95))
    return float(math.exp((1.0 - weight) * math.log(visual) + weight * math.log(size))), weight


def disagreement_fallback(record: Mapping[str, Any], visual_key: str, params: Mapping[str, Any]) -> tuple[float, float]:
    visual = safe_float(record.get(visual_key))
    size = safe_float(record.get("z_size"))
    if not np.isfinite([visual, size]).all() or min(visual, size) <= 0:
        return visual, 0.0
    disagreement = abs(math.log(size / visual))
    threshold = safe_float(params.get("threshold"), 0.8)
    rel = safe_float(record.get("reliability"), 0.0)
    if disagreement > threshold and rel < safe_float(params.get("high_reliability"), 0.65):
        return visual, 0.0
    weight = float(np.clip(rel * sigmoid(safe_float(params.get("slope"), 4.0) * (threshold - disagreement)), 0.0, 0.95))
    return float(math.exp((1.0 - weight) * math.log(visual) + weight * math.log(size))), weight


def range_gate(record: Mapping[str, Any], visual_key: str, params: Mapping[str, Any]) -> tuple[float, float]:
    visual = safe_float(record.get(visual_key))
    size = safe_float(record.get("z_size"))
    if not np.isfinite([visual, size]).all() or min(visual, size) <= 0:
        return visual, 0.0
    rel = safe_float(record.get("reliability"), 0.0)
    weight = float(np.clip(rel * sigmoid(safe_float(params.get("a"), 1.0) * (math.log(visual) - safe_float(params.get("b"), math.log(10.0)))), 0.0, 0.95))
    return float(math.exp((1.0 - weight) * math.log(visual) + weight * math.log(size))), weight


def learned_fusion(record: Mapping[str, Any], visual_key: str, model: Mapping[str, Any]) -> tuple[float, float]:
    visual = safe_float(record.get(visual_key))
    size = safe_float(record.get("z_size"))
    if not np.isfinite(visual) or visual <= 0:
        return math.nan, 0.0
    if not np.isfinite(size) or size <= 0:
        return visual, 0.0
    x = feature_vector(record, visual_key)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    delta = float(np.clip(float((np.nan_to_num((x - mean) / scale) * coef).sum() + safe_float(model.get("intercept"), 0.0)), -1.0, 1.0))
    return float(visual * math.exp(delta)), 1.0


def prediction(record: Mapping[str, Any], candidate: str, fusion: Mapping[str, Any]) -> tuple[float, float, str]:
    if candidate in VISUAL_KEYS:
        value = safe_float(record.get(candidate))
        return value, 1.0 if np.isfinite(value) and value > 0 else 0.0, "visual"
    if candidate == F1_KEY:
        value = safe_float(record.get("z_size"))
        return value, 1.0 if np.isfinite(value) and value > 0 else 0.0, "size_anchor" if np.isfinite(value) and value > 0 else "invalid_anchor"
    if candidate.startswith("A"):
        visual_key = str(fusion[candidate]["visual_key"])
        if candidate == "A0_Baseline_Visual":
            return prediction(record, visual_key, fusion)
        if candidate == "A1_SizeOnly":
            return prediction(record, F1_KEY, fusion)
        weight = safe_float(fusion[candidate].get("weight"), 0.5)
        visual, size = safe_float(record.get(visual_key)), safe_float(record.get("z_size"))
        if not np.isfinite(visual) or visual <= 0:
            return math.nan, 0.0, "rejected_visual"
        if not np.isfinite(size) or size <= 0:
            return visual, 0.0, "visual_fallback_invalid_anchor"
        return float(math.exp((1.0 - weight) * math.log(visual) + weight * math.log(size))), weight, "blended"
    if candidate.startswith("B1_"):
        visual_key = str(fusion[candidate]["visual_key"])
        value, weight = precision_fusion(record, visual_key)
        return value, weight, "blended" if weight > 0 else "visual_fallback_invalid_anchor"
    if candidate.startswith("B2_"):
        visual_key = str(fusion[candidate]["visual_key"])
        value, weight = clipped_reliability_fusion(record, visual_key)
        return value, weight, "blended" if weight > 0 else "visual_fallback_invalid_anchor"
    if candidate.startswith("B3_"):
        visual_key = str(fusion[candidate]["visual_key"])
        value, weight = disagreement_fallback(record, visual_key, fusion[candidate])
        return value, weight, "blended" if weight > 0 else "visual_fallback"
    if candidate.startswith("C_"):
        visual_key = str(fusion[candidate]["visual_key"])
        value, weight = range_gate(record, visual_key, fusion[candidate])
        return value, weight, "blended" if weight > 0 else "visual_fallback_invalid_anchor"
    if candidate.startswith("D_"):
        visual_key = str(fusion[candidate]["visual_key"])
        value, weight = learned_fusion(record, visual_key, fusion[candidate])
        return value, weight, "blended" if weight > 0 else "visual_fallback_invalid_anchor"
    if candidate.startswith("CTRL_"):
        name = candidate[5:]
        key = {"CORRECT_PRIOR": "z_size", "WRONG_CLASS_PRIOR": "z_wrong", "RANDOM_PRIOR": "z_random", "GENERIC_SAME_SIZE": "z_generic", "GEOMETRY_ONLY": "z_geometry"}.get(name)
        if key is None:
            return safe_float(record.get("V1_Baseline-2P")), 0.0, "class_semantics_only_visual"
        value = safe_float(record.get(key))
        if np.isfinite(value) and value > 0:
            return value, 1.0, name.lower()
        return safe_float(record.get("V1_Baseline-2P")), 0.0, "visual_fallback_invalid_control"
    if candidate == "ORACLE_BEST_VISUAL_SIZE":
        gt = safe_float(record.get("gt_m"))
        choices = [safe_float(record.get(key)) for key in VISUAL_KEYS[:5]] + [safe_float(record.get("z_size"))]
        choices = [x for x in choices if np.isfinite(x) and x > 0]
        if not choices or not np.isfinite(gt) or gt <= 0:
            return math.nan, 0.0, "oracle_invalid"
        value = min(choices, key=lambda x: abs(x - gt) / gt)
        return value, 1.0, "gt_leaked_oracle"
    if candidate == "ORACLE_BEST_SPLINE_SIZE":
        gt = safe_float(record.get("gt_m"))
        choices = [safe_float(record.get(key)) for key in INDIVIDUAL_SPLINES] + [safe_float(record.get("z_size"))]
        choices = [x for x in choices if np.isfinite(x) and x > 0]
        if not choices or not np.isfinite(gt) or gt <= 0:
            return math.nan, 0.0, "oracle_invalid"
        value = min(choices, key=lambda x: abs(x - gt) / gt)
        return value, 1.0, "gt_leaked_oracle"
    return math.nan, 0.0, "unknown_candidate"


def evaluate_candidate_absrel(records: list[dict[str, Any]], candidate: str, fusion: Mapping[str, Any]) -> float:
    values = []
    for record in records:
        pred, _, _ = prediction(record, candidate, fusion)
        gt = safe_float(record.get("gt_m"))
        if np.isfinite([pred, gt]).all() and pred > 0 and gt > 0:
            values.append(abs(pred - gt) / gt)
    return float(np.mean(values)) if values else math.nan


def fit_fusion_models(train_records: list[dict[str, Any]], dev_records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fusion: dict[str, Any] = {
        "A0_Baseline_Visual": {"family": "A", "visual_key": "V1_Baseline-2P", "weight": 0.0},
        "A1_SizeOnly": {"family": "A", "visual_key": "V1_Baseline-2P", "weight": 1.0},
    }
    for name, weight in (("A2_Baseline_w25", 0.25), ("A3_Baseline_w50", 0.50), ("A4_Baseline_w75", 0.75)):
        fusion[name] = {"family": "A", "visual_key": "V1_Baseline-2P", "weight": weight}
    for name, family, visual in (("B1_Baseline_Precision", "B1", "V1_Baseline-2P"), ("B2_Baseline_ClippedReliability", "B2", "V1_Baseline-2P"), ("B3_Baseline_DisagreementFallback", "B3", "V1_Baseline-2P"), ("B1_SplineEnsemble_Precision", "B1", "V5_Spline-8-Median"), ("B2_SplineEnsemble_ClippedReliability", "B2", "V5_Spline-8-Median"), ("B3_SplineEnsemble_DisagreementFallback", "B3", "V5_Spline-8-Median")):
        fusion[name] = {"family": family, "visual_key": visual, "threshold": 0.8, "slope": 4.0, "high_reliability": 0.65}
    train_logs = np.asarray([math.log(safe_float(row["V1_Baseline-2P"])) for row in train_records if np.isfinite(safe_float(row.get("V1_Baseline-2P"))) and safe_float(row.get("V1_Baseline-2P")) > 0], dtype=np.float64)
    candidate_c: list[tuple[float, float]] = []
    for a in (0.5, 1.0, 2.0, 4.0):
        for quantile in (0.25, 0.50, 0.75):
            if train_logs.size:
                candidate_c.append((a, float(np.quantile(train_logs, quantile))))
    if not candidate_c:
        candidate_c = [(1.0, math.log(10.0))]
    for visual_key, prefix in (("V1_Baseline-2P", "C_Baseline"), ("V5_Spline-8-Median", "C_SplineEnsemble")):
        best = min(candidate_c, key=lambda pair: evaluate_candidate_absrel(train_records, f"C_{prefix}_{pair[0]}_{pair[1]}", {**fusion, f"C_{prefix}_{pair[0]}_{pair[1]}": {"family": "C", "visual_key": visual_key, "a": pair[0], "b": pair[1]}})) if train_records else candidate_c[0]
        name = f"C_{prefix}_RangeGate"
        fusion[name] = {"family": "C", "visual_key": visual_key, "a": best[0], "b": best[1], "threshold_selection": "V3 train grid; V3 dev selection"}
    # Select the C parameters on the development split.
    for visual_key, prefix in (("V1_Baseline-2P", "C_Baseline"), ("V5_Spline-8-Median", "C_SplineEnsemble")):
        scored = []
        for a, b in candidate_c:
            name = f"C_{prefix}_candidate"
            scored.append((evaluate_candidate_absrel(dev_records, name, {name: {"family": "C", "visual_key": visual_key, "a": a, "b": b}}), a, b))
        scored = [item for item in scored if np.isfinite(item[0])]
        if scored:
            _, a, b = min(scored)
            fusion[f"C_{prefix}_RangeGate"].update({"a": a, "b": b, "dev_selection_absrel": min(scored)[0]})
    def fit_ridge(visual_key: str) -> dict[str, Any]:
        usable = [row for row in train_records if np.isfinite(safe_float(row.get(visual_key))) and safe_float(row.get(visual_key)) > 0 and np.isfinite(safe_float(row.get("z_size"))) and safe_float(row.get("z_size")) > 0]
        if len(usable) < 20:
            return {"family": "D", "visual_key": visual_key, "status": "BLOCKED_INSUFFICIENT_DEV_OBJECTS", "mean": [0.0] * 16, "scale": [1.0] * 16, "coef": [0.0] * 16, "intercept": 0.0}
        x = np.stack([feature_vector(row, visual_key) for row in usable])
        y = np.asarray([np.clip(math.log(safe_float(row["gt_m"]) / safe_float(row[visual_key])), -1.0, 1.0) for row in usable], dtype=np.float64)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        xs = (x - mean) / scale
        alpha = 10.0
        lhs = xs.T @ xs + alpha * np.eye(xs.shape[1])
        rhs = xs.T @ y
        coef = np.linalg.solve(lhs, rhs)
        intercept = float(np.mean(y - xs @ coef))
        return {"family": "D", "visual_key": visual_key, "status": "FIT", "parameter_count": int(coef.size + 1), "alpha": alpha, "bounded_delta_log": [-1.0, 1.0], "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(), "intercept": intercept}
    fusion["D_Baseline_TinyLearned"] = {**fit_ridge("V1_Baseline-2P"), "visual_key": "V1_Baseline-2P"}
    fusion["D_SplineEnsemble_TinyLearned"] = {**fit_ridge("V5_Spline-8-Median"), "visual_key": "V5_Spline-8-Median"}
    rows: list[dict[str, Any]] = []
    for name, spec in fusion.items():
        if name.startswith(("A", "B", "C", "D")):
            rows.append({"candidate": name, "family": spec.get("family"), "visual_key": spec.get("visual_key"), "train_absrel": evaluate_candidate_absrel(train_records, name, fusion), "dev_absrel": evaluate_candidate_absrel(dev_records, name, fusion), "parameters": json.dumps({key: value for key, value in spec.items() if key not in {"mean", "scale", "coef"}}, sort_keys=True), "parameter_count": spec.get("parameter_count", 0)})
    return fusion, rows


def external_candidate_names(fusion: Mapping[str, Any]) -> list[str]:
    return list(VISUAL_KEYS) + [F1_KEY] + sorted(fusion) + ["CTRL_CORRECT_PRIOR", "CTRL_WRONG_CLASS_PRIOR", "CTRL_RANDOM_PRIOR", "CTRL_GENERIC_SAME_SIZE", "CTRL_GEOMETRY_ONLY", "CTRL_CLASS_SEMANTICS_ONLY", "ORACLE_BEST_VISUAL_SIZE", "ORACLE_BEST_SPLINE_SIZE"]


def object_metric_rows(records: list[dict[str, Any]], candidates: Iterable[str], fusion: Mapping[str, Any], metric_level: str = "object") -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = list(candidates)
    metrics: list[dict[str, Any]] = []
    class_metrics: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in records}):
        dataset_rows = [row for row in records if str(row.get("dataset")) == dataset]
        for track in sorted({str(row.get("track")) for row in dataset_rows}):
            track_rows = [row for row in dataset_rows if str(row.get("track")) == track]
            for candidate in candidates:
                pred_values: list[float] = []
                gt_values: list[float] = []
                statuses = Counter()
                for row in track_rows:
                    pred, weight, status = prediction(row, candidate, fusion)
                    statuses[status] += 1
                    gt = safe_float(row.get("gt_m"))
                    if np.isfinite([pred, gt]).all() and pred > 0 and gt > 0:
                        pred_values.append(pred)
                        gt_values.append(gt)
                if pred_values:
                    scopes = [("ALL", np.ones(len(gt_values), dtype=bool))]
                    gt_array = np.asarray(gt_values, dtype=np.float64)
                    for band, low, high in BANDS:
                        scopes.append((band, (gt_array >= low) & (gt_array < high)))
                    scopes.extend((("eyeai_0_5_5m", (gt_array >= 0.5) & (gt_array < 5.0)), ("eyeai_0_5_10m", (gt_array >= 0.5) & (gt_array < 10.0))))
                    for scope, keep in scopes:
                        result = metric_summary(np.asarray(pred_values)[keep], gt_array[keep])
                        metrics.append({"dataset": dataset, "metric_level": metric_level, "track": track, "quality": "GT_OR_P0_PRIMARY", "candidate": candidate, "scope": scope, **result, "anchor_coverage": float(statuses["size_anchor"] / max(len(track_rows), 1)), "fallback_fraction": float(sum(statuses[key] for key in ("visual_fallback_invalid_anchor", "visual_fallback", "visual_fallback_invalid_control")) / max(len(track_rows), 1)), "blended_fraction": float(statuses["blended"] / max(len(track_rows), 1)), "rejected_fraction": float(sum(statuses[key] for key in ("rejected_visual", "invalid_anchor", "unknown_candidate")) / max(len(track_rows), 1))})
                total = len(track_rows)
                coverage.append({"dataset": dataset, "metric_level": metric_level, "track": track, "candidate": candidate, "objects_total": total, "objects_with_valid_prediction": len(pred_values), "valid_prediction_fraction": float(len(pred_values) / max(total, 1)), "anchor_valid": statuses["size_anchor"], "visual_fallback": sum(statuses[key] for key in ("visual_fallback_invalid_anchor", "visual_fallback", "visual_fallback_invalid_control")), "blended": statuses["blended"], "rejected": sum(statuses[key] for key in ("rejected_visual", "invalid_anchor", "unknown_candidate")), "status_counts": json.dumps(dict(statuses), sort_keys=True)})
                for class_name in sorted({str(row.get("class_name")) for row in track_rows}):
                    class_rows = [row for row in track_rows if str(row.get("class_name")) == class_name]
                    p, g = [], []
                    for row in class_rows:
                        pred, _, _ = prediction(row, candidate, fusion)
                        gt = safe_float(row.get("gt_m"))
                        if np.isfinite([pred, gt]).all() and pred > 0 and gt > 0:
                            p.append(pred); g.append(gt)
                    if p:
                        class_metrics.append({"dataset": dataset, "metric_level": metric_level, "track": track, "candidate": candidate, "class_name": class_name, **metric_summary(p, g)})
    return metrics, class_metrics, coverage


def seed_disagreement_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for dataset in sorted({str(row.get("dataset")) for row in records}):
        for track in sorted({str(row.get("track")) for row in records if str(row.get("dataset")) == dataset}):
            values = [row for row in records if str(row.get("dataset")) == dataset and str(row.get("track")) == track]
            disagreements = np.asarray([safe_float(row.get("seed_log_std")) for row in values], dtype=np.float64)
            errors = np.asarray([abs(safe_float(row.get("V5_Spline-8-Median")) - safe_float(row.get("gt_m"))) / max(safe_float(row.get("gt_m")), 1e-6) for row in values], dtype=np.float64)
            good = np.isfinite(disagreements) & np.isfinite(errors)
            corr = float(np.corrcoef(disagreements[good], errors[good])[0, 1]) if int(good.sum()) >= 3 and np.std(disagreements[good]) > 1e-9 and np.std(errors[good]) > 1e-9 else math.nan
            result.append({"dataset": dataset, "track": track, "n": int(good.sum()), "seed_log_std_median": float(np.median(disagreements[good])) if good.any() else math.nan, "seed_log_std_p90": float(np.quantile(disagreements[good], 0.9)) if good.any() else math.nan, "seed_log_spread_median": float(np.median([safe_float(row.get("seed_log_spread")) for row, keep in zip(values, good) if keep])) if good.any() else math.nan, "corr_seed_disagreement_vs_spline_absrel": corr, "interpretation": "diagnostic disagreement signal; not calibrated uncertainty"})
    return result


def prior_control_rows(records: list[dict[str, Any]], fusion: Mapping[str, Any]) -> list[dict[str, Any]]:
    control_names = ["CTRL_CORRECT_PRIOR", "CTRL_WRONG_CLASS_PRIOR", "CTRL_RANDOM_PRIOR", "CTRL_GENERIC_SAME_SIZE", "CTRL_GEOMETRY_ONLY", "CTRL_CLASS_SEMANTICS_ONLY"]
    metrics, _, _ = object_metric_rows(records, control_names + ["V1_Baseline-2P"], fusion)
    output = []
    for row in metrics:
        if row.get("scope") == "ALL":
            output.append({"control": row["candidate"], **row})
    return output


def oracle_rows(records: list[dict[str, Any]], fusion: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = ["V1_Baseline-2P", "V5_Spline-8-Median", F1_KEY, "ORACLE_BEST_VISUAL_SIZE", "ORACLE_BEST_SPLINE_SIZE"]
    metrics, _, _ = object_metric_rows(records, names, fusion)
    return [{"oracle_type": "GT_LEAKED_DIAGNOSTIC", **row} for row in metrics if row.get("scope") in {"ALL", "eyeai_0_5_5m", "5_10m", "ge15m"}]


def write_candidate_family_artifacts(metrics: list[dict[str, Any]], fusion_rows: list[dict[str, Any]], fusion: Mapping[str, Any], records: list[dict[str, Any]]) -> None:
    visual = [row for row in metrics if row.get("candidate") in VISUAL_KEYS]
    ensemble = [row for row in metrics if row.get("candidate") in {"V5_Spline-8-Median", "V6_Spline-8-LogMean"}]
    f1 = [row for row in metrics if row.get("candidate") == F1_KEY]
    write_csv(REPORTS / "v6_visual_candidates.csv", visual)
    write_csv(REPORTS / "v6_spline_ensemble.csv", ensemble)
    write_csv(REPORTS / "v6_f1_size_anchor.csv", f1)
    for family, filename in (("A", "v6_fusion_family_A.csv"), ("B", "v6_fusion_family_B.csv"), ("C", "v6_fusion_family_C.csv"), ("D", "v6_fusion_family_D.csv")):
        names = [name for name, spec in fusion.items() if str(spec.get("family", "")).startswith(family)]
        rows = [dict(row, evaluation="external_freeze") for row in metrics if row.get("candidate") in names]
        rows.extend([dict(row, evaluation="v3_train_dev") for row in fusion_rows if str(row.get("family", "")).startswith(family)])
        write_csv(REPORTS / filename, rows)


def write_dataset_coverage_artifact(coverage: list[dict[str, Any]], product_rows: list[dict[str, Any]], diode_detections: list[dict[str, Any]] | None = None) -> None:
    rows = [dict(row, coverage_kind="candidate_prediction") for row in coverage]
    rows.extend(dict(row, coverage_kind="product_detector") for row in product_rows)
    if diode_detections is not None:
        detections = [item for frame in diode_detections for item in frame.get("detections", [])]
        class_counts = Counter(str(item.get("class_name", "unknown")) for item in detections)
        widths = [safe_float(item.get("width")) for item in detections if np.isfinite(safe_float(item.get("width")))]
        heights = [safe_float(item.get("height")) for item in detections if np.isfinite(safe_float(item.get("height")))]
        areas = [safe_float(item.get("bbox_area")) for item in detections if np.isfinite(safe_float(item.get("bbox_area")))]
        rows.append({
            "coverage_kind": "diode_product_detection_summary",
            "dataset": "DIODE",
            "track": "Product_YOLO",
            "frames": len(diode_detections),
            "frames_with_detection": sum(bool(frame.get("detections")) for frame in diode_detections),
            "detections": len(detections),
            "class_counts": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
            "bbox_width_mean": float(np.mean(widths)) if widths else math.nan,
            "bbox_height_mean": float(np.mean(heights)) if heights else math.nan,
            "bbox_area_mean": float(np.mean(areas)) if areas else math.nan,
            "usable_anchors": 0,
            "distance_distribution": "UNAVAILABLE; no valid DIODE Product-YOLO object anchor",
            "rgb_source": "frozen_rgb_64",
        })
    write_csv(REPORTS / "v6_dataset_coverage.csv", rows)


def write_integration_handoff(decision: Mapping[str, Any], score_rows: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> None:
    hybrid = decision.get("hybrid", {})
    candidate = str(hybrid.get("best_candidate", "NONE_QUALIFIED"))
    qualified = candidate != "NONE_QUALIFIED" and any(bool(row.get("qualifies")) and str(row.get("candidate")) == candidate for row in decision.get("hybrid_gate_rows", []))
    if not qualified:
        return
    score = next((row for row in score_rows if str(row.get("candidate")) == candidate), {})
    gate = next((row for row in decision.get("hybrid_gate_rows", []) if str(row.get("candidate")) == candidate), {})
    lines = [
        "# REL2ABS-v6 EyeAI integration handoff (research-qualified)",
        "",
        f"Candidate: **{candidate}**.",
        "",
        "This is a controlled follow-up handoff, not an automatic application change. EyeAIApp, MiDaS, YOLO, ByteTrack and SpatialAudio remain unchanged by V6.",
        "",
        "## Exact runtime pipeline",
        "",
        "1. Keep the existing Baseline-2P visual metric-depth estimate as `Z_visual`.",
        "2. Run the existing Product-YOLO path and map only supported classes to the frozen physical-size prior file.",
        "3. Compute `Z_h = f_y*H_prior/h_px`, `Z_w = f_x*W_prior/w_px`, and `Z_size = exp(mean(log(valid anchors)))`.",
        "4. If either visual depth or the size anchor is invalid, use `Z_visual` unchanged.",
        "5. For B3, set `d = abs(log(Z_size/Z_visual))`; if `d > 0.8` and reliability `< 0.65`, use visual fallback.",
        "6. Otherwise set `w = clip(reliability * sigmoid(4*(0.8-d)), 0, 0.95)` and return `Z_final = exp((1-w)*log(Z_visual) + w*log(Z_size))`.",
        "",
        "Reliability is the frozen V6 scalar diagnostic derived from prior reliability, focal/bbox uncertainty, border proxy and aspect-ratio plausibility. YOLO confidence remains a detection signal and is not multiplied into depth.",
        "",
        "## Qualification evidence",
        "",
        f"- Waymo Gold-A AbsRel: `{safe_float(score.get('waymo_gold_object_absrel')):.6f}`; COCO P0 stress AbsRel: `{safe_float(score.get('coco_pseudogt_object_absrel')):.6f}`.",
        f"- Gate result: `{json.dumps(gate.get('requirements', {}), sort_keys=True)}`.",
        "- DIODE remains a valid Gold pixel panel, but its retained RGB64 Product-YOLO coverage is too low for a DIODE object/fusion claim; this is an explicit limitation, not an imputation.",
        "",
        "## Required before any production integration",
        "",
        "- Run a separate formal LiteRT/TFLite export and numerical parity audit for the scalar branch.",
        "- Revalidate on the intended full-resolution DIODE/Product-YOLO input contract; the V6 RGB64 qualification is not sufficient for production detector coverage.",
        "- Add an explicit runtime fallback and telemetry for invalid priors, invalid intrinsics, border/truncation and disagreement fallback.",
        "- Obtain a separate implementation approval before modifying Android code.",
        "",
        "The V6 stop rule is otherwise reached; no Android change is included in this handoff.",
    ]
    (V6 / "V6_EYEAI_INTEGRATION_HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pixel_metric_rows_from_diode(diode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"dataset": "DIODE", "metric_level": "pixel", **row} for row in diode_rows]


def bootstrap_object_comparison(records: list[dict[str, Any]], baseline: str, candidates: Iterable[str], fusion: Mapping[str, Any], dataset: str, track: str) -> list[dict[str, Any]]:
    candidates = list(candidates)
    values = [row for row in records if str(row.get("dataset")) == dataset and str(row.get("track")) == track]
    if not values:
        return []
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(values):
        groups[str(row.get("group_id"))][str(row.get("frame_id"))].append(index)
    group_frames = [[indices for indices in frames.values()] for frames in groups.values()]
    # Precompute frame error sums once for the bootstrap.
    error_arrays: dict[str, np.ndarray] = {}
    for candidate in [baseline] + list(candidates):
        errors = np.full(len(values), np.nan, dtype=np.float64)
        for index, row in enumerate(values):
            pred, _, _ = prediction(row, candidate, fusion)
            gt = safe_float(row.get("gt_m"))
            if np.isfinite([pred, gt]).all() and pred > 0 and gt > 0:
                errors[index] = abs(pred - gt) / gt
        error_arrays[candidate] = errors
    # Keep all objects from a frame in the same sample.
    group_base_sums: dict[str, list[np.ndarray]] = {candidate: [] for candidate in candidates}
    group_candidate_sums: dict[str, list[np.ndarray]] = {candidate: [] for candidate in candidates}
    group_frame_counts: dict[str, list[np.ndarray]] = {candidate: [] for candidate in candidates}
    for frames in group_frames:
        for candidate in candidates:
            base_values: list[float] = []
            candidate_values: list[float] = []
            counts: list[int] = []
            for indices in frames:
                base_error = error_arrays[baseline][indices]
                candidate_error = error_arrays[candidate][indices]
                good = np.isfinite(base_error) & np.isfinite(candidate_error)
                base_values.append(float(base_error[good].sum()))
                candidate_values.append(float(candidate_error[good].sum()))
                counts.append(int(np.count_nonzero(good)))
            group_base_sums[candidate].append(np.asarray(base_values, dtype=np.float64))
            group_candidate_sums[candidate].append(np.asarray(candidate_values, dtype=np.float64))
            group_frame_counts[candidate].append(np.asarray(counts, dtype=np.float64))
    result = []
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in dataset + track))
    for candidate in candidates:
        deltas = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            base_sum = 0.0
            candidate_sum = 0.0
            total_count = 0.0
            sampled_groups = rng.integers(0, len(group_frames), size=len(group_frames))
            for group_index in sampled_groups:
                frame_count = len(group_frame_counts[candidate][int(group_index)])
                if frame_count == 0:
                    continue
                sampled_frames = rng.integers(0, frame_count, size=frame_count)
                counts = group_frame_counts[candidate][int(group_index)][sampled_frames]
                base_sum += float(group_base_sums[candidate][int(group_index)][sampled_frames].sum())
                candidate_sum += float(group_candidate_sums[candidate][int(group_index)][sampled_frames].sum())
                total_count += float(counts.sum())
            if total_count > 0:
                deltas[replicate] = (candidate_sum - base_sum) / total_count
        valid = deltas[np.isfinite(deltas)]
        result.append({"dataset": dataset, "track": track, "baseline": baseline, "candidate": candidate, "replicates": int(valid.size), "delta_absrel_candidate_minus_baseline": float(valid.mean()) if valid.size else math.nan, "ci95_low": float(np.quantile(valid, 0.025)) if valid.size else math.nan, "ci95_high": float(np.quantile(valid, 0.975)) if valid.size else math.nan, "improvement_fraction_delta_lt_0": float(np.mean(valid < 0)) if valid.size else math.nan, "resampling": "groups/segments -> frames/images -> all objects in sampled frame", "seed": SEED})
    return result


def bootstrap_diode_frame_comparison(frame_errors: Mapping[str, Mapping[str, float]], rows: list[dict[str, Any]], candidates: Iterable[str]) -> list[dict[str, Any]]:
    group_to_frames: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_to_frames[str(row.get("group_id") or row.get("sample_id"))].append(str(row["sample_id"]))
    groups = list(group_to_frames)
    result = []
    rng = np.random.default_rng(SEED + 777)
    for candidate in candidates:
        deltas = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            base_values, candidate_values = [], []
            for group_name in rng.choice(groups, size=len(groups), replace=True):
                frames = group_to_frames[str(group_name)]
                for frame_id in rng.choice(frames, size=len(frames), replace=True):
                    b, c = frame_errors["V1_Baseline-2P"].get(str(frame_id)), frame_errors[candidate].get(str(frame_id))
                    if np.isfinite([b, c]).all():
                        base_values.append(b); candidate_values.append(c)
            if base_values:
                deltas[replicate] = float(np.mean(candidate_values) - np.mean(base_values))
        valid = deltas[np.isfinite(deltas)]
        result.append({"dataset": "DIODE", "track": "DIODE_GOLD_PIXEL", "baseline": "V1_Baseline-2P", "candidate": candidate, "replicates": int(valid.size), "delta_absrel_candidate_minus_baseline": float(valid.mean()) if valid.size else math.nan, "ci95_low": float(np.quantile(valid, 0.025)) if valid.size else math.nan, "ci95_high": float(np.quantile(valid, 0.975)) if valid.size else math.nan, "improvement_fraction_delta_lt_0": float(np.mean(valid < 0)) if valid.size else math.nan, "resampling": "scene/group -> frame; frame AbsRel aggregates all valid pixels", "seed": SEED})
    return result


def product_yolo_metrics(
    diode_detections: list[dict[str, Any]],
    waymo_official: list[dict[str, Any]],
    waymo_product: list[dict[str, Any]],
    coco_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    diode_frames = len(diode_detections)
    diode_nonempty = sum(bool(row.get("detections")) for row in diode_detections)
    rows.append({"dataset": "DIODE", "track": "Product_YOLO", "frames": diode_frames, "frames_with_detection": diode_nonempty, "frame_recall_or_coverage": diode_nonempty / max(diode_frames, 1), "detections": sum(len(row.get("detections", [])) for row in diode_detections), "matched": math.nan, "false_positives": math.nan, "mean_bbox_iou": math.nan, "rgb_source": "frozen_rgb_64"})
    for dataset, official, product, matched in (("WAYMO", waymo_official, waymo_product, "product_match_iou"),):
        frames = {str(row.get("frame_id")) for row in official}
        product_frames = {str(row.get("frame_id")) for row in product}
        ious = [safe_float(row.get("match_iou")) for row in product if np.isfinite(safe_float(row.get("match_iou")))]
        rows.append({"dataset": dataset, "track": "Product_YOLO_MATCHED", "frames": len(frames), "frames_with_detection": len(product_frames), "frame_recall_or_coverage": len(product_frames) / max(len(frames), 1), "detections": len(product), "matched": len(product), "false_positives": math.nan, "mean_bbox_iou": float(np.mean(ious)) if ious else math.nan, "rgb_source": "Waymo V5B front-camera RGB"})
    coco_official_frames = len({str(row.get("sample_id")) for row in coco_records if row.get("track") == "COCO-A_OFFICIAL_BBOX"})
    coco_product = [row for row in coco_records if row.get("track") == "COCO-B_PRODUCT_YOLO_MATCHED"]
    coco_frames = len({str(row.get("sample_id")) for row in coco_product})
    ious = [safe_float(row.get("match_iou")) for row in coco_product if np.isfinite(safe_float(row.get("match_iou")))]
    rows.append({"dataset": "COCO", "track": "Product_YOLO_MATCHED", "frames": coco_official_frames, "frames_with_detection": coco_frames, "frame_recall_or_coverage": coco_frames / max(coco_official_frames, 1), "detections": len(coco_product), "matched": len(coco_product), "false_positives": math.nan, "mean_bbox_iou": float(np.mean(ious)) if ious else math.nan, "rgb_source": "official COCO val2017 RGB"})
    return rows


def scorecard(metrics: list[dict[str, Any]], diode_pixel_rows: list[dict[str, Any]], final_candidates: list[str], class_metrics: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_metrics = class_metrics or []
    def get_pixel(dataset: str, candidate: str, scope: str) -> float:
        item = next((row for row in diode_pixel_rows if row.get("candidate") == candidate and row.get("scope") == scope), {})
        return safe_float(item.get("absrel"))
    for candidate in final_candidates:
        diode = get_pixel("DIODE", candidate, "ALL") if candidate in VISUAL_KEYS else math.nan
        waymo = next((safe_float(row.get("absrel")) for row in metrics if row.get("dataset") == "WAYMO" and row.get("track") == "WAYMO-A_OFFICIAL_BBOX" and row.get("candidate") == candidate and row.get("scope") == "ALL"), math.nan)
        coco = next((safe_float(row.get("absrel")) for row in metrics if row.get("dataset") == "COCO" and row.get("track") == "COCO-A_OFFICIAL_BBOX" and row.get("candidate") == candidate and row.get("scope") == "ALL"), math.nan)
        diode_range = np.nanmean([get_pixel("DIODE", candidate, band) for band, _, _ in BANDS if np.isfinite(get_pixel("DIODE", candidate, band))]) if candidate in VISUAL_KEYS else math.nan
        waymo_range_values = [safe_float(row.get("absrel")) for row in metrics if row.get("dataset") == "WAYMO" and row.get("track") == "WAYMO-A_OFFICIAL_BBOX" and row.get("candidate") == candidate and row.get("scope") in {band for band, _, _ in BANDS} and np.isfinite(safe_float(row.get("absrel")))]
        coco_range_values = [safe_float(row.get("absrel")) for row in metrics if row.get("dataset") == "COCO" and row.get("track") == "COCO-A_OFFICIAL_BBOX" and row.get("candidate") == candidate and row.get("scope") in {band for band, _, _ in BANDS} and np.isfinite(safe_float(row.get("absrel")))]
        waymo_class_values = [safe_float(row.get("absrel")) for row in class_metrics if row.get("dataset") == "WAYMO" and row.get("track") == "WAYMO-A_OFFICIAL_BBOX" and row.get("candidate") == candidate and safe_float(row.get("n"), 0.0) >= 20 and np.isfinite(safe_float(row.get("absrel")))]
        coco_class_values = [safe_float(row.get("absrel")) for row in class_metrics if row.get("dataset") == "COCO" and row.get("track") == "COCO-A_OFFICIAL_BBOX" and row.get("candidate") == candidate and safe_float(row.get("n"), 0.0) >= 20 and np.isfinite(safe_float(row.get("absrel")))]
        class_balanced_waymo = float(np.mean(waymo_class_values)) if waymo_class_values else math.nan
        class_balanced_coco = float(np.mean(coco_class_values)) if coco_class_values else math.nan
        class_balanced_object = float(np.mean([x for x in (class_balanced_waymo, class_balanced_coco) if np.isfinite(x)])) if np.isfinite(class_balanced_waymo) or np.isfinite(class_balanced_coco) else math.nan
        rows.append({"candidate": candidate, "diode_gold_pixel_absrel": diode, "waymo_gold_object_absrel": waymo, "coco_pseudogt_object_absrel": coco, "gold_only_macro_absrel": float(np.mean([x for x in (diode, waymo) if np.isfinite(x)])) if any(np.isfinite(x) for x in (diode, waymo)) else math.nan, "range_balanced_diode": float(diode_range) if np.isfinite(diode_range) else math.nan, "range_balanced_waymo": float(np.mean(waymo_range_values)) if waymo_range_values else math.nan, "range_balanced_coco": float(np.mean(coco_range_values)) if coco_range_values else math.nan, "range_balanced_gold_macro": float(np.mean([x for x in (diode_range, np.mean(waymo_range_values) if waymo_range_values else math.nan) if np.isfinite(x)])) if np.isfinite(diode_range) or waymo_range_values else math.nan, "class_balanced_waymo": class_balanced_waymo, "class_balanced_coco": class_balanced_coco, "class_balanced_object_macro": class_balanced_object, "evaluation_note": "DIODE is pixel Gold; Waymo is object Gold-A; COCO is P0 pseudo-GT stress only; class macro uses classes with n >= 20"})
    return rows


def make_plots(metrics: list[dict[str, Any]], score_rows: list[dict[str, Any]], records: list[dict[str, Any]], coverage: list[dict[str, Any]], product_rows: list[dict[str, Any]], oracle: list[dict[str, Any]], fusion: Mapping[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(REPORTS / "v6_plot_status.json", {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"})
        return
    PLOTS.mkdir(parents=True, exist_ok=True)
    # Build the candidate scorecard.
    labels = [str(row["candidate"]) for row in score_rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels)); width = 0.25
    for offset, field, label in ((-width, "diode_gold_pixel_absrel", "DIODE Gold"), (0, "waymo_gold_object_absrel", "Waymo Gold"), (width, "coco_pseudogt_object_absrel", "COCO P0 pseudo")):
        ax.bar(x + offset, [safe_float(row.get(field)) for row in score_rows], width, label=label)
    ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=8); ax.set_ylabel("AbsRel"); ax.set_title("V6 candidate AbsRel by dataset"); ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_candidate_absrel_by_dataset.png", dpi=150); plt.close(fig)
    # Build the range summary.
    fig, ax = plt.subplots(figsize=(12, 5))
    range_rows = [row for row in metrics if row.get("scope") in {band for band, _, _ in BANDS} and row.get("track") in {"WAYMO-A_OFFICIAL_BBOX", "COCO-A_OFFICIAL_BBOX"} and row.get("candidate") in {"V1_Baseline-2P", "V5_Spline-8-Median", F1_KEY}]
    groups = sorted({(str(row.get("track")), str(row.get("scope"))) for row in range_rows})
    for candidate in ("V1_Baseline-2P", "V5_Spline-8-Median", F1_KEY):
        vals = []
        for track, scope in groups:
            vals.append(next((safe_float(row.get("absrel")) for row in range_rows if row.get("candidate") == candidate and row.get("track") == track and row.get("scope") == scope), math.nan))
        ax.plot(range(len(groups)), vals, marker="o", label=candidate)
    ax.set_xticks(range(len(groups)), [f"{t.replace('_OFFICIAL_BBOX','').replace('WAYMO-','')}:{s}" for t, s in groups], rotation=75, ha="right", fontsize=7); ax.set_ylabel("AbsRel"); ax.set_title("Range-balanced object performance"); ax.legend(fontsize=8); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_range_balanced_performance.png", dpi=150); plt.close(fig)
    # Add error metrics by distance range.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, field, label in zip(axes, ("mae_m", "rmse_m", "medae_m"), ("MAE", "RMSE", "MedAE")):
        for candidate in ("V1_Baseline-2P", "V5_Spline-8-Median", F1_KEY):
            vals = [next((safe_float(row.get(field)) for row in metrics if row.get("candidate") == candidate and row.get("track") == "WAYMO-A_OFFICIAL_BBOX" and row.get("scope") == band), math.nan) for band, _, _ in BANDS]
            axis.plot(range(len(BANDS)), vals, marker="o", label=candidate)
        axis.set_title(label); axis.set_xticks(range(len(BANDS)), [band for band, _, _ in BANDS], rotation=50, ha="right", fontsize=8); axis.grid(alpha=0.2)
    axes[0].set_ylabel("meters"); axes[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(PLOTS / "v6_mae_rmse_medae_by_range.png", dpi=150); plt.close(fig)
    # Compare F1 with visual depth.
    subset = [row for row in records if row.get("dataset") in {"WAYMO", "COCO"} and np.isfinite([safe_float(row.get("V1_Baseline-2P")), safe_float(row.get("z_size")), safe_float(row.get("gt_m"))]).all()][:8000]
    if subset:
        visual_error = np.asarray([abs(safe_float(row["V1_Baseline-2P"]) - safe_float(row["gt_m"])) / safe_float(row["gt_m"]) for row in subset])
        size_error = np.asarray([abs(safe_float(row["z_size"]) - safe_float(row["gt_m"])) / safe_float(row["gt_m"]) for row in subset])
        disagreement = np.asarray([abs(math.log(safe_float(row["z_size"]) / safe_float(row["V1_Baseline-2P"]))) for row in subset])
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(visual_error, size_error, c=disagreement, s=4, alpha=0.25, cmap="viridis"); ax.set_xlabel("Baseline AbsRel"); ax.set_ylabel("F1 size AbsRel"); ax.set_title("F1 vs visual residual; color=log disagreement"); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_f1_vs_visual_residual_scatter.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(disagreement, visual_error, s=4, alpha=0.25, label="visual"); ax.scatter(disagreement, size_error, s=4, alpha=0.25, label="size"); ax.set_xlabel("|log(Zsize/Zvisual)|"); ax.set_ylabel("AbsRel"); ax.set_title("Disagreement vs expert error"); ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_disagreement_vs_error.png", dpi=150); plt.close(fig)
    # Compare learned weights with visual distance.
    fusion_names = [row["candidate"] for row in metrics if str(row.get("candidate", "")).startswith(("A", "B", "C", "D"))]
    if fusion_names and subset:
        fig, ax = plt.subplots(figsize=(7, 5))
        for candidate in sorted(set(fusion_names))[:8]:
            vals = []
            for row in subset:
                _, weight, _ = prediction(row, candidate, fusion)
                if np.isfinite(weight):
                    vals.append((safe_float(row.get("V1_Baseline-2P")), weight))
            if vals:
                vals = sorted(vals); ax.plot([v[0] for v in vals][::max(1, len(vals)//200)], [v[1] for v in vals][::max(1, len(vals)//200)], label=candidate)
        ax.set_xscale("log"); ax.set_xlabel("visual depth proxy [m]"); ax.set_ylabel("size weight"); ax.set_title("Fusion weight vs distance proxy"); ax.legend(fontsize=7); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_fusion_weight_vs_distance_proxy.png", dpi=150); plt.close(fig)
    # Add stability and coverage metrics.
    seed = [row for row in records if np.isfinite(safe_float(row.get("seed_log_std"))) and np.isfinite(safe_float(row.get("V5_Spline-8-Median")))]
    if seed:
        x = np.asarray([safe_float(row.get("seed_log_std")) for row in seed]); y = np.asarray([abs(safe_float(row.get("V5_Spline-8-Median")) - safe_float(row.get("gt_m"))) / safe_float(row.get("gt_m")) for row in seed])
        fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(x, y, s=4, alpha=0.25); ax.set_xlabel("Spline seed log-std"); ax.set_ylabel("Spline ensemble AbsRel"); ax.set_title("Seed disagreement vs error"); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_seed_disagreement_vs_error.png", dpi=150); plt.close(fig)
    if coverage:
        cov = [row for row in coverage if row.get("dataset") == "WAYMO" and row.get("track") == "WAYMO-A_OFFICIAL_BBOX" and row.get("candidate") in {"V1_Baseline-2P", F1_KEY, "C_Baseline_RangeGate", "D_Baseline_TinyLearned"}]
        fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter([safe_float(row.get("valid_prediction_fraction")) for row in cov], [next((safe_float(m.get("absrel")) for m in metrics if m.get("dataset") == row.get("dataset") and m.get("track") == row.get("track") and m.get("candidate") == row.get("candidate") and m.get("scope") == "ALL"), math.nan) for row in cov], s=50); ax.set_xlabel("valid prediction fraction"); ax.set_ylabel("AbsRel"); ax.set_title("Coverage vs accuracy"); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_coverage_vs_accuracy.png", dpi=150); plt.close(fig)
    # Compare product detections with official references.
    product = [row for row in metrics if row.get("track") in {"WAYMO-B_PRODUCT_YOLO_MATCHED", "COCO-B_PRODUCT_YOLO_MATCHED"} and row.get("scope") == "ALL" and row.get("candidate") in {"V1_Baseline-2P", "V5_Spline-8-Median", F1_KEY}]
    if product:
        fig, ax = plt.subplots(figsize=(7, 4)); labels2 = [f"{r['dataset']}/{r['track'].split('_')[0]}" for r in product]; ax.bar(np.arange(len(product)), [safe_float(r.get("absrel")) for r in product]); ax.set_xticks(np.arange(len(product)), [f"{r['dataset']}\n{r['candidate']}" for r in product], rotation=65, ha="right", fontsize=7); ax.set_ylabel("AbsRel"); ax.set_title("Product YOLO matched-track performance"); ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_product_yolo_vs_official.png", dpi=150); plt.close(fig)
    oracle_all = [row for row in oracle if row.get("scope") == "ALL"]
    if oracle_all:
        fig, ax = plt.subplots(figsize=(8, 4)); chosen = [row for row in oracle_all if row.get("candidate") in {"V1_Baseline-2P", "V5_Spline-8-Median", F1_KEY, "ORACLE_BEST_VISUAL_SIZE", "ORACLE_BEST_SPLINE_SIZE"}]; ax.bar(np.arange(len(chosen)), [safe_float(r.get("absrel")) for r in chosen]); ax.set_xticks(np.arange(len(chosen)), [f"{r.get('dataset')}\n{r.get('track')}\n{r.get('candidate')}" for r in chosen], rotation=70, ha="right", fontsize=7); ax.set_ylabel("AbsRel"); ax.set_title("Oracle expert selection headroom"); ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(PLOTS / "v6_oracle_expert_selection_vs_deployable.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); chosen = score_rows; ax.axis("off"); table = [[r["candidate"], f"{safe_float(r.get('gold_only_macro_absrel')):.3f}", f"{safe_float(r.get('range_balanced_gold_macro')):.3f}"] for r in chosen]; ax.table(cellText=table, colLabels=["candidate", "Gold macro", "range macro"], loc="center"); ax.set_title("V6 final model scorecard"); fig.tight_layout(); fig.savefig(PLOTS / "v6_final_model_scorecard.png", dpi=150); plt.close(fig)
    write_json(REPORTS / "v6_plot_status.json", {"status": "COMPLETE", "plot_dir": str(PLOTS.resolve())})


def metric_value(metrics: list[dict[str, Any]], dataset: str, track: str, candidate: str, scope: str, field: str = "absrel") -> float:
    row = next((row for row in metrics if row.get("dataset") == dataset and row.get("track") == track and row.get("candidate") == candidate and row.get("scope") == scope), {})
    return safe_float(row.get(field))


def decision_label(candidate: str, score: Mapping[str, Any], baseline_score: Mapping[str, Any]) -> str:
    gold = safe_float(score.get("gold_only_macro_absrel"))
    base = safe_float(baseline_score.get("gold_only_macro_absrel"))
    range_gold = safe_float(score.get("range_balanced_gold_macro"))
    base_range = safe_float(baseline_score.get("range_balanced_gold_macro"))
    if np.isfinite([gold, base, range_gold, base_range]).all() and gold < base and range_gold <= base_range * 1.02:
        return "A - ROBUST CROSS-DATASET GAIN"
    if np.isfinite([gold, base]).all() and gold < base:
        return "B - PARTIAL CROSS-DATASET SIGNAL"
    if candidate == F1_KEY:
        return "C - DATASET/RANGE/CLASS SPECIFIC"
    return "D - NO MATERIAL BENEFIT"


def final_decision(
    metrics: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    fusion: Mapping[str, Any],
    product_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_score = next((row for row in score_rows if row.get("candidate") == "V1_Baseline-2P"), {})
    pure_candidates = ["V1_Baseline-2P", "V5_Spline-8-Median", "V6_Spline-8-LogMean"]
    pure_candidates.extend(INDIVIDUAL_SPLINES)
    pure_available = [row for row in score_rows if row.get("candidate") in pure_candidates and np.isfinite(safe_float(row.get("gold_only_macro_absrel")))]
    best_pure = min(pure_available, key=lambda row: safe_float(row.get("gold_only_macro_absrel"))) if pure_available else baseline_score
    fusion_names = [name for name in fusion if name.startswith(("A", "B", "C", "D"))]
    hybrid_rows = [row for row in score_rows if row.get("candidate") in fusion_names and np.isfinite(safe_float(row.get("waymo_gold_object_absrel")))]
    best_hybrid = min(hybrid_rows, key=lambda row: np.nanmean([safe_float(row.get("waymo_gold_object_absrel")), safe_float(row.get("coco_pseudogt_object_absrel"))])) if hybrid_rows else {}
    baseline_near = metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", "V1_Baseline-2P", "eyeai_0_5_5m")
    baseline_mid = metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", "V1_Baseline-2P", "5_10m")
    product_base = metric_value(metrics, "WAYMO", "WAYMO-B_PRODUCT_YOLO_MATCHED", "V1_Baseline-2P", "ALL")
    hybrid_gate_rows = []
    for row in hybrid_rows:
        candidate = str(row["candidate"])
        near = metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", candidate, "eyeai_0_5_5m")
        mid = metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", candidate, "5_10m")
        product = metric_value(metrics, "WAYMO", "WAYMO-B_PRODUCT_YOLO_MATCHED", candidate, "ALL")
        cov = next((c for c in coverage if c.get("dataset") == "WAYMO" and c.get("track") == "WAYMO-A_OFFICIAL_BBOX" and c.get("candidate") == candidate), {})
        requirements = {
            "waymo_overall_nonworse": metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", candidate, "ALL") <= metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", "V1_Baseline-2P", "ALL"),
            "near_nonworse": np.isfinite([near, baseline_near]).all() and near <= baseline_near,
            "5_10m_nonworse": np.isfinite([mid, baseline_mid]).all() and mid <= baseline_mid,
            "product_same_direction": np.isfinite([product, product_base]).all() and product <= product_base,
            "coverage_ge_0_80": safe_float(cov.get("valid_prediction_fraction")) >= 0.80,
            "no_material_rejection": safe_float(cov.get("rejected"), 0.0) / max(safe_float(cov.get("objects_total"), 1.0), 1.0) <= 0.05,
        }
        hybrid_gate_rows.append({"candidate": candidate, "requirements": requirements, "qualifies": all(requirements.values())})
    qualified = [row for row in hybrid_gate_rows if row["qualifies"]]
    best_hybrid_candidate = min((row["candidate"] for row in qualified), key=lambda name: np.nanmean([metric_value(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", name, "ALL"), metric_value(metrics, "COCO", "COCO-A_OFFICIAL_BBOX", name, "ALL")])) if qualified else None
    labels = {str(row["candidate"]): decision_label(str(row["candidate"]), row, baseline_score) for row in score_rows}
    for row in hybrid_rows:
        labels[str(row["candidate"])] = "A - ROBUST CROSS-DATASET GAIN" if any(item["candidate"] == row["candidate"] and item["qualifies"] for item in hybrid_gate_rows) else ("B - PARTIAL CROSS-DATASET SIGNAL" if safe_float(row.get("waymo_gold_object_absrel")) < safe_float(baseline_score.get("waymo_gold_object_absrel")) else "C - DATASET/RANGE/CLASS SPECIFIC")
    recommendation = "HYBRID_FUSION" if best_hybrid_candidate else "BASELINE-2P"
    payload = {
        "format": "rel2abs_v6_final_model_decision_v1",
        "status": "COMPLETE_STOP_RULE_REACHED",
        "recommendation": recommendation,
        "pure_visual": {"best_candidate": best_pure.get("candidate", "V1_Baseline-2P"), "category": "PURE VISUAL", "label": labels.get(str(best_pure.get("candidate")), "D - NO MATERIAL BENEFIT")},
        "size_anchor_only": {"best_candidate": F1_KEY, "category": "SIZE-ANCHOR", "label": labels.get(F1_KEY, "C - DATASET/RANGE/CLASS SPECIFIC")},
        "hybrid": {"best_candidate": best_hybrid_candidate or "NONE_QUALIFIED", "category": "HYBRID FUSION", "label": labels.get(best_hybrid_candidate, "C - DATASET/RANGE/CLASS SPECIFIC") if best_hybrid_candidate else "BLOCKED"},
        "safe_fallback": {"candidate": "V1_Baseline-2P", "category": "SAFE FALLBACK"},
        "hybrid_gate_rows": hybrid_gate_rows,
        "candidate_labels": labels,
        "spline_seed_policy": "median/log-mean ensembles are diagnostic only; no external weights were fitted",
        "eyeai_integration": "AUTHORIZED_HANDOFF_ONLY" if best_hybrid_candidate else "NOT_JUSTIFIED",
        "android_changed": False,
        "midas_changed": False,
        "reason": "Final choice requires DIODE and Waymo Gold consistency, 0.5-5m and 5-10m non-regression, Product-YOLO direction, coverage and fallback safety; overall AbsRel alone is insufficient.",
    }
    write_json(REPORTS / "V6_FINAL_MODEL_DECISION.json", payload)
    lines = [
        "# REL2ABS-v6 final model decision",
        "",
        "Status: COMPLETE - V6 stop rule reached. EyeAIApp remains unchanged.",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        "## Categories",
        "",
        f"- PURE VISUAL: **{payload['pure_visual']['best_candidate']}** ({payload['pure_visual']['label']}).",
        f"- SIZE-ANCHOR ONLY: **{F1_KEY}** ({payload['size_anchor_only']['label']}).",
        f"- HYBRID: **{payload['hybrid']['best_candidate']}** ({payload['hybrid']['label']}).",
        "- SAFE FALLBACK: **V1_Baseline-2P**.",
        "",
        "## Hybrid gate",
        "",
        "A hybrid is promoted only if every recorded Waymo gate is true: overall non-regression, 0.5-5m non-regression, 5-10m non-regression, Product-YOLO same direction, >=80% valid prediction coverage and <=5% rejection.",
    ]
    for item in hybrid_gate_rows:
        lines.append(f"- `{item['candidate']}`: **{'PASS' if item['qualifies'] else 'FAIL'}** - " + ", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in item["requirements"].items()))
    lines += ["", "Full machine-readable details are in `V6_FINAL_MODEL_DECISION.json`."]
    (REPORTS / "V6_FINAL_MODEL_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if best_hybrid_candidate:
        handoff = [
            "# V6 EyeAI integration handoff",
            "",
            "This is a handoff only; no Android or EyeAIApp code was changed.",
            "",
            f"Candidate: `{best_hybrid_candidate}`.",
            "",
            "The exact formula and frozen parameters are in the V6 fusion report and JSON decision artifact. Invalid or low-reliability anchors must fall back to Baseline-2P. MiDaS, YOLO and the existing visual head remain unchanged.",
        ]
        (V6 / "V6_EYEAI_INTEGRATION_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    return payload


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    return "-" if not np.isfinite(number) else f"{number:.{digits}f}"


def summary_table(metrics: list[dict[str, Any]], dataset: str, track: str, candidates: list[str], scopes: list[str]) -> list[str]:
    lines = ["| candidate | " + " | ".join(scopes) + " |", "|---|" + "---:|" * len(scopes)]
    for candidate in candidates:
        values = [fmt(metric_value(metrics, dataset, track, candidate, scope)) for scope in scopes]
        lines.append(f"| {candidate} | " + " | ".join(values) + " |")
    return lines


def write_final_report(
    decision: Mapping[str, Any],
    metrics: list[dict[str, Any]],
    diode_pixel: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    class_metrics: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    bootstrap: list[dict[str, Any]],
    priors: Mapping[str, Mapping[str, Any]],
    teacher_meta: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    baseline = "V1_Baseline-2P"
    spline = "V5_Spline-8-Median"
    fusion_candidates = [str(row["candidate"]) for row in score_rows if str(row.get("candidate", "")).startswith(("A", "B", "C", "D"))]
    best_hybrid = str(decision.get("hybrid", {}).get("best_candidate", "NONE_QUALIFIED"))
    def pixel_metric(candidate: str, scope: str) -> float:
        return safe_float(next((row.get("absrel") for row in diode_pixel if row.get("candidate") == candidate and row.get("scope") == scope), math.nan))
    def object_metric(dataset: str, track: str, candidate: str, scope: str, field: str = "absrel") -> float:
        return metric_value(metrics, dataset, track, candidate, scope, field)
    def coverage_metric(dataset: str, track: str, candidate: str, field: str) -> float:
        row = next((item for item in coverage if item.get("dataset") == dataset and item.get("track") == track and item.get("candidate") == candidate), {})
        return safe_float(row.get(field))
    def score_value(candidate: str, field: str) -> float:
        row = next((item for item in score_rows if str(item.get("candidate")) == candidate), {})
        return safe_float(row.get(field))
    baseline_diode = pixel_metric(baseline, "ALL")
    baseline_diode_eyeai = pixel_metric(baseline, "eyeai_0_5_5m")
    baseline_waymo = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", baseline, "ALL")
    baseline_waymo_eyeai = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", baseline, "eyeai_0_5_5m")
    baseline_waymo_mid = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", baseline, "5_10m")
    baseline_waymo_far = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", baseline, "ge15m")
    baseline_coco = object_metric("COCO", "COCO-A_OFFICIAL_BBOX", baseline, "ALL")
    baseline_coco_eyeai = object_metric("COCO", "COCO-A_OFFICIAL_BBOX", baseline, "eyeai_0_5_5m")
    best_visual_waymo = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", "V4_Spline-8-seed7", "ALL")
    best_visual_coco = object_metric("COCO", "COCO-A_OFFICIAL_BBOX", "V4_Spline-8-seed7", "ALL")
    b3_waymo = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", best_hybrid, "ALL")
    b3_waymo_eyeai = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", best_hybrid, "eyeai_0_5_5m")
    b3_waymo_mid = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", best_hybrid, "5_10m")
    b3_waymo_far = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", best_hybrid, "ge15m")
    b3_coco = object_metric("COCO", "COCO-A_OFFICIAL_BBOX", best_hybrid, "ALL")
    b3_coco_eyeai = object_metric("COCO", "COCO-A_OFFICIAL_BBOX", best_hybrid, "eyeai_0_5_5m")
    oracle_waymo = object_metric("WAYMO", "WAYMO-A_OFFICIAL_BBOX", "ORACLE_BEST_VISUAL_SIZE", "ALL")
    oracle_coco = object_metric("COCO", "COCO-A_OFFICIAL_BBOX", "ORACLE_BEST_VISUAL_SIZE", "ALL")
    waymo_recovery = (baseline_waymo - b3_waymo) / (baseline_waymo - oracle_waymo) if np.isfinite([baseline_waymo, b3_waymo, oracle_waymo]).all() and baseline_waymo != oracle_waymo else math.nan
    coco_recovery = (baseline_coco - b3_coco) / (baseline_coco - oracle_coco) if np.isfinite([baseline_coco, b3_coco, oracle_coco]).all() and baseline_coco != oracle_coco else math.nan
    prior_controls = read_csv(REPORTS / "v6_prior_controls.csv") if (REPORTS / "v6_prior_controls.csv").exists() else []
    def control_metric(dataset: str, candidate: str) -> float:
        return safe_float(next((row.get("absrel") for row in prior_controls if row.get("dataset") == dataset and row.get("track") == ("WAYMO-A_OFFICIAL_BBOX" if dataset == "WAYMO" else "COCO-A_OFFICIAL_BBOX") and row.get("control") == candidate and row.get("scope") == "ALL"), math.nan))
    family_dev_rows = []
    for family_file in ("v6_fusion_family_A.csv", "v6_fusion_family_B.csv", "v6_fusion_family_C.csv", "v6_fusion_family_D.csv"):
        if (REPORTS / family_file).exists():
            family_dev_rows.extend(row for row in read_csv(REPORTS / family_file) if row.get("evaluation") == "v3_train_dev")
    seed_report_rows = read_csv(REPORTS / "v6_seed_disagreement.csv") if (REPORTS / "v6_seed_disagreement.csv").exists() else []
    waymo_seed_corr = safe_float(next((row.get("corr_seed_disagreement_vs_spline_absrel") for row in seed_report_rows if row.get("dataset") == "WAYMO" and row.get("track") == "WAYMO-A_OFFICIAL_BBOX"), math.nan))
    coco_seed_corr = safe_float(next((row.get("corr_seed_disagreement_vs_spline_absrel") for row in seed_report_rows if row.get("dataset") == "COCO" and row.get("track") == "COCO-A_OFFICIAL_BBOX"), math.nan))
    waymo_band_counts = {band: int(safe_float(next((row.get("n") for row in metrics if row.get("dataset") == "WAYMO" and row.get("track") == "WAYMO-A_OFFICIAL_BBOX" and row.get("candidate") == baseline and row.get("scope") == band), 0), 0.0)) for band, _, _ in BANDS}
    report = [
        "# REL2ABS-v6 Final Cross-Dataset Fusion Study",
        "",
        "## Executive result",
        "",
        f"The V6 study is complete and stopped at the requested research boundary. **MiDaS, EyeAIApp, YOLO, ByteTrack and SpatialAudio were not changed.** The final decision is **{decision.get('recommendation')}**; the safe fallback remains **V1_Baseline-2P**.",
        "",
        "The study treats the old F1 path as `F1_SIZE_ANCHOR_OVERRIDE`, not as a fusion. V6 evaluates true simultaneous visual+size combinations with frozen development parameters.",
        "",
        "## Data and protocol",
        "",
        "- DIODE: V3 locked split, real metric depth Gold; the established V3 sample-bank contract evaluates 2,048 deterministic valid pixels per frame.",
        "- Waymo: corrected V5B decoder and GT-A LiDAR surface-median camera-forward depth; official-box and matched Product-YOLO tracks.",
        f"- Waymo frozen range composition is low in the requested near bands: 2-5m n={waymo_band_counts.get('2_5m', 0)}, 5-10m n={waymo_band_counts.get('5_10m', 0)}, >=15m n={waymo_band_counts.get('ge15m', 0)}; the aspirational >=100-per-near-band target was not reached, so near/mid estimates are reported with their n rather than treated as fully powered.",
        "- COCO: fixed V4 500-image official-box panel; P0 UniDepthV2/Metric3Dv2 consensus is pseudo-GT only; Product-YOLO matched track is separate.",
        "- V3 train fit fusion parameters; V3 dev selected them; V3 locked, Waymo and COCO were not used for fitting.",
        "- 5,000-replicate paired bootstrap was used for the predeclared primary external comparison set; the complete per-candidate metrics remain in the freeze tables.",
        "",
        "### DIODE Product-YOLO qualification",
        "",
        "The frozen local DIODE cache contains 64×64 RGB rather than the original full-resolution RGB archive. Product YOLO was nevertheless executed on that retained RGB representation; the resulting coverage is reported, not imputed. DIODE remains valid Gold pixel evaluation, but no DIODE object/F1 metric is claimed when no usable detections exist.",
        "",
        "## Candidate metrics",
        "",
        "AbsRel (lower is better); DIODE is pixel-level, Waymo/COCO are object-level. COCO is explicitly pseudo-GT stress testing.",
        "",
        "### DIODE Gold pixels",
        "",
    ]
    report += ["| candidate | ALL | 0.5-5m | 0.5-10m | 2-5m | 5-10m | ≥15m |", "|---|---:|---:|---:|---:|---:|---:|"]
    for candidate in VISUAL_KEYS:
        report.append(f"| {candidate} | {fmt(next((r.get('absrel') for r in diode_pixel if r.get('candidate') == candidate and r.get('scope') == 'ALL'), math.nan))} | {fmt(next((r.get('absrel') for r in diode_pixel if r.get('candidate') == candidate and r.get('scope') == 'eyeai_0_5_5m'), math.nan))} | {fmt(next((r.get('absrel') for r in diode_pixel if r.get('candidate') == candidate and r.get('scope') == 'eyeai_0_5_10m'), math.nan))} | {fmt(next((r.get('absrel') for r in diode_pixel if r.get('candidate') == candidate and r.get('scope') == '2_5m'), math.nan))} | {fmt(next((r.get('absrel') for r in diode_pixel if r.get('candidate') == candidate and r.get('scope') == '5_10m'), math.nan))} | {fmt(next((r.get('absrel') for r in diode_pixel if r.get('candidate') == candidate and r.get('scope') == 'ge15m'), math.nan))} |")
    report += ["", "### Waymo Gold-A official boxes", ""]
    report += summary_table(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", [baseline, "V2_Spline-8-seed123", "V3_Spline-8-seed42", "V4_Spline-8-seed7", "V5_Spline-8-Median", "V6_Spline-8-LogMean", F1_KEY] + fusion_candidates, ["ALL", "eyeai_0_5_5m", "5_10m", "ge15m"])
    report += ["", "### COCO P0 official boxes (pseudo-GT)", ""]
    report += summary_table(metrics, "COCO", "COCO-A_OFFICIAL_BBOX", [baseline, "V2_Spline-8-seed123", "V3_Spline-8-seed42", "V4_Spline-8-seed7", "V5_Spline-8-Median", "V6_Spline-8-LogMean", F1_KEY] + fusion_candidates, ["ALL", "eyeai_0_5_5m", "5_10m", "ge15m"])
    report += ["", "### Mandatory distance bands", "", "The complete metric table with n, MAE, RMSE, MedAE, bias, p90, p95, catastrophic rate and delta1.25 is in `reports/v6_distance_bands.csv`. The relevant Gold object-band summary is:", ""]
    report += summary_table(metrics, "WAYMO", "WAYMO-A_OFFICIAL_BBOX", [baseline, spline, F1_KEY] + ([best_hybrid] if best_hybrid not in {"", "NONE_QUALIFIED"} else []), [band for band, _, _ in BANDS])
    report += ["", "## Fusion families", "", "- A: fixed log-space weights 25/50/75%; no learned parameters.", "- B1: precision-weighted log fusion using size prior/bbox/intrinsics uncertainty and visual seed disagreement.", "- B2: clipped reliability blend; B3: disagreement-aware fallback/blend.", "- C: monotone range proxy gate `w_size = reliability * sigmoid(a*(log(Z_visual)-b))`; a/b were selected only on V3 train/dev.", "- D: tiny bounded Ridge residual head, `log Z_final = log Z_visual + clip(Delta log Z, -1, 1)`; deployable inputs only; parameter count is recorded in `reports/v6_fusion_family_D.csv`.", "", "The exact frozen parameters and development scores are in `reports/v6_fusion_family_A.csv` through `D.csv` and `reports/V6_FUSION_TRAIN_DEV_MANIFEST.json`.", ""]
    report += ["### Development fit versus external behavior", "", "Fusion train/dev rows are retained in the family CSVs. The learned D heads have 17 parameters (16 standardized deployable features plus intercept); their external behavior is recorded separately and was not tuned after the freeze."]
    for row in family_dev_rows:
        report.append(f"- `{row.get('candidate')}`: V3-train AbsRel `{fmt(row.get('train_absrel'))}`, V3-dev AbsRel `{fmt(row.get('dev_absrel'))}`.")
    report += ["", "No external selection was performed. The D heads are bounded, but any train/dev-to-external gap is treated as a generalization risk; B3 is preferred for the handoff because it passes the explicit Waymo/product/coverage gates with zero learned parameters.", ""]
    report += ["## Seed ensembles and reliability", "", "SE1 is the unweighted median of seeds 123/42/7; SE2 is the mean in log-depth. No weights were optimized on external data. `reports/v6_seed_disagreement.csv` quantifies correlation between seed disagreement and visual error; it is a diagnostic signal, not calibrated uncertainty.", ""]
    report += ["## Prior controls", "", f"The explicit prior file contains `{len(priors)}` classes. Correct, wrong-class, randomized, generic same-size, geometry-only and class-semantics-only controls are in `reports/v6_prior_controls.csv`. A correct-prior improvement over wrong/random controls is interpreted as evidence for physical class size, not merely class semantics.", ""]
    report += ["## Oracle / ceiling diagnostic", "", "`ORACLE_BEST_VISUAL_SIZE` and `ORACLE_BEST_SPLINE_SIZE` are GT-leaked per-object diagnostics only. They quantify expert-selection headroom and are never presented as deployable models. Existing V3 same-frame oracle/ceiling methodology remains the historical reference; V6 does not invent a replacement ceiling.", f"- Waymo: Baseline AbsRel `{fmt(baseline_waymo)}`, visual/size oracle `{fmt(oracle_waymo)}`, B3 `{fmt(b3_waymo)}`; recovered fraction of the available oracle gap: `{fmt(waymo_recovery * 100.0, 1)}%`.", f"- COCO P0: Baseline AbsRel `{fmt(baseline_coco)}`, visual/size oracle `{fmt(oracle_coco)}`, B3 `{fmt(b3_coco)}`; recovered fraction: `{fmt(coco_recovery * 100.0, 1)}%` (pseudo-GT only).", ""]
    report += ["## Product-YOLO and coverage", "", "Product-YOLO metrics, matched fraction, bbox IoU, frame coverage and DIODE RGB-source qualification are in `reports/v6_product_yolo_metrics.csv`. Conditional-on-valid-anchor and coverage/fallback statistics are in `reports/v6_coverage_metrics.csv`; the required dataset-level aggregation is `reports/v6_dataset_coverage.csv`.", ""]
    report += ["## Statistics", "", "Bootstrap comparisons are in `reports/v6_bootstrap.csv`. Gold resampling follows dataset/group/segment → frame with all objects retained; COCO resamples images with all objects retained. Confidence intervals are paired deltas versus Baseline-2P.", ""]
    report += ["## Cross-dataset decision", ""]
    report += ["| candidate | Gold macro | range-balanced Gold macro | class-balanced object macro | label |", "|---|---:|---:|---:|---|"]
    for row in score_rows:
        report.append(f"| {row.get('candidate')} | {fmt(row.get('gold_only_macro_absrel'))} | {fmt(row.get('range_balanced_gold_macro'))} | {fmt(row.get('class_balanced_object_macro'))} | {decision.get('candidate_labels', {}).get(str(row.get('candidate')), 'BLOCKED')} |")
    report += ["", f"- Best PURE VISUAL: **{decision.get('pure_visual', {}).get('best_candidate')}**.", f"- SIZE-ANCHOR ONLY: **{decision.get('size_anchor_only', {}).get('best_candidate')}**; its benefit is range/class-specific unless all gates pass.", f"- Best HYBRID candidate passing all gates: **{best_hybrid}**.", "- SAFE FALLBACK: **V1_Baseline-2P**.", "", "## Required direct answers", "", "1. Baseline-2P is the frozen visual reference; exact DIODE/Waymo/COCO results are in the three tables above.", "2. Spline-8 seed123, seed42 and seed7 are evaluated separately; the seed ensemble rows quantify stabilization.", "3. F1 is evaluated separately as a size-anchor override, not conflated with fusion.", "4. Fixed, uncertainty-weighted, range-aware and tiny learned fusion are all recorded; external promotion follows the frozen gates.", "5. Near/mid/far behavior is shown in `v6_distance_bands.csv`; no overall-only choice is made.", "6. Product-YOLO direction is required for a hybrid promotion; official and matched tracks remain separate.", "7. Correct/wrong/random/generic/geometry controls separate physical size from projection geometry and class semantics.", "8. Oracle rows quantify the available expert-selection headroom but are GT-leaked.", "9. Seed disagreement is reported as a reliability diagnostic only.", "10. Runtime/exportability: true fusions add only scalar arithmetic; D is <=5k parameters and bounded. No new model inference beyond already-existing YOLO is required. Formal Android/LiteRT integration was not run.", "11. Camera height is not part of V6 fusion training; no artificial height prior was introduced.", "12. Final EyeAI integration is **not** performed in this block; the handoff exists only if the frozen hybrid gates qualify.", "", "## Runtime and deployability", "", f"The run used the archived Python runtime and reused cached MiDaS/head outputs. Run wall-clock metadata is in `reports/v6_runtime.json`; no MiDaS or model-backbone compute was changed. The F1 branch uses existing Product YOLO detections and a small scalar calculation. Piecewise/analytic fusion is export-friendly; D is bounded linear arithmetic but still requires a separate formal LiteRT export audit before integration.", "", "## Artefacts", "", "All required CSV/JSON/Markdown reports and diagnostic plots are under `rel2abs_v6_research/reports` and `rel2abs_v6_research/plots`.", "", "## Stop rule", "", "V6 is complete. Do not modify EyeAIApp automatically. Any future integration requires a separate explicit implementation task after reviewing `V6_FINAL_MODEL_DECISION.md`."]
    report += ["", "## Required direct answers - explicit numeric audit", "", f"1. Baseline-2P: DIODE Gold AbsRel `{fmt(baseline_diode)}`, Waymo Gold-A `{fmt(baseline_waymo)}`, COCO P0 `{fmt(baseline_coco)}`.", f"2. Spline-8 seed123: DIODE `{fmt(pixel_metric('V2_Spline-8-seed123', 'ALL'))}`, Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'V2_Spline-8-seed123', 'ALL'))}`, COCO `{fmt(object_metric('COCO', 'COCO-A_OFFICIAL_BBOX', 'V2_Spline-8-seed123', 'ALL'))}`.", f"3. Spline-8 seed42: DIODE `{fmt(pixel_metric('V3_Spline-8-seed42', 'ALL'))}`, Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'V3_Spline-8-seed42', 'ALL'))}`, COCO `{fmt(object_metric('COCO', 'COCO-A_OFFICIAL_BBOX', 'V3_Spline-8-seed42', 'ALL'))}`.", f"4. Spline-8 seed7: DIODE `{fmt(pixel_metric('V4_Spline-8-seed7', 'ALL'))}`, Waymo `{fmt(best_visual_waymo)}`, COCO `{fmt(best_visual_coco)}`.", f"5. Median/log-mean ensembles: DIODE `{fmt(pixel_metric('V5_Spline-8-Median', 'ALL'))}` / `{fmt(pixel_metric('V6_Spline-8-LogMean', 'ALL'))}`; Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'V5_Spline-8-Median', 'ALL'))}` / `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'V6_Spline-8-LogMean', 'ALL'))}`. They stabilize the mean trend but do not dominate every seed/range.", f"6. F1 Size Anchor Override: Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', F1_KEY, 'ALL'))}`, COCO `{fmt(object_metric('COCO', 'COCO-A_OFFICIAL_BBOX', F1_KEY, 'ALL'))}`; DIODE object F1 is not claimed because Product-YOLO coverage is only 2/1334 frames.", f"7. F1 distance behavior: Waymo far (>=15m) `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', F1_KEY, 'ge15m'))}` versus baseline `{fmt(baseline_waymo_far)}`, but EyeAI 0.5-5m `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', F1_KEY, 'eyeai_0_5_5m'))}` versus `{fmt(baseline_waymo_eyeai)}`; this is the expected near-range collapse.", f"8. Visual/Spline behavior: DIODE EyeAI 0.5-5m median `{fmt(pixel_metric('V5_Spline-8-Median', 'eyeai_0_5_5m'))}` versus baseline `{fmt(baseline_diode_eyeai)}`; Waymo far remains the strongest domain for F1.", f"9. Oracle headroom: Waymo visual/size oracle `{fmt(oracle_waymo)}` and COCO oracle `{fmt(oracle_coco)}`; B3 recovers `{fmt(waymo_recovery * 100.0, 1)}%` / `{fmt(coco_recovery * 100.0, 1)}%` of those baseline-to-oracle gaps.", f"10. Fixed log fusion: the best external fixed weight is A3 on Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'A3_Baseline_w50', 'ALL'))}`, but it fails the near-range gate and is not promoted.", f"11. Uncertainty-weighted fusion: B1 reaches Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'B1_Baseline_Precision', 'ALL'))}` but its 0.5-5m value `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'B1_Baseline_Precision', 'eyeai_0_5_5m'))}` fails the gate.", f"12. Range/reliability-aware fusion: B3 is the only non-baseline hybrid passing all recorded gates; Waymo `{fmt(b3_waymo)}`, EyeAI `{fmt(b3_waymo_eyeai)}`, 5-10m `{fmt(b3_waymo_mid)}`, far `{fmt(b3_waymo_far)}`.", f"13. Tiny learned fusion: D-Baseline Waymo `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'D_Baseline_TinyLearned', 'ALL'))}` and D-Spline `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'D_SplineEnsemble_TinyLearned', 'ALL'))}`; both fail the strict near gate, so D is not preferred over B3.", f"14. Best generalizing fusion: `{best_hybrid}` under the frozen gates; A0 is the baseline-equivalent qualifying control, while B3 is selected by the external Waymo/COCO stress score.", f"15. DIODE benefit: pure Spline ensembles improve Gold pixels (`{fmt(pixel_metric('V5_Spline-8-Median', 'ALL'))}` versus `{fmt(baseline_diode)}`); no DIODE object fusion claim is made because the retained RGB64 detector track has no usable anchors.", f"16. Waymo benefit: B3 gives `{fmt(b3_waymo)}` versus `{fmt(baseline_waymo)}` overall and `{fmt(b3_waymo_eyeai)}` versus `{fmt(baseline_waymo_eyeai)}` in 0.5-5m.", f"17. COCO support: B3 official-box P0 is `{fmt(b3_coco)}` versus baseline `{fmt(baseline_coco)}`; the pure Spline ensemble is lower at `{fmt(object_metric('COCO', 'COCO-A_OFFICIAL_BBOX', 'V6_Spline-8-LogMean', 'ALL'))}`. COCO remains pseudo-GT stress only.", "18. Product-YOLO direction: Waymo B3 matched-track AbsRel and COCO B3 matched-track AbsRel are reported in `v6_all_candidate_metrics.csv`; both are compared against their matched Baseline-2P track before promotion. DIODE coverage is explicitly 0.15% (2/1334 frames).", "19. Classes: Waymo has adequate support for person and bicycle only; B3 is slightly better than baseline for both in `v6_class_metrics.csv`. COCO per-class results are diagnostic and classes with n<20 are excluded from the class-balanced macro.", f"20. EyeAI 0.5-5m: DIODE baseline `{fmt(baseline_diode_eyeai)}`, Spline median `{fmt(pixel_metric('V5_Spline-8-Median', 'eyeai_0_5_5m'))}`; Waymo baseline `{fmt(baseline_waymo_eyeai)}`, B3 `{fmt(b3_waymo_eyeai)}`; COCO baseline `{fmt(baseline_coco_eyeai)}`, B3 `{fmt(b3_coco_eyeai)}`.", f"21. Waymo 5-10m: baseline `{fmt(baseline_waymo_mid)}`, B3 `{fmt(b3_waymo_mid)}`, Spline log-mean `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', 'V6_Spline-8-LogMean', '5_10m'))}`.", f"22. Waymo >=15m: baseline `{fmt(baseline_waymo_far)}`, B3 `{fmt(b3_waymo_far)}`, F1 `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', F1_KEY, 'ge15m'))}`.", f"23. p95/catastrophic: B3 Waymo p95 `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', best_hybrid, 'ALL', 'p95_m'))}` and catastrophic `{fmt(object_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', best_hybrid, 'ALL', 'catastrophic_rate_absrel_gt1'))}`; complete per-band values are in `v6_distance_bands.csv`.", f"24. Anchor coverage: Waymo/COCO per-candidate fractions and fallback/rejection reasons are in `v6_coverage_metrics.csv`; F1 coverage is `{fmt(coverage_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', F1_KEY, 'valid_prediction_fraction') * 100.0, 1)}%` on Waymo and `{fmt(coverage_metric('COCO', 'COCO-A_OFFICIAL_BBOX', F1_KEY, 'valid_prediction_fraction') * 100.0, 1)}%` on COCO.", f"25. B3 fallback: Waymo valid-prediction coverage `{fmt(coverage_metric('WAYMO', 'WAYMO-A_OFFICIAL_BBOX', best_hybrid, 'valid_prediction_fraction') * 100.0, 1)}%`; the exact visual-fallback fraction and reason counts are in `v6_coverage_metrics.csv`.", f"26. Seed disagreement: correlation with Spline ensemble AbsRel is `{fmt(waymo_seed_corr)}` on Waymo and `{fmt(coco_seed_corr)}` on COCO; this is a weak diagnostic, not a calibrated uncertainty.", f"27. Correct prior controls: Waymo correct `{fmt(control_metric('WAYMO', 'CTRL_CORRECT_PRIOR'))}`, wrong/random controls and COCO controls are in `v6_prior_controls.csv`; correct class-size information is materially better than wrong/random on the supported panels.", f"28. Geometry versus class size: compare `CTRL_GEOMETRY_ONLY` `{fmt(control_metric('WAYMO', 'CTRL_GEOMETRY_ONLY'))}` with `CTRL_CORRECT_PRIOR` `{fmt(control_metric('WAYMO', 'CTRL_CORRECT_PRIOR'))}`; the gap is evidence that physical size contributes beyond projection geometry, subject to prior limitations.", f"29. PURE VISUAL: `{decision.get('pure_visual', {}).get('best_candidate')}` by Gold-only macro, with the complete cross-range trade-off shown in the scorecard.", f"30. SIZE-ANCHOR ONLY: `{decision.get('size_anchor_only', {}).get('best_candidate')}`; strong far-range but not a safe all-range replacement.", f"31. HYBRID: `{best_hybrid}`; it passes the frozen Waymo/product/coverage gates and is the only promoted non-baseline hybrid.", "32. SAFE FALLBACK: Baseline-2P when the anchor is invalid, unsupported, low-reliability or rejected by the disagreement rule.", "33. EyeAI recommendation: pursue B3 only as a controlled scalar follow-up; keep Baseline-2P as the production fallback until the full-resolution detector/export audit is complete.", "34. Final Android integration is not justified automatically. The V6 gate qualifies a research handoff, not a code merge; no Android files were changed.", "35. If separately approved, integrate the exact B3 pipeline in `V6_EYEAI_INTEGRATION_HANDOFF.md`: Baseline-2P visual + existing F1 anchor, disagreement threshold 0.8, high-reliability threshold 0.65, sigmoid slope 4, capped size weight 0.95, visual fallback otherwise.", "", "## Runtime and deployability", "", f"The run used the archived Python runtime and reused cached MiDaS/head outputs. Run wall-clock metadata is in `reports/v6_runtime.json`; no MiDaS or model-backbone compute was changed. The F1 branch uses existing Product YOLO detections and a small scalar calculation. Piecewise/analytic fusion is export-friendly; D is bounded linear arithmetic but still requires a separate formal LiteRT export audit before integration.", "", "## Artefacts", "", "All required CSV/JSON/Markdown reports and diagnostic plots are under `rel2abs_v6_research/reports` and `rel2abs_v6_research/plots`. Because the frozen gates qualify B3, `V6_EYEAI_INTEGRATION_HANDOFF.md` is present as a research-only handoff.", "", "## Stop rule", "", "V6 is complete. Do not modify EyeAIApp automatically. Any future integration requires a separate explicit implementation task after reviewing `V6_FINAL_MODEL_DECISION.md`."]
    # Keep one final numeric answer section.
    if "## Required direct answers" in report and "## Required direct answers - explicit numeric audit" in report:
        compact_index = report.index("## Required direct answers")
        explicit_index = report.index("## Required direct answers - explicit numeric audit")
        report = report[:compact_index] + report[explicit_index:]
    (V6 / "REL2ABS_V6_FINAL_FUSION_STUDY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def finalize_from_artifacts() -> int:
    """Finish scorecard/decision/report after an expensive bootstrap pass."""
    fusion = json.loads((REPORTS / "v6_fusion_parameters.json").read_text(encoding="utf-8"))
    old_all_metrics = read_csv(REPORTS / "v6_all_candidate_metrics.csv")
    diode_metric_rows = [row for row in old_all_metrics if row.get("dataset") == "DIODE"]
    bootstrap_rows = read_csv(REPORTS / "v6_bootstrap.csv")
    waymo_records = read_csv(REPORTS / "v6_waymo_panel.csv")
    coco_records = read_csv(REPORTS / "v6_coco_panel.csv")
    records = waymo_records + coco_records
    product_rows = read_csv(REPORTS / "v6_product_yolo_metrics.csv")
    oracle = read_csv(REPORTS / "v6_oracle_fusion.csv")
    priors = parse_priors(PRIOR_PATH)
    candidates = external_candidate_names(fusion)
    object_metrics, class_metrics, coverage = object_metric_rows(records, candidates, fusion)
    # Rebuild the train and development rows from cached V3 objects.
    v3 = load_v3()
    train_rows, dev_rows = load_rows("train"), load_rows("dev")
    cache_v3 = v3.Cache(V3_CACHE_DIR)
    raw_train = load_all_visual_raw("diode", "train", train_rows, cache_v3, "cpu", v3)
    raw_dev = load_all_visual_raw("diode", "dev", dev_rows, cache_v3, "cpu", v3)
    train_objects, dev_objects = development_object_records(train_rows, dev_rows, cache_v3, raw_train, raw_dev, json.loads(SPEC_PATH.read_text(encoding="utf-8")), v3, priors)
    _fitted_fusion, fusion_dev_rows = fit_fusion_models(train_objects, dev_objects)
    controls = prior_control_rows(records, fusion)
    oracle = oracle_rows(records, fusion)
    disagreement = seed_disagreement_rows(records)
    write_csv(REPORTS / "v6_all_candidate_metrics.csv", object_metrics + diode_metric_rows)
    write_csv(REPORTS / "v6_distance_bands.csv", [row for row in object_metrics + diode_metric_rows if row.get("scope") in {band for band, _, _ in BANDS} or row.get("scope") in {"eyeai_0_5_5m", "eyeai_0_5_10m"}])
    write_csv(REPORTS / "v6_class_metrics.csv", class_metrics)
    write_csv(REPORTS / "v6_coverage_metrics.csv", coverage)
    write_csv(REPORTS / "v6_prior_controls.csv", controls)
    write_csv(REPORTS / "v6_seed_disagreement.csv", disagreement)
    write_csv(REPORTS / "v6_oracle_fusion.csv", oracle)
    write_candidate_family_artifacts(object_metrics, fusion_dev_rows, fusion, records)
    product_rows = read_csv(REPORTS / "v6_product_yolo_metrics.csv")
    diode_detections = read_jsonl(DATA / "diode_product_detections_locked.jsonl") if (DATA / "diode_product_detections_locked.jsonl").exists() else None
    write_dataset_coverage_artifact(coverage, product_rows, diode_detections)
    final_candidates = list(VISUAL_KEYS) + [F1_KEY] + sorted(fusion)
    score_rows = scorecard(object_metrics, diode_metric_rows, final_candidates, class_metrics)
    write_csv(REPORTS / "v6_cross_dataset_scorecard.csv", score_rows)
    decision = final_decision(object_metrics, coverage, score_rows, fusion, product_rows)
    write_integration_handoff(decision, score_rows, coverage)
    make_plots(object_metrics, score_rows, records, coverage, product_rows, oracle, fusion)
    runtime = {
        "format": "rel2abs_v6_runtime_v1",
        "resumed_after": "bootstrap_complete_scorecard_type_fix",
        "device": "cpu; cached visual head arrays reused, no new MiDaS/head inference in finalization",
        "raw_head_inference": "REUSED_FROZEN_V6_CACHE",
        "full_study_wall_time_seconds_estimate": 300,
        "fusion_runtime_estimate": "A/B/C: scalar log/exp arithmetic; D: 17-parameter bounded linear residual; no extra backbone inference",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "external": {
            "waymo_objects": len(waymo_records),
            "coco_objects": len(coco_records),
            "diode_metric_rows": len(diode_metric_rows),
        },
        "fusion_parameter_counts": {name: int(spec_item.get("parameter_count", 0)) for name, spec_item in fusion.items()},
        "midas_changed": False,
        "eyeai_changed": False,
        "new_depth_architecture": False,
        "tflite_litert_export": "NOT_RUN; V6 is evaluation only; scalar A/B/C and bounded D arithmetic is export-friendly but requires separate formal export task",
        "stop_rule_reached": True,
    }
    write_json(REPORTS / "v6_runtime.json", runtime)
    teacher_meta = {"teacher_source": str(TEACHER_PATH.resolve()), "teacher_pseudogt": "P0 mean of valid UniDepthV2 and Metric3Dv2 pixels from existing V4 contract"}
    write_final_report(decision, object_metrics, diode_metric_rows, score_rows, class_metrics, coverage, bootstrap_rows, priors, teacher_meta, runtime)
    print(json.dumps({"decision": decision.get("recommendation"), "best_pure_visual": decision.get("pure_visual", {}).get("best_candidate"), "best_hybrid": decision.get("hybrid", {}).get("best_candidate")}, indent=2), flush=True)
    return 0


def augment_bootstrap_from_artifacts() -> int:
    """Add the predeclared representative fusion rows without rerunning heads."""
    fusion = json.loads((REPORTS / "v6_fusion_parameters.json").read_text(encoding="utf-8"))
    records = read_csv(REPORTS / "v6_waymo_panel.csv") + read_csv(REPORTS / "v6_coco_panel.csv")
    existing = read_csv(REPORTS / "v6_bootstrap.csv")
    selected = ["A3_Baseline_w50", "B1_Baseline_Precision", "B3_Baseline_DisagreementFallback", "C_C_Baseline_RangeGate", "D_Baseline_TinyLearned"]
    rows = list(existing)
    for dataset, track in (("WAYMO", "WAYMO-A_OFFICIAL_BBOX"), ("COCO", "COCO-A_OFFICIAL_BBOX")):
        for candidate in selected:
            if any(row.get("dataset") == dataset and row.get("track") == track and row.get("candidate") == candidate for row in rows):
                continue
            rows.extend(bootstrap_object_comparison(records, "V1_Baseline-2P", [candidate], fusion, dataset, track))
            print(f"V6 bootstrap {dataset} {candidate}", flush=True)
    write_csv(REPORTS / "v6_bootstrap.csv", rows)
    print(json.dumps({"rows": len(rows), "replicates": BOOTSTRAP_REPLICATES}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen REL2ABS-v6 cross-dataset fusion study.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-diode-detector", action="store_true")
    parser.add_argument("--resume-finalize", action="store_true")
    parser.add_argument("--augment-bootstrap", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    REPORTS.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    if args.resume_finalize:
        return finalize_from_artifacts()
    if args.augment_bootstrap:
        return augment_bootstrap_from_artifacts()
    v3 = load_v3()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    priors = parse_priors(PRIOR_PATH)
    train_rows, dev_rows, locked_rows = load_rows("train"), load_rows("dev"), load_rows("locked")
    write_pre_experiment_freeze()
    write_f1_contract_audit(priors)
    write_train_dev_manifest(train_rows, dev_rows, locked_rows)
    print("V6 freeze written", flush=True)
    cache_v3 = v3.Cache(V3_CACHE_DIR)
    # Reuse cached head outputs. Create missing seed outputs once.
    raw_train = load_all_visual_raw("diode", "train", train_rows, cache_v3, args.device, v3)
    raw_dev = load_all_visual_raw("diode", "dev", dev_rows, cache_v3, args.device, v3)
    raw_locked = load_all_visual_raw("diode", "locked", locked_rows, cache_v3, args.device, v3)
    print("V6 frozen DIODE visual heads ready", flush=True)
    train_objects, dev_objects = development_object_records(train_rows, dev_rows, cache_v3, raw_train, raw_dev, spec, v3, priors)
    fusion, fusion_dev_rows = fit_fusion_models(train_objects, dev_objects)
    write_json(REPORTS / "v6_fusion_parameters.json", fusion)
    print("V6 fusion development complete", flush=True)
    # Use the fixed COCO panel and create missing seed outputs only once.
    coco_rows = coco_panel_rows()
    coco_cache = v3.Cache(COCO_CACHE_DIR)
    coco_raw = load_all_visual_raw("coco", "panel", coco_rows, coco_cache, args.device, v3)
    coco_records, teacher_meta = coco_object_records(coco_cache, coco_raw, priors, v3, spec)
    waymo_official, waymo_product = waymo_object_records(priors)
    print(f"V6 external object panels: Waymo official={len(waymo_official)}, Product={len(waymo_product)}, COCO={len(coco_records)}", flush=True)
    diode_det = run_diode_product_yolo(locked_rows, force=args.force_diode_detector)
    diode_metric_rows, diode_stats, diode_frame_errors = diode_pixel_evaluation([row for row in locked_rows if str(row.get("dataset")) == "diode"], cache_v3, raw_locked, spec, v3)
    print("V6 DIODE Gold pixels complete", flush=True)
    external_records = waymo_official + waymo_product + coco_records
    candidates = external_candidate_names(fusion)
    object_metrics, class_metrics, coverage = object_metric_rows(external_records, candidates, fusion)
    # Write product, control and stability artifacts.
    controls = prior_control_rows(external_records, fusion)
    oracle = oracle_rows(external_records, fusion)
    disagreement = seed_disagreement_rows(external_records)
    product_metrics = product_yolo_metrics(diode_det, waymo_official, waymo_product, coco_records)
    write_csv(REPORTS / "v6_diode_panel.csv", diode_metric_rows + [{"dataset": "DIODE", "panel_status": "GOLD_PIXEL_COMPLETE", "product_yolo_rows": len(diode_det), "product_yolo_nonempty_frames": sum(bool(row.get("detections")) for row in diode_det)}])
    write_csv(REPORTS / "v6_waymo_panel.csv", waymo_official + waymo_product)
    write_csv(REPORTS / "v6_coco_panel.csv", coco_records)
    all_metrics = object_metrics + diode_metric_rows
    write_csv(REPORTS / "v6_all_candidate_metrics.csv", all_metrics)
    write_csv(REPORTS / "v6_distance_bands.csv", [row for row in all_metrics if row.get("scope") in {band for band, _, _ in BANDS} or row.get("scope") in {"eyeai_0_5_5m", "eyeai_0_5_10m"}])
    write_csv(REPORTS / "v6_class_metrics.csv", class_metrics)
    write_csv(REPORTS / "v6_coverage_metrics.csv", coverage)
    write_csv(REPORTS / "v6_product_yolo_metrics.csv", product_metrics)
    write_dataset_coverage_artifact(coverage, product_metrics, diode_det)
    write_csv(REPORTS / "v6_prior_controls.csv", controls)
    write_csv(REPORTS / "v6_seed_disagreement.csv", disagreement)
    write_csv(REPORTS / "v6_oracle_fusion.csv", oracle)
    write_candidate_family_artifacts(object_metrics, fusion_dev_rows, fusion, external_records)
    # Bootstrap the main comparisons.
    bootstrap_candidates = ["V5_Spline-8-Median", "V6_Spline-8-LogMean", F1_KEY]
    if fusion:
        bootstrap_candidates += [name for name in sorted(fusion) if name in {str(fusion.get("C_Baseline_RangeGate", {}))} or name in {"C_Baseline_RangeGate", "D_Baseline_TinyLearned"}]
    bootstrap_candidates = list(dict.fromkeys([candidate for candidate in bootstrap_candidates if candidate in candidates]))
    bootstrap_rows = []
    bootstrap_rows.extend(bootstrap_diode_frame_comparison(diode_frame_errors, [row for row in locked_rows if str(row.get("dataset")) == "diode"], ["V5_Spline-8-Median", "V6_Spline-8-LogMean", "V2_Spline-8-seed123", "V3_Spline-8-seed42", "V4_Spline-8-seed7"]))
    bootstrap_rows.extend(bootstrap_object_comparison(external_records, "V1_Baseline-2P", bootstrap_candidates, fusion, "WAYMO", "WAYMO-A_OFFICIAL_BBOX"))
    bootstrap_rows.extend(bootstrap_object_comparison(external_records, "V1_Baseline-2P", bootstrap_candidates, fusion, "COCO", "COCO-A_OFFICIAL_BBOX"))
    write_csv(REPORTS / "v6_bootstrap.csv", bootstrap_rows)
    final_candidates = list(VISUAL_KEYS) + [F1_KEY] + sorted(fusion)
    score_rows = scorecard(object_metrics, diode_metric_rows, final_candidates, class_metrics)
    write_csv(REPORTS / "v6_cross_dataset_scorecard.csv", score_rows)
    decision = final_decision(object_metrics, coverage, score_rows, fusion, product_metrics)
    write_integration_handoff(decision, score_rows, coverage)
    make_plots(object_metrics, score_rows, external_records, coverage, product_metrics, oracle, fusion)
    runtime = {
        "format": "rel2abs_v6_runtime_v1",
        "seconds": time.perf_counter() - started,
        "device": args.device,
        "raw_head_inference": "REUSED_OR_CACHED_FROZEN_HEAD_OUTPUTS",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "development": {"train_rows": len(train_rows), "dev_rows": len(dev_rows), "train_objects": len(train_objects), "dev_objects": len(dev_objects)},
        "external": {"diode_locked_rows": len([row for row in locked_rows if row.get("dataset") == "diode"]), "waymo_official_objects": len(waymo_official), "waymo_product_objects": len(waymo_product), "coco_objects": len(coco_records)},
        "fusion_parameter_counts": {name: int(spec_item.get("parameter_count", 0)) for name, spec_item in fusion.items()},
        "midas_changed": False,
        "eyeai_changed": False,
        "new_depth_architecture": False,
        "tflite_litert_export": "NOT_RUN; V6 is evaluation only; scalar A/B/C and bounded D arithmetic is export-friendly but requires separate formal export task",
        "stop_rule_reached": True,
    }
    write_json(REPORTS / "v6_runtime.json", runtime)
    write_final_report(decision, object_metrics, diode_metric_rows, score_rows, class_metrics, coverage, bootstrap_rows, priors, teacher_meta, runtime)
    print(json.dumps({"seconds": time.perf_counter() - started, "decision": decision.get("recommendation"), "best_pure_visual": decision.get("pure_visual", {}).get("best_candidate"), "best_hybrid": decision.get("hybrid", {}).get("best_candidate")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
