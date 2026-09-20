from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
V6 = PROJECT / "rel2abs_v6_research"
V2 = PROJECT / "rel2abs_v2_research"
V4 = PROJECT / "rel2abs_v4_research"
V5B = PROJECT / "rel2abs_v5b_research"
SRC = V6 / "src"
REPORTS = V6 / "reports"
PLOTS = V6 / "plots"
SEED = 20260919
HIDDEN = 16
DEFAULT_BOOTSTRAP = 1000
OBJECT_FEATURE_GATE_FORMAT = "rel2abs_v6_object_feature_gate_v2"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import run_v6_cross_dataset_neural_gate as prior_gate
import run_v6_fusion_study as v6


try:
    import torch
    import torch.nn as nn
except Exception as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    TORCH_IMPORT_ERROR = None


FEATURE_GROUPS = (
    "HEIGHT",
    "WIDTH",
    "SHAPE_POSITION",
    "ANCHOR",
    "DETECTION",
    "SEGMENTATION",
)

GROUP_DESCRIPTIONS = {
    "HEIGHT": "normalized bbox height only",
    "WIDTH": "normalized bbox width only",
    "SHAPE_POSITION": "bbox area/aspect and normalized center/bottom/border",
    "ANCHOR": "F1 height/width decomposition, reliability and uncertainty",
    "DETECTION": "frame object count, object-size and detector-confidence context",
    "SEGMENTATION": "global and bbox-local semantic segmentation summaries",
}

DEPTH_FEATURE_NAMES = (
    "depth_log_visual",
    "depth_log_f1",
    "depth_abs_log_disagreement",
    "depth_signed_log_disagreement",
)

GROUP_FEATURE_NAMES = {
    "HEIGHT": (
        "bbox_log_height",
    ),
    "WIDTH": (
        "bbox_log_width",
    ),
    "SHAPE_POSITION": (
        "bbox_log_area",
        "bbox_log_aspect",
        "bbox_center_x",
        "bbox_center_y",
        "bbox_bottom_y",
        "bbox_border",
    ),
    "ANCHOR": (
        "anchor_log_z_height",
        "anchor_log_z_width",
        "anchor_log_z_generic",
        "anchor_log_z_geometry",
        "anchor_sigma_log",
        "anchor_reliability",
        "anchor_prior_reliability",
        "anchor_valid",
    ),
    "DETECTION": (
        "det_object_count_log1p",
        "det_supported_count_log1p",
        "det_mean_bbox_area",
        "det_max_bbox_area",
        "det_mean_bbox_width",
        "det_max_bbox_width",
        "det_mean_bbox_height",
        "det_max_bbox_height",
        "det_mean_confidence",
        "det_class_diversity",
        "det_border_fraction",
        "det_target_confidence",
        "det_target_class_supported",
    ),
    "SEGMENTATION": (
        "seg_available",
        "seg_present_count",
        "seg_max_area",
        "seg_entropy",
        "seg_road_area",
        "seg_sidewalk_area",
        "seg_building_area",
        "seg_vegetation_area",
        "seg_sky_area",
        "seg_people_area",
        "seg_vehicle_area",
        "seg_obj_available",
        "seg_obj_present_count",
        "seg_obj_max_area",
        "seg_obj_entropy",
        "seg_obj_class_supported",
        "seg_obj_class_area",
        "seg_obj_road_area",
        "seg_obj_building_area",
        "seg_obj_vegetation_area",
        "seg_obj_sky_area",
        "seg_obj_people_area",
        "seg_obj_vehicle_area",
    ),
}

SEGMENTATION_LABELS = tuple(prior_gate.SEMANTIC_LABELS)
SEGMENTATION_INDEX = {name: index for index, name in enumerate(SEGMENTATION_LABELS)}
SUPPORTED_DETECTION_CLASSES = {"person", "bicycle", "car"}
PEOPLE_CLASSES = {"person", "rider"}
VEHICLE_CLASSES = {"car", "truck", "bus", "train", "motorcycle", "bicycle"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def safe(value: Any, default: float = math.nan) -> float:
    return v6.safe_float(value, default)


def finite_positive(value: Any) -> bool:
    number = safe(value)
    return bool(np.isfinite(number) and number > 0)


def bounded_log(value: Any, floor: float = 1e-6, default: float = 0.0) -> float:
    number = safe(value, math.nan)
    return float(math.log(max(number, floor))) if np.isfinite(number) and number > 0 else default


def aliases(row: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("sample_id", "frame_id", "image_id", "group_id"):
        value = row.get(key)
        if value is not None and str(value) not in {"", "None", "nan"}:
            result.add(str(value))
    return result


def build_lookup(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for key in aliases(row):
            result[key] = row
    return result


def segmentation_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    values = values[values > 0]
    if not values.size:
        return 0.0
    return float(-(values * np.log(np.maximum(values, 1e-12))).sum() / math.log(max(len(SEGMENTATION_LABELS), 2)))


def class_area(values: np.ndarray, *names: str) -> float:
    return float(sum(values[SEGMENTATION_INDEX[name]] for name in names if name in SEGMENTATION_INDEX and SEGMENTATION_INDEX[name] < values.size))


def bbox_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def object_segmentation_features(segmentation: Mapping[str, Any] | None, record: Mapping[str, Any]) -> dict[str, float]:
    """Project the existing 4x4 scene summary into the current box."""
    zero = {
        "seg_available": 0.0,
        "seg_present_count": 0.0,
        "seg_max_area": 0.0,
        "seg_entropy": 0.0,
        "seg_road_area": 0.0,
        "seg_sidewalk_area": 0.0,
        "seg_building_area": 0.0,
        "seg_vegetation_area": 0.0,
        "seg_sky_area": 0.0,
        "seg_people_area": 0.0,
        "seg_vehicle_area": 0.0,
        "seg_obj_available": 0.0,
        "seg_obj_present_count": 0.0,
        "seg_obj_max_area": 0.0,
        "seg_obj_entropy": 0.0,
        "seg_obj_class_supported": 0.0,
        "seg_obj_class_area": 0.0,
        "seg_obj_road_area": 0.0,
        "seg_obj_building_area": 0.0,
        "seg_obj_vegetation_area": 0.0,
        "seg_obj_sky_area": 0.0,
        "seg_obj_people_area": 0.0,
        "seg_obj_vehicle_area": 0.0,
    }
    if not segmentation:
        return zero
    global_values = np.asarray(segmentation.get("global_area_fraction") or [], dtype=np.float64)
    global_values = np.clip(global_values, 0.0, 1.0)
    if global_values.size:
        zero.update({
            "seg_available": 1.0,
            "seg_present_count": float(np.sum(global_values > 1e-6)),
            "seg_max_area": float(global_values.max()),
            "seg_entropy": segmentation_entropy(global_values),
            "seg_road_area": class_area(global_values, "road"),
            "seg_sidewalk_area": class_area(global_values, "sidewalk"),
            "seg_building_area": class_area(global_values, "building"),
            "seg_vegetation_area": class_area(global_values, "vegetation"),
            "seg_sky_area": class_area(global_values, "sky"),
            "seg_people_area": class_area(global_values, *PEOPLE_CLASSES),
            "seg_vehicle_area": class_area(global_values, *VEHICLE_CLASSES),
        })

    grid_values = np.asarray(segmentation.get("grid_area_fraction") or [], dtype=np.float64)
    grid_shape = tuple(int(x) for x in (segmentation.get("grid_shape") or [4, 4, len(SEGMENTATION_LABELS)]))
    if len(grid_shape) != 3 or grid_shape[2] != len(SEGMENTATION_LABELS) or grid_values.size != int(np.prod(grid_shape)):
        return zero
    grid = np.clip(grid_values.reshape(grid_shape), 0.0, 1.0)
    xc = np.clip(safe(record.get("bbox_center_x"), 0.5), 0.0, 1.0)
    yc = np.clip(safe(record.get("bbox_center_y"), 0.5), 0.0, 1.0)
    bw = max(0.0, min(1.0, safe(record.get("bbox_width"), 0.0)))
    bh = max(0.0, min(1.0, safe(record.get("bbox_height"), 0.0)))
    x0, x1 = max(0.0, xc - bw / 2.0), min(1.0, xc + bw / 2.0)
    y0, y1 = max(0.0, yc - bh / 2.0), min(1.0, yc + bh / 2.0)
    bbox_area = max((x1 - x0) * (y1 - y0), 1e-8)
    projected = np.zeros(len(SEGMENTATION_LABELS), dtype=np.float64)
    gh, gw = grid_shape[0], grid_shape[1]
    for row_index in range(gh):
        cy0, cy1 = row_index / gh, (row_index + 1) / gh
        overlap_y = bbox_overlap(y0, y1, cy0, cy1)
        if overlap_y <= 0:
            continue
        for col_index in range(gw):
            cx0, cx1 = col_index / gw, (col_index + 1) / gw
            overlap = overlap_y * bbox_overlap(x0, x1, cx0, cx1)
            if overlap > 0:
                projected += (overlap / bbox_area) * grid[row_index, col_index]
    projected = np.clip(projected, 0.0, 1.0)
    class_name = str(record.get("class_name", ""))
    expected_index = SEGMENTATION_INDEX.get(class_name)
    zero.update({
        "seg_obj_available": 1.0,
        "seg_obj_present_count": float(np.sum(projected > 1e-6)),
        "seg_obj_max_area": float(projected.max()) if projected.size else 0.0,
        "seg_obj_entropy": segmentation_entropy(projected),
        "seg_obj_class_supported": float(expected_index is not None),
        "seg_obj_class_area": float(projected[expected_index]) if expected_index is not None and expected_index < projected.size else 0.0,
        "seg_obj_road_area": class_area(projected, "road"),
        "seg_obj_building_area": class_area(projected, "building"),
        "seg_obj_vegetation_area": class_area(projected, "vegetation"),
        "seg_obj_sky_area": class_area(projected, "sky"),
        "seg_obj_people_area": class_area(projected, *PEOPLE_CLASSES),
        "seg_obj_vehicle_area": class_area(projected, *VEHICLE_CLASSES),
    })
    return zero


def normalized_bbox_hint(item: Mapping[str, Any], width: float = 1.0, height: float = 1.0) -> dict[str, float]:
    if "bbox" in item:
        values = [safe(value, 0.0) for value in item.get("bbox", [])]
        if len(values) >= 4:
            x, y, bw, bh = values[:4]
            return {"bbox_center_x": (x + bw / 2.0) / max(width, 1.0), "bbox_center_y": (y + bh / 2.0) / max(height, 1.0), "bbox_width": bw / max(width, 1.0), "bbox_height": bh / max(height, 1.0)}
    return {
        "bbox_center_x": safe(item.get("x_center"), safe(item.get("bbox_center_x_norm"), 0.5)),
        "bbox_center_y": safe(item.get("y_center"), safe(item.get("bbox_center_y_norm"), 0.5)),
        "bbox_width": safe(item.get("width"), safe(item.get("bbox_width_norm"), 0.0)),
        "bbox_height": safe(item.get("height"), safe(item.get("bbox_height_norm"), 0.0)),
    }


def detection_bbox_hints(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    hints: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        detections = list(row.get("detections") or [])
        width, height = max(safe(row.get("width"), 1.0), 1.0), max(safe(row.get("height"), 1.0), 1.0)
        for index, item in enumerate(detections):
            for key in (str(row.get("sample_id", "")), str(row.get("frame_id", ""))):
                if key and key != "None":
                    hints[f"{key}:{index}"] = normalized_bbox_hint(item, width, height)
    return hints


def rich_context_sources() -> dict[str, dict[str, Mapping[str, Any]]]:
    """Load the existing detection and segmentation caches."""
    v3_det = read_jsonl(V2 / "cache" / "scene_context" / "detections.jsonl")
    v3_seg = read_jsonl(V2 / "cache" / "scene_context" / "segmentation.jsonl")
    coco_det = read_jsonl(V4 / "data" / "coco" / "product_vision" / "detections.jsonl")
    coco_seg = read_jsonl(V4 / "data" / "coco" / "product_vision" / "segmentation.jsonl")
    waymo_det = read_jsonl(V5B / "data" / "waymo_product_detections.jsonl")
    waymo_seg = read_jsonl(REPORTS / "v6_waymo_semantic_context.jsonl")
    sources: dict[str, dict[str, Mapping[str, Any]]] = {}
    for dataset in {str(row.get("dataset", "")).upper() for row in v3_det + v3_seg} - {""}:
        det_rows = [row for row in v3_det if str(row.get("dataset", "")).upper() == dataset]
        seg_rows = [row for row in v3_seg if str(row.get("dataset", "")).upper() == dataset]
        sources[dataset] = {
            "detections": build_lookup(det_rows),
            "segmentations": build_lookup(seg_rows),
            "bbox_hints": detection_bbox_hints(det_rows),
        }
    # Use the complete V2 cache for V3 training and development data.
    diode_det = [row for row in v3_det if str(row.get("dataset", "")).upper() == "DIODE"]
    diode_seg = [row for row in v3_seg if str(row.get("dataset", "")).upper() == "DIODE"]
    sources["DIODE"] = {
        "detections": build_lookup(diode_det),
        "segmentations": build_lookup(diode_seg),
        "bbox_hints": detection_bbox_hints(diode_det),
    }
    sources["COCO"] = {
        "detections": build_lookup(coco_det),
        "segmentations": build_lookup(coco_seg),
        "bbox_hints": detection_bbox_hints(coco_det),
    }
    for image in v6.coco_panel_rows():
        sample_id = str(image.get("sample_id", ""))
        for index, item in enumerate(image.get("objects", [])):
            sources["COCO"]["bbox_hints"][f"{sample_id}:{index}"] = normalized_bbox_hint(item, safe(image.get("width"), 1.0), safe(image.get("height"), 1.0))
    waymo_object_rows = read_jsonl(V5B / "data" / "waymo_object_predictions.jsonl") + read_jsonl(V5B / "data" / "waymo_product_object_predictions.jsonl")
    waymo_hints = detection_bbox_hints(waymo_det)
    for row in waymo_object_rows:
        key = str(row.get("object_key", ""))
        if key:
            waymo_hints[key] = normalized_bbox_hint(row)
    sources["WAYMO"] = {
        "detections": build_lookup(waymo_det),
        "segmentations": build_lookup(waymo_seg),
        "bbox_hints": waymo_hints,
    }
    return sources


def attach_rich_context(records: list[dict[str, Any]], sources: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in records:
        row = dict(original)
        dataset = str(row.get("dataset", "")).upper()
        source = sources.get(dataset, {})
        det_lookup = source.get("detections", {})
        seg_lookup = source.get("segmentations", {})
        bbox_lookup = source.get("bbox_hints", {})
        det_row = next((det_lookup[key] for key in (str(row.get("sample_id", "")), str(row.get("frame_id", "")), str(row.get("group_id", ""))) if key in det_lookup), None)
        seg_row = next((seg_lookup[key] for key in (str(row.get("sample_id", "")), str(row.get("frame_id", "")), str(row.get("group_id", ""))) if key in seg_lookup), None)
        bbox_keys = (
            str(row.get("object_key", "")),
            f"{row.get('track', '')}:{row.get('sample_id', '')}:{row.get('object_index', '')}",
            f"{row.get('sample_id', '')}:{row.get('object_index', '')}",
        )
        bbox_hint = next((bbox_lookup[key] for key in bbox_keys if key in bbox_lookup), None)
        if bbox_hint:
            for key in ("bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height"):
                if key in bbox_hint:
                    row[key] = float(bbox_hint[key])
        context = prior_gate.object_context(det_row, seg_row)
        detections = list((det_row or {}).get("detections") or [])
        areas = np.asarray([safe(item.get("bbox_area"), safe(item.get("width"), 0.0) * safe(item.get("height"), 0.0)) for item in detections], dtype=np.float64)
        widths = np.asarray([safe(item.get("width"), 0.0) for item in detections], dtype=np.float64)
        heights = np.asarray([safe(item.get("height"), 0.0) for item in detections], dtype=np.float64)
        context.update({
            "det_mean_bbox_width": float(widths.mean()) if widths.size else 0.0,
            "det_max_bbox_width": float(widths.max()) if widths.size else 0.0,
            "det_target_confidence": max(0.0, safe(row.get("detection_confidence"), 0.0)),
            "det_target_class_supported": float(str(row.get("class_name", "")) in SUPPORTED_DETECTION_CLASSES),
        })
        row.update(context)
        row.update(object_segmentation_features(seg_row, row))
        result.append(row)
    return result


def group_feature_values(record: Mapping[str, Any]) -> dict[str, list[float]]:
    visual = safe(record.get("V1_Baseline-2P"), 1e-6)
    f1 = safe(record.get("z_size"), 1e-6)
    signed_disagreement = math.log(max(f1, 1e-6) / max(visual, 1e-6))
    context = {
        "depth_log_visual": bounded_log(visual),
        "depth_log_f1": bounded_log(f1),
        "depth_abs_log_disagreement": abs(signed_disagreement),
        "depth_signed_log_disagreement": signed_disagreement,
        "bbox_log_height": bounded_log(record.get("bbox_height"), 1e-5),
        "bbox_log_width": bounded_log(record.get("bbox_width"), 1e-5),
        "bbox_log_area": bounded_log(record.get("bbox_area"), 1e-7),
        "bbox_log_aspect": bounded_log(record.get("bbox_aspect"), 1e-5),
        "bbox_center_x": float(np.clip(safe(record.get("bbox_center_x"), 0.5), 0.0, 1.0)),
        "bbox_center_y": float(np.clip(safe(record.get("bbox_center_y"), 0.5), 0.0, 1.0)),
        "bbox_bottom_y": float(np.clip(safe(record.get("bbox_center_y"), 0.5) + safe(record.get("bbox_height"), 0.0) / 2.0, 0.0, 1.0)),
        "bbox_border": safe(record.get("border"), 0.0),
        "anchor_log_z_height": bounded_log(record.get("z_height")),
        "anchor_log_z_width": bounded_log(record.get("z_width")),
        "anchor_log_z_generic": bounded_log(record.get("z_generic")),
        "anchor_log_z_geometry": bounded_log(record.get("z_geometry")),
        "anchor_sigma_log": max(0.0, safe(record.get("sigma_log"), 0.0)),
        "anchor_reliability": np.clip(safe(record.get("reliability"), 0.0), 0.0, 1.0),
        "anchor_prior_reliability": np.clip(safe(record.get("prior_reliability"), 0.0), 0.0, 1.0),
        "anchor_valid": float(safe(record.get("anchor_valid"), 0.0) > 0),
        "det_object_count_log1p": math.log1p(max(0.0, safe(record.get("ctx_object_count"), 0.0))),
        "det_supported_count_log1p": math.log1p(max(0.0, safe(record.get("ctx_supported_object_count"), 0.0))),
        "det_mean_bbox_area": max(0.0, safe(record.get("ctx_mean_bbox_area"), 0.0)),
        "det_max_bbox_area": max(0.0, safe(record.get("ctx_max_bbox_area"), 0.0)),
        "det_mean_bbox_width": max(0.0, safe(record.get("det_mean_bbox_width"), 0.0)),
        "det_max_bbox_width": max(0.0, safe(record.get("det_max_bbox_width"), 0.0)),
        "det_mean_bbox_height": max(0.0, safe(record.get("ctx_mean_bbox_height"), 0.0)),
        "det_max_bbox_height": max(0.0, safe(record.get("ctx_max_bbox_height"), 0.0)),
        "det_mean_confidence": max(0.0, safe(record.get("ctx_mean_confidence"), 0.0)),
        "det_class_diversity": max(0.0, safe(record.get("ctx_class_diversity"), 0.0)),
        "det_border_fraction": np.clip(safe(record.get("ctx_border_fraction"), 0.0), 0.0, 1.0),
        "det_target_confidence": np.clip(safe(record.get("det_target_confidence"), 0.0), 0.0, 1.0),
        "det_target_class_supported": np.clip(safe(record.get("det_target_class_supported"), 0.0), 0.0, 1.0),
    }
    for name in GROUP_FEATURE_NAMES["SEGMENTATION"]:
        context[name] = max(0.0, safe(record.get(name), 0.0))
    return {
        "DEPTH": [float(context[name]) for name in DEPTH_FEATURE_NAMES],
        **{group: [float(context[name]) for name in names] for group, names in GROUP_FEATURE_NAMES.items()},
    }


def feature_set_names() -> list[str]:
    names = ["DEPTH_ONLY"]
    for size in range(1, len(FEATURE_GROUPS) + 1):
        for combo in itertools.combinations(FEATURE_GROUPS, size):
            names.append("DEPTH+" + "+".join(combo))
    names.append("LEGACY_BASE")
    return names


def selected_groups(feature_set: str) -> tuple[str, ...]:
    if feature_set in {"DEPTH_ONLY", "LEGACY_BASE"}:
        return ()
    return tuple(feature_set.split("+")[1:])


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "LEGACY_BASE":
        return [f"legacy_base_{index}" for index in range(16)]
    names = list(DEPTH_FEATURE_NAMES)
    for group in selected_groups(feature_set):
        names.extend(GROUP_FEATURE_NAMES[group])
    return names


def feature_vector(record: Mapping[str, Any], feature_set: str) -> np.ndarray:
    if feature_set == "LEGACY_BASE":
        return v6.feature_vector(record, "V1_Baseline-2P")
    values = group_feature_values(record)
    output = list(values["DEPTH"])
    for group in selected_groups(feature_set):
        output.extend(values[group])
    return np.asarray(output, dtype=np.float64)


def valid_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        if all(finite_positive(row.get(key)) for key in ("V1_Baseline-2P", "z_size", "gt_m")):
            result.append(dict(row))
    return result


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (values - mean) / scale, mean, scale


_TinyGateBase = nn.Module if nn is not None else object


class TinyGate(_TinyGateBase):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, x: Any) -> Any:
        return torch.sigmoid(self.net(x)).reshape(-1)


def _numpy_sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def train_gate_numpy(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    feature_set: str,
    mixture: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train the same small gate with NumPy when Torch is unavailable."""
    train_records, dev_records = valid_records(train_records), valid_records(dev_records)
    if len(train_records) < 20 or len(dev_records) < 10:
        raise RuntimeError(f"Insufficient records for {feature_set}/{mixture}: train={len(train_records)} dev={len(dev_records)}")
    rng = np.random.default_rng(seed)
    names = feature_names(feature_set)
    raw_train = np.stack([feature_vector(row, feature_set) for row in train_records]).astype(np.float64)
    raw_dev = np.stack([feature_vector(row, feature_set) for row in dev_records]).astype(np.float64)
    train_x, mean, scale = standardize(raw_train)
    dev_x = (raw_dev - mean) / scale
    y_train = np.asarray([math.log(safe(row["gt_m"])) for row in train_records], dtype=np.float64)
    y_dev = np.asarray([math.log(safe(row["gt_m"])) for row in dev_records], dtype=np.float64)
    visual_train = np.asarray([math.log(safe(row["V1_Baseline-2P"])) for row in train_records], dtype=np.float64)
    f1_train = np.asarray([math.log(safe(row["z_size"])) for row in train_records], dtype=np.float64)
    dev_visual = np.asarray([math.log(safe(row["V1_Baseline-2P"])) for row in dev_records], dtype=np.float64)
    dev_f1 = np.asarray([math.log(safe(row["z_size"])) for row in dev_records], dtype=np.float64)

    input_dim = train_x.shape[1]
    w1 = rng.normal(0.0, math.sqrt(2.0 / max(input_dim, 1)), size=(HIDDEN, input_dim))
    b1 = np.zeros(HIDDEN, dtype=np.float64)
    w2 = rng.normal(0.0, math.sqrt(2.0 / HIDDEN), size=(HIDDEN,))
    b2 = 0.0
    moments = {"w1": np.zeros_like(w1), "b1": np.zeros_like(b1), "w2": np.zeros_like(w2), "b2": 0.0}
    velocities = {"w1": np.zeros_like(w1), "b1": np.zeros_like(b1), "w2": np.zeros_like(w2), "b2": 0.0}
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    best_dev = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    beta = 0.15
    lr = 0.01
    weight_decay = 1e-4
    for epoch in range(1, 501):
        z1 = train_x @ w1.T + b1
        hidden = np.maximum(z1, 0.0)
        gate = _numpy_sigmoid(hidden @ w2 + b2)
        log_prediction = (1.0 - gate) * visual_train + gate * f1_train
        difference = log_prediction - y_train
        abs_difference = np.abs(difference)
        loss = np.where(abs_difference <= beta, 0.5 * difference * difference / beta, abs_difference - 0.5 * beta).mean()
        d_prediction = np.where(abs_difference <= beta, difference / beta, np.sign(difference)) / max(len(train_records), 1)
        d_gate = d_prediction * (f1_train - visual_train)
        d_logit = d_gate * gate * (1.0 - gate)
        grad_w2 = hidden.T @ d_logit + weight_decay * w2
        grad_b2 = float(d_logit.sum())
        d_hidden = d_logit[:, None] * w2[None, :]
        d_z1 = d_hidden * (z1 > 0.0)
        grad_w1 = d_z1.T @ train_x + weight_decay * w1
        grad_b1 = d_z1.sum(axis=0)

        for name, gradient in (("w1", grad_w1), ("b1", grad_b1), ("w2", grad_w2), ("b2", grad_b2)):
            moments[name] = 0.9 * moments[name] + 0.1 * gradient
            velocities[name] = 0.999 * velocities[name] + 0.001 * (gradient * gradient)
        correction_1 = 1.0 - 0.9 ** epoch
        correction_2 = 1.0 - 0.999 ** epoch
        for name, value in (("w1", w1), ("b1", b1), ("w2", w2)):
            m_hat = moments[name] / correction_1
            v_hat = velocities[name] / correction_2
            value -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        m_hat = moments["b2"] / correction_1
        v_hat = velocities["b2"] / correction_2
        b2 -= lr * m_hat / (math.sqrt(max(v_hat, 0.0)) + 1e-8)

        dev_hidden = np.maximum(dev_x @ w1.T + b1, 0.0)
        dev_gate = _numpy_sigmoid(dev_hidden @ w2 + b2)
        dev_prediction = np.exp((1.0 - dev_gate) * dev_visual + dev_gate * dev_f1)
        dev_absrel = float(np.mean(np.abs(dev_prediction - np.exp(y_dev)) / np.maximum(np.exp(y_dev), 1e-8)))
        history.append({"epoch": epoch, "train_log_huber": float(loss), "dev_absrel": dev_absrel})
        if dev_absrel + 1e-7 < best_dev:
            best_dev = dev_absrel
            best_epoch = epoch
            best = (w1.copy(), b1.copy(), w2.copy(), float(b2))
            stale = 0
        else:
            stale += 1
        if stale >= 60:
            break
    if best is None:
        raise RuntimeError(f"No checkpoint selected for {feature_set}/{mixture}")
    w1, b1, w2, b2 = best
    spec = {
        "format": OBJECT_FEATURE_GATE_FORMAT,
        "feature_set": feature_set,
        "mixture": mixture,
        "groups": list(selected_groups(feature_set)),
        "feature_names": names,
        "input_dim": len(names),
        "hidden_dim": HIDDEN,
        "parameter_count": int(w1.size + b1.size + w2.size + 1),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weight_1": w1.tolist(),
        "bias_1": b1.tolist(),
        "weight_2": w2.reshape(1, -1).tolist(),
        "bias_2": [b2],
        "best_epoch": best_epoch,
        "train_rows": len(train_records),
        "dev_rows": len(dev_records),
        "output_contract": "g=sigmoid(MLP(x)); Z=exp((1-g)*log(Z_visual)+g*log(Z_F1))",
        "history_tail": history[-10:],
        "optimizer_backend": "numpy_adam_equivalent",
    }
    spec["train_absrel"] = gate_absrel(train_records, spec)
    spec["dev_absrel"] = gate_absrel(dev_records, spec)
    return spec, history


def train_gate(train_records: list[dict[str, Any]], dev_records: list[dict[str, Any]], feature_set: str, mixture: str, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if torch is None or nn is None:
        return train_gate_numpy(train_records, dev_records, feature_set, mixture, seed)
    train_records, dev_records = valid_records(train_records), valid_records(dev_records)
    if len(train_records) < 20 or len(dev_records) < 10:
        raise RuntimeError(f"Insufficient records for {feature_set}/{mixture}: train={len(train_records)} dev={len(dev_records)}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    names = feature_names(feature_set)
    raw_train = np.stack([feature_vector(row, feature_set) for row in train_records]).astype(np.float64)
    raw_dev = np.stack([feature_vector(row, feature_set) for row in dev_records]).astype(np.float64)
    train_x, mean, scale = standardize(raw_train)
    dev_x = (raw_dev - mean) / scale
    y_train = np.asarray([math.log(safe(row["gt_m"])) for row in train_records], dtype=np.float32)
    y_dev = np.asarray([math.log(safe(row["gt_m"])) for row in dev_records], dtype=np.float32)
    visual_train = np.asarray([math.log(safe(row["V1_Baseline-2P"])) for row in train_records], dtype=np.float32)
    f1_train = np.asarray([math.log(safe(row["z_size"])) for row in train_records], dtype=np.float32)
    model = TinyGate(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.15)
    tx, ty = torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(y_train)
    tv, tf1 = torch.from_numpy(visual_train), torch.from_numpy(f1_train)
    dev_tensor = torch.from_numpy(dev_x.astype(np.float32))
    dev_visual = torch.from_numpy(np.asarray([math.log(safe(row["V1_Baseline-2P"])) for row in dev_records], dtype=np.float32))
    dev_f1 = torch.from_numpy(np.asarray([math.log(safe(row["z_size"])) for row in dev_records], dtype=np.float32))
    best_state: dict[str, Any] | None = None
    best_dev = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        gate_value = model(tx)
        prediction = (1.0 - gate_value) * tv + gate_value * tf1
        loss = loss_fn(prediction, ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_gate = model(dev_tensor)
            dev_prediction = torch.exp((1.0 - dev_gate) * dev_visual + dev_gate * dev_f1).numpy()
        dev_gt = np.exp(y_dev)
        dev_absrel = float(np.mean(np.abs(dev_prediction - dev_gt) / np.maximum(dev_gt, 1e-8)))
        history.append({"epoch": epoch, "train_log_huber": float(loss.detach().cpu()), "dev_absrel": dev_absrel})
        if dev_absrel + 1e-7 < best_dev:
            best_dev, best_epoch, stale = dev_absrel, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 60:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {feature_set}/{mixture}")
    model.load_state_dict(best_state)
    state = model.state_dict()
    spec = {
        "format": OBJECT_FEATURE_GATE_FORMAT,
        "feature_set": feature_set,
        "mixture": mixture,
        "groups": list(selected_groups(feature_set)),
        "feature_names": names,
        "input_dim": len(names),
        "hidden_dim": HIDDEN,
        "parameter_count": int(sum(int(value.numel()) for value in state.values())),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weight_1": state["net.0.weight"].detach().cpu().numpy().tolist(),
        "bias_1": state["net.0.bias"].detach().cpu().numpy().tolist(),
        "weight_2": state["net.2.weight"].detach().cpu().numpy().tolist(),
        "bias_2": state["net.2.bias"].detach().cpu().numpy().tolist(),
        "best_epoch": best_epoch,
        "train_rows": len(train_records),
        "dev_rows": len(dev_records),
        "output_contract": "g=sigmoid(MLP(x)); Z=exp((1-g)*log(Z_visual)+g*log(Z_F1))",
        "history_tail": history[-10:],
    }
    spec["train_absrel"] = gate_absrel(train_records, spec)
    spec["dev_absrel"] = gate_absrel(dev_records, spec)
    return spec, history


def gate_prediction(record: Mapping[str, Any], spec: Mapping[str, Any]) -> tuple[float, float, str]:
    visual = safe(record.get("V1_Baseline-2P"))
    f1 = safe(record.get("z_size"))
    if not finite_positive(visual):
        return math.nan, 0.0, "rejected_visual"
    if not finite_positive(f1):
        return visual, 0.0, "visual_fallback_invalid_anchor"
    feature_set = str(spec["feature_set"])
    x = feature_vector(record, feature_set)
    mean = np.asarray(spec["mean"], dtype=np.float64)
    scale = np.asarray(spec["scale"], dtype=np.float64)
    w1 = np.asarray(spec["weight_1"], dtype=np.float64)
    b1 = np.asarray(spec["bias_1"], dtype=np.float64)
    w2 = np.asarray(spec["weight_2"], dtype=np.float64).reshape(-1)
    b2 = float(np.asarray(spec["bias_2"], dtype=np.float64).reshape(-1)[0])
    hidden = np.maximum(0.0, w1 @ ((x - mean) / scale) + b1)
    logit = float(w2 @ hidden + b2)
    gate_value = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
    value = math.exp((1.0 - gate_value) * math.log(visual) + gate_value * math.log(f1))
    return value, gate_value, "object_feature_neural_gate"


def gate_absrel(records: list[Mapping[str, Any]], spec: Mapping[str, Any]) -> float:
    values = []
    for row in records:
        pred, _, _ = gate_prediction(row, spec)
        gt = safe(row.get("gt_m"))
        if np.isfinite([pred, gt]).all() and pred > 0 and gt > 0:
            values.append(abs(pred - gt) / gt)
    return float(np.mean(values)) if values else math.nan


def candidate_prediction(
    row: Mapping[str, Any],
    candidate: str,
    fusion: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    legacy_models: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[float, float, str]:
    if candidate in models:
        return gate_prediction(row, models[candidate])
    if legacy_models and candidate in legacy_models:
        legacy = legacy_models[candidate]
        return prior_gate.neural_prediction(row, legacy, bool(legacy.get("with_context")))
    return v6.prediction(row, candidate, fusion)


def metric_rows(
    records: list[dict[str, Any]],
    candidates: list[str],
    fusion: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    split_name: str,
    legacy_models: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    band_defs = [("ALL", 0.0, math.inf), ("eyeai_0_5_5m", 0.5, 5.0), ("eyeai_0_5_10m", 0.5, 10.0)] + list(v6.BANDS)
    for dataset in sorted({str(row.get("dataset")) for row in records}):
        dataset_rows = [row for row in records if str(row.get("dataset")) == dataset]
        for track in sorted({str(row.get("track")) for row in dataset_rows}):
            track_rows = [row for row in dataset_rows if str(row.get("track")) == track]
            targets = np.asarray([safe(row.get("gt_m")) for row in track_rows], dtype=np.float64)
            target_valid = np.isfinite(targets) & (targets > 0)
            for candidate in candidates:
                # Compute each object once and reuse it for all distance bands.
                predictions = np.full(len(track_rows), math.nan, dtype=np.float64)
                weights = np.full(len(track_rows), math.nan, dtype=np.float64)
                for index, row in enumerate(track_rows):
                    predictions[index], weights[index], _ = candidate_prediction(row, candidate, fusion, models, legacy_models)
                prediction_valid = target_valid & np.isfinite(predictions) & (predictions > 0)
                learned = candidate in models or bool(legacy_models and candidate in legacy_models)
                spec = models.get(candidate) if candidate in models else (legacy_models or {}).get(candidate, {})
                for scope, low, high in band_defs:
                    mask = prediction_valid & (targets >= low) & (targets < high)
                    summary = v6.metric_summary(predictions[mask], targets[mask])
                    band_weights = weights[mask]
                    rows.append({
                        "split": split_name,
                        "dataset": dataset,
                        "track": track,
                        "candidate": candidate,
                        "feature_set": spec.get("feature_set", "LEGACY_CONTEXT" if candidate in (legacy_models or {}) else candidate),
                        "mixture": spec.get("mixture", "frozen"),
                        "candidate_type": "object_feature_gate" if candidate in models else ("legacy_neural_gate" if candidate in (legacy_models or {}) else "frozen_reference"),
                        "scope": scope,
                        "gate_weight_mean": float(np.nanmean(band_weights)) if learned and band_weights.size else math.nan,
                        "gate_weight_median": float(np.nanmedian(band_weights)) if learned and band_weights.size else math.nan,
                        **summary,
                    })
    return rows


def external_split(records: list[dict[str, Any]], dataset: str) -> dict[str, list[dict[str, Any]]]:
    frame_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = str(row.get("frame_id") or row.get("sample_id") or row.get("group_id"))
        frame_groups[key].append(row)
    keys = sorted(frame_groups, key=lambda value: int(hashlib.sha256(f"{SEED}|{dataset}|split|{value}".encode()).hexdigest()[:16], 16))
    n = len(keys)
    train_count = int(round(n * 0.60))
    dev_count = int(round(n * 0.20))
    return {
        "train": [row for key in keys[:train_count] for row in frame_groups[key]],
        "dev": [row for key in keys[train_count:train_count + dev_count] for row in frame_groups[key]],
        "test": [row for key in keys[train_count + dev_count:] for row in frame_groups[key]],
        "groups": {"train": keys[:train_count], "dev": keys[train_count:train_count + dev_count], "test": keys[train_count + dev_count:]},
    }


def split_summary(records: list[Mapping[str, Any]], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "rows": len(records),
        "frames": len({str(row.get("frame_id") or row.get("sample_id")) for row in records}),
        "seg_object_available": int(sum(safe(row.get("seg_obj_available"), 0.0) > 0 for row in records)),
        "det_context_available": int(sum(safe(row.get("ctx_object_count"), 0.0) > 0 for row in records)),
    }


def build_mixture_records(train_v3: list[dict[str, Any]], dev_v3: list[dict[str, Any]], splits: Mapping[str, Mapping[str, list[dict[str, Any]]]], mixture: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for dataset in prior_gate.MIXTURES[mixture]:
        if dataset == "V3":
            train.extend(train_v3)
            dev.extend(dev_v3)
        else:
            track = "COCO-A_OFFICIAL_BBOX" if dataset == "COCO" else "WAYMO-A_OFFICIAL_BBOX"
            train.extend(row for row in splits[dataset]["train"] if row.get("track") == track)
            dev.extend(row for row in splits[dataset]["dev"] if row.get("track") == track)
    return train, dev


def group_bootstrap(
    records: list[dict[str, Any]],
    candidates: list[str],
    fusion: Mapping[str, Any],
    models: Mapping[str, Mapping[str, Any]],
    dataset: str,
    track: str,
    seed: int,
    replicates: int,
    legacy_models: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    subset = [row for row in records if str(row.get("dataset")) == dataset and str(row.get("track")) == track]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        groups[str(row.get("frame_id") or row.get("sample_id") or row.get("group_id"))].append(row)
    keys = sorted(groups)
    if not keys:
        return []
    # Reuse the same frame samples for every candidate.
    candidate_sums: dict[str, np.ndarray] = {}
    baseline_sums: dict[str, np.ndarray] = {}
    candidate_counts: dict[str, np.ndarray] = {}
    for candidate in candidates:
        sums = np.zeros(len(keys), dtype=np.float64)
        baseline = np.zeros(len(keys), dtype=np.float64)
        counts = np.zeros(len(keys), dtype=np.float64)
        for index, key in enumerate(keys):
            rows = groups[key]
            current: list[float] = []
            reference: list[float] = []
            for row in rows:
                pred, _, _ = candidate_prediction(row, candidate, fusion, models, legacy_models)
                base, _, _ = candidate_prediction(row, "V1_Baseline-2P", fusion, models, legacy_models)
                gt = safe(row.get("gt_m"))
                if np.isfinite([pred, base, gt]).all() and min(pred, base, gt) > 0:
                    current.append(abs(pred - gt) / gt)
                    reference.append(abs(base - gt) / gt)
            if current:
                sums[index] = float(np.sum(current))
                baseline[index] = float(np.sum(reference))
                counts[index] = float(len(current))
        candidate_sums[candidate] = sums
        baseline_sums[candidate] = baseline
        candidate_counts[candidate] = counts
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(keys), size=(max(replicates, 0), len(keys)))
    output = []
    for candidate in candidates:
        counts = candidate_counts[candidate][selected].sum(axis=1) if replicates > 0 else np.asarray([], dtype=np.float64)
        current = candidate_sums[candidate][selected].sum(axis=1) if replicates > 0 else np.asarray([], dtype=np.float64)
        baseline = baseline_sums[candidate][selected].sum(axis=1) if replicates > 0 else np.asarray([], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = (current - baseline) / counts
        values = values[np.isfinite(values)]
        spec = models.get(candidate) if candidate in models else (legacy_models or {}).get(candidate, {})
        output.append({
            "dataset": dataset,
            "track": track,
            "candidate": candidate,
            "feature_set": spec.get("feature_set", candidate),
            "mixture": spec.get("mixture", "frozen"),
            "replicates": len(values),
            "delta_absrel_vs_baseline": float(np.mean(values)) if values.size else math.nan,
            "ci95_low": float(np.quantile(values, 0.025)) if values.size else math.nan,
            "ci95_high": float(np.quantile(values, 0.975)) if values.size else math.nan,
            "probability_improvement": float(np.mean(values < 0.0)) if values.size else math.nan,
            "group_count": len(keys),
            "object_count": len(subset),
        })
    return output


def aggregate_feature_scores(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create descriptive cross-dataset leaderboards without changing selection."""
    gate_rows = [row for row in metrics if row["candidate"].startswith("E_ObjectGate_") and row["scope"] in {"ALL", "eyeai_0_5_5m"} and row["track"] in {"COCO-A_OFFICIAL_BBOX", "WAYMO-A_OFFICIAL_BBOX"}]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        grouped[(str(row["candidate"]), str(row["scope"]))].append(row)
    output = []
    for (candidate, scope), values in grouped.items():
        valid = [row for row in values if safe(row.get("n"), 0.0) > 0 and np.isfinite(safe(row.get("absrel")))]
        gate_values = [safe(row.get("gate_weight_mean")) for row in valid if np.isfinite(safe(row.get("gate_weight_mean")))]
        count = sum(int(row["n"]) for row in valid)
        output.append({
            "candidate": candidate,
            "feature_set": values[0]["feature_set"],
            "mixture": values[0]["mixture"],
            "scope": scope,
            "datasets": "+".join(sorted(str(row["dataset"]) for row in valid)),
            "n": count,
            "weighted_absrel": float(sum(float(row["n"]) * float(row["absrel"]) for row in valid) / count) if count else math.nan,
            "mean_dataset_absrel": float(np.mean([float(row["absrel"]) for row in valid])) if valid else math.nan,
            "mean_gate_weight": float(np.mean(gate_values)) if gate_values else math.nan,
        })
    return output


def make_plot(metrics: list[dict[str, Any]], leaderboard: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [row for row in leaderboard if row["scope"] == "ALL"]
    rows = sorted(rows, key=lambda row: safe(row.get("weighted_absrel"), math.inf))[:20]
    if not rows:
        return
    labels = [f"{row['mixture']}\n{row['feature_set']}" for row in rows]
    values = [float(row["weighted_absrel"]) for row in rows]
    fig, axis = plt.subplots(figsize=(15, 6))
    axis.bar(np.arange(len(values)), values, color="#4c78a8")
    axis.set_xticks(np.arange(len(values)), labels, rotation=65, ha="right", fontsize=7)
    axis.set_ylabel("weighted official-track AbsRel")
    axis.set_title("Object-feature gate ablation: best held-out combinations")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "v6_object_feature_gate_ablation.png", dpi=160)
    plt.close(fig)


def render_report(runtime: Mapping[str, Any], train_dev: list[dict[str, Any]], metrics: list[dict[str, Any]], leaderboard: list[dict[str, Any]], bootstrap: list[dict[str, Any]]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        number = safe(value)
        return f"{number:.{digits}f}" if np.isfinite(number) else "-"

    report = [
        "# V6 Object-Feature Neural Gate Ablation",
        "",
        "This is a separate research-only ablation. MiDaS, all frozen visual experts, F1 and the existing B3 hybrid were not changed.",
        "",
        "## Gate contract",
        "",
        "Every learned candidate predicts only `g = sigmoid(MLP(features))`. The object depth remains the bounded log interpolation `Z = exp((1-g) log(Z_visual) + g log(Z_F1))`; it cannot leave the interval between the Baseline-2P visual depth and the deterministic F1 size anchor.",
        "",
        f"All models use a {HIDDEN}-unit ReLU hidden layer, Adam, the existing SmoothL1 log-depth objective, identical seed `{SEED}`, and the existing 60/20/20 frame-disjoint external split. The enumerated search contains `{len(feature_set_names()) - 1}` object-feature configurations plus the exact legacy V6 base gate.",
        "",
        "## Feature families",
        "",
        "`DEPTH_ONLY` is the mandatory inference-time candidate state: visual depth, F1 depth, absolute disagreement and signed disagreement. The optional families are:",
        "",
    ]
    for group in FEATURE_GROUPS:
        report.append(f"- `{group}`: {GROUP_DESCRIPTIONS[group]} ({len(GROUP_FEATURE_NAMES[group])} scalars).")
    report += [
        "",
        "`SEGMENTATION` uses the existing global semantic area vector and a deterministic bbox projection of the existing 4x4 semantic area grid. It is not a new segmentation network. Missing segmentation is represented by availability/zero features.",
        "",
        "## Training overview",
        "",
        "| mixture | feature set | train objects | dev objects | train AbsRel | dev AbsRel | input scalars | parameters | best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(train_dev, key=lambda item: (str(item["mixture"]), float(item["dev_absrel"]))):
        report.append(f"| {row['mixture']} | {row['feature_set']} | {row['train_rows']} | {row['dev_rows']} | {row['train_absrel']:.4f} | {row['dev_absrel']:.4f} | {row['input_dim']} | {row['parameter_count']} | {row['best_epoch']} |")
    report += [
        "",
        "## Held-out official-track leaderboard",
        "",
        "The leaderboard is descriptive freeze evaluation. COCO uses the existing P0 teacher pseudo-reference; Waymo uses the existing Gold-A object reference. Lower AbsRel is better.",
        "",
        "| mixture | feature set | scope | n | weighted AbsRel | mean dataset AbsRel | mean gate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted([item for item in leaderboard if item["scope"] in {"ALL", "eyeai_0_5_5m"}], key=lambda item: (item["scope"], safe(item.get("weighted_absrel"), math.inf)))[:40]:
        report.append(f"| {row['mixture']} | {row['feature_set']} | {row['scope']} | {row['n']} | {row['weighted_absrel']:.4f} | {row['mean_dataset_absrel']:.4f} | {row['mean_gate_weight']:.3f} |")
    report += [
        "",
        "## Full distance-band metric audit",
        "",
        "The table below shows the baseline, the existing deterministic B3 reference, F1, the best frozen prior hybrid (descriptive holdout minimum) and the best new object-feature gate (descriptive holdout minimum). The selection is not used for training; the complete, non-selected matrix is in `reports/v6_object_feature_gate_metrics.csv` and includes AbsRel, MAE and RMSE for every candidate.",
        "",
        "| dataset | scope | candidate | feature/mixture | n | AbsRel | MAE [m] | RMSE [m] |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    scope_order = [band for band, _, _ in v6.BANDS] + ["eyeai_0_5_5m", "ALL"]
    for dataset, track in (("COCO", "COCO-A_OFFICIAL_BBOX"), ("WAYMO", "WAYMO-A_OFFICIAL_BBOX")):
        for scope in scope_order:
            current = [row for row in metrics if row.get("dataset") == dataset and row.get("track") == track and row.get("scope") == scope]
            valid = [row for row in current if safe(row.get("n"), 0.0) > 0 and np.isfinite(safe(row.get("absrel")))]
            if not valid:
                continue
            by_name = {str(row.get("candidate")): row for row in valid}
            selected: list[dict[str, Any]] = []
            for name in ("V1_Baseline-2P", "B3_Baseline_DisagreementFallback", "F1_SIZE_ANCHOR_OVERRIDE"):
                if name in by_name:
                    selected.append(by_name[name])
            prior = [row for row in valid if str(row.get("candidate", "")).startswith(("A", "B", "C_", "D_")) and str(row.get("candidate")) not in {"A0_Baseline_Visual", "A1_SizeOnly"}]
            gates = [row for row in valid if str(row.get("candidate", "")).startswith("E_ObjectGate_")]
            if prior:
                selected.append(min(prior, key=lambda row: safe(row.get("absrel"), math.inf)))
            if gates:
                selected.append(min(gates, key=lambda row: safe(row.get("absrel"), math.inf)))
            seen: set[str] = set()
            for row in selected:
                candidate = str(row.get("candidate"))
                if candidate in seen:
                    continue
                seen.add(candidate)
                label = candidate
                if candidate.startswith("E_ObjectGate_"):
                    label = f"{row.get('feature_set')} / {row.get('mixture')}"
                report.append(f"| {dataset} | {scope} | {candidate} | {label} | {row.get('n', 0)} | {fmt(row.get('absrel'))} | {fmt(row.get('mae_m'))} | {fmt(row.get('rmse_m'))} |")
    report += [
        "",
        "## Baseline / hybrid / F1 reference",
        "",
        "All fixed references use exactly the same frame-disjoint external holdout rows as the learned gates. The reference set includes every V6 A/B/C/D fusion, the previous Base/Context neural gates when available, all frozen visual experts, the current Baseline-2P, and the pure F1 size anchor.",
        "",
        "## Paired bootstrap",
        "",
        "The paired bootstrap compares every selected feature configuration and reference with Baseline-2P on COCO/Waymo official and Product-YOLO matched tracks while resampling complete frames. Negative delta means improvement. Exact intervals are in `reports/v6_object_feature_gate_bootstrap.csv`.",
        "",
        "| dataset | track | candidate | feature set / mixture | delta AbsRel | 95% CI | P(improvement) |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in sorted(bootstrap, key=lambda item: (str(item["dataset"]), str(item.get("track")), safe(item.get("delta_absrel_vs_baseline"), math.inf)))[:60]:
        report.append(f"| {row['dataset']} | {row.get('track', '-')} | {row['candidate']} | {row['feature_set']} / {row['mixture']} | {fmt(row.get('delta_absrel_vs_baseline'))} | [{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}] | {fmt(row.get('probability_improvement'), 3)} |")
    report += [
        "",
        "## Interpretation limits",
        "",
        "- The 0.5-5 m Waymo support is sparse in the locked holdout and must not be overinterpreted; the row count is always reported.",
        "- A gate can only interpolate between the two frozen candidates. It cannot fix an error when both visual and F1 estimates are wrong in the same direction.",
        "- Object-local segmentation is an approximate projection of the existing 4x4 semantic summary, not a per-object mask. It is therefore a deployability-compatible diagnostic, not evidence of pixel-accurate instance segmentation.",
        "- The legacy base gate is retained as a control. The new feature families add parameters only in the gate; MiDaS and the depth heads are untouched.",
        "",
        "## Runtime/export implications",
        "",
        "The gate is a Linear-ReLU-Linear-Sigmoid graph with fixed-size tensors and no dynamic operators. It is suitable for ONNX/TFLite/LiteRT conversion after the feature-producing detector/segmentation contract is fixed. The segmentation family requires the existing semantic summary; without it, the availability features fall back deterministically to zero.",
        "",
        f"Runtime metadata: `{json.dumps(dict(runtime), ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return "\n".join(report)


def _numpy_sigmoid(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=np.float64), -60.0, 60.0)))


def _numpy_baseline_depth_frame(drel: np.ndarray, raw: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    """Decode the frozen V3 two-parameter head without importing Torch."""
    raw_values = np.asarray(raw, dtype=np.float64).reshape(-1)
    m = float(spec["m_min"]) + (float(spec["m_max"]) - float(spec["m_min"])) * _numpy_sigmoid(raw_values[0])
    delta = float(spec["delta_min"]) + (float(spec["delta_max"]) - float(spec["delta_min"])) * _numpy_sigmoid(raw_values[1])
    relative = np.clip(np.asarray(drel, dtype=np.float64), float(spec["r_low"]), float(spec["r_high"]))
    u0 = np.exp(m - delta / 2.0)
    u1 = np.exp(m + delta / 2.0)
    t = (relative - float(spec["r_low"])) / max(float(spec["r_high"]) - float(spec["r_low"]), 1e-8)
    u = (1.0 - t) * u0 + t * u1
    return (1.0 / np.maximum(u, float(spec["eps"]))).astype(np.float32)


def _numpy_spline_depth_frame(drel: np.ndarray, raw: np.ndarray, spec: Mapping[str, Any], knot_count: int = 8) -> np.ndarray:
    """NumPy/export equivalent of the V3 monotone piecewise-linear decoder."""
    raw_values = np.asarray(raw, dtype=np.float64).reshape(-1)[:knot_count]
    sigmoid = _numpy_sigmoid(raw_values[:2])
    start = float(spec["m_min"]) + (float(spec["m_max"]) - float(spec["m_min"])) * sigmoid[0]
    span = float(spec["delta_min"]) + (float(spec["delta_max"]) - float(spec["delta_min"])) * sigmoid[1]
    logits = np.concatenate((raw_values[2:knot_count], np.zeros(1, dtype=np.float64)))
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= max(float(weights.sum()), 1e-12)
    cumulative = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(weights * span)))
    if int(spec["direction"]) < 0:
        cumulative = -cumulative
    log_q = start + cumulative
    q_knots = np.exp(np.clip(log_q, -20.0, 20.0))
    relative = np.asarray(drel, dtype=np.float64)
    x = np.clip((relative - float(spec["r_low"])) / max(float(spec["r_high"]) - float(spec["r_low"]), 1e-8), 0.0, 1.0)
    scaled = x * float(knot_count - 1)
    left = np.minimum(np.floor(scaled).astype(np.int64), knot_count - 2)
    alpha = scaled - left
    q = (1.0 - alpha) * q_knots[left] + alpha * q_knots[left + 1]
    if str(spec["semantics"]) == "DIRECT_DEPTH":
        depth = q
    else:
        depth = 1.0 / np.maximum(q, float(spec["eps"]))
    return depth.astype(np.float32)


def _numpy_decode_v3_map(drel: np.ndarray, raw: np.ndarray, key: str, spec: Mapping[str, Any]) -> np.ndarray:
    if key in {"V0_Z1-frozen", "V1_Baseline-2P"}:
        return _numpy_baseline_depth_frame(drel, raw, spec)
    return _numpy_spline_depth_frame(drel, raw, spec, 8)


def numpy_v3_development_objects() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build V3 train/dev object records from frozen arrays using NumPy.

    The normal path delegates to the original V3 loader. This fallback keeps
    the same rows, raw heads, calibration decoder and object-record contract
    when a machine-level Torch DLL policy blocks the original import.
    """
    spec = json.loads(v6.SPEC_PATH.read_text(encoding="utf-8"))
    priors = v6.parse_priors(v6.PRIOR_PATH)
    train_rows, dev_rows = v6.load_rows("train"), v6.load_rows("dev")
    detection_rows = v6.load_product_detections_for_dev()
    cache_index_payload = json.loads((v6.V3_CACHE_DIR / "index.json").read_text(encoding="utf-8"))
    sample_to_cache_index = {str(item["sample_id"]): int(item["index"]) for item in cache_index_payload["index"]}
    drel_cache = np.load(v6.V3_CACHE_DIR / "d_rel_full_f16.npy", mmap_mode="r")
    depth_cache = np.load(v6.V3_CACHE_DIR / "depth_f16.npy", mmap_mode="r")
    raw_names = {
        "V0_Z1-frozen": "V0_Z1_frozen",
        "V1_Baseline-2P": "V1_Baseline_2P",
        "V2_Spline-8-seed123": "V2_Spline_8_seed123",
        "V3_Spline-8-seed42": "V3_Spline_8_seed42",
        "V4_Spline-8-seed7": "V4_Spline_8_seed7",
    }
    raw_by_split = {
        split: {key: np.load(V6 / "data" / "raw" / f"diode_{split}_{suffix}.npy", mmap_mode="r") for key, suffix in raw_names.items()}
        for split in ("train", "dev")
    }
    outputs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        raw = raw_by_split[split]
        for row_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            det_row = detection_rows.get(sample_id)
            cache_index = sample_to_cache_index.get(sample_id)
            if cache_index is None or not det_row or det_row.get("status") != "AVAILABLE" or not det_row.get("detections"):
                continue
            drel_map = np.asarray(drel_cache[cache_index], dtype=np.float32)
            gt_map = np.asarray(depth_cache[cache_index], dtype=np.float32)
            maps = {key: _numpy_decode_v3_map(drel_map, raw[key][row_index], key, spec) for key in v6.VISUAL_KEYS[:5]}
            spline_maps = np.stack([maps[key] for key in v6.INDIVIDUAL_SPLINES], axis=0)
            maps["V5_Spline-8-Median"] = np.median(spline_maps, axis=0).astype(np.float32)
            maps["V6_Spline-8-LogMean"] = np.exp(np.mean(np.log(np.maximum(spline_maps, 1e-6)), axis=0)).astype(np.float32)
            fx_px, fy_px, _ = v6.normalized_intrinsics(row)
            width = safe(row.get("image_width"), 256.0)
            height = safe(row.get("image_height"), 256.0)
            for object_index, item in enumerate(det_row.get("detections", [])):
                bbox = v6.bbox_values(item, width, height)
                y0, y1, x0, x1 = v6.crop_slice(bbox, drel_map.shape[1], drel_map.shape[0])
                gt_values = gt_map[y0:y1, x0:x1]
                gt_good = np.isfinite(gt_values) & (gt_values > 0)
                if not gt_good.any():
                    continue
                gt_m = float(np.median(gt_values[gt_good]))
                pred_values: dict[str, float] = {}
                for key in v6.VISUAL_KEYS:
                    predicted = np.asarray(maps[key][y0:y1, x0:x1], dtype=np.float64)
                    good = np.isfinite(predicted) & (predicted > 0)
                    pred_values[key] = float(np.median(predicted[good])) if good.any() else math.nan
                pred_values = v6.visual_from_predictions(pred_values)
                class_name = str(item.get("class_name", "unknown"))
                anchor = v6.anchor_values(class_name, bbox, width, height, fx_px, fy_px, priors)
                record = {
                    "dataset": str(row.get("dataset", "unknown")).upper(),
                    "split": split,
                    "track": "PRODUCT_YOLO_DEV",
                    "quality": "GOLD_GT_DEV",
                    "sample_id": sample_id,
                    "frame_id": sample_id,
                    "group_id": str(row.get("group_id") or sample_id),
                    "object_index": object_index,
                    "object_key": f"{sample_id}:{object_index}",
                    "class_name": class_name,
                    "gt_m": gt_m,
                    "band": v6.band_name(gt_m),
                    "bbox_area": float(bbox[2] * bbox[3]),
                    "bbox_width": float(bbox[2]),
                    "bbox_height": float(bbox[3]),
                    "bbox_aspect": float(bbox[2] / max(bbox[3], 1e-6)),
                    "detection_confidence": safe(item.get("confidence"), math.nan),
                    "matched_official": 1,
                    "match_iou": math.nan,
                    "fx_px": fx_px,
                    "fy_px": fy_px,
                    **anchor,
                    **pred_values,
                }
                record.update(v6.seed_stats(pred_values))
                outputs[split].append(record)
        print(f"V6 NumPy development objects {split}: {len(outputs[split])}", flush=True)
    return outputs["train"], outputs["dev"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exhaustive object-feature neural gate ablations.")
    parser.add_argument("--mixtures", nargs="+", choices=list(prior_gate.MIXTURES), default=list(prior_gate.MIXTURES))
    parser.add_argument("--feature-sets", nargs="+", choices=feature_set_names(), default=feature_set_names())
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    # Prefer Torch and use the NumPy fallback when Torch is unavailable.
    # Reuse the existing Waymo semantic summary.
    waymo_seg_rows, waymo_seg_meta = prior_gate.ensure_waymo_segmentation()
    sources = rich_context_sources()
    try:
        train_v3, dev_v3 = prior_gate.v3_development_objects()
    except Exception as exc:
        print(f"V3 Torch loader unavailable; using frozen NumPy decoder ({type(exc).__name__}: {exc})", flush=True)
        train_v3, dev_v3 = numpy_v3_development_objects()
    train_v3 = attach_rich_context(train_v3, sources)
    dev_v3 = attach_rich_context(dev_v3, sources)
    coco_all = attach_rich_context(prior_gate.read_csv(REPORTS / "v6_coco_panel.csv"), sources)
    waymo_all = attach_rich_context(prior_gate.read_csv(REPORTS / "v6_waymo_panel.csv"), sources)
    coco_split = external_split(coco_all, "COCO")
    waymo_split = external_split(waymo_all, "WAYMO")
    splits = {"COCO": coco_split, "WAYMO": waymo_split}
    fusion = json.loads((REPORTS / "v6_fusion_parameters.json").read_text(encoding="utf-8"))
    legacy_path = REPORTS / "v6_neural_gate_parameters.json"
    legacy_models: dict[str, dict[str, Any]] = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else {}
    selected_feature_sets = list(args.feature_sets)
    selected_mixtures = list(args.mixtures)
    models: dict[str, dict[str, Any]] = {}
    train_dev: list[dict[str, Any]] = []
    for mixture in selected_mixtures:
        train_records, dev_records = build_mixture_records(train_v3, dev_v3, splits, mixture)
        for feature_set in selected_feature_sets:
            candidate = f"E_ObjectGate_{feature_set}_{mixture}"
            spec, _ = train_gate(train_records, dev_records, feature_set, mixture, SEED)
            models[candidate] = spec
            train_dev.append({
                "model": candidate,
                "feature_set": feature_set,
                "mixture": mixture,
                "train_rows": len(valid_records(train_records)),
                "dev_rows": len(valid_records(dev_records)),
                "train_absrel": spec["train_absrel"],
                "dev_absrel": spec["dev_absrel"],
                "input_dim": spec["input_dim"],
                "parameter_count": spec["parameter_count"],
                "best_epoch": spec["best_epoch"],
            })
            print(f"trained {candidate}: dev_absrel={spec['dev_absrel']:.5f}", flush=True)
    write_json(REPORTS / "v6_object_feature_gate_parameters.json", models)
    write_csv(REPORTS / "v6_object_feature_gate_train_dev.csv", train_dev)
    test_records = coco_split["test"] + waymo_split["test"]
    # Compare the new gate with the fixed visual, F1 and hybrid references.
    fixed = list(v6.VISUAL_KEYS) + ["F1_SIZE_ANCHOR_OVERRIDE"] + sorted(fusion) + sorted(legacy_models)
    fixed = list(dict.fromkeys(fixed))
    all_candidates = fixed + sorted(models)
    metric = metric_rows(test_records, all_candidates, fusion, models, "external_group_holdout", legacy_models)
    write_csv(REPORTS / "v6_object_feature_gate_metrics.csv", metric)
    leaderboard = aggregate_feature_scores(metric)
    write_csv(REPORTS / "v6_object_feature_gate_leaderboard.csv", leaderboard)
    bootstrap: list[dict[str, Any]] = []
    bootstrap_candidates = [candidate for candidate in all_candidates if candidate != "V1_Baseline-2P"]
    if not args.skip_bootstrap and args.bootstrap_replicates > 0 and bootstrap_candidates:
        for dataset, track in (
            ("COCO", "COCO-A_OFFICIAL_BBOX"),
            ("COCO", "COCO-B_PRODUCT_YOLO_MATCHED"),
            ("WAYMO", "WAYMO-A_OFFICIAL_BBOX"),
            ("WAYMO", "WAYMO-B_PRODUCT_YOLO_MATCHED"),
        ):
            bootstrap.extend(group_bootstrap(test_records, bootstrap_candidates, fusion, models, dataset, track, SEED + len(bootstrap), args.bootstrap_replicates, legacy_models))
    write_csv(REPORTS / "v6_object_feature_gate_bootstrap.csv", bootstrap)
    make_plot(metric, leaderboard)
    runtime = {
        "format": OBJECT_FEATURE_GATE_FORMAT,
        "seconds": time.perf_counter() - started,
        "seed": SEED,
        "hidden_dim": HIDDEN,
        "feature_sets": selected_feature_sets,
        "feature_set_count": len(selected_feature_sets),
        "mixtures": selected_mixtures,
        "model_count": len(models),
        "reference_candidate_count": len(fixed),
        "legacy_reference_models": sorted(legacy_models),
        "bootstrap_replicates": 0 if args.skip_bootstrap else args.bootstrap_replicates,
        "holdout_rows": len(test_records),
        "context_sources": {dataset: {"detections": len(source.get("detections", {})), "segmentations": len(source.get("segmentations", {})), "bbox_hints": len(source.get("bbox_hints", {}))} for dataset, source in sources.items()},
        "v3_train": split_summary(train_v3, "V3_train"),
        "v3_dev": split_summary(dev_v3, "V3_dev"),
        "coco_all": split_summary(coco_all, "COCO_all"),
        "waymo_all": split_summary(waymo_all, "WAYMO_all"),
        "waymo_segmentation": waymo_seg_meta,
        "midas_changed": False,
        "eyeai_changed": False,
        "new_depth_architecture": False,
    }
    write_json(REPORTS / "v6_object_feature_gate_runtime.json", runtime)
    write_json(REPORTS / "v6_object_feature_gate_protocol.json", {
        "format": "rel2abs_v6_object_feature_gate_protocol_v2",
        "seed": SEED,
        "mixture_definitions": prior_gate.MIXTURES,
        "feature_groups": {group: {"description": GROUP_DESCRIPTIONS[group], "features": list(GROUP_FEATURE_NAMES[group])} for group in FEATURE_GROUPS},
        "depth_features": list(DEPTH_FEATURE_NAMES),
        "feature_sets": selected_feature_sets,
        "reference_candidates": fixed,
        "split_policy": "same V6 frame-disjoint 60/20/20 external splits; V3 train/dev retained",
        "labels": {"V3": "Gold metric depth", "WAYMO": "Gold-A LiDAR object depth", "COCO": "P0 teacher pseudo-reference"},
        "waymo_segmentation": waymo_seg_meta,
    })
    (V6 / "V6_OBJECT_FEATURE_GATE_ABLATION_REPORT.md").write_text(render_report(runtime, train_dev, metric, leaderboard, bootstrap), encoding="utf-8")
    print(json.dumps({"runtime_seconds": runtime["seconds"], "models": len(models), "metrics": len(metric), "leaderboard": str((REPORTS / "v6_object_feature_gate_leaderboard.csv").resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
