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
V2 = PROJECT / "rel2abs_v2_research"
V4 = PROJECT / "rel2abs_v4_research"
V5B = PROJECT / "rel2abs_v5b_research"
SRC = V6 / "src"
REPORTS = V6 / "reports"
PLOTS = V6 / "plots"
SEGMENTATION_CONTEXT = REPORTS / "v6_waymo_semantic_context.jsonl"
SPLIT_MANIFEST = REPORTS / "v6_neural_gate_cross_dataset_split.json"
SEED = 20260919
BOOTSTRAP_REPLICATES = 5000
HIDDEN = 16

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
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


SEMANTIC_LABELS = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)
CONTEXT_FEATURES = (
    "ctx_object_count_log1p",
    "ctx_supported_object_count_log1p",
    "ctx_mean_bbox_area",
    "ctx_max_bbox_area",
    "ctx_mean_bbox_height",
    "ctx_max_bbox_height",
    "ctx_mean_confidence",
    "ctx_class_diversity",
    "ctx_border_fraction",
    "ctx_seg_available",
    "ctx_seg_present_count",
    "ctx_seg_max_area",
    "ctx_seg_entropy",
    "ctx_seg_road_area",
    "ctx_seg_sidewalk_area",
    "ctx_seg_building_area",
    "ctx_seg_vegetation_area",
    "ctx_seg_sky_area",
    "ctx_seg_people_area",
    "ctx_seg_vehicle_area",
)
DATASET_ORDER = ("V3", "COCO", "WAYMO")
MIXTURES = {
    "V3_ONLY": ("V3",),
    "V3_COCO": ("V3", "COCO"),
    "V3_WAYMO": ("V3", "WAYMO"),
    "COCO_WAYMO": ("COCO", "WAYMO"),
    "V3_COCO_WAYMO": ("V3", "COCO", "WAYMO"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, allow_nan=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def stable_value(value: str, salt: str = "") -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()[:16], 16)


def safe(value: Any, default: float = math.nan) -> float:
    return v6.safe_float(value, default)


def aliases(row: Mapping[str, Any]) -> set[str]:
    values = set()
    for key in ("sample_id", "frame_id", "image_id", "group_id"):
        value = row.get(key)
        if value is not None and str(value) not in {"", "nan", "None"}:
            values.add(str(value))
    return values


def ordered_aliases(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("sample_id", "frame_id", "image_id", "group_id"):
        value = row.get(key)
        if value is not None and str(value) not in {"", "nan", "None"} and str(value) not in values:
            values.append(str(value))
    return values


def compact_segmentation(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"seg_available": 0, "global_area_fraction": [], "class_presence": []}
    area = row.get("global_area_fraction") or []
    presence = row.get("class_presence") or []
    return {
        "seg_available": int(bool(area)),
        "global_area_fraction": [float(x) for x in area],
        "class_presence": [int(x) for x in presence],
    }


def object_context(detection_row: Mapping[str, Any] | None, segmentation_row: Mapping[str, Any] | None) -> dict[str, Any]:
    detections = list((detection_row or {}).get("detections") or [])
    areas = np.asarray([safe(item.get("bbox_area"), safe(item.get("width"), 0.0) * safe(item.get("height"), 0.0)) for item in detections], dtype=np.float64)
    heights = np.asarray([safe(item.get("height"), 0.0) for item in detections], dtype=np.float64)
    widths = np.asarray([safe(item.get("width"), 0.0) for item in detections], dtype=np.float64)
    confidences = np.asarray([safe(item.get("confidence"), 0.0) for item in detections], dtype=np.float64)
    classes = [str(item.get("class_name", "unknown")) for item in detections]
    supported = sum(item in {"person", "bicycle", "car"} for item in classes)
    border_hits = 0
    for item in detections:
        cx = safe(item.get("x_center"), safe(item.get("cx"), 0.5))
        cy = safe(item.get("y_center"), safe(item.get("cy"), 0.5))
        w = safe(item.get("width"), safe(item.get("w"), 0.0))
        h = safe(item.get("height"), safe(item.get("h"), 0.0))
        if cx - w / 2.0 <= 0.01 or cx + w / 2.0 >= 0.99 or cy - h / 2.0 <= 0.01 or cy + h / 2.0 >= 0.99:
            border_hits += 1
    seg = compact_segmentation(segmentation_row)
    fraction = np.asarray(seg["global_area_fraction"], dtype=np.float64)
    if fraction.size:
        fraction = np.clip(fraction, 0.0, 1.0)
        entropy = float(-(fraction[fraction > 0] * np.log(np.maximum(fraction[fraction > 0], 1e-12))).sum() / math.log(max(len(SEMANTIC_LABELS), 2)))
        present = int(sum(bool(x) for x in seg["class_presence"]))
        max_area = float(fraction.max())
        def area_for(*names: str) -> float:
            return float(sum(fraction[SEMANTIC_LABELS.index(name)] for name in names if SEMANTIC_LABELS.index(name) < fraction.size))
        road = area_for("road")
        sidewalk = area_for("sidewalk")
        building = area_for("building")
        vegetation = area_for("vegetation")
        sky = area_for("sky")
        people = area_for("person", "rider")
        vehicles = area_for("car", "truck", "bus", "train", "motorcycle", "bicycle")
    else:
        entropy = 0.0
        present = 0
        max_area = road = sidewalk = building = vegetation = sky = people = vehicles = 0.0
    return {
        "ctx_object_count": len(detections),
        "ctx_supported_object_count": supported,
        "ctx_mean_bbox_area": float(np.mean(areas)) if areas.size else 0.0,
        "ctx_max_bbox_area": float(np.max(areas)) if areas.size else 0.0,
        "ctx_mean_bbox_height": float(np.mean(heights)) if heights.size else 0.0,
        "ctx_max_bbox_height": float(np.max(heights)) if heights.size else 0.0,
        "ctx_mean_confidence": float(np.mean(confidences)) if confidences.size else 0.0,
        "ctx_class_diversity": float(len(set(classes))),
        "ctx_border_fraction": float(border_hits / max(len(detections), 1)),
        "ctx_seg_available": int(seg["seg_available"]),
        "ctx_seg_present_count": present,
        "ctx_seg_max_area": max_area,
        "ctx_seg_entropy": entropy,
        "ctx_seg_road_area": road,
        "ctx_seg_sidewalk_area": sidewalk,
        "ctx_seg_building_area": building,
        "ctx_seg_vegetation_area": vegetation,
        "ctx_seg_sky_area": sky,
        "ctx_seg_people_area": people,
        "ctx_seg_vehicle_area": vehicles,
    }


def make_context_map(detection_rows: Iterable[Mapping[str, Any]], segmentation_rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    detections: dict[str, Mapping[str, Any]] = {}
    segmentations: dict[str, Mapping[str, Any]] = {}
    for row in detection_rows:
        for key in aliases(row):
            detections[key] = row
    for row in segmentation_rows:
        for key in aliases(row):
            segmentations[key] = row
    result: dict[str, dict[str, Any]] = {}
    for key in set(detections) | set(segmentations):
        result[key] = object_context(detections.get(key), segmentations.get(key))
    return result


def missing_context() -> dict[str, Any]:
    return object_context(None, None)


def context_for(record: Mapping[str, Any], context: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for key in ordered_aliases(record):
        if key in context:
            return dict(context[key])
    return missing_context()


def attach_context(records: list[dict[str, Any]], context: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        value = dict(row)
        value.update(context_for(row, context))
        result.append(value)
    return result


def ensure_waymo_segmentation(force: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if SEGMENTATION_CONTEXT.exists() and not force:
        rows = read_jsonl(SEGMENTATION_CONTEXT)
        return rows, {"status": "REUSED", "rows": len(rows), "path": str(SEGMENTATION_CONTEXT.resolve())}
    sys.path.insert(0, str((V2 / "src").resolve()))
    import build_scene_context_vision_cache as vision
    started = time.perf_counter()
    frame_payload = json.loads((V5B / "data" / "waymo_frame_rows.json").read_text(encoding="utf-8"))
    frame_rows = list(frame_payload.get("rows", []))
    _detector, segmenter, _det_labels, seg_labels, contract = vision.load_interpreters()
    output: list[dict[str, Any]] = []
    from PIL import Image
    for index, frame in enumerate(frame_rows):
        record = {"sample_id": frame.get("sample_id"), "frame_id": frame.get("frame_id"), "status": "ERROR", "global_area_fraction": [], "class_presence": []}
        try:
            with Image.open(str(frame["path"])) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            segmenter.set_tensor(segmenter.get_input_details()[0]["index"], vision.resize_chw(rgb, 256).astype(np.float32))
            segmenter.invoke()
            summary = vision.product_segmentation(segmenter.get_tensor(segmenter.get_output_details()[0]["index"]), seg_labels)
            record.update({"status": "AVAILABLE", "global_area_fraction": summary.get("global_area_fraction", []), "class_presence": summary.get("class_presence", []), "image_width": frame.get("width"), "image_height": frame.get("height")})
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        output.append(record)
        if (index + 1) % 50 == 0 or index + 1 == len(frame_rows):
            print(f"Waymo semantic context {index + 1}/{len(frame_rows)}", flush=True)
    write_jsonl(SEGMENTATION_CONTEXT, output)
    meta = {"status": "COMPUTED", "rows": len(output), "available": sum(row.get("status") == "AVAILABLE" for row in output), "seconds": time.perf_counter() - started, "contract": contract, "asset": str(vision.SEGMENTATION_MODEL.resolve())}
    write_json(REPORTS / "v6_waymo_semantic_context_runtime.json", meta)
    return output, meta


def context_sources() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    v2_det = read_jsonl(V2 / "cache" / "scene_context" / "detections.jsonl")
    v2_seg = read_jsonl(V2 / "cache" / "scene_context" / "segmentation.jsonl")
    coco_det = read_jsonl(V4 / "data" / "coco" / "product_vision" / "detections.jsonl")
    coco_seg = read_jsonl(V4 / "data" / "coco" / "product_vision" / "segmentation.jsonl")
    waymo_det = read_jsonl(V5B / "data" / "waymo_product_detections.jsonl")
    waymo_seg, waymo_meta = ensure_waymo_segmentation()
    return make_context_map(v2_det, v2_seg), make_context_map(coco_det, coco_seg), make_context_map(waymo_det, waymo_seg), {"waymo_segmentation": waymo_meta, "v2_detection_rows": len(v2_det), "v2_segmentation_rows": len(v2_seg), "coco_detection_rows": len(coco_det), "coco_segmentation_rows": len(coco_seg)}


def base_features(record: Mapping[str, Any]) -> np.ndarray:
    return v6.feature_vector(record, "V1_Baseline-2P")


def context_features(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([
        math.log1p(max(safe(record.get("ctx_object_count"), 0.0), 0.0)),
        math.log1p(max(safe(record.get("ctx_supported_object_count"), 0.0), 0.0)),
        safe(record.get("ctx_mean_bbox_area"), 0.0),
        safe(record.get("ctx_max_bbox_area"), 0.0),
        safe(record.get("ctx_mean_bbox_height"), 0.0),
        safe(record.get("ctx_max_bbox_height"), 0.0),
        safe(record.get("ctx_mean_confidence"), 0.0),
        safe(record.get("ctx_class_diversity"), 0.0),
        safe(record.get("ctx_border_fraction"), 0.0),
        safe(record.get("ctx_seg_available"), 0.0),
        safe(record.get("ctx_seg_present_count"), 0.0),
        safe(record.get("ctx_seg_max_area"), 0.0),
        safe(record.get("ctx_seg_entropy"), 0.0),
        safe(record.get("ctx_seg_road_area"), 0.0),
        safe(record.get("ctx_seg_sidewalk_area"), 0.0),
        safe(record.get("ctx_seg_building_area"), 0.0),
        safe(record.get("ctx_seg_vegetation_area"), 0.0),
        safe(record.get("ctx_seg_sky_area"), 0.0),
        safe(record.get("ctx_seg_people_area"), 0.0),
        safe(record.get("ctx_seg_vehicle_area"), 0.0),
    ], dtype=np.float64)


def gate_features(record: Mapping[str, Any], with_context: bool) -> np.ndarray:
    return np.concatenate([base_features(record), context_features(record)]) if with_context else base_features(record)


_TinyGateBase = nn.Module if nn is not None else object


class TinyGate(_TinyGateBase):
    def __init__(self, input_dim: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: Any) -> Any:
        return torch.sigmoid(self.net(x)).reshape(-1)


def valid_gate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in records:
        visual, size, gt = safe(row.get("V1_Baseline-2P")), safe(row.get("z_size")), safe(row.get("gt_m"))
        if np.isfinite([visual, size, gt]).all() and min(visual, size, gt) > 0:
            result.append(dict(row))
    return result


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (features - mean) / scale, mean, scale


def gate_metric(records: list[Mapping[str, Any]], model_spec: Mapping[str, Any], with_context: bool) -> float:
    values = []
    for row in records:
        pred, _, _ = neural_prediction(row, model_spec, with_context)
        gt = safe(row.get("gt_m"))
        if np.isfinite([pred, gt]).all() and pred > 0 and gt > 0:
            values.append(abs(pred - gt) / gt)
    return float(np.mean(values)) if values else math.nan


def state_spec(model: TinyGate, mean: np.ndarray, scale: np.ndarray, feature_names: list[str], mixture: str, with_context: bool, best_epoch: int, train_rows: int, dev_rows: int) -> dict[str, Any]:
    state = model.state_dict()
    return {
        "format": "rel2abs_v6_neural_gate_v1",
        "mixture": mixture,
        "with_context": with_context,
        "feature_names": feature_names,
        "input_dim": len(feature_names),
        "hidden_dim": HIDDEN,
        "parameter_count": int(sum(int(value.numel()) for value in state.values())),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weight_1": state["net.0.weight"].detach().cpu().numpy().tolist(),
        "bias_1": state["net.0.bias"].detach().cpu().numpy().tolist(),
        "weight_2": state["net.2.weight"].detach().cpu().numpy().tolist(),
        "bias_2": state["net.2.bias"].detach().cpu().numpy().tolist(),
        "best_epoch": best_epoch,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "output_contract": "g=sigmoid(MLP(x)); log(Z)=(1-g)log(Z_visual)+g log(Z_F1)",
    }


def train_gate(train_records: list[dict[str, Any]], dev_records: list[dict[str, Any]], mixture: str, with_context: bool, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if torch is None or nn is None:
        raise RuntimeError(f"PyTorch import failed: {TORCH_IMPORT_ERROR}")
    train_records = valid_gate_records(train_records)
    dev_records = valid_gate_records(dev_records)
    if len(train_records) < 20 or len(dev_records) < 10:
        raise RuntimeError(f"Insufficient gate records for {mixture}: train={len(train_records)} dev={len(dev_records)}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    feature_names = list(v6.feature_vector(train_records[0], "V1_Baseline-2P").shape[0] * ["base"])
    feature_names = [f"base_{index}" for index in range(16)]
    if with_context:
        feature_names.extend(CONTEXT_FEATURES)
    x_train_raw = np.stack([gate_features(row, with_context) for row in train_records]).astype(np.float32)
    x_dev_raw = np.stack([gate_features(row, with_context) for row in dev_records]).astype(np.float32)
    x_train, mean, scale = standardize(x_train_raw.astype(np.float64))
    x_train = x_train.astype(np.float32)
    x_dev = ((x_dev_raw.astype(np.float64) - mean) / scale).astype(np.float32)
    y_train = np.asarray([math.log(safe(row.get("gt_m"))) for row in train_records], dtype=np.float32)
    y_dev = np.asarray([math.log(safe(row.get("gt_m"))) for row in dev_records], dtype=np.float32)
    log_visual_train = np.asarray([math.log(safe(row.get("V1_Baseline-2P"))) for row in train_records], dtype=np.float32)
    log_size_train = np.asarray([math.log(safe(row.get("z_size"))) for row in train_records], dtype=np.float32)
    model = TinyGate(x_train.shape[1], HIDDEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    huber = nn.SmoothL1Loss(beta=0.15)
    tx = torch.from_numpy(x_train)
    ty = torch.from_numpy(y_train)
    tv = torch.from_numpy(log_visual_train)
    ts = torch.from_numpy(log_size_train)
    best_state: dict[str, Any] | None = None
    best_dev = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        gate = model(tx)
        log_pred = (1.0 - gate) * tv + gate * ts
        loss = huber(log_pred, ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dv_gate = model(torch.from_numpy(x_dev))
            dv_log = torch.from_numpy(np.asarray([math.log(safe(row.get("V1_Baseline-2P"))) for row in dev_records], dtype=np.float32))
            dv_size = torch.from_numpy(np.asarray([math.log(safe(row.get("z_size"))) for row in dev_records], dtype=np.float32))
            dv_pred = torch.exp((1.0 - dv_gate) * dv_log + dv_gate * dv_size).numpy()
        dev_gt = np.exp(y_dev)
        dev_absrel = float(np.mean(np.abs(dv_pred - dev_gt) / np.maximum(dev_gt, 1e-8)))
        history.append({"epoch": epoch, "train_log_huber": float(loss.detach().cpu()), "dev_absrel": dev_absrel})
        if dev_absrel + 1e-7 < best_dev:
            best_dev = dev_absrel
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 60:
            break
    if best_state is None:
        raise RuntimeError(f"No learned-gate checkpoint selected for {mixture}")
    model.load_state_dict(best_state)
    spec = state_spec(model, mean, scale, feature_names, mixture, with_context, best_epoch, len(train_records), len(dev_records))
    spec["train_absrel"] = gate_metric(train_records, spec, with_context)
    spec["dev_absrel"] = gate_metric(dev_records, spec, with_context)
    spec["history_tail"] = history[-10:]
    return spec, history


def neural_prediction(record: Mapping[str, Any], model_spec: Mapping[str, Any], with_context: bool) -> tuple[float, float, str]:
    visual = safe(record.get("V1_Baseline-2P"))
    size = safe(record.get("z_size"))
    if not np.isfinite(visual) or visual <= 0:
        return math.nan, 0.0, "rejected_visual"
    if not np.isfinite(size) or size <= 0:
        return visual, 0.0, "visual_fallback_invalid_anchor"
    x = gate_features(record, with_context)
    mean = np.asarray(model_spec["mean"], dtype=np.float64)
    scale = np.asarray(model_spec["scale"], dtype=np.float64)
    w1 = np.asarray(model_spec["weight_1"], dtype=np.float64)
    b1 = np.asarray(model_spec["bias_1"], dtype=np.float64)
    w2 = np.asarray(model_spec["weight_2"], dtype=np.float64).reshape(-1)
    b2 = float(np.asarray(model_spec["bias_2"], dtype=np.float64).reshape(-1)[0])
    hidden = np.maximum(0.0, w1 @ ((x - mean) / scale) + b1)
    logit = float(w2 @ hidden + b2)
    gate = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
    value = math.exp((1.0 - gate) * math.log(visual) + gate * math.log(size))
    return value, gate, "neural_gate"


def candidate_prediction(record: Mapping[str, Any], candidate: str, fusion: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]]) -> tuple[float, float, str]:
    if candidate in models:
        return neural_prediction(record, models[candidate], bool(models[candidate].get("with_context")))
    return v6.prediction(record, candidate, fusion)


def split_external(records: list[dict[str, Any]], dataset: str) -> dict[str, list[dict[str, Any]]]:
    frame_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        frame = str(row.get("frame_id") or row.get("sample_id") or row.get("group_id"))
        frame_groups[frame].append(row)
    groups = sorted(frame_groups, key=lambda key: stable_value(key, f"{SEED}|{dataset}|split"))
    n = len(groups)
    n_train = int(round(n * 0.60))
    n_dev = int(round(n * 0.20))
    train_groups = groups[:n_train]
    dev_groups = groups[n_train:n_train + n_dev]
    test_groups = groups[n_train + n_dev:]
    return {
        "train": [row for key in train_groups for row in frame_groups[key]],
        "dev": [row for key in dev_groups for row in frame_groups[key]],
        "test": [row for key in test_groups for row in frame_groups[key]],
        "groups": {"train": list(train_groups), "dev": list(dev_groups), "test": list(test_groups)},
    }


def v3_development_objects() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    v3 = v6.load_v3()
    spec = json.loads(v6.SPEC_PATH.read_text(encoding="utf-8"))
    priors = v6.parse_priors(v6.PRIOR_PATH)
    train_rows, dev_rows = v6.load_rows("train"), v6.load_rows("dev")
    cache = v3.Cache(v6.V3_CACHE_DIR)
    raw_train = v6.load_all_visual_raw("diode", "train", train_rows, cache, "cpu", v3)
    raw_dev = v6.load_all_visual_raw("diode", "dev", dev_rows, cache, "cpu", v3)
    return v6.development_object_records(train_rows, dev_rows, cache, raw_train, raw_dev, spec, v3, priors)


def build_mixture_records(train_v3: list[dict[str, Any]], dev_v3: list[dict[str, Any]], splits: Mapping[str, Mapping[str, list[dict[str, Any]]]], mixture: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for dataset in MIXTURES[mixture]:
        if dataset == "V3":
            train.extend(train_v3)
            dev.extend(dev_v3)
        else:
            official_track = "COCO-A_OFFICIAL_BBOX" if dataset == "COCO" else "WAYMO-A_OFFICIAL_BBOX"
            train.extend(row for row in splits[dataset]["train"] if row.get("track") == official_track)
            dev.extend(row for row in splits[dataset]["dev"] if row.get("track") == official_track)
    return train, dev


def metric_rows(records: list[dict[str, Any]], candidates: list[str], fusion: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]], split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    band_defs = [("ALL", 0.0, math.inf), ("eyeai_0_5_5m", 0.5, 5.0), ("eyeai_0_5_10m", 0.5, 10.0)] + [(name, low, high) for name, low, high in v6.BANDS]
    for dataset in sorted({str(row.get("dataset")) for row in records}):
        dataset_rows = [row for row in records if str(row.get("dataset")) == dataset]
        for track in sorted({str(row.get("track")) for row in dataset_rows}):
            track_rows = [row for row in dataset_rows if str(row.get("track")) == track]
            for candidate in candidates:
                for scope, low, high in band_defs:
                    pred, gt, weights = [], [], []
                    for row in track_rows:
                        value, weight, status = candidate_prediction(row, candidate, fusion, models)
                        target = safe(row.get("gt_m"))
                        if np.isfinite([value, target]).all() and value > 0 and target > 0 and low <= target < high:
                            pred.append(value); gt.append(target); weights.append(weight)
                    summary = v6.metric_summary(pred, gt)
                    is_learned_gate = candidate.startswith("E_NeuralGate_")
                    rows.append({"split": split_name, "dataset": dataset, "track": track, "candidate": candidate, "scope": scope, "gate_weight_mean": float(np.mean(weights)) if weights and is_learned_gate else math.nan, "gate_weight_median": float(np.median(weights)) if weights and is_learned_gate else math.nan, **summary})
    return rows


def group_bootstrap(records: list[dict[str, Any]], candidates: list[str], fusion: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]], dataset: str, track: str, seed: int) -> list[dict[str, Any]]:
    subset = [row for row in records if str(row.get("dataset")) == dataset and str(row.get("track")) == track]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset:
        key = str(row.get("frame_id") or row.get("sample_id") or row.get("group_id"))
        groups[key].append(row)
    keys = sorted(groups)
    if not keys:
        return []
    candidate_errors: dict[str, dict[str, np.ndarray]] = {}
    paired_baseline_errors: dict[str, dict[str, np.ndarray]] = {}
    for candidate in candidates:
        candidate_errors[candidate] = {}
        paired_baseline_errors[candidate] = {}
        for key, rows in groups.items():
            candidate_values = []
            baseline_values = []
            for row in rows:
                candidate_pred, _, _ = candidate_prediction(row, candidate, fusion, models)
                baseline_pred, _, _ = candidate_prediction(row, "V1_Baseline-2P", fusion, models)
                gt = safe(row.get("gt_m"))
                if np.isfinite([candidate_pred, baseline_pred, gt]).all() and candidate_pred > 0 and baseline_pred > 0 and gt > 0:
                    candidate_values.append(abs(candidate_pred - gt) / gt)
                    baseline_values.append(abs(baseline_pred - gt) / gt)
            candidate_errors[candidate][key] = np.asarray(candidate_values, dtype=np.float64)
            paired_baseline_errors[candidate][key] = np.asarray(baseline_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = {candidate: np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64) for candidate in candidates}
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.choice(keys, size=len(keys), replace=True)
        for candidate in candidates:
            values = np.concatenate([candidate_errors[candidate][key] for key in selected if candidate_errors[candidate][key].size], dtype=np.float64)
            baseline_values = np.concatenate([paired_baseline_errors[candidate][key] for key in selected if paired_baseline_errors[candidate][key].size], dtype=np.float64)
            deltas[candidate][index] = float(np.mean(values) - np.mean(baseline_values)) if values.size and baseline_values.size else math.nan
    rows = []
    for candidate, values in deltas.items():
        values = values[np.isfinite(values)]
        paired_count = sum(int(paired_baseline_errors[candidate][key].size) for key in keys)
        rows.append({"dataset": dataset, "track": track, "candidate": candidate, "replicates": len(values), "delta_absrel_vs_baseline": float(np.mean(values)), "ci95_low": float(np.quantile(values, 0.025)), "ci95_high": float(np.quantile(values, 0.975)), "probability_improvement": float(np.mean(values < 0.0)), "group_count": len(keys), "object_count": len(subset), "paired_object_count": paired_count})
    return rows


def split_summary(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return {"name": name, "rows": len(records), "frames": len({str(x.get("frame_id") or x.get("sample_id")) for x in records}), "bands": dict(sorted(Counter(str(x.get("band")) for x in records).items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cross-dataset neural gate experiment.")
    parser.add_argument("--force-waymo-segmentation", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if torch is None:
        raise RuntimeError(f"PyTorch is required for this experiment: {TORCH_IMPORT_ERROR}")
    waymo_seg_rows, waymo_seg_meta = ensure_waymo_segmentation(force=args.force_waymo_segmentation)
    v3_context, coco_context, waymo_context, context_meta = context_sources()
    train_v3, dev_v3 = v3_development_objects()
    train_v3 = attach_context(train_v3, v3_context)
    dev_v3 = attach_context(dev_v3, v3_context)
    coco_all = attach_context(read_csv(REPORTS / "v6_coco_panel.csv"), coco_context)
    waymo_all = attach_context(read_csv(REPORTS / "v6_waymo_panel.csv"), waymo_context)
    coco_by_split = split_external(coco_all, "COCO")
    waymo_by_split = split_external(waymo_all, "WAYMO")
    splits = {"COCO": coco_by_split, "WAYMO": waymo_by_split}
    manifest = {
        "format": "rel2abs_v6_neural_gate_cross_dataset_split_v1",
        "seed": SEED,
        "policy": "group-disjoint frame split; external train/dev may fit/select; external test is freeze-only",
        "ratios": {"train": 0.60, "dev": 0.20, "test": 0.20},
        "sources": {"V3_train": split_summary(train_v3, "V3_train"), "V3_dev": split_summary(dev_v3, "V3_dev"), "COCO_all": split_summary(coco_all, "COCO_all"), "WAYMO_all": split_summary(waymo_all, "WAYMO_all")},
        "splits": {dataset: {part: split_summary(values, f"{dataset}_{part}") for part, values in split.items() if part != "groups"} for dataset, split in splits.items()},
        "groups": {dataset: split["groups"] for dataset, split in splits.items()},
        "label_quality": {"V3": "Gold metric depth", "WAYMO": "Gold-A LiDAR object depth", "COCO": "P0 pseudo-GT; stress/training diagnostic, not metric Gold"},
        "context_features": list(CONTEXT_FEATURES),
        "context_meta": context_meta,
        "waymo_segmentation_meta": waymo_seg_meta,
    }
    write_json(SPLIT_MANIFEST, manifest)
    fusion = json.loads((REPORTS / "v6_fusion_parameters.json").read_text(encoding="utf-8"))
    models: dict[str, dict[str, Any]] = {}
    train_dev_rows: list[dict[str, Any]] = []
    for mixture, datasets in MIXTURES.items():
        train_records, dev_records = build_mixture_records(train_v3, dev_v3, splits, mixture)
        for with_context, feature_family in ((True, "Context"), (False, "Base")):
            model_name = f"E_NeuralGate_{feature_family}_{mixture}"
            spec, history = train_gate(train_records, dev_records, mixture, with_context, SEED)
            models[model_name] = spec
            train_dev_rows.append({"model": model_name, "mixture": mixture, "train_rows": len(valid_gate_records(train_records)), "dev_rows": len(valid_gate_records(dev_records)), "train_absrel": spec["train_absrel"], "dev_absrel": spec["dev_absrel"], "parameter_count": spec["parameter_count"], "best_epoch": spec["best_epoch"], "context": with_context, "source_datasets": "+".join(datasets)})
    write_json(REPORTS / "v6_neural_gate_parameters.json", models)
    write_csv(REPORTS / "v6_neural_gate_train_dev.csv", train_dev_rows)
    candidates = ["V1_Baseline-2P", "B3_Baseline_DisagreementFallback", "F1_SIZE_ANCHOR_OVERRIDE"] + sorted(models)
    test_records = coco_by_split["test"] + waymo_by_split["test"]
    metric = metric_rows(test_records, candidates, fusion, models, "external_group_holdout")
    write_csv(REPORTS / "v6_neural_gate_metrics.csv", metric)
    all_holdout_rows = []
    for dataset, split in (("COCO", coco_by_split), ("WAYMO", waymo_by_split)):
        all_holdout_rows.extend(split["test"])
    bootstrap: list[dict[str, Any]] = []
    for dataset, track in (("COCO", "COCO-A_OFFICIAL_BBOX"), ("COCO", "COCO-B_PRODUCT_YOLO_MATCHED"), ("WAYMO", "WAYMO-A_OFFICIAL_BBOX"), ("WAYMO", "WAYMO-B_PRODUCT_YOLO_MATCHED")):
        bootstrap.extend(group_bootstrap(all_holdout_rows, ["B3_Baseline_DisagreementFallback", "F1_SIZE_ANCHOR_OVERRIDE"] + sorted(models), fusion, models, dataset, track, SEED + stable_value(f"{dataset}|{track}") % 10000))
    write_csv(REPORTS / "v6_neural_gate_bootstrap.csv", bootstrap)
    gate_rows = []
    for row in test_records:
        output = dict(row)
        for candidate in sorted(models):
            value, weight, status = neural_prediction(row, models[candidate], bool(models[candidate].get("with_context")))
            output[f"{candidate}_prediction"] = value
            output[f"{candidate}_weight"] = weight
            output[f"{candidate}_status"] = status
        gate_rows.append(output)
    write_csv(REPORTS / "v6_neural_gate_holdout_rows.csv", gate_rows)
    try:
        import matplotlib.pyplot as plt
        plot_rows = [row for row in metric if row["scope"] in {"eyeai_0_5_5m", "5_10m", "ge15m"} and row["track"] in {"COCO-A_OFFICIAL_BBOX", "WAYMO-A_OFFICIAL_BBOX"}]
        if plot_rows:
            labels = sorted({row["candidate"] for row in plot_rows})
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for axis, dataset in zip(axes, ("COCO", "WAYMO")):
                scopes = ["eyeai_0_5_5m", "5_10m", "ge15m"]
                x = np.arange(len(scopes)); width = 0.8 / max(len(labels), 1)
                for index, candidate in enumerate(labels):
                    vals = [next((float(row["absrel"]) for row in plot_rows if row["dataset"] == dataset and row["candidate"] == candidate and row["scope"] == scope), math.nan) for scope in scopes]
                    axis.bar(x + (index - len(labels) / 2) * width + width / 2, vals, width, label=candidate.replace("E_NeuralGate_Context_", "E:"))
                axis.set_title(f"{dataset} held-out object AbsRel"); axis.set_xticks(x, scopes); axis.grid(axis="y", alpha=0.2)
            axes[0].legend(fontsize=6, loc="upper left"); fig.tight_layout(); fig.savefig(PLOTS / "v6_neural_gate_cross_dataset_holdout.png", dpi=150); plt.close(fig)
    except Exception:
        pass
    report: list[str] = [
        "# V6 Cross-Dataset Neural Gate Experiment",
        "",
        "This is a research-only comparison. MiDaS, all frozen visual experts, F1 and B3 were not changed.",
        "",
        "## Gate contract",
        "",
        "The network predicts only `g = sigmoid(MLP(features))`. The final object depth is `exp((1-g)*log(Z_visual) + g*log(Z_F1))`, so the learned output stays between the visual Baseline-2P and deterministic F1 estimates.",
        "",
        f"The network has a 16-unit hidden layer. Context gates add {len(CONTEXT_FEATURES)} frame-context features; Base gates omit them as an ablation. Exact parameter counts are in `reports/v6_neural_gate_parameters.json`.",
        "",
        "## Training mixtures",
        "",
        "- V3_ONLY",
        "- V3_COCO",
        "- V3_WAYMO",
        "- COCO_WAYMO",
        "- V3_COCO_WAYMO",
        "",
        "For each mixture, Context and Base are trained with identical data and seeds. Base is the no-context control; Context adds YOLO object/semantic statistics.",
        "",
        "COCO uses the existing P0 pseudo-GT labels and is not treated as metric Gold. COCO and Waymo are split by frame into 60% train, 20% selection and 20% holdout before any gate fitting.",
        "",
        "External gate fitting uses only the official-box tracks (`COCO-A_OFFICIAL_BBOX` and `WAYMO-A_OFFICIAL_BBOX`); Product-YOLO matched tracks are held out for robustness evaluation only.",
        "",
        "## Context features",
        "",
        "YOLO object count, supported-object count, bbox area/height statistics, confidence, class diversity and border fraction are included. Existing product semantic-segmentation global area fractions are reduced to presence count, entropy, largest segment and road/sidewalk/building/vegetation/sky/people/vehicle areas. Waymo semantic context was computed with the existing product segmentation model on the 500 local selected frames; no new model or data source was introduced.",
        "",
        "## Train/dev results",
        "",
        "| model | train objects | dev objects | train AbsRel | dev AbsRel | parameters | best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in train_dev_rows:
        report.append(f"| {row['mixture']} | {row['train_rows']} | {row['dev_rows']} | {row['train_absrel']:.4f} | {row['dev_absrel']:.4f} | {row['parameter_count']} | {row['best_epoch']} |")
    report += ["", "## Held-out comparison", "", "AbsRel is lower-is-better. Exact MAE/RMSE/MedAE, distance bands and gate weights are in `reports/v6_neural_gate_metrics.csv`.", ""]
    report += ["| dataset | track | candidate | scope | n | AbsRel | mean gate | catastrophic |", "|---|---|---|---|---:|---:|---:|---:|"]
    for row in metric:
        if row["scope"] in {"ALL", "eyeai_0_5_5m", "5_10m", "ge15m"} and row["track"] in {"COCO-A_OFFICIAL_BBOX", "WAYMO-A_OFFICIAL_BBOX"}:
            report.append(f"| {row['dataset']} | {row['track']} | {row['candidate']} | {row['scope']} | {row['n']} | {row['absrel']:.4f} | {row['gate_weight_mean']:.3f} | {row['catastrophic_rate_absrel_gt1']:.4f} |")
    report += ["", "## Paired bootstrap", "", "The bootstrap resamples complete frames, preserving all objects in each frame. Candidate deltas use the common valid object subset for Baseline-2P and the candidate. Negative delta means improvement over Baseline-2P.", "", "| dataset | track | candidate | delta | 95% CI | P(improvement) |", "|---|---|---|---:|---|---:|"]
    for row in bootstrap:
        report.append(f"| {row['dataset']} | {row['track']} | {row['candidate']} | {row['delta_absrel_vs_baseline']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['probability_improvement']:.3f} |")
    report += ["", "## Limitations", "", "- Waymo near/mid support remains small even after group splitting; the holdout n is reported explicitly and must not be overinterpreted.", "- COCO training and evaluation use P0 pseudo-GT, so cross-dataset gains involving COCO are stress evidence, not metric-depth certification.", "- The context gate uses no dataset identifier. Missing segmentation is represented by an availability feature; semantic features are therefore a deployability risk if the mobile pipeline does not provide them consistently.", "- This experiment does not modify EyeAIApp or perform Android/LiteRT integration.", ""]
    (V6 / "V6_NEURAL_GATE_CROSS_DATASET_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    runtime = {"format": "rel2abs_v6_neural_gate_runtime_v2", "seconds": time.perf_counter() - started, "seed": SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "mixtures": list(MIXTURES), "context_features": list(CONTEXT_FEATURES), "model_parameter_counts": {name: spec["parameter_count"] for name, spec in models.items()}, "waymo_segmentation": waymo_seg_meta, "holdout_rows": len(test_records), "midas_changed": False, "eyeai_changed": False, "new_depth_architecture": False}
    write_json(REPORTS / "v6_neural_gate_runtime.json", runtime)
    print(json.dumps({"seconds": runtime["seconds"], "models": list(models), "holdout_rows": len(test_records), "report": str((V6 / 'V6_NEURAL_GATE_CROSS_DATASET_REPORT.md').resolve())}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
