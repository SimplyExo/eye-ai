from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import run_v6_object_feature_gate_ablation as ablation


SEED = 20260919
CAMERA_HEIGHT_TENTHS = (16, 17, 18, 19, 20)
BASE_FEATURE_SET = "DEPTH+HEIGHT+WIDTH+SHAPE_POSITION+ANCHOR+DETECTION+SEGMENTATION"
CAMERA_FEATURE_NAME = "camera_height_m"
FEATURE_SET = f"{BASE_FEATURE_SET}+CAMERA_HEIGHT_CALIBRATION"
OUTPUT_PATH = ablation.REPORTS / "v6_object_feature_gate_camera_height_160_200_parameters.json"
METRICS_PATH = ablation.REPORTS / "v6_object_feature_gate_camera_height_160_200_metrics.json"

ACTIVE_CAMERA_HEIGHT_M = 1.70


def height_label(camera_height_tenths: int) -> str:
    return f"{camera_height_tenths * 10:03d}"


def model_id(camera_height_tenths: int) -> str:
    return f"E_ObjectGate_{BASE_FEATURE_SET}_V3_WAYMO_CAMERA_HEIGHT_{height_label(camera_height_tenths)}"


def camera_feature_names(feature_set: str) -> list[str]:
    if feature_set != FEATURE_SET:
        return ORIGINAL_FEATURE_NAMES(feature_set)
    return ORIGINAL_FEATURE_NAMES(BASE_FEATURE_SET) + [CAMERA_FEATURE_NAME]


def camera_feature_vector(record: Mapping[str, Any], feature_set: str) -> np.ndarray:
    if feature_set != FEATURE_SET:
        return ORIGINAL_FEATURE_VECTOR(record, feature_set)
    base = ORIGINAL_FEATURE_VECTOR(record, BASE_FEATURE_SET)
    return np.concatenate((base, np.asarray([ACTIVE_CAMERA_HEIGHT_M], dtype=np.float64)))


def camera_standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized, mean, scale = ORIGINAL_STANDARDIZE(values)
    if values.ndim == 2 and values.shape[1] == len(ORIGINAL_FEATURE_NAMES(BASE_FEATURE_SET)) + 1:
        # Keep the physical camera height in the exported model.
        # Each model is trained for one fixed height.
        mean = mean.copy()
        scale = scale.copy()
        mean[-1] = 0.0
        scale[-1] = 1.0
        normalized = (values - mean) / scale
    return normalized, mean, scale


def load_v3_waymo_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior_gate = ablation.prior_gate
    prior_gate.ensure_waymo_segmentation()
    sources = ablation.rich_context_sources()
    try:
        train_v3, dev_v3 = prior_gate.v3_development_objects()
    except Exception as exc:
        print(f"V3 Torch loader unavailable; using frozen NumPy decoder ({type(exc).__name__}: {exc})", flush=True)
        train_v3, dev_v3 = ablation.numpy_v3_development_objects()
    train_v3 = ablation.attach_rich_context(train_v3, sources)
    dev_v3 = ablation.attach_rich_context(dev_v3, sources)
    coco_all = ablation.attach_rich_context(prior_gate.read_csv(ablation.REPORTS / "v6_coco_panel.csv"), sources)
    waymo_all = ablation.attach_rich_context(prior_gate.read_csv(ablation.REPORTS / "v6_waymo_panel.csv"), sources)
    splits = {
        "COCO": ablation.external_split(coco_all, "COCO"),
        "WAYMO": ablation.external_split(waymo_all, "WAYMO"),
    }
    return ablation.build_mixture_records(train_v3, dev_v3, splits, "V3_WAYMO")


def main() -> int:
    global ACTIVE_CAMERA_HEIGHT_M, ORIGINAL_FEATURE_NAMES, ORIGINAL_FEATURE_VECTOR, ORIGINAL_STANDARDIZE
    ORIGINAL_FEATURE_NAMES = ablation.feature_names
    ORIGINAL_FEATURE_VECTOR = ablation.feature_vector
    ORIGINAL_STANDARDIZE = ablation.standardize

    # Reuse the main trainer and add the camera height input.
    ablation.feature_names = camera_feature_names
    ablation.feature_vector = camera_feature_vector
    ablation.standardize = camera_standardize

    train_records, dev_records = load_v3_waymo_records()
    models: dict[str, dict[str, Any]] = {}
    metrics: list[dict[str, Any]] = []
    for camera_height_tenths in CAMERA_HEIGHT_TENTHS:
        ACTIVE_CAMERA_HEIGHT_M = camera_height_tenths / 10.0
        spec, _ = ablation.train_gate(train_records, dev_records, FEATURE_SET, "V3_WAYMO", SEED)
        current_model_id = model_id(camera_height_tenths)
        spec.update({
            "calibration_parameter": f"camera_height_m_{ACTIVE_CAMERA_HEIGHT_M:.1f}",
            "camera_height_m": ACTIVE_CAMERA_HEIGHT_M,
            "camera_height_reference_m": ACTIVE_CAMERA_HEIGHT_M,
            "camera_height_tenths": camera_height_tenths,
            "camera_height_feature_name": CAMERA_FEATURE_NAME,
            "training_mixture": "V3_WAYMO",
            "optimizer_backend": "torch_adam",
        })
        models[current_model_id] = spec
        metric = {
            "model_id": current_model_id,
            "feature_set": FEATURE_SET,
            "input_dim": spec["input_dim"],
            "train_rows": spec["train_rows"],
            "dev_rows": spec["dev_rows"],
            "train_absrel": spec["train_absrel"],
            "dev_absrel": spec["dev_absrel"],
            "camera_height_m": ACTIVE_CAMERA_HEIGHT_M,
            "parameter_count": spec["parameter_count"],
        }
        metrics.append(metric)
        print(json.dumps(metric), flush=True)

    ablation.write_json(OUTPUT_PATH, models)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"parameters={OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
