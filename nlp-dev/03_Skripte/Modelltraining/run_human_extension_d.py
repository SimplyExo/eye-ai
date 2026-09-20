#!/usr/bin/env python3
"""Train the D model with the approved human clean extension.

The script has five separate steps: gate, promote, prepare, train and
finalize metadata. Existing A, B and C results stay unchanged. Human ASR
data is not used for D.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/robert/Dokumente/GitHub")
STUDY = WORKSPACE / "corrected_study"
D_ROOT = STUDY / "D_human_extension"
D_PROTOCOL = D_ROOT / "protocol"
CANONICAL = WORKSPACE / "Intent_Datasets"
PACKAGE = CANONICAL / "Neu/human_extension_package/EyeAI_Human_Intent_Extension"
CANDIDATE = PACKAGE / "eyeai_human_extension_clean.train"
FINAL_GATE_ROOT = PACKAGE / "final_gate"
FINAL_GATE_FILE = FINAL_GATE_ROOT / "human_extension_final.train"
FINAL_ACTIVE_ROOT = CANONICAL / "Aktuell_verwendet/training/extensions/human"
FINAL_ACTIVE_FILE = FINAL_ACTIVE_ROOT / "eyeai_human_extension_final.train"
GATE_CSV = FINAL_GATE_ROOT / "human_extension_final_gate.csv"
GATE_MANIFEST = FINAL_GATE_ROOT / "human_extension_final_manifest.json"
GATE_SUMMARY = FINAL_GATE_ROOT / "human_extension_gate_summary.json"
REGISTRY_PATH = CANONICAL / "dataset_registry.json"
REGISTRY_KEY = "HUMAN_EXTENSION_D_CLEAN"
DEV_SHA256 = "bdb2122acc9c7d8e42e9a84d593ea4162d9688d599c3f0a7eee66ad99f75663f"
DEV_COUNT = 659
SEEDS = (20260810, 20260811, 20260812, 20260813, 20260814)

sys.path.insert(0, str(STUDY))
import run_corrected_study as abc  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_locked_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite existing locked file: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def write_csv_locked(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    rendered_text = stream.getvalue()
    if path.exists() and path.read_text(encoding="utf-8") != rendered_text:
        raise RuntimeError(f"Refusing to overwrite existing locked file: {path}")
    if not path.exists():
        path.write_text(rendered_text, encoding="utf-8")


def normalize(text: str) -> str:
    return abc.normalize(text)


def registry_data() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def verify_registry() -> None:
    from dataset_registry import verify_registry as registry_verify

    registry_verify()


def read_manifest_rows() -> list[dict[str, str]]:
    path = PACKAGE / "eyeai_human_extension_manifest.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def read_registry_texts(skip_keys: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Collect all text entries from the registry, including ASR data."""

    skip_keys = skip_keys or set()
    registry = registry_data()
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, entry in registry.get("entries", {}).items():
        if key in skip_keys:
            continue
        kind = entry.get("count_kind", "labelled")
        if kind == "none":
            continue
        path = CANONICAL / entry["canonical_path"]
        if not path.is_file():
            raise RuntimeError(f"Registry text source missing: {path}")

        def add(text: str, label: str = "", field: str = "text", line: int = 0) -> None:
            text = text.strip()
            if not text:
                return
            indexed[normalize(text)].append(
                {
                    "registry_key": key,
                    "path": str(path),
                    "role": entry.get("role", ""),
                    "status": entry.get("status", ""),
                    "label": label.strip(),
                    "field": field,
                    "line": line,
                }
            )

        if kind == "qc_nonempty":
            with path.open(encoding="utf-8", newline="") as handle:
                for line, row in enumerate(csv.DictReader(handle), start=2):
                    add(row.get("clean_text", ""), row.get("gold_label", ""), "clean_text", line)
                    add(row.get("asr_text", ""), row.get("gold_label", ""), "asr_text", line)
            continue

        with path.open(encoding="utf-8-sig") as handle:
            for line, raw in enumerate(handle, start=1):
                value = raw.rstrip("\r\n")
                if not value.strip():
                    continue
                if kind == "inputs" or kind == "raw_lines":
                    add(value, "", "input", line)
                elif ";" in value:
                    text, label = value.rsplit(";", 1)
                    add(text, label, "text", line)
                else:
                    add(value, "", "text", line)
    return indexed


def candidate_rows() -> tuple[list[abc.baseline.Sample], list[dict[str, str]]]:
    if not CANDIDATE.is_file():
        raise RuntimeError(f"Human candidate missing: {CANDIDATE}")
    samples = abc.load_labelled(CANDIDATE, "human_extension_candidate", "human_clean")
    manifest = read_manifest_rows()
    if len(samples) != len(manifest):
        raise RuntimeError(f"Candidate/manifest count mismatch: {len(samples)} vs {len(manifest)}")
    return samples, manifest


def verify_existing_a_comparator() -> dict[str, Any]:
    """Check that A is the fixed comparator for D."""

    dev = abc.resolve_file("V2_DEV_CORRECTED")
    if sha256(dev) != DEV_SHA256 or len(abc.load_labelled(dev, "v2_dev_corrected", "development")) != DEV_COUNT:
        raise RuntimeError("V2_DEV_CORRECTED does not match the required 659-row frozen split")
    protocol_dev = abc.PROTOCOL / "dev_locked.val"
    if sha256(protocol_dev) != DEV_SHA256:
        raise RuntimeError("A protocol dev_locked.val differs from V2_DEV_CORRECTED")

    metric_files = sorted((abc.STATE_ROOTS["A"] / "metrics").glob("M*_T*_seed_*.json"))
    if len(metric_files) != 40:
        raise RuntimeError(f"Frozen A comparator has {len(metric_files)} metrics, expected 40")
    expected_train = {"M0": 2918, "M1": 4865, "M2": 4865, "M3": 1828}
    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("development_validation_sha256") != DEV_SHA256:
            raise RuntimeError(f"A artifact has wrong dev hash: {path}")
        if payload.get("development_validation_samples") != DEV_COUNT:
            raise RuntimeError(f"A artifact has wrong dev count: {path}")
        if payload.get("old_model_checkpoint_reused") is not False:
            raise RuntimeError(f"A artifact provenance is not a fresh controlled artifact: {path}")
        if payload.get("training_samples") != expected_train[payload["strategy"]]:
            raise RuntimeError(f"A artifact has unexpected training count: {path}")
    return {
        "metric_files": len(metric_files),
        "dev_path": str(dev),
        "dev_sha256": sha256(dev),
        "dev_count": DEV_COUNT,
        "training_counts": expected_train,
    }


def gate_human_candidates() -> dict[str, Any]:
    verify_registry()
    comparator = verify_existing_a_comparator()
    samples, manifest = candidate_rows()
    protected = read_registry_texts(skip_keys={REGISTRY_KEY})
    candidate_by_norm: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        candidate_by_norm[sample.normalized_text].append(index)

    fields = [
        "sample_id", "text", "label", "normalized_text", "source_persons", "source_files",
        "raw_occurrences", "gate_status", "included", "reason", "collision_registry_keys",
        "collision_paths", "collision_labels",
    ]
    gate_rows: list[dict[str, Any]] = []
    approved: list[abc.baseline.Sample] = []
    for index, sample in enumerate(samples):
        meta = manifest[index]
        collisions = protected.get(sample.normalized_text, [])
        internal = candidate_by_norm[sample.normalized_text]
        reasons: list[str] = []
        if len(internal) > 1:
            reasons.append("INTERNAL_NORMALIZED_DUPLICATE")
        if len({samples[item].label for item in internal}) > 1:
            reasons.append("INTERNAL_CROSS_LABEL_CONFLICT")
        if collisions:
            roles = {row["role"] for row in collisions}
            if any(role in {"train_clean", "train_vosk", "train_extension", "asr_generated"} for role in roles):
                reasons.append("EXISTING_TRAINING_DUPLICATE")
            elif any(role in {"development", "validation_eval", "challenge_eval", "blind_eval", "legacy_evaluation_only"} for role in roles):
                reasons.append("PROTECTED_EVALUATION_OVERLAP")
            else:
                reasons.append("REGISTRY_TEXT_OVERLAP")
        include = not reasons
        if include:
            approved.append(sample)
        gate_rows.append(
            {
                "sample_id": meta.get("sample_id", f"HUMAN_LINE_{index + 1:04d}"),
                "text": sample.text,
                "label": sample.label,
                "normalized_text": sample.normalized_text,
                "source_persons": meta.get("source_persons", ""),
                "source_files": meta.get("source_files", ""),
                "raw_occurrences": meta.get("raw_occurrences", ""),
                "gate_status": "INCLUDED" if include else "EXCLUDED",
                "included": str(include).lower(),
                "reason": "included_no_registry_overlap" if include else ";".join(reasons),
                "collision_registry_keys": "|".join(sorted({row["registry_key"] for row in collisions})),
                "collision_paths": "|".join(sorted({row["path"] for row in collisions})),
                "collision_labels": "|".join(sorted({row["label"] for row in collisions if row["label"]})),
            }
        )

    final_text = "".join(f"{sample.text};{sample.label}\n" for sample in approved)
    write_locked_text(FINAL_GATE_FILE, final_text)
    write_locked_text(FINAL_ACTIVE_FILE, final_text)
    write_csv_locked(GATE_CSV, gate_rows, fields)
    included_meta = [row for row in gate_rows if row["included"] == "true"]
    manifest_payload = {
        "experiment": "D = A + Human Clean Extension",
        "source_package": str(PACKAGE),
        "candidate_path": str(CANDIDATE),
        "candidate_sha256": sha256(CANDIDATE),
        "candidate_count": len(samples),
        "final_gate_path": str(FINAL_GATE_FILE),
        "final_active_path": str(FINAL_ACTIVE_FILE),
        "final_count": len(approved),
        "final_sha256": sha256(FINAL_GATE_FILE),
        "class_counts": dict(sorted(Counter(sample.label for sample in approved).items())),
        "source_persons": dict(sorted(Counter(row.get("source_persons", "") for row in included_meta).items())),
        "rows": included_meta,
    }
    write_json(GATE_MANIFEST, manifest_payload)
    summary = {
        "candidate_count": len(samples),
        "included_count": len(approved),
        "excluded_count": len(samples) - len(approved),
        "candidate_sha256": sha256(CANDIDATE),
        "final_sha256": sha256(FINAL_GATE_FILE),
        "final_count": len(approved),
        "class_counts": manifest_payload["class_counts"],
        "gate_csv": str(GATE_CSV),
        "final_manifest": str(GATE_MANIFEST),
        "final_gate_file": str(FINAL_GATE_FILE),
        "final_active_file": str(FINAL_ACTIVE_FILE),
        "registry_verified_before_gate": True,
        "a_comparator": comparator,
        "asr_sources_used": False,
        "reasons": dict(sorted(Counter(row["reason"] for row in gate_rows if row["included"] != "true").items())),
    }
    write_json(GATE_SUMMARY, summary)
    return summary


def promote_registry_entry() -> None:
    """Add the approved human data to the registry after review."""

    registry = registry_data()
    entry = {
        "canonical_path": str(FINAL_ACTIVE_FILE.relative_to(CANONICAL)),
        "sha256": sha256(FINAL_ACTIVE_FILE),
        "expected_count": int(json.loads(GATE_SUMMARY.read_text(encoding="utf-8"))["final_count"]),
        "count_kind": "labelled",
        "role": "train_extension_human",
        "status": "tested_extension_d",
    }
    existing = registry.setdefault("entries", {}).get(REGISTRY_KEY)
    if existing is not None and existing != entry:
        raise RuntimeError(f"Registry entry {REGISTRY_KEY} differs and will not be overwritten")
    registry["entries"][REGISTRY_KEY] = entry
    registry.setdefault("collections", {})["HUMAN_EXTENSION_D"] = [REGISTRY_KEY]
    registry.setdefault("notes", {})["human_extension_d"] = (
        "Final human Clean extension after full canonical registry gate; used only by state D, not A/B/C."
    )
    write_json(REGISTRY_PATH, registry)


def update_manifest() -> None:
    manifest_path = CANONICAL / "MANIFEST.json"
    csv_path = CANONICAL / "MANIFEST.csv"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_rel = str(FINAL_ACTIVE_FILE.relative_to(CANONICAL))
    summary = json.loads(GATE_SUMMARY.read_text(encoding="utf-8"))
    rows = json.loads(GATE_MANIFEST.read_text(encoding="utf-8"))["rows"]
    classes = dict(summary["class_counts"])
    item = {
        "canonical_path": canonical_rel,
        "original_path": str(CANDIDATE),
        "filename": FINAL_ACTIVE_FILE.name,
        "category": "CURRENT",
        "role": "train_extension_human",
        "status": "tested_extension_d",
        "sample_count": summary["final_count"],
        "sha256": summary["final_sha256"],
        "classes": classes,
        "used_in_v1": False,
        "used_in_v2_baseline": False,
        "used_in_previous_retraining": False,
        "used_for_training": True,
        "used_for_early_stopping": False,
        "used_for_evaluation": False,
        "duplicate_of": "",
        "notes": "Approved human Clean extension for isolated state D = A + Human; ASR sources are not used in D.",
    }
    existing = [row for row in data if row.get("canonical_path") == canonical_rel]
    if existing and existing[0] != item:
        raise RuntimeError("MANIFEST entry differs and will not be overwritten")
    if not existing:
        data.append(item)
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "canonical_path", "original_path", "filename", "category", "role", "status", "sample_count",
        "sha256", "classes", "used_in_v1", "used_in_v2_baseline", "used_in_previous_retraining",
        "used_for_training", "used_for_early_stopping", "used_for_evaluation", "duplicate_of", "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in data:
            row = dict(row)
            row["classes"] = json.dumps(row.get("classes", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(row)


def prepare_d_protocol() -> dict[str, Any]:
    verify_registry()
    if not GATE_SUMMARY.is_file() or not GATE_MANIFEST.is_file():
        raise RuntimeError("Run --gate first")
    summary = json.loads(GATE_SUMMARY.read_text(encoding="utf-8"))
    if summary["included_count"] <= 0:
        raise RuntimeError("Human gate approved no samples")
    active = abc.resolve_file(REGISTRY_KEY)
    if sha256(active) != summary["final_sha256"]:
        raise RuntimeError("Registered human final file hash differs from gate summary")
    comparator = verify_existing_a_comparator()
    human = abc.load_labelled(active, "human_extension_final", "human_clean")
    if len(human) != summary["included_count"]:
        raise RuntimeError("Registered human final count differs from gate summary")
    base_clean, base_vosk = original_training_sources("A", [])
    d_clean = [*base_clean, *human]
    mix = abc.expected_mix_counts(len(d_clean), len(base_vosk))
    eval_datasets = abc.load_evaluation_datasets(abc.resolve_file("V2_DEV_CORRECTED"))
    evaluation = {}
    registry = registry_data()
    for name, key in (
        ("dev_locked", "V2_DEV_CORRECTED"),
        ("v2_challenge_40", "V2_CHALLENGE_40"),
        ("v2_known_failure_9", "V2_BOUNDARY_9"),
        ("legacy_validation", "LEGACY_VALIDATION_EVAL"),
        ("new_validation", "NEW_VALIDATION"),
        ("semantic_gap_validation", "SEMANTIC_GAP_VALIDATION"),
        ("recommended_validation", "RECOMMENDED_VALIDATION"),
        ("human_blind_core", "BLIND_CORE_GOLD"),
        ("asr_validation", "ASR_VALIDATION_QC"),
        ("asr_human_blind_core", "ASR_BLIND_QC"),
        ("blind_hard_clean", "BLIND_HARD_GOLD"),
        ("blind_hard_asr", "ASR_BLIND_QC"),
        ("semantic_gap_blind_clean", "SEMANTIC_GAP_BLIND_GOLD"),
        ("semantic_gap_blind_asr", "ASR_BLIND_QC"),
    ):
        entry = registry["entries"][key]
        evaluation[name] = {
            "registry_key": key,
            "path": str(CANONICAL / entry["canonical_path"]),
            "sha256": entry["sha256"],
            "expected_count": entry.get("expected_count"),
            "loaded_count": len(eval_datasets[name]),
        }
    evaluation_lines = "".join(
        f"| {name} | `{item['registry_key']}` | {item['loaded_count']} | `{item['sha256']}` |\n"
        for name, item in evaluation.items()
    )
    protocol_text = f"""# HUMAN_EXTENSION_D_PROTOCOL — EyeAI Intent

Dieses Protokoll wurde nach dem vollständigen Human-Gate und vor jedem D-Training erzeugt. A/B/C bleiben unverändert. Die Human-ASR-Quellen werden in D nicht verwendet.

## Kontrollierter Vergleich

`D = A + Human Clean Extension` — nicht `C + Human`.

| Eigenschaft | A-Comparator | D |
|---|---|---|
| V2 Clean | 2918 | 2918 |
| Human Clean Extension | 0 | {len(human)} |
| V2 Vosk | 1828 | 1828 |
| neue ASR-Daten | 0 | 0 |
| Development | 659 | 659 |
| Development SHA-256 | `{DEV_SHA256}` | `{DEV_SHA256}` |
| Seeds | {', '.join(map(str, SEEDS))} | {', '.join(map(str, SEEDS))} |
| Tokenizer | T1/T2 eingefroren | T1/T2 eingefroren |
| Architektur/Hyperparameter | eingefroren | identisch |

## Human-Gate

- Kandidaten vor Registry-Gate: `{summary['candidate_count']}`
- Nach Registry-Gate: `{summary['included_count']}`
- Ausgeschlossen: `{summary['excluded_count']}`
- Finaler Trainingsfile-Hash: `{summary['final_sha256']}`
- Gate-Tabelle: `{GATE_CSV}`
- Finaler Manifest: `{GATE_MANIFEST}`
- ASR-Quellen verwendet: `false`
- Gate-Ausschlussgründe: `{json.dumps(summary['reasons'], ensure_ascii=False, sort_keys=True)}`

## D-Trainingsmengen

| Strategie | Clean-Quelle | Vosk-Quelle | erwartete Trainingszeilen |
|---|---:|---:|---:|
| M0 | {len(d_clean)} | — | {len(d_clean)} |
| M1 | {len(d_clean)} | {len(base_vosk)} | {mix['target_mixed_samples']} |
| M2 | {len(d_clean)} | {len(base_vosk)} | {mix['target_mixed_samples']} |
| M3 | M0-D-Checkpoint | {len(base_vosk)} | {len(base_vosk)} |

M0 bleibt Clean-only. M2/M3 verwenden den innerhalb von D frisch erzeugten M0-D-Checkpoint. Die deterministische bestehende ungefähr 60/40-Mixing-/Upsampling-Logik bleibt unverändert.

## Architektur und Training

- BaselineCNN unverändert: `Embedding(32) → Conv1D(32,kernel=3) → GlobalMax + GlobalMean → Dense(32) → Dropout(0.15) → Softmax(10)`.
- T1 Word-Level und T2 BPE werden getrennt trainiert.
- Seeds: `{', '.join(map(str, SEEDS))}`.
- Maximal 30 Epochen, Patience 5, Batch Size 64, Adam, SparseCategoricalCrossentropy.
- Lernraten: M0/M1 `0.001`, M2/M3 `0.0001`.
- Kein Hyperparameter-Tuning, keine nachträgliche Datensatz-/Labeländerung.

## Frozen A comparator

- A-Artefakte geprüft: `{comparator['metric_files']}`.
- A-Development-Hash/Count: `{comparator['dev_sha256']}` / `{comparator['dev_count']}`.
- A-Trainingszahlen: `{json.dumps(comparator['training_counts'], sort_keys=True)}`.
- A wird nicht neu trainiert.

## Evaluation

Alle D-Modelle verwenden dieselben 14 Evaluationssets in derselben Reihenfolge wie A/B/C. `DATASET.val` bleibt Evaluation-only. Blind-/Challenge-Daten werden nicht für Training, Early Stopping oder Auswahl verwendet.

| Set | Registry-Key | Samples | SHA-256 |
|---|---|---:|---|
{evaluation_lines}

## Wissenschaftliche Sperre

Nach diesem Protokoll werden keine Beispiele, Labels, Splits, Hyperparameter oder Auswahlregeln anhand von D-Ergebnissen verändert. Es werden 40 D-Artefakte erzeugt: M0–M3 × T1/T2 × fünf Seeds.
"""
    protocol_path = STUDY / "HUMAN_EXTENSION_D_PROTOCOL.md"
    write_locked_text(protocol_path, protocol_text)
    lock = {
        "protocol_version": 1,
        "protocol_sha256": sha256(protocol_path),
        "human_gate_sha256": sha256(GATE_CSV),
        "human_manifest_sha256": sha256(GATE_MANIFEST),
        "human_final_sha256": summary["final_sha256"],
        "human_final_count": summary["final_count"],
        "dev_sha256": DEV_SHA256,
        "dev_count": DEV_COUNT,
        "registry_entry": REGISTRY_KEY,
        "asr_sources_used": False,
        "a_comparator_metric_files": comparator["metric_files"],
        "seeds": list(SEEDS),
    }
    write_locked_text(D_PROTOCOL / "human_extension_d_protocol_lock.json", json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    write_json(D_PROTOCOL / "human_extension_d_dataset_snapshot.json", {
        "human_gate": summary,
        "human_manifest": str(GATE_MANIFEST),
        "v2_clean_samples": len(base_clean),
        "human_samples": len(human),
        "v2_vosk_samples": len(base_vosk),
        "d_clean_samples": len(d_clean),
        "d_mixed_sampling": mix,
        "development": {"path": str(abc.resolve_file("V2_DEV_CORRECTED")), "sha256": DEV_SHA256, "sample_count": DEV_COUNT},
        "evaluation": evaluation,
        "labels": abc.LABELS,
        "no_human_asr_training": True,
    })
    return lock


def original_training_sources(_state: str, _approved_asr: list[Any]) -> tuple[list[Any], list[Any]]:
    clean: list[Any] = []
    for key, source in (
        ("V2_CLEAN_ORGANIC", "v2_clean_organic"),
        ("V2_CLEAN_MAIN", "v2_clean_main"),
        ("V2_HARD_NEGATIVE_PATCH", "v2_hard_negative_patch"),
    ):
        clean.extend(abc.load_labelled(abc.resolve_file(key), source, "clean"))
    vosk: list[Any] = []
    for key, source in (("V2_VOSK_GEN1", "v2_vosk_generation1"), ("V2_VOSK_GEN2", "v2_vosk_generation2")):
        vosk.extend(abc.load_labelled(abc.resolve_file(key), source, "vosk"))
    return clean, vosk


def verify_d_lock() -> dict[str, Any]:
    lock_path = D_PROTOCOL / "human_extension_d_protocol_lock.json"
    protocol_path = STUDY / "HUMAN_EXTENSION_D_PROTOCOL.md"
    if not lock_path.is_file() or not protocol_path.is_file():
        raise RuntimeError("D protocol is missing; run --prepare first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_registry()
    verify_existing_a_comparator()
    final = abc.resolve_file(REGISTRY_KEY)
    if sha256(final) != lock["human_final_sha256"] or len(abc.load_labelled(final, "human_extension_final", "human_clean")) != lock["human_final_count"]:
        raise RuntimeError("Human final file no longer matches the locked D protocol")
    if sha256(abc.resolve_file("V2_DEV_CORRECTED")) != lock["dev_sha256"]:
        raise RuntimeError("D development split changed")
    if sha256(protocol_path) != lock["protocol_sha256"]:
        raise RuntimeError("HUMAN_EXTENSION_D_PROTOCOL.md changed after lock")
    if sha256(GATE_CSV) != lock["human_gate_sha256"] or sha256(GATE_MANIFEST) != lock["human_manifest_sha256"]:
        raise RuntimeError("Human gate artifacts changed after lock")
    if lock.get("asr_sources_used") is not False:
        raise RuntimeError("D protocol unexpectedly enables human ASR")
    return lock


def train_d() -> None:
    verify_d_lock()
    abc.STATE_ROOTS["D"] = D_ROOT
    abc.STATE_NAMES["D"] = "Human Clean Extension"
    original = abc.training_sources
    final_path = abc.resolve_file(REGISTRY_KEY)

    def d_sources(state: str, _approved_asr: list[Any]) -> tuple[list[Any], list[Any]]:
        if state != "D":
            return original(state, _approved_asr)
        clean, vosk = original_training_sources("A", [])
        human = abc.load_labelled(final_path, "human_extension_final", "human_clean")
        return [*clean, *human], vosk

    abc.training_sources = d_sources
    try:
        tokenizers, _ = abc.load_tokenizers()
        abc.run_state("D", [], tokenizers)
    finally:
        abc.training_sources = original


def finalize_d_artifact_metadata() -> None:
    """Mark the copied model artifacts as experiment D."""

    root = D_ROOT
    metric_files = sorted((root / "metrics").glob("M*_T*_seed_*.json"))
    if len(metric_files) != 40:
        raise RuntimeError(f"Expected 40 D metric files, found {len(metric_files)}")
    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiment"] = "eyeai_intent_human_extension_d"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in sorted((root / "models").glob("M*/T*/Seed */metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiment"] = "eyeai_intent_human_extension_d"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for path in sorted((root / "models").glob("M*/T*/Seed */model_identity.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiment"] = "eyeai_intent_human_extension_d"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path = root / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"] = "eyeai_intent_human_extension_d"
    config["human_asr_training_samples"] = 0
    config["human_extension_registry_key"] = REGISTRY_KEY
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--finalize-metadata", action="store_true")
    args = parser.parse_args()
    selected = sum(bool(value) for value in (args.gate, args.promote, args.prepare, args.train, args.finalize_metadata))
    if selected != 1:
        parser.error("choose exactly one of --gate, --promote, --prepare or --train")
    if args.gate:
        print(json.dumps(gate_human_candidates(), ensure_ascii=False, indent=2))
    elif args.promote:
        if not GATE_SUMMARY.is_file():
            raise RuntimeError("Run --gate first")
        promote_registry_entry()
        update_manifest()
        verify_registry()
        print(json.dumps({"promoted": True, "registry_key": REGISTRY_KEY, "path": str(FINAL_ACTIVE_FILE)}, ensure_ascii=False, indent=2))
    elif args.prepare:
        print(json.dumps(prepare_d_protocol(), ensure_ascii=False, indent=2))
    elif args.finalize_metadata:
        finalize_d_artifact_metadata()
        print(json.dumps({"finalized": True, "state": "D", "models": 40}, ensure_ascii=False, indent=2))
    else:
        train_d()
        print(json.dumps({"trained": True, "state": "D", "models": 40}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
