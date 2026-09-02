#!/usr/bin/env python3
"""Train four small, TFLite-builtins-only intent classifier experiments.

Text normalization and tokenization deliberately happen outside the Keras graph.
The exported end-to-end models therefore accept fixed-shape int32 token IDs.  An
optional NPU-core export starts after the embedding lookup and accepts float32
embeddings instead.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.feature_extraction.text import TfidfVectorizer


LABELS = [
    "TEXT_RECOGNITION",
    "OBJECT_DETECTION",
    "CHANGE_SPEECH_SPEED",
    "CHANGE_SPEAKER",
    "REDIRECT_TO_LLM",
    "OPEN_SETTINGS",
    "SET_FREQUENCY",
    "SET_BPS",
    "MEASURE_DISTANCE",
    "ABORT",
]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

PAD_TOKEN = "[PAD]"
OOV_TOKEN = "[OOV]"
PAD_ID = 0
OOV_ID = 1
DEFAULT_SEED = 20260810
DEFAULT_MAX_LEN = 24
DEFAULT_MAX_VOCAB_SIZE = 6000
EMBEDDING_DIM = 32
MANUAL_VALIDATION_SAMPLES_PER_DOMAIN = 300
COMPLEX_VALIDATION_SAMPLES_PER_CLASS = 6
COMBINED_DEVELOPMENT_VALIDATION_SIZE = 660
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.90

MODEL_SPECS = {
    "M0_clean": "Core Clean only, random initialization",
    "M1_mixed_from_scratch": "60% Clean + 40% Vosk, shared random initialization",
    "M2_clean_then_mixed": "Best M0 checkpoint -> 60% Clean + 40% Vosk fine-tuning",
    "M3_clean_then_vosk": "Best M0 checkpoint -> Vosk-only fine-tuning",
}

CRITICAL_PAIRS = [
    ("SET_FREQUENCY", "SET_BPS"),
    ("OBJECT_DETECTION", "MEASURE_DISTANCE"),
    ("TEXT_RECOGNITION", "OBJECT_DETECTION"),
    ("CHANGE_SPEAKER", "CHANGE_SPEECH_SPEED"),
]

# Manually reviewed, semantically valid rows from the legacy DATASET.val.  The
# leakage audit below remains authoritative: training/manual-validation changes
# can still disqualify a curated row and cause a hard failure.
CURATED_COMPLEX_VALIDATION_LINES = {
    "TEXT_RECOGNITION": [69, 73, 78, 81, 82, 95],
    "OBJECT_DETECTION": [108, 119, 144, 181, 185, 207],
    "CHANGE_SPEECH_SPEED": [226, 228, 230, 239, 240, 253],
    "CHANGE_SPEAKER": [317, 324, 329, 330, 356, 363],
    "REDIRECT_TO_LLM": [416, 422, 428, 452, 480, 494],
    "OPEN_SETTINGS": [516, 523, 541, 581, 582, 596],
    "SET_FREQUENCY": [720, 725, 730, 740, 747, 758],
    "SET_BPS": [825, 829, 833, 837, 843, 861],
    "MEASURE_DISTANCE": [930, 931, 936, 938, 946, 980],
    "ABORT": [643, 659, 679, 689, 700, 705],
}

_TF: Any | None = None


class PipelineError(RuntimeError):
    """A user-actionable, hard pipeline failure."""


@dataclass
class Sample:
    source: str
    path: Path
    line_no: int
    text: str
    label: str
    domain: str
    normalized_text: str
    base_reference: str | None = None
    base_reference_kind: str = "text"
    speaker_id: str | None = None

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]

    @property
    def group_id(self) -> str:
        if self.base_reference:
            kind = self.base_reference_kind
            base = (
                normalize_identifier(self.base_reference)
                if kind == "id"
                else normalize_text(self.base_reference)
            )
        else:
            base = self.normalized_text
            kind = "text"
        return f"{kind}:{base} || {self.label}"

    @property
    def sample_id(self) -> str:
        payload = (
            f"{self.source}\x1f{self.line_no}\x1f{self.text}\x1f{self.label}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line_no}"


def normalize_text(text: str) -> str:
    """Apply the deterministic normalization mirrored by tokenizer_config.json."""

    value = unicodedata.normalize("NFKC", text).lower()
    normalized_chars = []
    for character in value:
        if character.isspace() or unicodedata.category(character).startswith("P"):
            normalized_chars.append(" ")
        else:
            normalized_chars.append(character)
    return " ".join("".join(normalized_chars).split())


def normalize_identifier(identifier: str) -> str:
    """Normalize metadata IDs without discarding identity-bearing punctuation."""

    return " ".join(unicodedata.normalize("NFKC", identifier).lower().split())


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))


def token_length_statistics(samples: Sequence[Sample]) -> dict[str, float | int]:
    lengths = [len(sample.normalized_text.split()) for sample in samples]
    return {
        "mean": float(np.mean(lengths)) if lengths else 0.0,
        "median": float(np.median(lengths)) if lengths else 0.0,
        "p90": _percentile(lengths, 90),
        "p95": _percentile(lengths, 95),
        "p99": _percentile(lengths, 99),
        "maximum": max(lengths, default=0),
    }


def load_dataset(path: Path, source: str, domain: str) -> list[Sample]:
    """Load text;LABEL records, splitting only at the final semicolon."""

    if not path.is_file():
        raise PipelineError(f"Dataset not found: {path}")

    samples: list[Sample] = []
    errors: list[str] = []
    ignored_blank_lines = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                ignored_blank_lines += 1
                continue
            if ";" not in line:
                errors.append(f"{path}:{line_no}: missing ';LABEL' delimiter")
                continue
            text, label = line.rsplit(";", 1)
            text = text.strip()
            label = label.strip()
            if not text:
                errors.append(f"{path}:{line_no}: empty text")
                continue
            if label not in LABEL_TO_ID:
                errors.append(f"{path}:{line_no}: unknown label {label!r}")
                continue
            normalized = normalize_text(text)
            if not normalized:
                errors.append(
                    f"{path}:{line_no}: text is empty after deterministic normalization"
                )
                continue
            samples.append(
                Sample(
                    source=source,
                    path=path,
                    line_no=line_no,
                    text=text,
                    label=label,
                    domain=domain,
                    normalized_text=normalized,
                )
            )

    if errors:
        raise PipelineError("Invalid dataset rows:\n" + "\n".join(errors))
    if not samples:
        raise PipelineError(f"Dataset contains no samples: {path}")
    if ignored_blank_lines:
        print(
            f"Ignored {ignored_blank_lines} blank separator line(s) in {path}"
        )
    return samples


def _detect_csv_dialect(path: Path) -> csv.Dialect:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        preview = handle.read(8192)
    try:
        return csv.Sniffer().sniff(preview, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def _read_dict_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    dialect = _detect_csv_dialect(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames:
            raise PipelineError(f"CSV has no header: {path}")
        rows = [
            {
                str(key).strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            for row in reader
        ]
        return rows, [field.strip().lower() for field in reader.fieldnames]


def _first_value(row: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


TRANSCRIPT_COLUMNS = (
    "text",
    "transcript",
    "vosk_text",
    "asr_text",
    "generated_text",
    "utterance",
)
BASE_TEXT_COLUMNS = ("base_text", "original_text", "clean_text", "source_text")
BASE_ID_COLUMNS = ("base_id", "original_id", "source_id")
SPEAKER_COLUMNS = ("speaker_id", "speaker", "participant_id")


def attach_metadata(samples: list[Sample], metadata_path: Path | None) -> None:
    """Attach base/speaker metadata by safe row alignment or transcript matching."""

    if metadata_path is None or not metadata_path.is_file():
        if metadata_path is not None:
            print(
                f"WARNING: metadata not found; grouping falls back to normalized "
                f"transcripts: {metadata_path}",
                file=sys.stderr,
            )
        return

    rows, fields = _read_dict_csv(metadata_path)
    if not rows:
        raise PipelineError(f"Metadata CSV contains no rows: {metadata_path}")

    has_transcript = any(column in fields for column in TRANSCRIPT_COLUMNS)
    aligned = len(rows) == len(samples)
    if aligned:
        for sample, row in zip(samples, rows):
            transcript = _first_value(row, TRANSCRIPT_COLUMNS)
            row_label = row.get("label", "")
            if (
                has_transcript
                and transcript
                and normalize_text(transcript) != sample.normalized_text
            ):
                aligned = False
                break
            if row_label and row_label != sample.label:
                aligned = False
                break

    mapped_rows: list[dict[str, str]] = []
    if aligned:
        mapped_rows = rows
    else:
        if not has_transcript:
            raise PipelineError(
                f"Metadata {metadata_path} is not row-aligned and has none of the "
                f"transcript columns {TRANSCRIPT_COLUMNS}."
            )
        lookup: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            transcript = _first_value(row, TRANSCRIPT_COLUMNS)
            label = row.get("label", "")
            if transcript:
                lookup[(normalize_text(transcript), label)].append(row)

        for sample in samples:
            exact = lookup.get((sample.normalized_text, sample.label), [])
            without_label = lookup.get((sample.normalized_text, ""), [])
            candidates = exact or without_label
            if len(candidates) != 1:
                raise PipelineError(
                    f"Cannot map {sample.location} uniquely to metadata "
                    f"{metadata_path}; found {len(candidates)} matching rows."
                )
            mapped_rows.append(candidates[0])

    missing_base = 0
    for sample, row in zip(samples, mapped_rows):
        base_text = _first_value(row, BASE_TEXT_COLUMNS)
        base_id = _first_value(row, BASE_ID_COLUMNS)
        if base_text:
            sample.base_reference = base_text
            sample.base_reference_kind = "text"
        elif base_id:
            sample.base_reference = base_id
            sample.base_reference_kind = "id"
        else:
            missing_base += 1
        speaker_id = _first_value(row, SPEAKER_COLUMNS)
        if speaker_id:
            sample.speaker_id = speaker_id

    if missing_base:
        print(
            f"WARNING: {missing_base}/{len(samples)} rows in {metadata_path} have no "
            "base_text/base_id; those rows use normalized transcript grouping.",
            file=sys.stderr,
        )
    else:
        print(f"Metadata attached to all {len(samples)} rows from {metadata_path}")


def load_human_dataset(path: Path, source: str) -> list[Sample]:
    """Load human evaluation data from text;LABEL or a headed CSV."""

    if not path.is_file():
        raise PipelineError(f"Human evaluation dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        return load_dataset(path, source=source, domain="human")

    rows, _ = _read_dict_csv(path)
    errors: list[str] = []
    samples: list[Sample] = []
    for line_no, row in enumerate(rows, start=2):
        text = _first_value(row, TRANSCRIPT_COLUMNS)
        label = row.get("label", "")
        if not text:
            errors.append(f"{path}:{line_no}: missing text/transcript")
            continue
        if label not in LABEL_TO_ID:
            errors.append(f"{path}:{line_no}: unknown label {label!r}")
            continue
        normalized = normalize_text(text)
        if not normalized:
            errors.append(f"{path}:{line_no}: empty after normalization")
            continue
        samples.append(
            Sample(
                source=source,
                path=path,
                line_no=line_no,
                text=text,
                label=label,
                domain="human",
                normalized_text=normalized,
                speaker_id=_first_value(row, SPEAKER_COLUMNS) or None,
            )
        )
    if errors:
        raise PipelineError("Invalid human evaluation rows:\n" + "\n".join(errors))
    if not samples:
        raise PipelineError(f"Human evaluation dataset contains no samples: {path}")
    return samples


def _duplicate_row_count(groups: Iterable[Sequence[Sample]]) -> int:
    return sum(max(0, len(group) - 1) for group in groups)


def validate_dataset(samples: Sequence[Sample], report_name: str) -> dict[str, Any]:
    """Report consistency statistics and hard-fail normalized label conflicts."""

    exact_groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    normalized_groups: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    normalized_text_groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        exact_groups[(sample.text, sample.label)].append(sample)
        normalized_groups[(sample.normalized_text, sample.label)].append(sample)
        normalized_text_groups[sample.normalized_text].append(sample)

    conflicts = [
        group
        for group in normalized_text_groups.values()
        if len({sample.label for sample in group}) > 1
    ]
    source_counts = Counter(sample.source for sample in samples)
    class_counts = Counter(sample.label for sample in samples)
    length_stats = token_length_statistics(samples)
    report = {
        "name": report_name,
        "sample_count": len(samples),
        "samples_per_source": dict(sorted(source_counts.items())),
        "samples_per_class": {label: class_counts[label] for label in LABELS},
        "duplicates": {
            "exact_duplicate_rows": _duplicate_row_count(exact_groups.values()),
            "exact_duplicate_groups": sum(
                len(group) > 1 for group in exact_groups.values()
            ),
            "normalized_duplicate_rows": _duplicate_row_count(
                normalized_groups.values()
            ),
            "normalized_duplicate_groups": sum(
                len(group) > 1 for group in normalized_groups.values()
            ),
            "cross_label_conflict_groups": len(conflicts),
        },
        "token_lengths": length_stats,
    }

    print(f"\n=== Data validation: {report_name} ===")
    print(f"Samples: {len(samples)}")
    print("Per source: " + ", ".join(f"{k}={v}" for k, v in source_counts.items()))
    print("Per class: " + ", ".join(f"{label}={class_counts[label]}" for label in LABELS))
    print(
        "Duplicates: "
        f"exact rows={report['duplicates']['exact_duplicate_rows']}, "
        f"normalized rows={report['duplicates']['normalized_duplicate_rows']}, "
        f"cross-label groups={len(conflicts)}"
    )
    print(
        "Token lengths: "
        + ", ".join(
            f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in length_stats.items()
        )
    )

    if conflicts:
        details = []
        for group in conflicts:
            details.append(f"normalized text: {group[0].normalized_text!r}")
            details.extend(
                f"  {sample.location}: {sample.text!r};{sample.label}"
                for sample in group
            )
        raise PipelineError(
            "Cross-label conflicts detected (automatic semantic cleanup is disabled):\n"
            + "\n".join(details)
        )
    return report


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def write_training_manifest(
    samples: Sequence[Sample], manifest_path: Path, recreate: bool
) -> None:
    """Record that every selected Clean/Vosk source row is used for training."""

    expected = {sample.sample_id: sample for sample in samples}
    if manifest_path.exists() and not recreate:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        existing = {row.get("sample_id", ""): row for row in rows}
        if set(existing) != set(expected):
            missing = len(set(expected) - set(existing))
            extra = len(set(existing) - set(expected))
            raise PipelineError(
                f"Existing training manifest does not match current inputs "
                f"(missing={missing}, extra={extra}). Use --recreate-manifests or "
                "a new output directory."
            )
        for sample_id, sample in expected.items():
            row = existing[sample_id]
            if (
                row.get("label") != sample.label
                or row.get("domain") != sample.domain
                or row.get("group_id") != sample.group_id
                or row.get("split") != "train"
            ):
                raise PipelineError(
                    f"Existing training manifest differs for {sample.location}. "
                    "Use --recreate-manifests or a new output directory."
                )
        print(f"Reused all-training manifest: {manifest_path}")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "source",
        "source_file",
        "source_line",
        "domain",
        "split",
        "label",
        "group_id",
        "normalized_text",
        "text",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in sorted(samples, key=lambda item: (item.source, item.line_no)):
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "source": sample.source,
                    "source_file": sample.path.name,
                    "source_line": sample.line_no,
                    "domain": sample.domain,
                    "split": "train",
                    "label": sample.label,
                    "group_id": sample.group_id,
                    "normalized_text": sample.normalized_text,
                    "text": sample.text,
                }
            )
    print(f"Created all-training manifest with {len(samples)} rows: {manifest_path}")


def _nearest_reference_matches(
    candidates: Sequence[Sample], references: Sequence[Sample]
) -> list[tuple[float, Sample]]:
    """Find deterministic char-ngram TF-IDF nearest neighbors."""

    if not candidates or not references:
        raise PipelineError("Near-duplicate analysis requires non-empty datasets")
    reference_texts = [sample.normalized_text for sample in references]
    candidate_texts = [sample.normalized_text for sample in candidates]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        norm="l2",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(reference_texts + candidate_texts)
    reference_matrix = matrix[: len(references)]
    candidate_matrix = matrix[len(references) :]
    similarities = candidate_matrix @ reference_matrix.T
    maximum_scores = np.asarray(similarities.max(axis=1).toarray()).ravel()
    nearest_indices = np.asarray(similarities.argmax(axis=1)).ravel()
    return [
        (float(maximum_scores[index]), references[int(nearest_indices[index])])
        for index in range(len(candidates))
    ]


def audit_manual_validation_leakage(
    manual_sets: dict[str, Sequence[Sample]],
    training_samples: Sequence[Sample],
    report_path: Path,
) -> dict[str, Any]:
    """Hard-fail exact leakage and report near-neighbor similarity."""

    combined = [sample for samples in manual_sets.values() for sample in samples]
    normalized_counts = Counter(sample.normalized_text for sample in combined)
    duplicate_texts = {text for text, count in normalized_counts.items() if count > 1}
    if duplicate_texts:
        details = [
            f"{sample.location}: {sample.text!r};{sample.label}"
            for sample in combined
            if sample.normalized_text in duplicate_texts
        ]
        raise PipelineError(
            "Manual validation sets contain normalized duplicate texts:\n"
            + "\n".join(details)
        )

    matches = _nearest_reference_matches(combined, training_samples)
    exact_leaks = [
        (sample, nearest)
        for sample, (score, nearest) in zip(combined, matches)
        if sample.normalized_text == nearest.normalized_text or score >= 1.0 - 1e-12
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "validation_set",
        "sample_id",
        "source_file",
        "source_line",
        "label",
        "text",
        "nearest_training_source",
        "nearest_training_line",
        "nearest_training_label",
        "nearest_training_text",
        "char_ngram_tfidf_similarity",
    ]
    sample_to_set = {
        sample.sample_id: name for name, samples in manual_sets.items() for sample in samples
    }
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample, (score, nearest) in zip(combined, matches):
            writer.writerow(
                {
                    "validation_set": sample_to_set[sample.sample_id],
                    "sample_id": sample.sample_id,
                    "source_file": sample.path.name,
                    "source_line": sample.line_no,
                    "label": sample.label,
                    "text": sample.text,
                    "nearest_training_source": nearest.source,
                    "nearest_training_line": nearest.line_no,
                    "nearest_training_label": nearest.label,
                    "nearest_training_text": nearest.text,
                    "char_ngram_tfidf_similarity": score,
                }
            )
    if exact_leaks:
        details = [
            f"{sample.location} duplicates {nearest.location}: {sample.text!r}"
            for sample, nearest in exact_leaks
        ]
        raise PipelineError(
            "Exact normalized overlap between manual validation and training:\n"
            + "\n".join(details)
        )

    scores = np.asarray([score for score, _ in matches], dtype=np.float64)
    summary = {
        "sample_count": len(combined),
        "exact_training_overlaps": 0,
        "similarity_method": "character TF-IDF cosine, char_wb ngrams 3-5",
        "similarity_p50": float(np.percentile(scores, 50)),
        "similarity_p90": float(np.percentile(scores, 90)),
        "similarity_p95": float(np.percentile(scores, 95)),
        "similarity_p99": float(np.percentile(scores, 99)),
        "maximum_similarity": float(np.max(scores)),
    }
    print(
        "Manual validation leakage audit: 0 exact overlaps with training; "
        f"maximum near-neighbor similarity={summary['maximum_similarity']:.4f}"
    )
    return summary


def select_complex_validation(
    pool: Sequence[Sample],
    training_samples: Sequence[Sample],
    manual_validation: Sequence[Sample],
    *,
    samples_per_class: int,
    near_duplicate_threshold: float,
    manifest_path: Path,
    audit_path: Path,
    recreate: bool,
) -> tuple[list[Sample], dict[str, Any]]:
    """Select a fixed, balanced, structurally complex and leakage-safe subset."""

    training_matches = _nearest_reference_matches(pool, training_samples)
    manual_matches = _nearest_reference_matches(pool, manual_validation)
    manual_normalized = {sample.normalized_text for sample in manual_validation}
    rows: list[dict[str, Any]] = []
    for sample, (training_score, nearest_training), (
        manual_score,
        nearest_manual,
    ) in zip(pool, training_matches, manual_matches):
        exact_training = sample.normalized_text == nearest_training.normalized_text
        exact_manual = sample.normalized_text in manual_normalized
        eligible = (
            not exact_training
            and not exact_manual
            and training_score < near_duplicate_threshold
            and manual_score < near_duplicate_threshold
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source_line": sample.line_no,
                "label": sample.label,
                "text": sample.text,
                "token_length": len(sample.normalized_text.split()),
                "nearest_training_text": nearest_training.text,
                "nearest_training_label": nearest_training.label,
                "training_similarity": training_score,
                "nearest_manual_validation_text": nearest_manual.text,
                "manual_validation_similarity": manual_score,
                "exact_training_overlap": exact_training,
                "exact_manual_validation_overlap": exact_manual,
                "eligible": eligible,
                "selected": False,
            }
        )

    if samples_per_class != COMPLEX_VALIDATION_SAMPLES_PER_CLASS:
        raise PipelineError(
            "The reviewed complex-validation manifest is fixed at "
            f"{COMPLEX_VALIDATION_SAMPLES_PER_CLASS} samples per class; received "
            f"{samples_per_class}."
        )

    pool_by_line: dict[int, Sample] = {}
    for sample in pool:
        if sample.line_no in pool_by_line:
            raise PipelineError(
                f"Duplicate physical source line {sample.line_no} in complex pool."
            )
        pool_by_line[sample.line_no] = sample
    audit_by_id = {row["sample_id"]: row for row in rows}

    selected: list[Sample] = []
    selected_normalized: dict[str, Sample] = {}
    for label in LABELS:
        curated_lines = CURATED_COMPLEX_VALIDATION_LINES.get(label, [])
        if len(curated_lines) != samples_per_class or len(set(curated_lines)) != len(
            curated_lines
        ):
            raise PipelineError(
                f"Curated complex-validation lines for {label} must contain exactly "
                f"{samples_per_class} unique entries; found {curated_lines}."
            )
        for line_no in curated_lines:
            sample = pool_by_line.get(line_no)
            if sample is None:
                raise PipelineError(
                    f"Curated complex-validation line {line_no} for {label} no longer "
                    "exists in the source dataset."
                )
            if sample.label != label:
                raise PipelineError(
                    f"Curated complex-validation line {line_no} is labelled "
                    f"{sample.label}, expected {label}."
                )
            audit_row = audit_by_id[sample.sample_id]
            if not bool(audit_row["eligible"]):
                raise PipelineError(
                    f"Curated complex-validation line {line_no} ({label}) failed the "
                    "current leakage audit: "
                    f"exact_training={audit_row['exact_training_overlap']}, "
                    f"training_similarity={audit_row['training_similarity']:.4f}, "
                    f"exact_manual={audit_row['exact_manual_validation_overlap']}, "
                    f"manual_similarity={audit_row['manual_validation_similarity']:.4f}."
                )
            duplicate = selected_normalized.get(sample.normalized_text)
            if duplicate is not None:
                raise PipelineError(
                    f"Curated complex-validation lines {duplicate.line_no} and "
                    f"{sample.line_no} normalize to the same text."
                )
            selected.append(sample)
            selected_normalized[sample.normalized_text] = sample

    selected_ids = {sample.sample_id for sample in selected}
    for row in rows:
        row["selected"] = row["sample_id"] in selected_ids

    if manifest_path.exists() and not recreate:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            existing_ids = {
                row.get("sample_id", "") for row in csv.DictReader(handle)
            }
        if existing_ids != selected_ids:
            raise PipelineError(
                "Existing complex-validation selection differs from the current "
                "leakage audit. Use --recreate-manifests or a new output directory."
            )
        print(f"Reused fixed complex-validation manifest: {manifest_path}")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_fields = [
            "sample_id",
            "source_file",
            "source_line",
            "label",
            "text",
            "token_length",
            "training_similarity",
            "manual_validation_similarity",
        ]
        row_by_id = {row["sample_id"]: row for row in rows}
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            for sample in sorted(selected, key=lambda item: (item.label_id, item.line_no)):
                audit_row = row_by_id[sample.sample_id]
                writer.writerow(
                    {
                        "sample_id": sample.sample_id,
                        "source_file": sample.path.name,
                        "source_line": sample.line_no,
                        "label": sample.label,
                        "text": sample.text,
                        "token_length": audit_row["token_length"],
                        "training_similarity": audit_row["training_similarity"],
                        "manual_validation_similarity": audit_row[
                            "manual_validation_similarity"
                        ],
                    }
                )
        print(f"Created fixed complex-validation manifest: {manifest_path}")

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_fields = list(rows[0])
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "pool_samples": len(pool),
        "selected_samples": len(selected),
        "samples_per_class": samples_per_class,
        "selection_method": "fixed manually reviewed source lines after leakage audit",
        "near_duplicate_threshold": near_duplicate_threshold,
        "similarity_method": "character TF-IDF cosine, char_wb ngrams 3-5",
        "exact_training_overlaps_excluded": sum(
            bool(row["exact_training_overlap"]) for row in rows
        ),
        "near_training_duplicates_excluded": sum(
            not bool(row["exact_training_overlap"])
            and float(row["training_similarity"]) >= near_duplicate_threshold
            for row in rows
        ),
        "exact_manual_validation_overlaps_excluded": sum(
            bool(row["exact_manual_validation_overlap"]) for row in rows
        ),
        "near_manual_validation_duplicates_excluded": sum(
            not bool(row["exact_manual_validation_overlap"])
            and float(row["manual_validation_similarity"]) >= near_duplicate_threshold
            for row in rows
        ),
        "selected_mean_token_length": float(
            np.mean([len(sample.normalized_text.split()) for sample in selected])
        ),
    }
    print(
        f"Complex validation: selected {len(selected)} balanced samples; "
        f"excluded {summary['exact_training_overlaps_excluded']} exact and "
        f"{summary['near_training_duplicates_excluded']} near training overlaps."
    )
    return selected, summary


def build_vocabulary(
    training_samples: Sequence[Sample], max_vocab_size: int
) -> tuple[list[str], dict[str, int]]:
    """Build one deterministic vocabulary for M0-M3 from training samples only."""

    if max_vocab_size < 2:
        raise PipelineError("--max-vocab-size must be at least 2")
    token_counts = Counter(
        token
        for sample in training_samples
        for token in sample.normalized_text.split()
    )
    ordered_tokens = sorted(token_counts, key=lambda token: (-token_counts[token], token))
    vocabulary = [PAD_TOKEN, OOV_TOKEN] + ordered_tokens[: max_vocab_size - 2]
    return vocabulary, {token: index for index, token in enumerate(vocabulary)}


def encode_text(text: str, token_to_id: dict[str, int], max_len: int) -> np.ndarray:
    ids = [token_to_id.get(token, OOV_ID) for token in normalize_text(text).split()]
    encoded = np.full(max_len, PAD_ID, dtype=np.int32)
    kept = ids[:max_len]
    encoded[: len(kept)] = kept
    return encoded


def encode_samples(
    samples: Sequence[Sample], token_to_id: dict[str, int], max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack(
        [encode_text(sample.text, token_to_id, max_len) for sample in samples]
    ).astype(np.int32, copy=False)
    labels = np.asarray([sample.label_id for sample in samples], dtype=np.int32)
    return features, labels


def oov_statistics(
    samples: Sequence[Sample], token_to_id: dict[str, int]
) -> dict[str, int | float]:
    tokens = [
        token for sample in samples for token in sample.normalized_text.split()
    ]
    oov_count = sum(token not in token_to_id for token in tokens)
    return {
        "tokens": len(tokens),
        "oov_tokens": oov_count,
        "oov_rate": (oov_count / len(tokens)) if tokens else 0.0,
    }


def write_tokenizer_artifacts(
    output_dir: Path,
    vocabulary: Sequence[str],
    max_len: int,
    max_vocab_size: int,
) -> None:
    tokenizer_dir = output_dir / "tokenizer"
    _write_json(tokenizer_dir / "vocab.json", list(vocabulary))
    _write_json(tokenizer_dir / "labels.json", LABELS)
    _write_json(
        tokenizer_dir / "tokenizer_config.json",
        {
            "version": 1,
            "tokenizer_type": "deterministic_word_level",
            "normalization_steps_in_order": [
                "Unicode NFKC",
                "Unicode lowercase (Kotlin: lowercase(Locale.ROOT))",
                "replace every Unicode punctuation category P* with U+0020 SPACE",
                "replace every Unicode whitespace character with U+0020 SPACE",
                "collapse whitespace and trim",
            ],
            "unicode_punctuation_categories": [
                "Pc",
                "Pd",
                "Pe",
                "Pf",
                "Pi",
                "Po",
                "Ps",
            ],
            "kotlin_normalization_primitives": {
                "nfkc": "java.text.Normalizer.normalize(text, Normalizer.Form.NFKC)",
                "lowercase": "lowercase(Locale.ROOT)",
                "punctuation": "Character.getType(codePoint) in Pc/Pd/Pe/Pf/Pi/Po/Ps",
                "whitespace": "Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint)",
            },
            "preserved_examples": ["ä", "ö", "ü", "ß", "nicht", "nur", "doch", "sondern", "aber", "statt"],
            "split": "normalized whitespace",
            "padding": "post",
            "truncating": "post",
            "max_length": max_len,
            "reserved_tokens": {
                "PAD": {"token": PAD_TOKEN, "id": PAD_ID},
                "OOV": {"token": OOV_TOKEN, "id": OOV_ID},
            },
            "vocab_format": "JSON array; array index is the token ID",
            "vocab_order": "frequency descending, then Unicode code-point ascending",
            "configured_max_vocab_size": max_vocab_size,
            "actual_vocab_size": len(vocabulary),
            "model_input_dtype": "int32",
            "model_input_shape": [1, max_len],
            "label_order": "labels.json array index is the output class ID",
        },
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def require_tensorflow() -> Any:
    global _TF
    if _TF is None:
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as error:
            raise PipelineError(
                "TensorFlow is required for training/export. Install a TensorFlow "
                "version that provides tf.lite.TFLiteConverter, or use --prepare-only."
            ) from error
        _TF = tf
    return _TF


def set_global_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as error:
        print(
            f"WARNING: TensorFlow deterministic ops could not be enabled: {error}",
            file=sys.stderr,
        )


def build_model(
    vocab_size: int,
    max_len: int,
    *,
    batch_size: int | None = None,
    name: str = "intent_classifier",
) -> Any:
    """Build the required numerical CNN classifier without masking/text ops."""

    tf = require_tensorflow()
    tokens = tf.keras.Input(
        shape=(max_len,),
        batch_size=batch_size,
        dtype=tf.int32,
        name="tokens",
    )
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=False,
        name="embedding",
    )(tokens)
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max")(x)
    mean_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_mean")(x)
    x = tf.keras.layers.Concatenate(name="pool_concat")([max_pool, mean_pool])
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(0.15, name="dropout")(x)
    probabilities = tf.keras.layers.Dense(
        len(LABELS), activation="softmax", name="intent_probabilities"
    )(x)
    return tf.keras.Model(tokens, probabilities, name=name)


def build_fixed_batch_inference_model(
    trained_model: Any, vocab_size: int, max_len: int
) -> Any:
    inference_model = build_model(
        vocab_size,
        max_len,
        batch_size=1,
        name=f"{trained_model.name}_fixed_batch",
    )
    inference_model.set_weights(trained_model.get_weights())
    return inference_model


def build_npu_core_model(trained_model: Any, max_len: int) -> Any:
    """Build the optional float32 graph beginning after embedding lookup."""

    tf = require_tensorflow()
    embeddings = tf.keras.Input(
        shape=(max_len, EMBEDDING_DIM),
        batch_size=1,
        dtype=tf.float32,
        name="embedded_tokens",
    )
    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(embeddings)
    max_pool = tf.keras.layers.GlobalMaxPooling1D(name="global_max")(x)
    mean_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_mean")(x)
    x = tf.keras.layers.Concatenate(name="pool_concat")([max_pool, mean_pool])
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_hidden")(x)
    probabilities = tf.keras.layers.Dense(
        len(LABELS), activation="softmax", name="intent_probabilities"
    )(x)
    core_model = tf.keras.Model(embeddings, probabilities, name="intent_npu_core")
    for layer_name in ("conv1", "dense_hidden", "intent_probabilities"):
        core_model.get_layer(layer_name).set_weights(
            trained_model.get_layer(layer_name).get_weights()
        )
    return core_model


def _make_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
    training: bool,
) -> Any:
    tf = require_tensorflow()
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(features), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size, drop_remainder=False).prefetch(1)


def make_domain_mix(
    clean_features: np.ndarray,
    clean_labels: np.ndarray,
    vosk_features: np.ndarray,
    vosk_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Include every row and upsample deterministically to approximately 60:40."""

    clean_count = len(clean_features)
    vosk_count = len(vosk_features)
    if not clean_count or not vosk_count:
        raise PipelineError("Mixed-domain training requires non-empty Clean and Vosk data")

    scale = max(clean_count / 0.60, vosk_count / 0.40)
    target_clean = max(clean_count, int(math.ceil(scale * 0.60)))
    target_vosk = max(vosk_count, int(math.ceil(scale * 0.40)))
    rng = np.random.default_rng(seed)

    def expanded_indices(length: int, target: int) -> np.ndarray:
        base = np.arange(length, dtype=np.int64)
        if target > length:
            base = np.concatenate(
                [base, rng.choice(length, size=target - length, replace=True)]
            )
        return base

    clean_indices = expanded_indices(clean_count, target_clean)
    vosk_indices = expanded_indices(vosk_count, target_vosk)
    features = np.concatenate(
        [clean_features[clean_indices], vosk_features[vosk_indices]], axis=0
    )
    labels = np.concatenate(
        [clean_labels[clean_indices], vosk_labels[vosk_indices]], axis=0
    )
    permutation = rng.permutation(len(features))
    features = features[permutation]
    labels = labels[permutation]
    clean_fraction = target_clean / (target_clean + target_vosk)
    stats: dict[str, float | int] = {
        "clean_source_samples": clean_count,
        "vosk_source_samples": vosk_count,
        "clean_sampled": target_clean,
        "vosk_sampled": target_vosk,
        "clean_fraction": clean_fraction,
        "vosk_fraction": 1.0 - clean_fraction,
    }
    print(
        "Mixed-domain sample: "
        f"Clean={target_clean} ({clean_fraction:.2%}), "
        f"Vosk={target_vosk} ({1.0 - clean_fraction:.2%})"
    )
    return features, labels, stats


def _compile_model(model: Any, learning_rate: float) -> None:
    tf = require_tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def _train_phase(
    model: Any,
    model_dir: Path,
    train_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    *,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    verbose: int,
) -> Any:
    tf = require_tensorflow()
    model_dir.mkdir(parents=True, exist_ok=True)
    _compile_model(model, learning_rate)
    checkpoint_path = model_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1 if verbose else 0,
        ),
        tf.keras.callbacks.CSVLogger(str(model_dir / "history.csv"), append=False),
    ]
    train_dataset = _make_tf_dataset(
        train_data[0], train_data[1], batch_size, seed, training=True
    )
    validation_dataset = _make_tf_dataset(
        validation_data[0], validation_data[1], batch_size, seed, training=False
    )
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    if not checkpoint_path.is_file():
        raise PipelineError(f"Best checkpoint was not written: {checkpoint_path}")
    # ModelCheckpoint and EarlyStopping should agree; explicitly reloading makes
    # the provenance of M2/M3 unambiguous.
    model.load_weights(checkpoint_path)
    model.save(model_dir / "model.keras")
    return model


def train_m0(
    model: Any,
    model_dir: Path,
    clean_train: tuple[np.ndarray, np.ndarray],
    development_val: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> Any:
    return _train_phase(
        model,
        model_dir,
        clean_train,
        development_val,
        learning_rate=1e-3,
        epochs=args.epochs_m0,
        patience=args.patience_m0,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=args.verbose,
    )


def train_m1(
    model: Any,
    model_dir: Path,
    mixed_train: tuple[np.ndarray, np.ndarray],
    development_val: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> Any:
    return _train_phase(
        model,
        model_dir,
        mixed_train,
        development_val,
        learning_rate=1e-3,
        epochs=args.epochs_m1,
        patience=args.patience_m1,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        verbose=args.verbose,
    )


def train_m2(
    model: Any,
    model_dir: Path,
    mixed_train: tuple[np.ndarray, np.ndarray],
    development_val: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> Any:
    return _train_phase(
        model,
        model_dir,
        mixed_train,
        development_val,
        learning_rate=1e-4,
        epochs=args.epochs_m2,
        patience=args.patience_finetune,
        batch_size=args.batch_size,
        seed=args.seed + 2,
        verbose=args.verbose,
    )


def train_m3(
    model: Any,
    model_dir: Path,
    vosk_train: tuple[np.ndarray, np.ndarray],
    development_val: tuple[np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> Any:
    return _train_phase(
        model,
        model_dir,
        vosk_train,
        development_val,
        learning_rate=1e-4,
        epochs=args.epochs_m3,
        patience=args.patience_finetune,
        batch_size=args.batch_size,
        seed=args.seed + 3,
        verbose=args.verbose,
    )


def _critical_confusions(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first, second in CRITICAL_PAIRS:
        for actual, predicted in ((first, second), (second, first)):
            actual_id = LABEL_TO_ID[actual]
            predicted_id = LABEL_TO_ID[predicted]
            support = int(matrix[actual_id].sum())
            count = int(matrix[actual_id, predicted_id])
            rows.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "actual_support": support,
                    "rate": count / support if support else 0.0,
                }
            )

    abort_id = LABEL_TO_ID["ABORT"]
    abort_support = int(matrix[abort_id].sum())
    abort_misses = int(abort_support - matrix[abort_id, abort_id])
    rows.append(
        {
            "actual": "ABORT",
            "predicted": "ANY_OTHER",
            "count": abort_misses,
            "actual_support": abort_support,
            "rate": abort_misses / abort_support if abort_support else 0.0,
        }
    )
    for predicted_id, predicted in enumerate(LABELS):
        if predicted == "ABORT":
            continue
        count = int(matrix[abort_id, predicted_id])
        rows.append(
            {
                "actual": "ABORT",
                "predicted": predicted,
                "count": count,
                "actual_support": abort_support,
                "rate": count / abort_support if abort_support else 0.0,
            }
        )
    return rows


def evaluate_model(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    if not len(features):
        raise PipelineError("Cannot evaluate an empty dataset")
    probabilities = np.asarray(
        model.predict(features, batch_size=batch_size, verbose=0), dtype=np.float32
    )
    predictions = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(LABELS)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=np.arange(len(LABELS)),
        zero_division=0,
    )
    clipped = np.clip(
        probabilities[np.arange(len(labels)), labels], 1e-7, 1.0
    )
    metrics = {
        "sample_count": len(labels),
        "loss": float(-np.mean(np.log(clipped))),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(len(LABELS)),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "critical_confusions": _critical_confusions(matrix),
    }
    return metrics, probabilities


def _write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])


def write_evaluation_artifacts(
    model_dir: Path, evaluations: dict[str, dict[str, Any]]
) -> None:
    # The unsuffixed matrix is the combined external development-validation view.
    primary_name = "combined_val" if "combined_val" in evaluations else next(iter(evaluations))
    _write_confusion_matrix(
        model_dir / "confusion_matrix.csv",
        evaluations[primary_name]["confusion_matrix"],
    )
    for dataset_name, metrics in evaluations.items():
        _write_confusion_matrix(
            model_dir / f"confusion_matrix_{dataset_name}.csv",
            metrics["confusion_matrix"],
        )

    with (model_dir / "per_class_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["dataset", "label", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_name, metrics in evaluations.items():
            for label in LABELS:
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "label": label,
                        **metrics["per_class"][label],
                    }
                )

    with (model_dir / "critical_confusions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "dataset",
            "actual",
            "predicted",
            "count",
            "actual_support",
            "rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_name, metrics in evaluations.items():
            for row in metrics["critical_confusions"]:
                writer.writerow({"dataset": dataset_name, **row})


def convert_builtin_tflite(inference_model: Any, output_path: Path) -> bytes:
    """Convert with builtins only; never fall back to Select TF/Flex ops."""

    tf = require_tensorflow()
    converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = False
    try:
        model_bytes = converter.convert()
    except Exception as error:  # TensorFlow raises several converter error types.
        raise PipelineError(
            f"Builtins-only TFLite conversion failed for {output_path}; no Select "
            f"TF Ops fallback was attempted: {error}"
        ) from error
    output_path.write_bytes(model_bytes)
    return model_bytes


def _analyze_tflite(model_path: Path) -> str:
    tf = require_tensorflow()
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            result = tf.lite.experimental.Analyzer.analyze(
                model_path=str(model_path), gpu_compatibility=False
            )
            if result:
                capture.write(str(result))
    except Exception as error:
        raise PipelineError(f"TFLite Analyzer failed for {model_path}: {error}") from error
    analyzer_text = capture.getvalue()
    forbidden = []
    if re.search(r"\bFlex[A-Za-z0-9_]*\b", analyzer_text):
        forbidden.append("Flex*")
    if "SELECT_TF_OPS" in analyzer_text:
        forbidden.append("SELECT_TF_OPS")
    if re.search(r"\bCUSTOM\b", analyzer_text):
        forbidden.append("CUSTOM")
    if forbidden:
        raise PipelineError(
            f"Forbidden operators found in {model_path}: {', '.join(forbidden)}"
        )
    return analyzer_text


def _tflite_predict(model_path: Path, features: np.ndarray) -> np.ndarray:
    tf = require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise PipelineError(
            f"Expected one input and one output tensor in {model_path}, got "
            f"{len(input_details)} and {len(output_details)}"
        )
    expected_shape = tuple(int(value) for value in features.shape[1:])
    actual_shape = tuple(int(value) for value in input_details[0]["shape"])
    if actual_shape != (1, *expected_shape):
        raise PipelineError(
            f"Unexpected fixed input shape in {model_path}: {actual_shape}; "
            f"expected {(1, *expected_shape)}"
        )
    if input_details[0]["dtype"] != features.dtype:
        raise PipelineError(
            f"Unexpected input dtype in {model_path}: {input_details[0]['dtype']}; "
            f"expected {features.dtype}"
        )
    output_shape = tuple(int(value) for value in output_details[0]["shape"])
    if output_shape != (1, len(LABELS)) or output_details[0]["dtype"] != np.float32:
        raise PipelineError(
            f"Unexpected output tensor in {model_path}: shape={output_shape}, "
            f"dtype={output_details[0]['dtype']}; expected {(1, len(LABELS))} "
            "float32"
        )

    outputs = np.empty((len(features), len(LABELS)), dtype=np.float32)
    for index, row in enumerate(features):
        interpreter.set_tensor(input_details[0]["index"], row[np.newaxis, ...])
        interpreter.invoke()
        outputs[index] = interpreter.get_tensor(output_details[0]["index"])[0]
    return outputs


def _parity_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "top1_agreement": float(
            np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
        ),
        "mean_absolute_probability_difference": float(np.mean(difference)),
        "max_absolute_probability_difference": float(np.max(difference)),
    }


def verify_tflite(
    trained_model: Any,
    model_path: Path,
    verification_features: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    keras_probabilities = np.asarray(
        trained_model.predict(verification_features, verbose=0), dtype=np.float32
    )
    tflite_probabilities = _tflite_predict(model_path, verification_features)
    metrics = _parity_metrics(keras_probabilities, tflite_probabilities)
    if (
        metrics["top1_agreement"] != 1.0
        or metrics["max_absolute_probability_difference"] > tolerance
    ):
        raise PipelineError(
            f"Keras/TFLite parity failed for {model_path}: {metrics}; "
            f"tolerance={tolerance}"
        )
    return metrics


def export_embedding(model: Any, model_dir: Path) -> np.ndarray:
    embedding = np.asarray(
        model.get_layer("embedding").get_weights()[0], dtype=np.float32
    )
    np.save(model_dir / "embedding.npy", embedding, allow_pickle=False)
    little_endian = embedding.astype("<f4", copy=False)
    little_endian.tofile(model_dir / "embedding_f32.bin")
    _write_json(
        model_dir / "embedding_metadata.json",
        {
            "rows": embedding.shape[0],
            "cols": embedding.shape[1],
            "dtype": "float32",
            "endianness": "little",
        },
    )
    return embedding


def verify_npu_core(
    trained_model: Any,
    npu_core_path: Path,
    embedding: np.ndarray,
    verification_tokens: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    reference = np.asarray(
        trained_model.predict(verification_tokens, verbose=0), dtype=np.float32
    )
    embedded = embedding[verification_tokens].astype(np.float32, copy=False)
    core_probabilities = _tflite_predict(npu_core_path, embedded)
    metrics = _parity_metrics(reference, core_probabilities)
    if (
        metrics["top1_agreement"] != 1.0
        or metrics["max_absolute_probability_difference"] > tolerance
    ):
        raise PipelineError(
            f"Full-model/NPU-core parity failed for {npu_core_path}: {metrics}; "
            f"tolerance={tolerance}"
        )
    return metrics


def _choose_verification_samples(
    features: np.ndarray, count: int, seed: int
) -> np.ndarray:
    if len(features) < count:
        raise PipelineError(
            f"Only {len(features)} development-validation samples are available; at "
            f"least {count} are required for export verification."
        )
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(features), size=count, replace=False))
    return features[indices]


def _read_training_summary(model_dir: Path) -> dict[str, Any]:
    history_path = model_dir / "history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "val_loss" not in rows[0]:
        raise PipelineError(f"Training history is missing validation loss: {history_path}")
    best_row = min(rows, key=lambda row: float(row["val_loss"]))
    return {
        "epochs_ran": len(rows),
        "best_epoch": int(best_row["epoch"]) + 1,
        "best_validation_loss_during_fit": float(best_row["val_loss"]),
        "best_validation_accuracy_during_fit": float(best_row["val_accuracy"]),
    }


def finalize_model(
    model_name: str,
    model: Any,
    model_dir: Path,
    evaluation_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    verification_features: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    evaluations: dict[str, dict[str, Any]] = {}
    for dataset_name, (features, labels) in evaluation_sets.items():
        metrics, _ = evaluate_model(model, features, labels, args.batch_size)
        evaluations[dataset_name] = metrics
        print(
            f"{model_name} {dataset_name}: accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}"
        )
    write_evaluation_artifacts(model_dir, evaluations)

    fixed_model = build_fixed_batch_inference_model(
        model, args.actual_vocab_size, args.max_len
    )
    builtin_path = model_dir / "model_builtin.tflite"
    convert_builtin_tflite(fixed_model, builtin_path)
    analyzer_sections = [
        "=== model_builtin.tflite ===\n" + _analyze_tflite(builtin_path)
    ]
    tflite_metrics = verify_tflite(
        model,
        builtin_path,
        verification_features,
        args.parity_tolerance,
    )

    npu_metrics: dict[str, float] | None = None
    if not args.skip_npu_core:
        export_embedding(model, model_dir)
        embedding = np.load(model_dir / "embedding.npy", allow_pickle=False)
        npu_core = build_npu_core_model(model, args.max_len)
        npu_core_path = model_dir / "model_npu_core.tflite"
        convert_builtin_tflite(npu_core, npu_core_path)
        analyzer_sections.append(
            "=== model_npu_core.tflite ===\n" + _analyze_tflite(npu_core_path)
        )
        npu_metrics = verify_npu_core(
            model,
            npu_core_path,
            embedding,
            verification_features,
            args.parity_tolerance,
        )

    (model_dir / "tflite_analyzer.txt").write_text(
        "\n\n".join(analyzer_sections), encoding="utf-8"
    )
    result = {
        "model": model_name,
        "training_strategy": MODEL_SPECS[model_name],
        "parameter_count": int(model.count_params()),
        "training_summary": _read_training_summary(model_dir),
        "evaluations": evaluations,
        "tflite_verification": tflite_metrics,
        "npu_core_verification": npu_metrics,
        "builtins_only_conversion": True,
    }
    _write_json(model_dir / "metrics.json", _json_ready(result))
    return result


def write_comparison_report(
    output_dir: Path, results: Sequence[dict[str, Any]]
) -> None:
    fields = [
        "Model",
        "Training strategy",
        "Parameter count",
        ".keras size",
        ".tflite size",
        "Epochs run",
        "Best epoch",
        "Best combined val loss during fit",
        "Semantic val loss",
        "Semantic val accuracy",
        "Semantic val Macro-F1",
        "ASR val loss",
        "ASR val accuracy",
        "ASR val Macro-F1",
        "Complex val loss",
        "Complex val accuracy",
        "Complex val Macro-F1",
        "Combined val loss",
        "Combined val accuracy",
        "Combined val Macro-F1",
        "Human Vosk val accuracy",
        "Human Vosk val Macro-F1",
        "Human Vosk test accuracy",
        "Human Vosk test Macro-F1",
        "TFLite/Keras top1 agreement",
        "Builtins-only conversion yes/no",
    ]
    with (output_dir / "model_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            model_name = result["model"]
            model_dir = output_dir / model_name
            evaluations = result["evaluations"]
            human_val = evaluations.get("human_vosk_val", {})
            human_test = evaluations.get("human_vosk_test", {})
            training_summary = result["training_summary"]
            writer.writerow(
                {
                    "Model": model_name,
                    "Training strategy": result["training_strategy"],
                    "Parameter count": result["parameter_count"],
                    ".keras size": (model_dir / "model.keras").stat().st_size,
                    ".tflite size": (model_dir / "model_builtin.tflite").stat().st_size,
                    "Epochs run": training_summary["epochs_ran"],
                    "Best epoch": training_summary["best_epoch"],
                    "Best combined val loss during fit": training_summary[
                        "best_validation_loss_during_fit"
                    ],
                    "Semantic val loss": evaluations["semantic_val"]["loss"],
                    "Semantic val accuracy": evaluations["semantic_val"]["accuracy"],
                    "Semantic val Macro-F1": evaluations["semantic_val"]["macro_f1"],
                    "ASR val loss": evaluations["asr_val"]["loss"],
                    "ASR val accuracy": evaluations["asr_val"]["accuracy"],
                    "ASR val Macro-F1": evaluations["asr_val"]["macro_f1"],
                    "Complex val loss": evaluations["complex_val"]["loss"],
                    "Complex val accuracy": evaluations["complex_val"]["accuracy"],
                    "Complex val Macro-F1": evaluations["complex_val"]["macro_f1"],
                    "Combined val loss": evaluations["combined_val"]["loss"],
                    "Combined val accuracy": evaluations["combined_val"]["accuracy"],
                    "Combined val Macro-F1": evaluations["combined_val"]["macro_f1"],
                    "Human Vosk val accuracy": human_val.get("accuracy", ""),
                    "Human Vosk val Macro-F1": human_val.get("macro_f1", ""),
                    "Human Vosk test accuracy": human_test.get("accuracy", ""),
                    "Human Vosk test Macro-F1": human_test.get("macro_f1", ""),
                    "TFLite/Keras top1 agreement": result["tflite_verification"]["top1_agreement"],
                    "Builtins-only conversion yes/no": "yes"
                    if result["builtins_only_conversion"]
                    else "no",
                }
            )

    with (output_dir / "validation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["model", "dataset", "sample_count", "loss", "accuracy", "macro_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for dataset in ("semantic_val", "asr_val", "complex_val", "combined_val"):
                metrics = result["evaluations"][dataset]
                writer.writerow(
                    {
                        "model": result["model"],
                        "dataset": dataset,
                        "sample_count": metrics["sample_count"],
                        "loss": metrics["loss"],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                    }
                )

    with (output_dir / "per_class_recall_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["model", "dataset", "label", "recall", "support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for dataset in ("semantic_val", "asr_val", "complex_val", "combined_val"):
                per_class = result["evaluations"][dataset]["per_class"]
                for label in LABELS:
                    writer.writerow(
                        {
                            "model": result["model"],
                            "dataset": dataset,
                            "label": label,
                            "recall": per_class[label]["recall"],
                            "support": per_class[label]["support"],
                        }
                    )


def _resolve_manual_clean_replacement(args: argparse.Namespace) -> None:
    default_name = "02_Saubere_Trainingsdaten_bereinigt_dedupliziert.txt"
    if args.clean_main.name != default_name:
        return
    candidates = sorted(
        candidate
        for candidate in args.clean_main.parent.glob("02*manuell*.txt")
        if candidate.resolve() != args.clean_main.resolve()
    )
    if len(candidates) == 1:
        print(f"Using manually cleaned replacement for Clean Main: {candidates[0]}")
        args.clean_main = candidates[0]
    elif len(candidates) > 1:
        raise PipelineError(
            "Several manually cleaned replacements for Clean Main were found; "
            "select one explicitly with --clean-main:\n"
            + "\n".join(str(path) for path in candidates)
        )


def _validate_cli(args: argparse.Namespace) -> None:
    if args.max_len <= 0:
        raise PipelineError("--max-len must be positive")
    if args.max_vocab_size < 2:
        raise PipelineError("--max-vocab-size must be at least 2")
    if args.batch_size <= 0:
        raise PipelineError("--batch-size must be positive")
    if not 0.0 < args.near_duplicate_threshold < 1.0:
        raise PipelineError("--near-duplicate-threshold must be between 0 and 1")
    if args.verification_samples < 100:
        raise PipelineError("--verification-samples must be at least 100")
    if args.parity_tolerance <= 0:
        raise PipelineError("--parity-tolerance must be positive")
    for name in ("epochs_m0", "epochs_m1", "epochs_m2", "epochs_m3"):
        if getattr(args, name) <= 0:
            raise PipelineError(f"--{name.replace('_', '-')} must be positive")
    for name in ("patience_m0", "patience_m1", "patience_finetune"):
        if getattr(args, name) < 0:
            raise PipelineError(f"--{name.replace('_', '-')} must not be negative")


def _check_human_speaker_disjoint(
    human_val: Sequence[Sample], human_test: Sequence[Sample]
) -> None:
    if not human_val or not human_test:
        return
    val_speakers = {sample.speaker_id for sample in human_val if sample.speaker_id}
    test_speakers = {sample.speaker_id for sample in human_test if sample.speaker_id}
    if not val_speakers and not test_speakers:
        return
    if any(not sample.speaker_id for sample in (*human_val, *human_test)):
        print(
            "WARNING: some human evaluation rows lack speaker_id; full speaker "
            "disjointness cannot be proven.",
            file=sys.stderr,
        )
    overlap = val_speakers & test_speakers
    if overlap:
        raise PipelineError(
            "Human validation/test are not speaker-disjoint. Overlapping speaker_id "
            f"values: {sorted(overlap)}"
        )


def _concatenate_many_encoded(
    datasets: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not datasets:
        raise PipelineError("At least one encoded dataset is required")
    return (
        np.concatenate([dataset[0] for dataset in datasets], axis=0),
        np.concatenate([dataset[1] for dataset in datasets], axis=0),
    )


def _require_balanced_validation(
    samples: Sequence[Sample], expected_total: int, expected_per_class: int, name: str
) -> None:
    counts = Counter(sample.label for sample in samples)
    if len(samples) != expected_total or any(
        counts[label] != expected_per_class for label in LABELS
    ):
        details = ", ".join(f"{label}={counts[label]}" for label in LABELS)
        raise PipelineError(
            f"{name} must contain exactly {expected_total} samples with "
            f"{expected_per_class} per class; found {len(samples)} ({details})."
        )


def write_development_validation_manifest(
    validation_sets: dict[str, Sequence[Sample]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "validation_set",
        "sample_id",
        "source_file",
        "source_line",
        "label",
        "normalized_text",
        "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for validation_set, samples in validation_sets.items():
            for sample in samples:
                writer.writerow(
                    {
                        "validation_set": validation_set,
                        "sample_id": sample.sample_id,
                        "source_file": sample.path.name,
                        "source_line": sample.line_no,
                        "label": sample.label,
                        "normalized_text": sample.normalized_text,
                        "text": sample.text,
                    }
                )


def run_pipeline(args: argparse.Namespace) -> None:
    _validate_cli(args)
    _resolve_manual_clean_replacement(args)

    clean_sources = [
        ("clean_organic", args.clean_organic),
        ("clean_main", args.clean_main),
    ]
    if args.use_old_clean:
        clean_sources.append(("clean_old_auxiliary", args.old_clean))
        print("Old template-heavy Clean source is ENABLED for this explicit ablation.")
    else:
        print("Old template-heavy Clean source is excluded (default).")

    clean_samples: list[Sample] = []
    for source, path in clean_sources:
        clean_samples.extend(load_dataset(path, source=source, domain="clean"))

    vosk_gen1 = load_dataset(args.vosk_gen1, source="vosk_gen1", domain="vosk")
    vosk_gen2 = load_dataset(args.vosk_gen2, source="vosk_gen2", domain="vosk")
    attach_metadata(vosk_gen1, args.gen1_metadata)
    attach_metadata(vosk_gen2, args.gen2_metadata)
    vosk_samples = vosk_gen1 + vosk_gen2
    training_source_samples = clean_samples + vosk_samples

    data_report = {
        "training_sources": validate_dataset(
            training_source_samples, "Core Clean + synthetic Vosk"
        )
    }

    semantic_val_samples = load_dataset(
        args.semantic_val, source="semantic_val", domain="validation"
    )
    asr_val_samples = load_dataset(
        args.asr_val, source="asr_val", domain="validation"
    )
    _require_balanced_validation(
        semantic_val_samples,
        MANUAL_VALIDATION_SAMPLES_PER_DOMAIN,
        MANUAL_VALIDATION_SAMPLES_PER_DOMAIN // len(LABELS),
        "Semantic validation",
    )
    _require_balanced_validation(
        asr_val_samples,
        MANUAL_VALIDATION_SAMPLES_PER_DOMAIN,
        MANUAL_VALIDATION_SAMPLES_PER_DOMAIN // len(LABELS),
        "ASR validation",
    )
    data_report["semantic_val"] = validate_dataset(
        semantic_val_samples, "External Semantic validation"
    )
    data_report["asr_val"] = validate_dataset(
        asr_val_samples, "External ASR validation"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = args.output_dir / "validation"
    manual_validation_sets = {
        "semantic_val": semantic_val_samples,
        "asr_val": asr_val_samples,
    }
    manual_leakage_summary = audit_manual_validation_leakage(
        manual_validation_sets,
        training_source_samples,
        validation_dir / "manual_validation_leakage_report.csv",
    )

    complex_pool = load_dataset(
        args.complex_val_source,
        source="complex_val_pool",
        domain="validation",
    )
    data_report["complex_val_pool"] = validate_dataset(
        complex_pool, "Legacy complex-validation candidate pool"
    )
    complex_val_samples, complex_selection_summary = select_complex_validation(
        complex_pool,
        training_source_samples,
        semantic_val_samples + asr_val_samples,
        samples_per_class=COMPLEX_VALIDATION_SAMPLES_PER_CLASS,
        near_duplicate_threshold=args.near_duplicate_threshold,
        manifest_path=validation_dir / "complex_validation_manifest.csv",
        audit_path=validation_dir / "complex_validation_leakage_audit.csv",
        recreate=args.recreate_manifests,
    )
    _require_balanced_validation(
        complex_val_samples,
        COMPLEX_VALIDATION_SAMPLES_PER_CLASS * len(LABELS),
        COMPLEX_VALIDATION_SAMPLES_PER_CLASS,
        "Complex validation",
    )
    data_report["complex_val"] = validate_dataset(
        complex_val_samples, "Selected Complex validation"
    )

    development_validation_sets: dict[str, Sequence[Sample]] = {
        "semantic_val": semantic_val_samples,
        "asr_val": asr_val_samples,
        "complex_val": complex_val_samples,
    }
    combined_val_samples = [
        sample
        for samples in development_validation_sets.values()
        for sample in samples
    ]
    if len(combined_val_samples) != COMBINED_DEVELOPMENT_VALIDATION_SIZE:
        raise PipelineError(
            f"Combined development validation must contain exactly "
            f"{COMBINED_DEVELOPMENT_VALIDATION_SIZE} samples; found "
            f"{len(combined_val_samples)}."
        )
    combined_validation_report = validate_dataset(
        combined_val_samples, "Combined external development validation"
    )
    if combined_validation_report["duplicates"]["normalized_duplicate_rows"]:
        raise PipelineError(
            "Combined development validation contains normalized duplicate rows."
        )
    data_report["combined_val"] = combined_validation_report
    write_development_validation_manifest(
        development_validation_sets,
        validation_dir / "development_validation_manifest.csv",
    )
    write_training_manifest(
        training_source_samples,
        args.output_dir / "splits" / "training_manifest.csv",
        recreate=args.recreate_manifests,
    )

    human_val_samples = (
        load_human_dataset(args.human_val, "human_vosk_val")
        if args.human_val
        else []
    )
    human_test_samples = (
        load_human_dataset(args.human_test, "human_vosk_test")
        if args.human_test
        else []
    )
    if human_val_samples:
        data_report["human_vosk_val"] = validate_dataset(
            human_val_samples, "Human Vosk validation"
        )
    if human_test_samples:
        data_report["human_vosk_test"] = validate_dataset(
            human_test_samples, "Human Vosk test"
        )
    if human_val_samples and human_test_samples:
        validate_dataset(
            human_val_samples + human_test_samples,
            "Combined Human Vosk evaluation consistency",
        )
    _check_human_speaker_disjoint(human_val_samples, human_test_samples)
    vocabulary, token_to_id = build_vocabulary(
        training_source_samples, args.max_vocab_size
    )
    args.actual_vocab_size = len(vocabulary)
    write_tokenizer_artifacts(
        args.output_dir,
        vocabulary,
        max_len=args.max_len,
        max_vocab_size=args.max_vocab_size,
    )

    truncation_count = sum(
        len(sample.normalized_text.split()) > args.max_len
        for sample in training_source_samples
    )
    truncation_rate = truncation_count / len(training_source_samples)
    if truncation_rate > 0.01:
        print(
            f"WARNING: {truncation_count}/{len(training_source_samples)} "
            f"({truncation_rate:.2%}) training-source sentences exceed MAX_LEN="
            f"{args.max_len}. Consider --max-len 32; it was not changed "
            "automatically.",
            file=sys.stderr,
        )
    else:
        print(
            f"Truncation at MAX_LEN={args.max_len}: {truncation_count}/"
            f"{len(training_source_samples)} ({truncation_rate:.2%})"
        )

    oov_sets: dict[str, Sequence[Sample]] = {
        "clean_train": clean_samples,
        "vosk_train": vosk_samples,
        "all_training": training_source_samples,
        **development_validation_sets,
        "combined_val": combined_val_samples,
    }
    if human_val_samples:
        oov_sets["human_vosk_val"] = human_val_samples
    if human_test_samples:
        oov_sets["human_vosk_test"] = human_test_samples
    oov_report = {
        name: oov_statistics(samples, token_to_id)
        for name, samples in oov_sets.items()
    }
    for name, statistics in oov_report.items():
        print(
            f"OOV {name}: {statistics['oov_tokens']}/{statistics['tokens']} "
            f"({statistics['oov_rate']:.2%})"
        )

    data_report.update(
        {
            "seed": args.seed,
            "max_len": args.max_len,
            "max_vocab_size": args.max_vocab_size,
            "actual_vocab_size": len(vocabulary),
            "vocabulary_built_from": [
                "all Clean training samples",
                "all Vosk Gen1 training samples",
                "all Vosk Gen2 training samples",
            ],
            "validation_excluded_from_vocabulary": True,
            "truncation": {
                "count": truncation_count,
                "rate": truncation_rate,
            },
            "oov": oov_report,
            "training_counts": {
                "clean_train": len(clean_samples),
                "vosk_train": len(vosk_samples),
                "all_training": len(training_source_samples),
            },
            "development_validation_counts": {
                name: len(samples)
                for name, samples in development_validation_sets.items()
            },
            "combined_development_validation_count": len(combined_val_samples),
            "manual_validation_leakage_audit": manual_leakage_summary,
            "complex_validation_selection": complex_selection_summary,
        }
    )
    _write_json(args.output_dir / "data_report.json", _json_ready(data_report))
    _write_json(
        args.output_dir / "run_config.json",
        {
            "seed": args.seed,
            "clean_organic": str(args.clean_organic),
            "clean_main": str(args.clean_main),
            "old_clean": str(args.old_clean),
            "use_old_clean": args.use_old_clean,
            "vosk_gen1": str(args.vosk_gen1),
            "vosk_gen2": str(args.vosk_gen2),
            "gen1_metadata": str(args.gen1_metadata),
            "gen2_metadata": str(args.gen2_metadata),
            "semantic_val": str(args.semantic_val),
            "asr_val": str(args.asr_val),
            "complex_val_source": str(args.complex_val_source),
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "human_val": str(args.human_val) if args.human_val else None,
            "human_test": str(args.human_test) if args.human_test else None,
            "output_dir": str(args.output_dir),
            "max_len": args.max_len,
            "max_vocab_size": args.max_vocab_size,
            "actual_vocab_size": len(vocabulary),
            "batch_size": args.batch_size,
            "epochs": {
                "M0": args.epochs_m0,
                "M1": args.epochs_m1,
                "M2": args.epochs_m2,
                "M3": args.epochs_m3,
            },
            "patience": {
                "M0": args.patience_m0,
                "M1": args.patience_m1,
                "M2": args.patience_finetune,
                "M3": args.patience_finetune,
            },
            "early_stopping_validation": (
                "combined external development validation (660 samples)"
            ),
            "skip_npu_core": args.skip_npu_core,
        },
    )
    if args.prepare_only:
        print(
            f"Preparation complete. Artifacts written to {args.output_dir}; "
            "training was skipped by --prepare-only."
        )
        return

    set_global_determinism(args.seed)
    clean_train = encode_samples(clean_samples, token_to_id, args.max_len)
    vosk_train = encode_samples(vosk_samples, token_to_id, args.max_len)
    semantic_val = encode_samples(
        semantic_val_samples, token_to_id, args.max_len
    )
    asr_val = encode_samples(asr_val_samples, token_to_id, args.max_len)
    complex_val = encode_samples(complex_val_samples, token_to_id, args.max_len)
    combined_val = _concatenate_many_encoded(
        [semantic_val, asr_val, complex_val]
    )
    mixed_features, mixed_labels, mixed_stats = make_domain_mix(
        *clean_train,
        *vosk_train,
        seed=args.seed,
    )
    _write_json(args.output_dir / "mixed_sampling.json", _json_ready(mixed_stats))
    mixed_train = (mixed_features, mixed_labels)

    evaluation_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "semantic_val": semantic_val,
        "asr_val": asr_val,
        "complex_val": complex_val,
        "combined_val": combined_val,
    }
    if human_val_samples:
        evaluation_sets["human_vosk_val"] = encode_samples(
            human_val_samples, token_to_id, args.max_len
        )
    if human_test_samples:
        evaluation_sets["human_vosk_test"] = encode_samples(
            human_test_samples, token_to_id, args.max_len
        )

    verification_features = _choose_verification_samples(
        combined_val[0], args.verification_samples, args.seed
    )

    initial_model = build_model(
        len(vocabulary), args.max_len, name="shared_random_initialization"
    )
    initial_weights = [np.array(weight, copy=True) for weight in initial_model.get_weights()]
    initial_model.save_weights(args.output_dir / "initial_weights.weights.h5")

    results: list[dict[str, Any]] = []

    m0_dir = args.output_dir / "M0_clean"
    m0 = build_model(len(vocabulary), args.max_len, name="M0_clean")
    m0.set_weights([np.array(weight, copy=True) for weight in initial_weights])
    print("\n=== Training M0_clean ===")
    train_m0(m0, m0_dir, clean_train, combined_val, args)
    results.append(
        finalize_model(
            "M0_clean", m0, m0_dir, evaluation_sets, verification_features, args
        )
    )

    m1_dir = args.output_dir / "M1_mixed_from_scratch"
    m1 = build_model(len(vocabulary), args.max_len, name="M1_mixed_from_scratch")
    m1.set_weights([np.array(weight, copy=True) for weight in initial_weights])
    print("\n=== Training M1_mixed_from_scratch ===")
    train_m1(m1, m1_dir, mixed_train, combined_val, args)
    results.append(
        finalize_model(
            "M1_mixed_from_scratch",
            m1,
            m1_dir,
            evaluation_sets,
            verification_features,
            args,
        )
    )

    m0_checkpoint = m0_dir / "best.weights.h5"
    m2_dir = args.output_dir / "M2_clean_then_mixed"
    m2 = build_model(len(vocabulary), args.max_len, name="M2_clean_then_mixed")
    m2.load_weights(m0_checkpoint)
    print("\n=== Training M2_clean_then_mixed ===")
    train_m2(m2, m2_dir, mixed_train, combined_val, args)
    results.append(
        finalize_model(
            "M2_clean_then_mixed",
            m2,
            m2_dir,
            evaluation_sets,
            verification_features,
            args,
        )
    )

    m3_dir = args.output_dir / "M3_clean_then_vosk"
    m3 = build_model(len(vocabulary), args.max_len, name="M3_clean_then_vosk")
    m3.load_weights(m0_checkpoint)
    print("\n=== Training M3_clean_then_vosk ===")
    train_m3(m3, m3_dir, vosk_train, combined_val, args)
    results.append(
        finalize_model(
            "M3_clean_then_vosk",
            m3,
            m3_dir,
            evaluation_sets,
            verification_features,
            args,
        )
    )

    write_comparison_report(args.output_dir, results)
    print(f"\nAll four experiments completed: {args.output_dir}")
    if not human_test_samples:
        print(
            "No automatic winner was selected. Final model selection requires "
            "independent real Human-Vosk test data."
        )


def build_argument_parser(script_dir: Path) -> argparse.ArgumentParser:
    data_dir = script_dir / "Trainingsdaten"
    validation_dir = script_dir
    parser = argparse.ArgumentParser(
        description=(
            "Train M0-M3 numerical intent classifiers and export fixed-batch, "
            "TFLITE_BUILTINS-only models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clean-organic",
        type=Path,
        default=data_dir / "01_Daten_mit_Widerspruechen_bereinigt_dedupliziert.txt",
    )
    parser.add_argument(
        "--clean-main",
        type=Path,
        default=data_dir / "02_Saubere_Trainingsdaten_bereinigt_dedupliziert.txt",
    )
    parser.add_argument(
        "--old-clean",
        type=Path,
        default=data_dir / "03_Streng_gefilterte_alte_Daten_bereinigt_dedupliziert.txt",
    )
    parser.add_argument(
        "--use-old-clean",
        action="store_true",
        help="Include the older template-heavy Clean source for an explicit ablation.",
    )
    parser.add_argument(
        "--vosk-gen1",
        type=Path,
        default=data_dir / "vosk_generation1_final_manuell_kuratiert.txt",
    )
    parser.add_argument(
        "--vosk-gen2",
        type=Path,
        default=data_dir / "vosk_generation2_streng_manuell_gefiltert.txt",
    )
    parser.add_argument(
        "--gen1-metadata",
        type=Path,
        default=data_dir / "vosk_generation1_final_manuell_kuratiert_metadata.csv",
    )
    parser.add_argument(
        "--gen2-metadata",
        type=Path,
        default=data_dir / "vosk_generation2_streng_manuell_gefiltert_metadata.csv",
    )
    parser.add_argument(
        "--semantic-val",
        type=Path,
        default=validation_dir / "validation_manual_semantic_300.val",
    )
    parser.add_argument(
        "--asr-val",
        type=Path,
        default=validation_dir / "validation_manual_asr_300.val",
    )
    parser.add_argument(
        "--complex-val-source",
        type=Path,
        default=script_dir / "Altes NLP Modell" / "DATASET.val",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
        help="Exclude complex-validation candidates at or above this char-ngram cosine similarity.",
    )
    parser.add_argument("--human-val", type=Path, default=None)
    parser.add_argument("--human-test", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "intent_models_next",
    )
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument(
        "--max-vocab-size", type=int, default=DEFAULT_MAX_VOCAB_SIZE
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs-m0", type=int, default=30)
    parser.add_argument("--epochs-m1", type=int, default=30)
    parser.add_argument("--epochs-m2", type=int, default=30)
    parser.add_argument("--epochs-m3", type=int, default=30)
    parser.add_argument("--patience-m0", type=int, default=5)
    parser.add_argument("--patience-m1", type=int, default=5)
    parser.add_argument("--patience-finetune", type=int, default=5)
    parser.add_argument("--verification-samples", type=int, default=100)
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--skip-npu-core",
        action="store_true",
        help="Skip the optional post-embedding NPU-core export.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate, audit leakage, select Complex-Val, and write tokenizer artifacts without training.",
    )
    parser.add_argument(
        "--recreate-manifests",
        action="store_true",
        help="Replace existing training/validation manifests after validating current inputs.",
    )
    parser.add_argument(
        "--verbose", type=int, choices=(0, 1, 2), default=2, help="Keras verbosity."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = build_argument_parser(script_dir)
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
