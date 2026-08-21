#!/usr/bin/env python3
"""Reproducible diagnosis and M0-only architecture comparison.

This script is intentionally separate from the production M0-M3 pipeline.  It
never edits source datasets or existing model artifacts and refuses to reuse an
existing output directory.  Challenge-40 and the appended nine hard cases are
evaluation-only data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import train_intent_models_npu as baseline


SEED = 20260810
MAX_LEN = 24
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3

CURRENT_RUN = "intent_models_current"
DEFAULT_OUTPUT = "intent_analysis_next"
CURRENT_CHALLENGE = "challenge_regression_holdout_clean_40.val"
KNOWN_FAILURE_CHALLENGE_LINES = (36, 1, 2, 3, 4, 5, 6, 7, 9)

ARCHITECTURE_DESCRIPTIONS = {
    "baseline_cnn": (
        "Embedding(32) -> Conv1D(32,k=3) -> global max+mean -> Dense(32)"
    ),
    "strong_cnn": (
        "Embedding(48) -> parallel Conv1D(32,k=2/3/5) -> global max pools "
        "+ k=3 global mean -> Dense(48)"
    ),
    "small_gru": (
        "Embedding(56, mask_zero=True) -> unidirectional GRU(40, unroll=True) "
        "-> Dense(40)"
    ),
}

STRUCTURAL_PROBES = [
    (
        "Nicht die Entfernung messen, sondern das Objekt erkennen",
        "OBJECT_DETECTION",
        "distance_object",
    ),
    (
        "Nicht das Objekt erkennen, sondern die Entfernung messen",
        "MEASURE_DISTANCE",
        "distance_object",
    ),
    (
        "Nicht die Stimme ändern, sondern die Sprechgeschwindigkeit ändern",
        "CHANGE_SPEECH_SPEED",
        "speaker_speed",
    ),
    (
        "Nicht die Sprechgeschwindigkeit ändern, sondern die Stimme ändern",
        "CHANGE_SPEAKER",
        "speaker_speed",
    ),
    (
        "Nicht die Frequenz ändern, sondern die Pulsrate ändern",
        "SET_BPS",
        "frequency_bps",
    ),
    (
        "Nicht die Pulsrate ändern, sondern die Frequenz ändern",
        "SET_FREQUENCY",
        "frequency_bps",
    ),
    (
        "Nicht den Text lesen, sondern das Schild finden",
        "OBJECT_DETECTION",
        "text_object",
    ),
    (
        "Nicht das Schild finden, sondern den Text lesen",
        "TEXT_RECOGNITION",
        "text_object",
    ),
]

EXPLICIT_PATTERNS: dict[str, list[tuple[str, ...]]] = {
    "was_ist": [("was", "ist")],
    "wie_viel": [("wie", "viel")],
    "wo_finde": [("wo", "finde")],
    "wer": [("wer",)],
    "was_denkst": [("was", "denkst")],
    "meinung": [("meinung",)],
    "uhr": [("uhr",)],
    "uhrzeit": [("uhrzeit",)],
    "wochentag_oder_tagname": [
        ("wochentag",),
        ("montag",),
        ("dienstag",),
        ("mittwoch",),
        ("donnerstag",),
        ("freitag",),
        ("samstag",),
        ("sonntag",),
    ],
    "person_oder_bild": [("person",), ("bild",)],
    "wortlaut_oder_geschrieben": [
        ("wortlaut",),
        ("wortlaute",),
        ("geschrieben",),
    ],
    "stimme": [("stimme",)],
    "was_steht": [("was", "steht")],
    "wo_befindet": [("wo", "befindet"), ("wo", "befinde")],
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline._json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_complex_validation(manifest_path: Path) -> list[baseline.Sample]:
    samples: list[baseline.Sample] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["validation_set"] != "complex_val":
                continue
            text = row["text"]
            samples.append(
                baseline.Sample(
                    source="complex_val",
                    path=Path(row["source_file"]),
                    line_no=int(row["source_line"]),
                    text=text,
                    label=row["label"],
                    domain="validation",
                    normalized_text=baseline.normalize_text(text),
                )
            )
    if len(samples) != 60:
        raise baseline.PipelineError(
            f"Expected 60 complex validation rows in {manifest_path}, found {len(samples)}"
        )
    return samples


def load_structural_probes() -> list[baseline.Sample]:
    return [
        baseline.Sample(
            source=f"structural_probe:{pair}",
            path=Path("generated_structural_probes"),
            line_no=index,
            text=text,
            label=label,
            domain="diagnostic",
            normalized_text=baseline.normalize_text(text),
        )
        for index, (text, label, pair) in enumerate(STRUCTURAL_PROBES, start=1)
    ]


def load_all_inputs(root: Path, current_dir: Path) -> dict[str, list[baseline.Sample]]:
    data_dir = root / "Trainingsdaten"
    clean_organic = baseline.load_dataset(
        data_dir / "01_Daten_mit_Widerspruechen_bereinigt_dedupliziert.txt",
        source="clean_organic",
        domain="clean",
    )
    clean_main = baseline.load_dataset(
        data_dir / "02_Saubere_Trainingsdaten_bereinigt_dedupliziert.txt",
        source="clean_main",
        domain="clean",
    )
    vosk_gen1 = baseline.load_dataset(
        data_dir / "vosk_generation1_final_manuell_kuratiert.txt",
        source="vosk_gen1",
        domain="vosk",
    )
    vosk_gen2 = baseline.load_dataset(
        data_dir / "vosk_generation2_streng_manuell_gefiltert.txt",
        source="vosk_gen2",
        domain="vosk",
    )
    semantic = baseline.load_dataset(
        root / "validation_manual_semantic_300.val",
        source="semantic_val",
        domain="validation",
    )
    asr = baseline.load_dataset(
        root / "validation_manual_asr_300.val",
        source="asr_val",
        domain="validation",
    )
    challenge = baseline.load_dataset(
        root / CURRENT_CHALLENGE,
        source="challenge_40",
        domain="holdout",
    )
    if len(challenge) != 40:
        raise baseline.PipelineError(
            f"{CURRENT_CHALLENGE} must contain 40 rows, found {len(challenge)}"
        )
    hard_cases = [challenge[line_number - 1] for line_number in KNOWN_FAILURE_CHALLENGE_LINES]
    complex_val = load_complex_validation(
        current_dir / "validation" / "development_validation_manifest.csv"
    )
    return {
        "clean_organic": clean_organic,
        "clean_main": clean_main,
        "clean_train": clean_organic + clean_main,
        "vosk_train": vosk_gen1 + vosk_gen2,
        "all_training": clean_organic + clean_main + vosk_gen1 + vosk_gen2,
        "semantic_val": semantic,
        "asr_val": asr,
        "complex_val": complex_val,
        "combined_val": semantic + asr + complex_val,
        "challenge_40": challenge,
        "hard_cases_9": hard_cases,
        # Compatibility view for the historical diagnostic report. The final
        # tokenizer runner treats these nine rows only as a Challenge-40 subset.
        "challenge_49": [*challenge, *hard_cases],
        "structural_probes": load_structural_probes(),
    }


def load_tokenizer(current_dir: Path) -> tuple[list[str], dict[str, int]]:
    vocabulary = read_json(current_dir / "tokenizer" / "vocab.json")
    labels = read_json(current_dir / "tokenizer" / "labels.json")
    tokenizer_config = read_json(current_dir / "tokenizer" / "tokenizer_config.json")
    if labels != baseline.LABELS:
        raise baseline.PipelineError("Saved labels.json does not match training script labels")
    if tokenizer_config["max_length"] != MAX_LEN:
        raise baseline.PipelineError(
            f"Expected MAX_LEN={MAX_LEN}, found {tokenizer_config['max_length']}"
        )
    if vocabulary[:2] != [baseline.PAD_TOKEN, baseline.OOV_TOKEN]:
        raise baseline.PipelineError("Unexpected reserved-token ordering in current vocabulary")
    return vocabulary, {token: index for index, token in enumerate(vocabulary)}


def contiguous_match(tokens: Sequence[str], phrase: Sequence[str]) -> bool:
    width = len(phrase)
    return any(tuple(tokens[index : index + width]) == tuple(phrase) for index in range(len(tokens) - width + 1))


def pattern_matches(tokens: Sequence[str], variants: Sequence[Sequence[str]], mode: str) -> bool:
    if mode == "starts_with":
        return any(tuple(tokens[: len(variant)]) == tuple(variant) for variant in variants)
    return any(contiguous_match(tokens, variant) for variant in variants)


def token_diagnostics(
    datasets: dict[str, list[baseline.Sample]],
    token_to_id: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    challenge = datasets["challenge_49"]
    rows: list[dict[str, Any]] = []
    first_occurrence: dict[tuple[str, str], int] = {}
    duplicate_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(challenge, start=1):
        tokens = sample.normalized_text.split()
        token_ids = [token_to_id.get(token, baseline.OOV_ID) for token in tokens]
        kept_tokens = tokens[:MAX_LEN]
        kept_ids = token_ids[:MAX_LEN]
        padded_ids = kept_ids + [baseline.PAD_ID] * (MAX_LEN - len(kept_ids))
        key = (sample.normalized_text, sample.label)
        if key in first_occurrence:
            duplicate_rows.append(
                {
                    "challenge_line": index,
                    "duplicates_line": first_occurrence[key],
                    "text": sample.text,
                    "label": sample.label,
                }
            )
        else:
            first_occurrence[key] = index
        rows.append(
            {
                "challenge_line": index,
                "subset": "challenge_40" if index <= 40 else "hard_cases_9",
                "expected_label": sample.label,
                "original_text": sample.text,
                "normalized_text": sample.normalized_text,
                "tokens": json.dumps(tokens, ensure_ascii=False),
                "token_ids": json.dumps(token_ids),
                "encoded_ids_length_24": json.dumps(padded_ids),
                "oov_tokens": json.dumps(
                    [token for token in tokens if token not in token_to_id],
                    ensure_ascii=False,
                ),
                "oov_count": sum(token not in token_to_id for token in tokens),
                "truncated_tokens": json.dumps(tokens[MAX_LEN:], ensure_ascii=False),
                "truncated_count": max(0, len(tokens) - MAX_LEN),
            }
        )
    write_csv(
        output_dir / "tokenizer" / "challenge_49_tokenization.csv",
        rows,
        list(rows[0]),
    )
    write_csv(
        output_dir / "tokenizer" / "challenge_duplicate_audit.csv",
        duplicate_rows,
        ["challenge_line", "duplicates_line", "text", "label"],
    )

    overlap_rows: list[dict[str, Any]] = []
    reference_sets = {
        "training": datasets["all_training"],
        "development_validation": datasets["combined_val"],
    }
    for reference_name, reference_samples in reference_sets.items():
        reference_index: dict[tuple[str, str], list[baseline.Sample]] = defaultdict(list)
        for reference in reference_samples:
            reference_index[(reference.normalized_text, reference.label)].append(reference)
        for index, sample in enumerate(challenge[:40], start=1):
            for reference in reference_index.get((sample.normalized_text, sample.label), []):
                overlap_rows.append(
                    {
                        "challenge_line": index,
                        "text": sample.text,
                        "label": sample.label,
                        "overlap_set": reference_name,
                        "reference_source": reference.source,
                        "reference_line": reference.line_no,
                        "reference_text": reference.text,
                    }
                )
    write_csv(
        output_dir / "tokenizer" / "challenge_exact_overlap_audit.csv",
        overlap_rows,
        [
            "challenge_line",
            "text",
            "label",
            "overlap_set",
            "reference_source",
            "reference_line",
            "reference_text",
        ],
    )

    oov_sets = {
        name: datasets[name]
        for name in (
            "clean_train",
            "vosk_train",
            "all_training",
            "semantic_val",
            "asr_val",
            "complex_val",
            "challenge_40",
            "hard_cases_9",
            "challenge_49",
        )
    }
    oov_summary: dict[str, Any] = {}
    oov_token_rows: list[dict[str, Any]] = []
    for name, samples in oov_sets.items():
        statistics_row = baseline.oov_statistics(samples, token_to_id)
        sentence_oov = sum(
            any(token not in token_to_id for token in sample.normalized_text.split())
            for sample in samples
        )
        token_counts: Counter[str] = Counter(
            token
            for sample in samples
            for token in sample.normalized_text.split()
            if token not in token_to_id
        )
        oov_summary[name] = {
            **statistics_row,
            "sentences": len(samples),
            "sentences_with_oov": sentence_oov,
            "sentence_oov_rate": sentence_oov / len(samples) if samples else 0.0,
            "unique_oov_tokens": len(token_counts),
        }
        for token, count in token_counts.most_common():
            examples = [
                sample.text
                for sample in samples
                if token in sample.normalized_text.split()
            ][:3]
            oov_token_rows.append(
                {
                    "dataset": name,
                    "token": token,
                    "count": count,
                    "examples": json.dumps(examples, ensure_ascii=False),
                }
            )
    write_json(output_dir / "tokenizer" / "oov_summary.json", oov_summary)
    write_csv(
        output_dir / "tokenizer" / "oov_tokens.csv",
        oov_token_rows,
        ["dataset", "token", "count", "examples"],
    )

    target_words = [
        "meinung",
        "autohaus",
        "montag",
        "wortlaut",
        "wortlaute",
        "uhr",
        "uhrzeit",
        "befinde",
        "befindet",
        "nervt",
        "wochentag",
        "person",
        "bild",
        "geschrieben",
        "stimme",
    ]
    training_token_counts = Counter(
        token
        for sample in datasets["all_training"]
        for token in sample.normalized_text.split()
    )
    target_rows = [
        {
            "token": token,
            "in_vocabulary": token in token_to_id,
            "token_id": token_to_id.get(token, baseline.OOV_ID),
            "training_occurrences": training_token_counts[token],
        }
        for token in target_words
    ]
    write_csv(
        output_dir / "tokenizer" / "focus_word_vocabulary.csv",
        target_rows,
        ["token", "in_vocabulary", "token_id", "training_occurrences"],
    )
    return {
        "vocabulary_size": len(token_to_id),
        "max_len": MAX_LEN,
        "oov": oov_summary,
        "challenge_duplicate_count": len(duplicate_rows),
        "challenge_duplicates": duplicate_rows,
        "challenge_40_exact_training_overlaps": sum(
            row["overlap_set"] == "training" for row in overlap_rows
        ),
        "challenge_40_exact_development_validation_overlaps": sum(
            row["overlap_set"] == "development_validation" for row in overlap_rows
        ),
        "challenge_exact_overlaps": overlap_rows,
        "focus_words": target_rows,
    }


def metrics_from_probabilities(
    samples: Sequence[baseline.Sample], probabilities: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray([sample.label_id for sample in samples], dtype=np.int32)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.arange(len(baseline.LABELS))
    matrix = confusion_matrix(truth, predictions, labels=labels)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    present = np.unique(truth)
    correct = predictions == truth
    true_probabilities = np.clip(
        probabilities[np.arange(len(truth)), truth], 1e-7, 1.0
    )
    predicted_confidences = np.max(probabilities, axis=1)
    return {
        "sample_count": len(samples),
        "loss": float(-np.mean(np.log(true_probabilities))),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1_all_10_classes": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1_present_truth_classes": float(
            f1_score(
                truth,
                predictions,
                labels=present,
                average="macro",
                zero_division=0,
            )
        ),
        "correct_count": int(np.sum(correct)),
        "wrong_count": int(np.sum(~correct)),
        "mean_confidence_correct": (
            float(np.mean(predicted_confidences[correct])) if np.any(correct) else None
        ),
        "mean_confidence_wrong": (
            float(np.mean(predicted_confidences[~correct])) if np.any(~correct) else None
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(baseline.LABELS)
        },
        "confusion_matrix": matrix.tolist(),
    }


def prediction_rows(
    model_name: str,
    dataset_name: str,
    samples: Sequence[baseline.Sample],
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (sample, distribution) in enumerate(zip(samples, probabilities), start=1):
        predicted_id = int(np.argmax(distribution))
        row: dict[str, Any] = {
            "model": model_name,
            "dataset": dataset_name,
            "row": index,
            "source_line": sample.line_no,
            "text": sample.text,
            "normalized_text": sample.normalized_text,
            "expected_label": sample.label,
            "predicted_label": baseline.LABELS[predicted_id],
            "confidence": float(distribution[predicted_id]),
            "correct": predicted_id == sample.label_id,
        }
        row.update(
            {
                f"p_{label}": float(distribution[label_id])
                for label_id, label in enumerate(baseline.LABELS)
            }
        )
        rows.append(row)
    return rows


def write_confusion(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    rows = []
    for actual, values in zip(baseline.LABELS, matrix):
        rows.append(
            {"actual_label": actual, **dict(zip(baseline.LABELS, values))}
        )
    write_csv(path, rows, ["actual_label", *baseline.LABELS])


def structural_pair_metrics(
    samples: Sequence[baseline.Sample], probabilities: np.ndarray
) -> dict[str, Any]:
    rows = prediction_rows("", "structural_probes", samples, probabilities)
    pair_results: dict[str, bool] = {}
    for pair_name in {pair for _, _, pair in STRUCTURAL_PROBES}:
        indices = [
            index
            for index, (_, _, candidate_pair) in enumerate(STRUCTURAL_PROBES)
            if candidate_pair == pair_name
        ]
        pair_results[pair_name] = all(bool(rows[index]["correct"]) for index in indices)
    return {
        "accuracy": float(np.mean([bool(row["correct"]) for row in rows])),
        "correct_count": sum(bool(row["correct"]) for row in rows),
        "resolved_pairs": sum(pair_results.values()),
        "pair_count": len(pair_results),
        "pairs": pair_results,
    }


def explicit_shortcut_analysis(
    datasets: dict[str, list[baseline.Sample]], output_dir: Path
) -> list[dict[str, Any]]:
    scopes = {
        "clean_train": datasets["clean_train"],
        "vosk_train": datasets["vosk_train"],
        "all_training": datasets["all_training"],
    }
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for scope_name, samples in scopes.items():
        overall = Counter(sample.label for sample in samples)
        for pattern_name, variants in EXPLICIT_PATTERNS.items():
            for mode in ("starts_with", "contains"):
                matches = [
                    sample
                    for sample in samples
                    if pattern_matches(sample.normalized_text.split(), variants, mode)
                ]
                counts = Counter(sample.label for sample in matches)
                dominant_label, dominant_count = (
                    counts.most_common(1)[0] if counts else ("", 0)
                )
                purity = dominant_count / len(matches) if matches else 0.0
                prior = (
                    overall[dominant_label] / len(samples)
                    if matches and dominant_label
                    else 0.0
                )
                summary_rows.append(
                    {
                        "scope": scope_name,
                        "pattern": pattern_name,
                        "match_mode": mode,
                        "support": len(matches),
                        "dominant_label": dominant_label,
                        "dominant_count": dominant_count,
                        "purity": purity,
                        "dominant_class_prior": prior,
                        "lift_over_prior": purity / prior if prior else 0.0,
                        **{
                            f"count_{label}": counts[label]
                            for label in baseline.LABELS
                        },
                    }
                )
                seen_per_label: Counter[str] = Counter()
                for sample in matches:
                    if seen_per_label[sample.label] >= 2:
                        continue
                    seen_per_label[sample.label] += 1
                    example_rows.append(
                        {
                            "scope": scope_name,
                            "pattern": pattern_name,
                            "match_mode": mode,
                            "label": sample.label,
                            "source": sample.source,
                            "source_line": sample.line_no,
                            "text": sample.text,
                        }
                    )
    fields = [
        "scope",
        "pattern",
        "match_mode",
        "support",
        "dominant_label",
        "dominant_count",
        "purity",
        "dominant_class_prior",
        "lift_over_prior",
        *[f"count_{label}" for label in baseline.LABELS],
    ]
    write_csv(output_dir / "shortcuts" / "explicit_pattern_statistics.csv", summary_rows, fields)
    write_csv(
        output_dir / "shortcuts" / "explicit_pattern_examples.csv",
        example_rows,
        ["scope", "pattern", "match_mode", "label", "source", "source_line", "text"],
    )
    return summary_rows


def sample_features(tokens: Sequence[str]) -> set[str]:
    features: set[str] = set()
    for width in (1, 2, 3):
        kind = "token" if width == 1 else f"ngram_{width}"
        for index in range(len(tokens) - width + 1):
            features.add(f"{kind}:{' '.join(tokens[index:index + width])}")
        if len(tokens) >= width:
            features.add(f"prefix_{width}:{' '.join(tokens[:width])}")
    return features


def discovered_shortcut_analysis(
    datasets: dict[str, list[baseline.Sample]], output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Counter[str]]]]:
    all_rows: list[dict[str, Any]] = []
    feature_maps: dict[str, dict[str, Counter[str]]] = {}
    for scope_name in ("clean_train", "vosk_train", "all_training"):
        samples = datasets[scope_name]
        overall = Counter(sample.label for sample in samples)
        feature_labels: dict[str, Counter[str]] = defaultdict(Counter)
        feature_examples: dict[str, list[str]] = defaultdict(list)
        for sample in samples:
            for feature in sample_features(sample.normalized_text.split()):
                feature_labels[feature][sample.label] += 1
                if len(feature_examples[feature]) < 3:
                    feature_examples[feature].append(sample.text)
        feature_maps[scope_name] = feature_labels
        for feature, counts in feature_labels.items():
            support = sum(counts.values())
            if support < 5:
                continue
            dominant_label, dominant_count = counts.most_common(1)[0]
            purity = dominant_count / support
            prior = overall[dominant_label] / len(samples)
            lift = purity / prior if prior else 0.0
            recall_of_label = dominant_count / overall[dominant_label]
            feature_kind, feature_value = feature.split(":", 1)
            association_score = (
                math.log2(support + 1.0) * purity * math.log2(max(lift, 1.0))
            )
            all_rows.append(
                {
                    "scope": scope_name,
                    "feature_kind": feature_kind,
                    "feature": feature_value,
                    "support": support,
                    "dominant_label": dominant_label,
                    "dominant_count": dominant_count,
                    "purity": purity,
                    "lift_over_prior": lift,
                    "recall_of_dominant_label": recall_of_label,
                    "association_score": association_score,
                    "class_counts": json.dumps(counts, ensure_ascii=False, sort_keys=True),
                    "examples": json.dumps(feature_examples[feature], ensure_ascii=False),
                }
            )
    all_rows.sort(
        key=lambda row: (
            -float(row["association_score"]),
            -int(row["support"]),
            str(row["scope"]),
            str(row["feature_kind"]),
            str(row["feature"]),
        )
    )
    write_csv(
        output_dir / "shortcuts" / "discovered_feature_associations.csv",
        all_rows,
        [
            "scope",
            "feature_kind",
            "feature",
            "support",
            "dominant_label",
            "dominant_count",
            "purity",
            "lift_over_prior",
            "recall_of_dominant_label",
            "association_score",
            "class_counts",
            "examples",
        ],
    )
    return all_rows, feature_maps


def write_challenge_feature_evidence(
    challenge: Sequence[baseline.Sample],
    feature_map: dict[str, Counter[str]],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for line_no, sample in enumerate(challenge, start=1):
        for feature in sorted(sample_features(sample.normalized_text.split())):
            counts = feature_map.get(feature)
            if not counts:
                continue
            support = sum(counts.values())
            if support < 2:
                continue
            dominant_label, dominant_count = counts.most_common(1)[0]
            kind, value = feature.split(":", 1)
            rows.append(
                {
                    "challenge_line": line_no,
                    "subset": "challenge_40" if line_no <= 40 else "hard_cases_9",
                    "text": sample.text,
                    "expected_label": sample.label,
                    "feature_kind": kind,
                    "feature": value,
                    "training_support": support,
                    "dominant_label": dominant_label,
                    "purity": dominant_count / support,
                    "class_counts": json.dumps(counts, ensure_ascii=False, sort_keys=True),
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["challenge_line"]),
            -float(row["purity"]),
            -int(row["training_support"]),
        )
    )
    write_csv(
        output_dir / "shortcuts" / "challenge_feature_evidence.csv",
        rows,
        [
            "challenge_line",
            "subset",
            "text",
            "expected_label",
            "feature_kind",
            "feature",
            "training_support",
            "dominant_label",
            "purity",
            "class_counts",
        ],
    )


def evaluate_current_models(
    current_dir: Path,
    datasets: dict[str, list[baseline.Sample]],
    token_to_id: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    evaluation_names = ("challenge_40", "hard_cases_9", "challenge_49", "structural_probes")
    encoded = {
        name: baseline.encode_samples(datasets[name], token_to_id, MAX_LEN)[0]
        for name in evaluation_names
    }
    model_dirs = [
        "M0_clean",
        "M1_mixed_from_scratch",
        "M2_clean_then_mixed",
        "M3_clean_then_vosk",
    ]
    summary: dict[str, Any] = {}
    all_prediction_rows: list[dict[str, Any]] = []
    challenge_predictions: dict[str, np.ndarray] = {}
    for model_name in model_dirs:
        model = tf.keras.models.load_model(current_dir / model_name / "model.keras", compile=False)
        model_summary: dict[str, Any] = {
            "parameter_count": int(model.count_params()),
            "evaluations": {},
        }
        for dataset_name in evaluation_names:
            probabilities = np.asarray(
                model.predict(encoded[dataset_name], batch_size=BATCH_SIZE, verbose=0),
                dtype=np.float32,
            )
            metrics = metrics_from_probabilities(datasets[dataset_name], probabilities)
            if dataset_name == "structural_probes":
                metrics["structural_pairs"] = structural_pair_metrics(
                    datasets[dataset_name], probabilities
                )
            model_summary["evaluations"][dataset_name] = metrics
            rows = prediction_rows(
                model_name, dataset_name, datasets[dataset_name], probabilities
            )
            all_prediction_rows.extend(rows)
            if dataset_name == "challenge_49":
                challenge_predictions[model_name] = np.argmax(probabilities, axis=1)
            write_confusion(
                output_dir / "current_models" / model_name / f"confusion_matrix_{dataset_name}.csv",
                metrics["confusion_matrix"],
            )
        summary[model_name] = model_summary
        del model
        tf.keras.backend.clear_session()
    write_json(output_dir / "current_models" / "summary.json", summary)
    prediction_fields = [
        "model",
        "dataset",
        "row",
        "source_line",
        "text",
        "normalized_text",
        "expected_label",
        "predicted_label",
        "confidence",
        "correct",
        *[f"p_{label}" for label in baseline.LABELS],
    ]
    write_csv(
        output_dir / "current_models" / "full_softmax_predictions.csv",
        all_prediction_rows,
        prediction_fields,
    )

    truth = np.asarray(
        [sample.label_id for sample in datasets["challenge_49"]], dtype=np.int32
    )
    m0_predictions = challenge_predictions["M0_clean"]
    transition_rows: list[dict[str, Any]] = []
    transition_counts: dict[str, Counter[str]] = {}
    for model_name in model_dirs[1:]:
        current = challenge_predictions[model_name]
        counts: Counter[str] = Counter()
        for index, (expected, m0_pred, candidate_pred) in enumerate(
            zip(truth, m0_predictions, current), start=1
        ):
            if m0_pred == expected and candidate_pred == expected:
                transition = "correct_in_both"
            elif m0_pred != expected and candidate_pred == expected:
                transition = "m0_error_fixed"
            elif m0_pred == expected and candidate_pred != expected:
                transition = "new_error_after_m0"
            elif m0_pred == candidate_pred:
                transition = "same_error_as_m0"
            else:
                transition = "different_error_from_m0"
            counts[transition] += 1
            transition_rows.append(
                {
                    "model": model_name,
                    "challenge_line": index,
                    "subset": "challenge_40" if index <= 40 else "hard_cases_9",
                    "text": datasets["challenge_49"][index - 1].text,
                    "expected_label": baseline.LABELS[int(expected)],
                    "m0_prediction": baseline.LABELS[int(m0_pred)],
                    "model_prediction": baseline.LABELS[int(candidate_pred)],
                    "transition": transition,
                }
            )
        transition_counts[model_name] = counts
    write_csv(
        output_dir / "current_models" / "m0_error_transitions.csv",
        transition_rows,
        [
            "model",
            "challenge_line",
            "subset",
            "text",
            "expected_label",
            "m0_prediction",
            "model_prediction",
            "transition",
        ],
    )
    write_json(output_dir / "current_models" / "m0_error_transition_summary.json", transition_counts)
    return summary


def build_baseline_cnn(
    vocab_size: int, max_len: int, batch_size: int | None = None
) -> Any:
    return baseline.build_model(
        vocab_size,
        max_len,
        batch_size=batch_size,
        name="diagnostic_baseline_cnn",
    )


def build_strong_cnn(
    vocab_size: int, max_len: int, batch_size: int | None = None
) -> Any:
    tf = baseline.require_tensorflow()
    tokens = tf.keras.Input(
        shape=(max_len,), batch_size=batch_size, dtype=tf.int32, name="tokens"
    )
    embeddings = tf.keras.layers.Embedding(
        vocab_size, 48, mask_zero=False, name="embedding"
    )(tokens)
    pooled = []
    for kernel_size in (2, 3, 5):
        branch = tf.keras.layers.Conv1D(
            32,
            kernel_size,
            padding="same",
            activation="relu",
            name=f"conv_k{kernel_size}",
        )(embeddings)
        pooled.append(
            tf.keras.layers.GlobalMaxPooling1D(name=f"global_max_k{kernel_size}")(
                branch
            )
        )
        if kernel_size == 3:
            pooled.append(
                tf.keras.layers.GlobalAveragePooling1D(name="global_mean_k3")(
                    branch
                )
            )
    x = tf.keras.layers.Concatenate(name="pool_concat")(pooled)
    x = tf.keras.layers.Dense(48, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(0.15, name="dropout")(x)
    probabilities = tf.keras.layers.Dense(
        len(baseline.LABELS), activation="softmax", name="intent_probabilities"
    )(x)
    return tf.keras.Model(tokens, probabilities, name="diagnostic_strong_cnn")


def build_small_gru(
    vocab_size: int, max_len: int, batch_size: int | None = None
) -> Any:
    tf = baseline.require_tensorflow()
    tokens = tf.keras.Input(
        shape=(max_len,), batch_size=batch_size, dtype=tf.int32, name="tokens"
    )
    x = tf.keras.layers.Embedding(
        vocab_size,
        56,
        mask_zero=True,
        name="embedding",
    )(tokens)
    x = tf.keras.layers.GRU(
        40,
        return_sequences=False,
        dropout=0.0,
        recurrent_dropout=0.0,
        unroll=True,
        reset_after=True,
        name="gru",
    )(x)
    x = tf.keras.layers.Dense(40, activation="relu", name="dense_hidden")(x)
    x = tf.keras.layers.Dropout(0.15, name="dropout")(x)
    probabilities = tf.keras.layers.Dense(
        len(baseline.LABELS), activation="softmax", name="intent_probabilities"
    )(x)
    return tf.keras.Model(tokens, probabilities, name="diagnostic_small_gru")


def history_summary(history_path: Path) -> dict[str, Any]:
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise baseline.PipelineError(f"Empty training history: {history_path}")
    best_index = min(range(len(rows)), key=lambda index: float(rows[index]["val_loss"]))
    best = rows[best_index]
    return {
        "epochs_ran": len(rows),
        "best_epoch_one_based": best_index + 1,
        "best_epoch_csv_index": int(float(best["epoch"])),
        "best_validation_loss": float(best["val_loss"]),
        "best_validation_accuracy": float(best["val_accuracy"]),
        "final_training_loss": float(rows[-1]["loss"]),
        "final_training_accuracy": float(rows[-1]["accuracy"]),
    }


def tflite_operator_summary(model_path: Path) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    details = interpreter._get_ops_details()  # Diagnostic-only private API.
    counts = Counter(str(detail["op_name"]) for detail in details)
    return {
        "operator_invocations": len(details),
        "unique_operators": sorted(counts),
        "operator_counts": dict(sorted(counts.items())),
        "contains_control_flow": any(name in counts for name in ("WHILE", "IF")),
        "contains_sequence_rnn_builtin": any(
            "LSTM" in name or "RNN" in name or "GRU" in name for name in counts
        ),
    }


def benchmark_tflite(
    model_path: Path,
    features: np.ndarray,
    *,
    warmup_runs: int = 25,
    measured_runs: int = 250,
) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    for index in range(warmup_runs):
        row = features[index % len(features)]
        interpreter.set_tensor(input_detail["index"], row[np.newaxis, ...])
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
    durations_ms: list[float] = []
    for index in range(measured_runs):
        row = features[index % len(features)]
        start = time.perf_counter_ns()
        interpreter.set_tensor(input_detail["index"], row[np.newaxis, ...])
        interpreter.invoke()
        interpreter.get_tensor(output_detail["index"])
        durations_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "environment": "host CPU, TensorFlow Lite Python interpreter, one thread",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "minimum_ms": float(np.min(durations_ms)),
        "maximum_ms": float(np.max(durations_ms)),
        "android_npu_latency_claim": False,
    }


def export_architecture(
    architecture_name: str,
    model: Any,
    builder: Callable[[int, int, int | None], Any],
    vocab_size: int,
    verification_features: np.ndarray,
    model_dir: Path,
) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    result: dict[str, Any] = {
        "builtins_only_requested": True,
        "success": False,
    }
    try:
        fixed_model = builder(vocab_size, MAX_LEN, 1)
        fixed_model.set_weights(model.get_weights())
        model_path = model_dir / "model_builtin.tflite"
        baseline.convert_builtin_tflite(fixed_model, model_path)
        analyzer_text = baseline._analyze_tflite(model_path)
        (model_dir / "tflite_analyzer.txt").write_text(
            analyzer_text, encoding="utf-8"
        )
        tflite_probabilities = baseline._tflite_predict(
            model_path, verification_features
        )
        keras_probabilities = np.asarray(
            model.predict(verification_features, verbose=0), dtype=np.float32
        )
        parity = baseline._parity_metrics(
            keras_probabilities, tflite_probabilities
        )
        result.update(
            {
                "success": True,
                "model_size_bytes": model_path.stat().st_size,
                "parity": parity,
                "operators": tflite_operator_summary(model_path),
                "host_cpu_latency": benchmark_tflite(
                    model_path, verification_features
                ),
            }
        )
        del fixed_model
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        (model_dir / "tflite_export_error.txt").write_text(
            result["error"] + "\n", encoding="utf-8"
        )
    finally:
        tf.keras.backend.clear_session()
    write_json(model_dir / "export_summary.json", result)
    return result


def train_architecture_comparison(
    datasets: dict[str, list[baseline.Sample]],
    token_to_id: dict[str, int],
    output_dir: Path,
    verbose: int,
) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    encoded = {
        name: baseline.encode_samples(samples, token_to_id, MAX_LEN)
        for name, samples in datasets.items()
        if name
        in {
            "clean_train",
            "semantic_val",
            "asr_val",
            "complex_val",
            "combined_val",
            "challenge_40",
            "hard_cases_9",
            "challenge_49",
            "structural_probes",
        }
    }
    train_data = encoded["clean_train"]
    validation_data = encoded["combined_val"]
    evaluation_names = (
        "semantic_val",
        "asr_val",
        "complex_val",
        "combined_val",
        "challenge_40",
        "hard_cases_9",
        "challenge_49",
        "structural_probes",
    )
    builders: list[tuple[str, Callable[[int, int, int | None], Any]]] = [
        ("baseline_cnn", build_baseline_cnn),
        ("strong_cnn", build_strong_cnn),
        ("small_gru", build_small_gru),
    ]
    results: dict[str, Any] = {}
    all_prediction_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for architecture_name, builder in builders:
        tf.keras.backend.clear_session()
        baseline.set_global_determinism(SEED)
        model_dir = output_dir / "architecture_comparison" / architecture_name
        model = builder(len(token_to_id), MAX_LEN, None)
        parameter_count = int(model.count_params())
        print(
            f"\n=== Diagnostic M0 training: {architecture_name} "
            f"({parameter_count:,} parameters) ==="
        )
        baseline._train_phase(
            model,
            model_dir,
            train_data,
            validation_data,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            patience=PATIENCE,
            batch_size=BATCH_SIZE,
            seed=SEED,
            verbose=verbose,
        )
        model_result: dict[str, Any] = {
            "architecture": ARCHITECTURE_DESCRIPTIONS[architecture_name],
            "parameter_count": parameter_count,
            "training": history_summary(model_dir / "history.csv"),
            "evaluations": {},
        }
        for dataset_name in evaluation_names:
            probabilities = np.asarray(
                model.predict(
                    encoded[dataset_name][0], batch_size=BATCH_SIZE, verbose=0
                ),
                dtype=np.float32,
            )
            metrics = metrics_from_probabilities(
                datasets[dataset_name], probabilities
            )
            if dataset_name == "structural_probes":
                metrics["structural_pairs"] = structural_pair_metrics(
                    datasets[dataset_name], probabilities
                )
            model_result["evaluations"][dataset_name] = metrics
            all_prediction_rows.extend(
                prediction_rows(
                    architecture_name,
                    dataset_name,
                    datasets[dataset_name],
                    probabilities,
                )
            )
            write_confusion(
                model_dir / f"confusion_matrix_{dataset_name}.csv",
                metrics["confusion_matrix"],
            )
            for label in baseline.LABELS:
                recall_rows.append(
                    {
                        "model": architecture_name,
                        "dataset": dataset_name,
                        "label": label,
                        "recall": metrics["per_class"][label]["recall"],
                        "support": metrics["per_class"][label]["support"],
                    }
                )
        verification_features = encoded["combined_val"][0][:100]
        model_result["tflite_export"] = export_architecture(
            architecture_name,
            model,
            builder,
            len(token_to_id),
            verification_features,
            model_dir,
        )
        write_json(model_dir / "metrics.json", model_result)
        results[architecture_name] = model_result
        hard = model_result["evaluations"]["hard_cases_9"]
        comparison_rows.append(
            {
                "model": architecture_name,
                "parameters": parameter_count,
                "best_epoch": model_result["training"]["best_epoch_one_based"],
                "semantic_accuracy": model_result["evaluations"]["semantic_val"]["accuracy"],
                "semantic_macro_f1": model_result["evaluations"]["semantic_val"]["macro_f1_all_10_classes"],
                "asr_accuracy": model_result["evaluations"]["asr_val"]["accuracy"],
                "asr_macro_f1": model_result["evaluations"]["asr_val"]["macro_f1_all_10_classes"],
                "complex_accuracy": model_result["evaluations"]["complex_val"]["accuracy"],
                "complex_macro_f1": model_result["evaluations"]["complex_val"]["macro_f1_all_10_classes"],
                "combined_accuracy": model_result["evaluations"]["combined_val"]["accuracy"],
                "combined_macro_f1": model_result["evaluations"]["combined_val"]["macro_f1_all_10_classes"],
                "challenge_40_accuracy": model_result["evaluations"]["challenge_40"]["accuracy"],
                "hard_cases_correct": hard["correct_count"],
                "hard_cases_accuracy": hard["accuracy"],
                "hard_mean_confidence_correct": hard["mean_confidence_correct"],
                "hard_mean_confidence_wrong": hard["mean_confidence_wrong"],
                "structural_probe_accuracy": model_result["evaluations"]["structural_probes"]["accuracy"],
                "structural_pairs_resolved": model_result["evaluations"]["structural_probes"]["structural_pairs"]["resolved_pairs"],
                "tflite_builtins_export": model_result["tflite_export"]["success"],
                "tflite_size_bytes": model_result["tflite_export"].get("model_size_bytes"),
                "host_cpu_median_ms": model_result["tflite_export"].get("host_cpu_latency", {}).get("median_ms"),
            }
        )
        del model
        tf.keras.backend.clear_session()
    write_json(output_dir / "architecture_comparison" / "all_metrics.json", results)
    write_csv(
        output_dir / "architecture_comparison" / "model_comparison.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    write_csv(
        output_dir / "architecture_comparison" / "per_class_recall.csv",
        recall_rows,
        ["model", "dataset", "label", "recall", "support"],
    )
    write_csv(
        output_dir / "architecture_comparison" / "full_softmax_predictions.csv",
        all_prediction_rows,
        [
            "model",
            "dataset",
            "row",
            "source_line",
            "text",
            "normalized_text",
            "expected_label",
            "predicted_label",
            "confidence",
            "correct",
            *[f"p_{label}" for label in baseline.LABELS],
        ],
    )
    return results


def write_run_protocol(
    current_dir: Path,
    output_dir: Path,
    datasets: dict[str, list[baseline.Sample]],
    vocabulary_size: int,
) -> None:
    protocol = {
        "purpose": "diagnosis plus M0-only fair architecture comparison",
        "source_model_directory": str(current_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "seed": SEED,
        "max_len": MAX_LEN,
        "vocabulary_size": vocabulary_size,
        "training": {
            "dataset": "clean_organic + clean_main only",
            "sample_count": len(datasets["clean_train"]),
            "epochs_max": EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "early_stopping_monitor": "combined_val loss",
        },
        "development_validation": {
            "semantic_val": len(datasets["semantic_val"]),
            "asr_val": len(datasets["asr_val"]),
            "complex_val": len(datasets["complex_val"]),
            "combined_val": len(datasets["combined_val"]),
        },
        "untouched_evaluation_only": {
            "challenge_40": len(datasets["challenge_40"]),
            "hard_cases_9": len(datasets["hard_cases_9"]),
            "note": (
                "The nine appended hard cases duplicate rows within Challenge-40; "
                "the exact-overlap audit is written by token_diagnostics."
            ),
        },
        "architectures": ARCHITECTURE_DESCRIPTIONS,
        "constraints": {
            "datasets_modified": False,
            "existing_models_overwritten": False,
            "full_m0_m3_retrain": False,
            "new_m0_only_models_trained": 3,
        },
    }
    write_json(output_dir / "run_protocol.json", protocol)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose current intent models and compare three M0 architectures."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Write tokenizer/current-model/shortcut diagnostics only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    current_dir = root / CURRENT_RUN
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if output_dir.exists():
        print(
            f"ERROR: output directory already exists; refusing to overwrite: {output_dir}",
            file=os.sys.stderr,
        )
        return 2
    if not current_dir.is_dir():
        print(f"ERROR: current model directory not found: {current_dir}", file=os.sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    try:
        datasets = load_all_inputs(root, current_dir)
        vocabulary, token_to_id = load_tokenizer(current_dir)
        write_run_protocol(
            current_dir, output_dir, datasets, len(vocabulary)
        )
        tokenizer_summary = token_diagnostics(
            datasets, token_to_id, output_dir
        )
        write_json(output_dir / "tokenizer" / "summary.json", tokenizer_summary)
        explicit_rows = explicit_shortcut_analysis(datasets, output_dir)
        discovered_rows, feature_maps = discovered_shortcut_analysis(
            datasets, output_dir
        )
        write_challenge_feature_evidence(
            datasets["challenge_49"],
            feature_maps["all_training"],
            output_dir,
        )
        write_json(
            output_dir / "shortcuts" / "summary.json",
            {
                "explicit_pattern_rows": len(explicit_rows),
                "discovered_association_rows": len(discovered_rows),
                "minimum_discovered_feature_support": 5,
                "scopes": ["clean_train", "vosk_train", "all_training"],
            },
        )
        evaluate_current_models(
            current_dir, datasets, token_to_id, output_dir
        )
        if not args.skip_training:
            train_architecture_comparison(
                datasets, token_to_id, output_dir, args.verbose
            )
        print(f"\nDiagnostic artifacts completed: {output_dir}")
        return 0
    except Exception as error:
        (output_dir / "FAILED.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
