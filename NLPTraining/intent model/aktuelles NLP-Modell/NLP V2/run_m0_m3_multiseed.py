#!/usr/bin/env python3
"""Train and analyze the frozen M0-M3 x T1/T2 five-seed experiment.

The current NLP V2 data and tokenizer artifacts are treated as immutable inputs.
Every exported model is the fixed BaselineCNN with an external tokenizer and a
fixed int32[1, 24] TFLite input.  Challenge-40 is evaluated only. It is never
used for training, early stopping, ranking, or representative-seed selection in order to guarantee 
a critical validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "experiment_config.json"


class ExperimentConfigError(RuntimeError):
    """Raised when the external experiment configuration is malformed."""


def read_experiment_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExperimentConfigError(f"Experiment configuration is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ExperimentConfigError(f"Invalid JSON in experiment configuration {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ExperimentConfigError(
            f"Unsupported experiment configuration schema in {path}; expected schema_version 1"
        )
    return payload


BOOTSTRAP_CONFIG = read_experiment_config(DEFAULT_CONFIG_PATH)
NLP_V1 = ROOT / BOOTSTRAP_CONFIG["experiment"]["legacy_implementation_directory"]
if not NLP_V1.is_dir():
    raise RuntimeError(f"Required implementation directory not found: {NLP_V1}")
sys.path.insert(0, str(NLP_V1))

import compare_intent_tokenizers as tokenizers_v1  # noqa: E402
import diagnose_intent_model_architectures as diagnosis  # noqa: E402
import train_intent_models_npu as baseline  # noqa: E402


CONFIG_PATH = DEFAULT_CONFIG_PATH
EXPERIMENT_CONFIG: dict[str, Any] = {}
SEEDS: tuple[int, ...]
MAX_LEN: int
BATCH_SIZE: int
EPOCHS: int
PATIENCE: int
PHASE1_STRATEGY: str
PARITY_TOLERANCE: float
PARITY_VERIFICATION_SAMPLES: int
STRATEGIES: tuple[str, ...]
STRATEGY_CONFIGS: dict[str, dict[str, Any]]
STRATEGY_DESCRIPTIONS: dict[str, str]
LEARNING_RATES: dict[str, float]
TOKENIZER_VARIANTS: tuple[str, ...]
TOKENIZER_CONFIGS: dict[str, dict[str, Any]]
MODEL_FOLDER_NAMES: dict[tuple[str, str], str]
TRAINING_FILES: dict[str, Path]
VALIDATION_FILES: dict[str, Path]
TRAINING_GROUPS: dict[str, tuple[str, ...]]
TOKENIZER_SOURCE_DIRS: dict[str, Path]
EXPECTED_INPUTS: dict[Path, tuple[int, str]]
EXPECTED_TOKENIZER_HASHES: dict[str, dict[str, str]]
EXPECTED_COUNTS: dict[str, int]
PRIMARY_EVALUATIONS: tuple[str, ...]
ALL_EVALUATIONS: tuple[str, ...]
DEVELOPMENT_EVALUATIONS: tuple[str, ...]
CHALLENGE_EVALUATION: str
KNOWN_BOUNDARY_SUBSET: str
DISPLAY_DATASETS: dict[str, str]
BASELINE_ARCHITECTURE: dict[str, Any]
TFLITE_TENSOR_CONTRACT: dict[str, Any]
REPRESENTATIVE_SEED_RULE: dict[str, Any]
RANKING_RULE: dict[str, Any]
DEFAULT_OUTPUT_DIR: Path


def configure_experiment(path: Path) -> None:
    """Load the experiment definition and derive runtime lookup tables from it."""
    global CONFIG_PATH, EXPERIMENT_CONFIG, SEEDS, MAX_LEN, BATCH_SIZE, EPOCHS, PATIENCE
    global PHASE1_STRATEGY
    global PARITY_TOLERANCE, PARITY_VERIFICATION_SAMPLES, STRATEGIES, STRATEGY_CONFIGS
    global STRATEGY_DESCRIPTIONS, LEARNING_RATES, TOKENIZER_VARIANTS, TOKENIZER_CONFIGS
    global MODEL_FOLDER_NAMES, TRAINING_FILES, VALIDATION_FILES, TRAINING_GROUPS
    global TOKENIZER_SOURCE_DIRS, EXPECTED_INPUTS, EXPECTED_TOKENIZER_HASHES
    global EXPECTED_COUNTS, PRIMARY_EVALUATIONS, ALL_EVALUATIONS, DEVELOPMENT_EVALUATIONS
    global CHALLENGE_EVALUATION, KNOWN_BOUNDARY_SUBSET, DISPLAY_DATASETS
    global BASELINE_ARCHITECTURE, TFLITE_TENSOR_CONTRACT, REPRESENTATIVE_SEED_RULE
    global RANKING_RULE, DEFAULT_OUTPUT_DIR

    CONFIG_PATH = path.resolve()
    config = read_experiment_config(CONFIG_PATH)
    EXPERIMENT_CONFIG = config
    config_root = CONFIG_PATH.parent
    try:
        training = config["training"]
        strategy_rows = config["strategies"]
        tokenizer_rows = config["tokenizers"]
        datasets = config["datasets"]
        evaluations = datasets["evaluations"]
        training_rows = datasets["training"]
        validation_rows = datasets["validation"]
        STRATEGIES = tuple(str(row["id"]) for row in strategy_rows)
        STRATEGY_CONFIGS = {str(row["id"]): dict(row) for row in strategy_rows}
        STRATEGY_DESCRIPTIONS = {
            strategy: str(row["description"])
            for strategy, row in STRATEGY_CONFIGS.items()
        }
        LEARNING_RATES = {
            strategy: float(row["learning_rate"])
            for strategy, row in STRATEGY_CONFIGS.items()
        }
        TOKENIZER_VARIANTS = tuple(str(row["id"]) for row in tokenizer_rows)
        TOKENIZER_CONFIGS = {str(row["id"]): dict(row) for row in tokenizer_rows}
        SEEDS = tuple(int(seed) for seed in training["seeds"])
        MAX_LEN = int(training["max_len"])
        BATCH_SIZE = int(training["batch_size"])
        EPOCHS = int(training["epochs"])
        PATIENCE = int(training["patience"])
        PHASE1_STRATEGY = str(training["phase1_strategy"])
        PARITY_TOLERANCE = float(training["parity_tolerance"])
        PARITY_VERIFICATION_SAMPLES = int(training["parity_verification_samples"])
        TRAINING_FILES = {
            str(row["id"]): config_root / str(row["path"])
            for row in training_rows
        }
        VALIDATION_FILES = {
            str(row["id"]): config_root / str(row["path"])
            for row in validation_rows
        }
        TRAINING_GROUPS = {
            str(group): tuple(str(source) for source in sources)
            for group, sources in datasets["training_groups"].items()
        }
        TOKENIZER_SOURCE_DIRS = {
            variant: config_root / str(row["source_directory"])
            for variant, row in TOKENIZER_CONFIGS.items()
        }
        MODEL_FOLDER_NAMES = {
            (strategy, variant): str(TOKENIZER_CONFIGS[variant]["model_folder_name"]).format(
                strategy=strategy, tokenizer=variant
            )
            for strategy in STRATEGIES
            for variant in TOKENIZER_VARIANTS
        }
        expected_rows = [*training_rows, *validation_rows]
        expected_paths = {str(row["id"]): row for row in expected_rows}
        EXPECTED_INPUTS = {
            (TRAINING_FILES | VALIDATION_FILES)[name]: (
                int(row["expected_samples"]), str(row["sha256"])
            )
            for name, row in expected_paths.items()
        }
        EXPECTED_TOKENIZER_HASHES = {
            variant: {
                str(filename): str(expected_hash)
                for filename, expected_hash in row["expected_files"].items()
            }
            for variant, row in TOKENIZER_CONFIGS.items()
        }
        EXPECTED_COUNTS = {
            str(name): int(count)
            for name, count in datasets["expected_aggregate_counts"].items()
        }
        PRIMARY_EVALUATIONS = tuple(str(name) for name in evaluations["primary"])
        DEVELOPMENT_EVALUATIONS = tuple(str(name) for name in evaluations["development"])
        KNOWN_BOUNDARY_SUBSET = str(evaluations["known_boundary_subset"])
        CHALLENGE_EVALUATION = str(evaluations["challenge"])
        ALL_EVALUATIONS = (*PRIMARY_EVALUATIONS, KNOWN_BOUNDARY_SUBSET)
        DISPLAY_DATASETS = {
            str(row["id"]): str(row["display_name"])
            for row in validation_rows
        }
        BASELINE_ARCHITECTURE = dict(config["audit"]["baseline_architecture"])
        TFLITE_TENSOR_CONTRACT = dict(config["audit"]["tflite_tensor_contract"])
        REPRESENTATIVE_SEED_RULE = dict(config["selection"]["representative_seed_rule"])
        RANKING_RULE = dict(config["selection"]["ranking_rule"])
        DEFAULT_OUTPUT_DIR = config_root / str(config["experiment"]["default_output_directory"])
    except (KeyError, TypeError, ValueError) as error:
        raise ExperimentConfigError(
            f"Incomplete or invalid experiment configuration {CONFIG_PATH}: {error}"
        ) from error

    if not STRATEGIES or not TOKENIZER_VARIANTS or not SEEDS:
        raise ExperimentConfigError("Strategies, tokenizers, and seeds must not be empty")
    if PHASE1_STRATEGY not in STRATEGY_CONFIGS:
        raise ExperimentConfigError("The phase-1 strategy must be one of the configured strategies")
    if len(STRATEGIES) != len(STRATEGY_CONFIGS) or len(TOKENIZER_VARIANTS) != len(TOKENIZER_CONFIGS):
        raise ExperimentConfigError("Strategy and tokenizer IDs must be unique")
    if set(PRIMARY_EVALUATIONS) - set(VALIDATION_FILES):
        raise ExperimentConfigError("Primary evaluations must reference configured validation files")
    if set(DEVELOPMENT_EVALUATIONS) - set(PRIMARY_EVALUATIONS):
        raise ExperimentConfigError("Development evaluations must be primary evaluations")
    if CHALLENGE_EVALUATION not in PRIMARY_EVALUATIONS:
        raise ExperimentConfigError("The challenge evaluation must be a primary evaluation")
    if KNOWN_BOUNDARY_SUBSET not in VALIDATION_FILES:
        raise ExperimentConfigError("The configured boundary subset is not a validation file")
    tokenizer_kinds = [str(row["kind"]) for row in TOKENIZER_CONFIGS.values()]
    if tokenizer_kinds.count("word") != 1 or tokenizer_kinds.count("bpe") != 1:
        raise ExperimentConfigError("Exactly one word tokenizer and one BPE tokenizer are required")
    if any(row["training_dataset"] not in {"clean_train", "vosk_train", "mixed_train"} for row in STRATEGY_CONFIGS.values()):
        raise ExperimentConfigError("Each strategy must use clean_train, vosk_train, or mixed_train")
    if any(row["initialization"] not in {"initial_weights", "m0_checkpoint"} for row in STRATEGY_CONFIGS.values()):
        raise ExperimentConfigError("Each strategy must use initial_weights or m0_checkpoint")


configure_experiment(DEFAULT_CONFIG_PATH)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline._json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_checksum(samples: Sequence[baseline.Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.normalized_text.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(sample.label.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise baseline.PipelineError(message)


def sample_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def combination_id(strategy: str, tokenizer: str) -> str:
    return f"{strategy}_{tokenizer}"


def tokenizer_variant_by_kind(kind: str) -> str:
    return next(
        variant
        for variant in TOKENIZER_VARIANTS
        if TOKENIZER_CONFIGS[variant]["kind"] == kind
    )


def expected_model_count() -> int:
    return len(SEEDS) * len(STRATEGIES) * len(TOKENIZER_VARIANTS)


def model_dir(output_dir: Path, seed: int, strategy: str, tokenizer: str) -> Path:
    return output_dir / "Modelle" / f"Seed {seed}" / MODEL_FOLDER_NAMES[(strategy, tokenizer)]


def load_datasets() -> dict[str, list[baseline.Sample]]:
    clean_train: list[baseline.Sample] = []
    for name in TRAINING_GROUPS["clean"]:
        clean_train.extend(
            baseline.load_dataset(TRAINING_FILES[name], source=name, domain="clean")
        )
    vosk_train: list[baseline.Sample] = []
    for name in TRAINING_GROUPS["vosk"]:
        vosk_train.extend(
            baseline.load_dataset(TRAINING_FILES[name], source=name, domain="vosk")
        )
    datasets: dict[str, list[baseline.Sample]] = {
        "clean_train": clean_train,
        "vosk_train": vosk_train,
        "final_train": [*clean_train, *vosk_train],
    }
    for name, path in VALIDATION_FILES.items():
        datasets[name] = baseline.load_dataset(
            path,
            source=name,
            domain="challenge" if name.startswith("challenge") else "validation",
        )
    datasets["combined_val"] = [
        sample
        for name in DEVELOPMENT_EVALUATIONS
        for sample in datasets[name]
    ]
    return datasets


def normalized_index(
    samples: Sequence[baseline.Sample], *, include_label: bool
) -> dict[Any, list[baseline.Sample]]:
    result: dict[Any, list[baseline.Sample]] = defaultdict(list)
    for sample in samples:
        key: Any = (
            (sample.normalized_text, sample.label)
            if include_label
            else sample.normalized_text
        )
        result[key].append(sample)
    return result


def verify_frozen_files(datasets: dict[str, list[baseline.Sample]]) -> dict[str, Any]:
    file_rows: list[dict[str, Any]] = []
    for path, (expected_samples, expected_hash) in EXPECTED_INPUTS.items():
        require(path.is_file(), f"Frozen input file is missing: {path}")
        actual_hash = sha256_file(path)
        require(
            actual_hash == expected_hash,
            f"Frozen input hash changed for {path}: {actual_hash} != {expected_hash}",
        )
        source_name = next(
            (name for name, candidate in {**TRAINING_FILES, **VALIDATION_FILES}.items() if candidate == path),
            path.name,
        )
        actual_samples = (
            len(datasets[source_name])
            if source_name in datasets
            else len(
                baseline.load_dataset(
                    path,
                    source=source_name,
                    domain="clean" if "vosk" not in source_name else "vosk",
                )
            )
        )
        require(
            actual_samples == expected_samples,
            f"Frozen sample count changed for {path}: {actual_samples} != {expected_samples}",
        )
        file_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "samples": actual_samples,
                "sha256": actual_hash,
                "matches_frozen_reference": True,
            }
        )
    for name, expected in EXPECTED_COUNTS.items():
        require(
            len(datasets[name]) == expected,
            f"{name} must contain {expected} samples; found {len(datasets[name])}",
        )
    return {"files": file_rows, "all_hashes_match": True}


def load_tokenizers(
    datasets: dict[str, list[baseline.Sample]],
) -> tuple[dict[str, tokenizers_v1.Tokenizer], dict[str, Any]]:
    hash_report: dict[str, Any] = {}
    for variant, expected_files in EXPECTED_TOKENIZER_HASHES.items():
        source_dir = TOKENIZER_SOURCE_DIRS[variant]
        hash_report[variant] = {}
        for filename, expected_hash in expected_files.items():
            path = source_dir / filename
            require(path.is_file(), f"Frozen tokenizer artifact is missing: {path}")
            actual_hash = sha256_file(path)
            require(
                actual_hash == expected_hash,
                f"Frozen tokenizer hash changed for {path}: {actual_hash} != {expected_hash}",
            )
            hash_report[variant][filename] = actual_hash

    tokenizers: dict[str, tokenizers_v1.Tokenizer] = {}
    word_variant = next(
        (variant for variant in TOKENIZER_VARIANTS if TOKENIZER_CONFIGS[variant]["kind"] == "word"),
        None,
    )
    bpe_variant = next(
        (variant for variant in TOKENIZER_VARIANTS if TOKENIZER_CONFIGS[variant]["kind"] == "bpe"),
        None,
    )
    require(word_variant is not None, "No word tokenizer is configured")
    require(bpe_variant is not None, "No BPE tokenizer is configured")

    word_config = TOKENIZER_CONFIGS[word_variant]
    word_vocab = read_json(TOKENIZER_SOURCE_DIRS[word_variant] / "vocab.json")
    t1 = tokenizers_v1.WordTokenizer(str(word_config["name"]), word_vocab, MAX_LEN)
    tokenizers[word_variant] = t1

    bpe_config = TOKENIZER_CONFIGS[bpe_variant]
    bpe_vocab = read_json(TOKENIZER_SOURCE_DIRS[bpe_variant] / "vocab.json")
    raw_merges = read_json(TOKENIZER_SOURCE_DIRS[bpe_variant] / "merges.json")
    t2_merges = [
        tokenizers_v1.BpeMerge(str(row["left"]), str(row["right"]), str(row["merged"]))
        for expected_rank, row in enumerate(raw_merges)
        if int(row["rank"]) == expected_rank
    ]
    require(len(t2_merges) == len(raw_merges), "BPE merge ranks are not contiguous")
    t2 = tokenizers_v1.BpeTokenizer(
        str(bpe_config["name"]),
        bpe_vocab,
        t2_merges,
        MAX_LEN,
        int(bpe_config["vocabulary_size"]),
    )
    tokenizers[bpe_variant] = t2

    training_words = {
        word
        for sample in datasets["final_train"]
        for word in sample.normalized_text.split()
    }
    require(
        set(t1.vocabulary[2:]) == training_words,
        "Word-tokenizer vocabulary no longer equals all frequency>=1 frozen training words",
    )
    training_characters = sorted({character for word in training_words for character in word})
    expected_t2_vocab = [
        tokenizers_v1.PAD_TOKEN,
        tokenizers_v1.UNK_TOKEN,
        tokenizers_v1.BPE_BOUNDARY,
        *training_characters,
        *(merge.merged for merge in t2.merges),
    ]
    require(t2.vocabulary == expected_t2_vocab, "BPE vocabulary/merge chain audit failed")
    evaluation_words = {
        word
        for name in PRIMARY_EVALUATIONS
        for sample in datasets[name]
        for word in sample.normalized_text.split()
    }
    evaluation_only_in_t1 = sorted((evaluation_words - training_words) & set(t1.vocabulary))
    require(not evaluation_only_in_t1, "Evaluation-only words leaked into T1 vocabulary")
    return tokenizers, {
        "hashes": hash_report,
        word_variant: {
            "vocabulary_size": len(t1.vocabulary),
            "training_unique_words": len(training_words),
            "exact_training_word_boundary": True,
            "evaluation_only_words_in_vocabulary": [],
        },
        bpe_variant: {
            "vocabulary_size": len(t2.vocabulary),
            "merge_count": len(t2.merges),
            "training_base_characters": len(training_characters),
            "valid_training_derived_merge_chain": True,
        },
        "max_len": MAX_LEN,
    }


def audit_data(
    datasets: dict[str, list[baseline.Sample]], output_dir: Path
) -> dict[str, Any]:
    validation_reports = {
        name: baseline.validate_dataset(samples, name)
        for name, samples in datasets.items()
        if name != "combined_val"
    }
    train_by_text = normalized_index(datasets["final_train"], include_label=False)
    overlap_rows: list[dict[str, Any]] = []
    for dataset in PRIMARY_EVALUATIONS:
        for row, sample in enumerate(datasets[dataset], start=1):
            for training in train_by_text.get(sample.normalized_text, []):
                overlap_rows.append(
                    {
                        "dataset": dataset,
                        "row": row,
                        "text": sample.text,
                        "expected_label": sample.label,
                        "training_source": training.source,
                        "training_line": training.line_no,
                        "training_label": training.label,
                    }
                )
    require(not overlap_rows, "Frozen training/evaluation normalized overlap detected")

    challenge_index = normalized_index(datasets[CHALLENGE_EVALUATION], include_label=True)
    mapping_rows: list[dict[str, Any]] = []
    for subset_row, sample in enumerate(
        datasets[KNOWN_BOUNDARY_SUBSET], start=1
    ):
        matches = challenge_index.get((sample.normalized_text, sample.label), [])
        require(
            len(matches) == 1,
            f"Boundary subset row {subset_row} must map once into {DISPLAY_DATASETS[CHALLENGE_EVALUATION]}",
        )
        challenge_row = datasets[CHALLENGE_EVALUATION].index(matches[0]) + 1
        mapping_rows.append(
            {
                "boundary_row": subset_row,
                "challenge_row": challenge_row,
                "text": sample.text,
                "expected_label": sample.label,
                "relationship": f"member_of_{CHALLENGE_EVALUATION}_not_independent_testset",
            }
        )
    write_csv(
        output_dir / "Analysen" / "data" / "known_boundary_subset_mapping.csv",
        mapping_rows,
        ["boundary_row", "challenge_row", "text", "expected_label", "relationship"],
    )
    write_csv(
        output_dir / "Analysen" / "data" / "training_evaluation_overlap_audit.csv",
        overlap_rows,
        [
            "dataset",
            "row",
            "text",
            "expected_label",
            "training_source",
            "training_line",
            "training_label",
        ],
    )
    return {
        "validation_reports": validation_reports,
        "training_evaluation_normalized_overlaps": 0,
        "known_boundary_subset_samples": len(mapping_rows),
        "known_boundary_subset_is_independent": False,
        "training_counts": {
            "clean": len(datasets["clean_train"]),
            "vosk": len(datasets["vosk_train"]),
            "joint": len(datasets["final_train"]),
        },
        "evaluation_counts": {
            name: len(datasets[name]) for name in PRIMARY_EVALUATIONS
        },
        "training_corpus_checksum_sha256": corpus_checksum(datasets["final_train"]),
    }


def architecture_audit(tokenizers: dict[str, tokenizers_v1.Tokenizer]) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    report: dict[str, Any] = {}
    for variant, tokenizer in tokenizers.items():
        tf.keras.backend.clear_session()
        baseline.set_global_determinism(SEEDS[0])
        model = baseline.build_model(
            len(tokenizer.vocabulary), MAX_LEN, name=f"architecture_audit_{variant}"
        )
        layers = {layer.name: layer for layer in model.layers}
        require(tuple(model.input_shape) == (None, MAX_LEN), "BaselineCNN input shape changed")
        require(tuple(model.output_shape) == (None, len(baseline.LABELS)), "BaselineCNN output shape changed")
        require(
            layers["embedding"].output_dim == BASELINE_ARCHITECTURE["embedding_dimension"],
            "Embedding dimension changed",
        )
        require(layers["embedding"].mask_zero is False, "Embedding mask_zero changed")
        require(
            layers["conv1"].filters == BASELINE_ARCHITECTURE["conv_filters"],
            "Conv1D filters changed",
        )
        require(
            tuple(layers["conv1"].kernel_size)
            == (BASELINE_ARCHITECTURE["conv_kernel_size"],),
            "Conv1D kernel changed",
        )
        require(layers["conv1"].padding == "same", "Conv1D padding changed")
        require(
            layers["dense_hidden"].units == BASELINE_ARCHITECTURE["dense_units"],
            "Dense width changed",
        )
        require(
            math.isclose(layers["dropout"].rate, BASELINE_ARCHITECTURE["dropout_rate"]),
            "Dropout rate changed",
        )
        report[variant] = {
            "architecture": (
                f"Embedding({BASELINE_ARCHITECTURE['embedding_dimension']}) -> "
                f"Conv1D({BASELINE_ARCHITECTURE['conv_filters']},"
                f"k={BASELINE_ARCHITECTURE['conv_kernel_size']},same,relu) -> "
                f"GlobalMax+GlobalMean -> Dense({BASELINE_ARCHITECTURE['dense_units']},relu) -> "
                f"Dropout({BASELINE_ARCHITECTURE['dropout_rate']}) -> "
                f"Dense({len(baseline.LABELS)},softmax)"
            ),
            "parameter_count": int(model.count_params()),
            "input_shape": [None, MAX_LEN],
            "input_dtype": "int32",
            "output_shape": [None, len(baseline.LABELS)],
            "output_dtype": "float32",
            "matches_frozen_baseline": True,
        }
        del model
    tf.keras.backend.clear_session()
    return report


def copy_tokenizer_artifacts(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)
    write_json(target_dir / "labels.json", baseline.LABELS)


def encode_all(
    tokenizer: tokenizers_v1.Tokenizer,
    datasets: dict[str, list[baseline.Sample]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: tokenizers_v1.encode_samples(tokenizer, samples)
        for name, samples in datasets.items()
    }


def concatenate_encoded(
    sets: Iterable[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    materialized = list(sets)
    return (
        np.concatenate([features for features, _labels in materialized], axis=0),
        np.concatenate([labels for _features, labels in materialized], axis=0),
    )


def tflite_tensor_contract(path: Path) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    require(len(inputs) == 1 and len(outputs) == 1, f"Unexpected tensor count: {path}")
    contract = {
        "input_count": 1,
        "input_shape": [int(value) for value in inputs[0]["shape"]],
        "input_dtype": str(inputs[0]["dtype"].__name__),
        "output_count": 1,
        "output_shape": [int(value) for value in outputs[0]["shape"]],
        "output_dtype": str(outputs[0]["dtype"].__name__),
    }
    require(
        contract["input_shape"] == TFLITE_TENSOR_CONTRACT["input_shape"],
        f"TFLite input shape changed: {path}",
    )
    require(
        contract["input_dtype"] == TFLITE_TENSOR_CONTRACT["input_dtype"],
        f"TFLite input dtype changed: {path}",
    )
    require(contract["output_shape"] == [1, len(baseline.LABELS)], f"TFLite output shape changed: {path}")
    require(
        contract["output_dtype"] == TFLITE_TENSOR_CONTRACT["output_dtype"],
        f"TFLite output dtype changed: {path}",
    )
    return contract


def training_args(seed: int, verbose: int, vocab_size: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        batch_size=BATCH_SIZE,
        epochs_m0=EPOCHS,
        epochs_m1=EPOCHS,
        epochs_m2=EPOCHS,
        epochs_m3=EPOCHS,
        patience_m0=PATIENCE,
        patience_m1=PATIENCE,
        patience_finetune=PATIENCE,
        verbose=verbose,
        actual_vocab_size=vocab_size,
        max_len=MAX_LEN,
        parity_tolerance=PARITY_TOLERANCE,
        skip_npu_core=True,
    )


def model_complete(path: Path) -> bool:
    required = (
        path / "metrics.json",
        path / "model.keras",
        path / "best.weights.h5",
        path / "model_builtin.tflite",
        path / "predictions_all_evaluation_sets.csv",
        path / "tokenizer" / "vocab.json",
        path / "tokenizer" / "tokenizer_config.json",
        path / "tokenizer" / "labels.json",
    )
    return all(candidate.is_file() for candidate in required)


def train_strategy(
    strategy: str,
    variant: str,
    seed: int,
    tokenizer: tokenizers_v1.Tokenizer,
    encoded: dict[str, tuple[np.ndarray, np.ndarray]],
    datasets: dict[str, list[baseline.Sample]],
    mixed_train: tuple[np.ndarray, np.ndarray],
    initial_weights: list[np.ndarray],
    m0_checkpoint: Path,
    output_dir: Path,
    verbose: int,
) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    strategy_config = STRATEGY_CONFIGS[strategy]
    destination = model_dir(output_dir, seed, strategy, variant)
    if model_complete(destination):
        stored = read_json(destination / "metrics.json")
        require(
            stored.get("seed") == seed
            and stored.get("strategy") == strategy
            and stored.get("tokenizer_variant") == variant,
            f"Existing completed model identity mismatch: {destination}",
        )
        print(f"SKIP complete {strategy}/{variant}/seed {seed}", flush=True)
        return stored
    require(
        not destination.exists(),
        f"Refusing partial model directory without an explicit clean restart: {destination}",
    )
    working_dir = destination.with_name(destination.name + ".partial")
    require(
        not working_dir.exists(),
        f"Refusing stale partial model directory: {working_dir}",
    )

    baseline.set_global_determinism(seed + STRATEGIES.index(strategy))
    model = baseline.build_model(
        len(tokenizer.vocabulary),
        MAX_LEN,
        name=f"{strategy}_{variant}_seed_{seed}",
    )
    if strategy_config["initialization"] == "initial_weights":
        model.set_weights([np.array(weight, copy=True) for weight in initial_weights])
    else:
        require(m0_checkpoint.is_file(), f"M0 phase-1 checkpoint missing: {m0_checkpoint}")
        model.load_weights(m0_checkpoint)

    args = training_args(seed, verbose, len(tokenizer.vocabulary))
    available_training_data = {
        "clean_train": encoded["clean_train"],
        "mixed_train": mixed_train,
        "vosk_train": encoded["vosk_train"],
    }
    train_data = available_training_data[strategy_config["training_dataset"]]
    print(
        f"\n=== TRAIN {strategy}/{variant}/seed {seed}: samples={len(train_data[0])}, "
        f"lr={LEARNING_RATES[strategy]:g} ===",
        flush=True,
    )
    train_function = getattr(baseline, str(strategy_config["training_function"]))
    train_function(model, working_dir, train_data, encoded["combined_val"], args)

    evaluations: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for dataset in ALL_EVALUATIONS:
        probabilities = np.asarray(
            model.predict(encoded[dataset][0], batch_size=BATCH_SIZE, verbose=0),
            dtype=np.float32,
        )
        metrics = diagnosis.metrics_from_probabilities(datasets[dataset], probabilities)
        evaluations[dataset] = metrics
        diagnosis.write_confusion(
            working_dir / f"confusion_matrix_{dataset}.csv",
            metrics["confusion_matrix"],
        )
        rows = diagnosis.prediction_rows(
            combination_id(strategy, variant), dataset, datasets[dataset], probabilities
        )
        for row in rows:
            row.update({"seed": seed, "strategy": strategy, "tokenizer": variant})
        prediction_rows.extend(rows)
        print(
            f"EVAL {strategy}/{variant}/{seed} {DISPLAY_DATASETS[dataset]}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1_present_truth_classes' if dataset == 'challenge_40' else 'macro_f1_all_10_classes']:.4f}",
            flush=True,
        )

    prediction_fields = [
        "seed",
        "strategy",
        "tokenizer",
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
        working_dir / "predictions_all_evaluation_sets.csv",
        prediction_rows,
        prediction_fields,
    )
    write_csv(
        working_dir / "misclassifications.csv",
        [row for row in prediction_rows if not bool(row["correct"])],
        prediction_fields,
    )

    verification_features = baseline._choose_verification_samples(
        encoded["combined_val"][0], PARITY_VERIFICATION_SAMPLES, seed
    )
    tflite = tokenizers_v1.export_model(
        model,
        len(tokenizer.vocabulary),
        MAX_LEN,
        verification_features,
        working_dir,
    )
    parity_features = np.concatenate(
        [encoded[name][0] for name in PRIMARY_EVALUATIONS], axis=0
    )
    keras_probabilities = np.asarray(
        model.predict(parity_features, batch_size=BATCH_SIZE, verbose=0), dtype=np.float32
    )
    lite_probabilities = baseline._tflite_predict(
        working_dir / "model_builtin.tflite", parity_features
    )
    full_parity = baseline._parity_metrics(keras_probabilities, lite_probabilities)
    require(
        full_parity["top1_agreement"] == 1.0
        and full_parity["max_absolute_probability_difference"] <= PARITY_TOLERANCE,
        f"Full primary-evaluation Keras/TFLite parity failed: {destination}: {full_parity}",
    )
    tflite["full_evaluation_parity"] = {
        **full_parity,
        "samples": len(parity_features),
    }
    tflite["tensor_contract"] = tflite_tensor_contract(
        working_dir / "model_builtin.tflite"
    )

    copy_tokenizer_artifacts(TOKENIZER_SOURCE_DIRS[variant], working_dir / "tokenizer")
    write_json(working_dir / "labels.json", baseline.LABELS)
    training_summary = diagnosis.history_summary(working_dir / "history.csv")
    phase1 = None
    if strategy_config["initialization"] == "m0_checkpoint":
        phase1 = {
            "strategy": f"{PHASE1_STRATEGY} Clean-only pretraining",
            "checkpoint": str(m0_checkpoint),
            "checkpoint_sha256": sha256_file(m0_checkpoint),
            "shared_with_newly_trained_M0_same_seed_and_tokenizer": True,
        }
    result = {
        "experiment": EXPERIMENT_CONFIG["experiment"]["id"],
        "seed": seed,
        "strategy": strategy,
        "strategy_description": strategy_config["description"],
        "tokenizer_variant": variant,
        "tokenizer_name": tokenizer.name,
        "training_samples_phase2_or_single_phase": len(train_data[0]),
        "phase1": phase1,
        "learning_rate": strategy_config["learning_rate"],
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "parameter_count": int(model.count_params()),
        "training": training_summary,
        "evaluations": evaluations,
        "tflite": tflite,
        "keras_size_bytes": (working_dir / "model.keras").stat().st_size,
        "tflite_size_bytes": (working_dir / "model_builtin.tflite").stat().st_size,
        "builtins_only_compatible": True,
    }
    write_json(working_dir / "metrics.json", result)
    write_json(
        working_dir / "model_identity.json",
        {
            "experiment": result["experiment"],
            "seed": seed,
            "strategy": strategy,
            "tokenizer_variant": variant,
            "architecture": "BaselineCNN frozen",
            "max_len": MAX_LEN,
            "fully_retrained": True,
            "old_model_checkpoint_reused": False,
            "phase1_checkpoint": phase1,
        },
    )
    working_dir.rename(destination)
    del model
    tf.keras.backend.clear_session()
    return result


def train_all(
    output_dir: Path,
    datasets: dict[str, list[baseline.Sample]],
    tokenizers: dict[str, tokenizers_v1.Tokenizer],
    verbose: int,
) -> None:
    tf = baseline.require_tensorflow()
    for seed in SEEDS:
        for variant in TOKENIZER_VARIANTS:
            tokenizer = tokenizers[variant]
            encoded = encode_all(tokenizer, datasets)
            mixed_features, mixed_labels, mixed_stats = baseline.make_domain_mix(
                *encoded["clean_train"],
                *encoded["vosk_train"],
                seed=seed,
            )
            seed_dir = output_dir / "Modelle" / f"Seed {seed}"
            write_json(seed_dir / f"mixed_sampling_{variant}.json", mixed_stats)
            baseline.set_global_determinism(seed)
            initial_model = baseline.build_model(
                len(tokenizer.vocabulary),
                MAX_LEN,
                name=f"initial_{variant}_seed_{seed}",
            )
            initial_weights = [
                np.array(weight, copy=True) for weight in initial_model.get_weights()
            ]
            initial_model.save_weights(
                seed_dir / f"initial_weights_{variant}_seed_{seed}.weights.h5"
            )
            del initial_model
            tf.keras.backend.clear_session()

            m0_path = model_dir(output_dir, seed, PHASE1_STRATEGY, variant)
            train_strategy(
                PHASE1_STRATEGY,
                variant,
                seed,
                tokenizer,
                encoded,
                datasets,
                (mixed_features, mixed_labels),
                initial_weights,
                m0_path / "best.weights.h5",
                output_dir,
                verbose,
            )
            for strategy in (item for item in STRATEGIES if item != PHASE1_STRATEGY):
                train_strategy(
                    strategy,
                    variant,
                    seed,
                    tokenizer,
                    encoded,
                    datasets,
                    (mixed_features, mixed_labels),
                    initial_weights,
                    m0_path / "best.weights.h5",
                    output_dir,
                    verbose,
                )
            del encoded, mixed_features, mixed_labels, initial_weights
            tf.keras.backend.clear_session()


def collect_results(output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            for variant in TOKENIZER_VARIANTS:
                path = model_dir(output_dir, seed, strategy, variant)
                require(model_complete(path), f"Incomplete model output: {path}")
                result = read_json(path / "metrics.json")
                require(
                    result["seed"] == seed
                    and result["strategy"] == strategy
                    and result["tokenizer_variant"] == variant,
                    f"Stored identity mismatch: {path}",
                )
                result["artifact_dir"] = str(path)
                results.append(result)
    require(
        len(results) == expected_model_count(),
        f"Expected {expected_model_count()} completed models; found {len(results)}",
    )
    return results


def metric_f1(metrics: dict[str, Any], dataset: str) -> float:
    key = (
        "macro_f1_present_truth_classes"
        if dataset == CHALLENGE_EVALUATION
        else "macro_f1_all_10_classes"
    )
    return float(metrics[key])


def aggregate_metrics(
    results: Sequence[dict[str, Any]], analysis_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for result in results:
        seed = int(result["seed"])
        strategy = str(result["strategy"])
        variant = str(result["tokenizer_variant"])
        combo = combination_id(strategy, variant)
        for dataset in PRIMARY_EVALUATIONS:
            metrics = result["evaluations"][dataset]
            metric_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    "tokenizer": variant,
                    "combination": combo,
                    "dataset": dataset,
                    "display_name": DISPLAY_DATASETS[dataset],
                    "samples": metrics["sample_count"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metric_f1(metrics, dataset),
                    "macro_f1_all_10_classes": metrics["macro_f1_all_10_classes"],
                    "macro_f1_present_truth_classes": metrics[
                        "macro_f1_present_truth_classes"
                    ],
                    "correct": metrics["correct_count"],
                    "wrong": metrics["wrong_count"],
                    "best_epoch": result["training"]["best_epoch_one_based"],
                    "best_validation_loss": result["training"]["best_validation_loss"],
                }
            )
            for label in baseline.LABELS:
                class_metrics = metrics["per_class"][label]
                recall_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "tokenizer": variant,
                        "combination": combo,
                        "dataset": dataset,
                        "label": label,
                        "recall": class_metrics["recall"],
                        "precision": class_metrics["precision"],
                        "f1": class_metrics["f1"],
                        "support": class_metrics["support"],
                    }
                )
        tflite = result["tflite"]
        resource_rows.append(
            {
                "seed": seed,
                "strategy": strategy,
                "tokenizer": variant,
                "combination": combo,
                "parameters": result["parameter_count"],
                "keras_size_bytes": result["keras_size_bytes"],
                "tflite_size_bytes": result["tflite_size_bytes"],
                "tflite_mean_ms_host": tflite["host_cpu_latency"]["mean_ms"],
                "tflite_median_ms_host": tflite["host_cpu_latency"]["median_ms"],
                "tflite_p95_ms_host": tflite["host_cpu_latency"]["p95_ms"],
                "keras_tflite_top1_agreement_700": tflite[
                    "full_evaluation_parity"
                ]["top1_agreement"],
                "keras_tflite_max_abs_probability_difference_700": tflite[
                    "full_evaluation_parity"
                ]["max_absolute_probability_difference"],
                "builtins_only": result["builtins_only_compatible"],
                "operators": json.dumps(tflite["operators"]["unique_operators"]),
                "input_shape": json.dumps(tflite["tensor_contract"]["input_shape"]),
                "input_dtype": tflite["tensor_contract"]["input_dtype"],
                "output_shape": json.dumps(tflite["tensor_contract"]["output_shape"]),
                "output_dtype": tflite["tensor_contract"]["output_dtype"],
            }
        )

    write_csv(analysis_dir / "metrics_per_seed.csv", metric_rows, list(metric_rows[0]))
    write_csv(
        analysis_dir / "per_class_recall_per_seed.csv",
        recall_rows,
        list(recall_rows[0]),
    )
    write_csv(
        analysis_dir / "resource_metrics_per_seed.csv",
        resource_rows,
        list(resource_rows[0]),
    )

    mean_rows: list[dict[str, Any]] = []
    for (strategy, variant, dataset), group_iterator in itertools.groupby(
        sorted(metric_rows, key=lambda row: (row["strategy"], row["tokenizer"], row["dataset"])),
        key=lambda row: (row["strategy"], row["tokenizer"], row["dataset"]),
    ):
        group = list(group_iterator)
        accuracies = [float(row["accuracy"]) for row in group]
        f1_values = [float(row["macro_f1"]) for row in group]
        mean_rows.append(
            {
                "strategy": strategy,
                "tokenizer": variant,
                "combination": combination_id(strategy, variant),
                "dataset": dataset,
                "display_name": DISPLAY_DATASETS[dataset],
                "seed_count": len(group),
                "seeds": json.dumps(SEEDS),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std_sample_ddof1": sample_std(accuracies),
                "accuracy_min": min(accuracies),
                "accuracy_max": max(accuracies),
                "macro_f1_mean": float(np.mean(f1_values)),
                "macro_f1_std_sample_ddof1": sample_std(f1_values),
                "macro_f1_min": min(f1_values),
                "macro_f1_max": max(f1_values),
                "macro_f1_scope": (
                    "present_truth_classes"
                    if dataset == CHALLENGE_EVALUATION
                    else "all_10_classes"
                ),
            }
        )
    write_csv(analysis_dir / "metrics_mean_std.csv", mean_rows, list(mean_rows[0]))
    write_json(analysis_dir / "metrics_mean_std.json", mean_rows)

    recall_mean_rows: list[dict[str, Any]] = []
    recall_sort = sorted(
        recall_rows,
        key=lambda row: (
            row["strategy"], row["tokenizer"], row["dataset"], row["label"]
        ),
    )
    for key, group_iterator in itertools.groupby(
        recall_sort,
        key=lambda row: (
            row["strategy"], row["tokenizer"], row["dataset"], row["label"]
        ),
    ):
        strategy, variant, dataset, label = key
        group = list(group_iterator)
        values = [float(row["recall"]) for row in group]
        recall_mean_rows.append(
            {
                "strategy": strategy,
                "tokenizer": variant,
                "combination": combination_id(strategy, variant),
                "dataset": dataset,
                "label": label,
                "support_per_seed": int(group[0]["support"]),
                "recall_mean": float(np.mean(values)),
                "recall_std_sample_ddof1": sample_std(values),
                "recall_min": min(values),
                "recall_max": max(values),
            }
        )
    write_csv(
        analysis_dir / "per_class_recall_mean_std.csv",
        recall_mean_rows,
        list(recall_mean_rows[0]),
    )

    resource_mean_rows: list[dict[str, Any]] = []
    for (strategy, variant), group_iterator in itertools.groupby(
        sorted(resource_rows, key=lambda row: (row["strategy"], row["tokenizer"])),
        key=lambda row: (row["strategy"], row["tokenizer"]),
    ):
        group = list(group_iterator)
        latencies = [float(row["tflite_median_ms_host"]) for row in group]
        sizes = [int(row["tflite_size_bytes"]) for row in group]
        resource_mean_rows.append(
            {
                "strategy": strategy,
                "tokenizer": variant,
                "combination": combination_id(strategy, variant),
                "seed_count": len(group),
                "parameters": int(group[0]["parameters"]),
                "tflite_size_bytes_mean": float(np.mean(sizes)),
                "tflite_size_bytes_std_sample_ddof1": sample_std(sizes),
                "tflite_median_ms_host_mean": float(np.mean(latencies)),
                "tflite_median_ms_host_std_sample_ddof1": sample_std(latencies),
                "keras_tflite_top1_agreement_min": min(
                    float(row["keras_tflite_top1_agreement_700"]) for row in group
                ),
                "keras_tflite_max_abs_probability_difference_max": max(
                    float(row["keras_tflite_max_abs_probability_difference_700"])
                    for row in group
                ),
                "all_builtins_only": all(bool(row["builtins_only"]) for row in group),
                "tensor_contract": "int32[1,24] -> float32[1,10]",
            }
        )
    write_csv(
        analysis_dir / "resource_metrics_mean_std.csv",
        resource_mean_rows,
        list(resource_mean_rows[0]),
    )
    return metric_rows, mean_rows, recall_mean_rows


def sum_confusion_matrices(
    results: Sequence[dict[str, Any]], analysis_dir: Path
) -> None:
    target = analysis_dir / "Confusion Matrices"
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            matching = [
                result
                for result in results
                if result["strategy"] == strategy
                and result["tokenizer_variant"] == variant
            ]
            for dataset in PRIMARY_EVALUATIONS:
                matrix = np.sum(
                    [
                        np.asarray(
                            result["evaluations"][dataset]["confusion_matrix"],
                            dtype=np.int64,
                        )
                        for result in matching
                    ],
                    axis=0,
                )
                diagnosis.write_confusion(
                    target
                    / f"{combination_id(strategy, variant)}_{dataset}_sum_5_seeds.csv",
                    matrix.tolist(),
                )


def collect_predictions(
    output_dir: Path, analysis_dir: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    index: dict[tuple[str, str, int, str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        for strategy in STRATEGIES:
            for variant in TOKENIZER_VARIANTS:
                path = model_dir(output_dir, seed, strategy, variant)
                for raw in read_csv(path / "predictions_all_evaluation_sets.csv"):
                    row: dict[str, Any] = dict(raw)
                    row["seed"] = int(raw["seed"])
                    row["row"] = int(raw["row"])
                    row["correct"] = parse_bool(raw["correct"])
                    row["confidence"] = float(raw["confidence"])
                    key = (strategy, variant, seed, str(row["dataset"]), int(row["row"]))
                    require(key not in index, f"Duplicate prediction key: {key}")
                    index[key] = row
                    rows.append(row)
    fields = list(rows[0])
    write_csv(analysis_dir / "predictions_all_seeds.csv", rows, fields)
    write_csv(
        analysis_dir / "misclassifications_all_seeds.csv",
        [row for row in rows if not bool(row["correct"])],
        fields,
    )
    return rows, index


def analyze_error_stability(
    prediction_rows: Sequence[dict[str, Any]], analysis_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        key = (
            str(row["strategy"]),
            str(row["tokenizer"]),
            str(row["dataset"]),
            int(row["row"]),
            str(row["text"]),
            str(row["expected_label"]),
        )
        grouped[key].append(row)
    stability_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        strategy, variant, dataset, row_number, text, expected = key
        require(
            len(group) == len(SEEDS),
            f"Expected {len(SEEDS)} predictions for {key}; found {len(group)}",
        )
        wrong = [row for row in group if not bool(row["correct"])]
        summary = {
            "strategy": strategy,
            "tokenizer": variant,
            "combination": combination_id(strategy, variant),
            "dataset": dataset,
            "row": row_number,
            "text": text,
            "expected_label": expected,
            "correct_seed_count": len(SEEDS) - len(wrong),
            "wrong_seed_count": len(wrong),
            "wrong_seeds": json.dumps(sorted(int(row["seed"]) for row in wrong)),
            "predicted_label_counts": json.dumps(
                Counter(str(row["predicted_label"]) for row in group),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "stable_correct_5_of_5": not wrong,
            "stable_wrong_5_of_5": len(wrong) == len(SEEDS),
        }
        if wrong:
            stability_rows.append(summary)
        if dataset == KNOWN_BOUNDARY_SUBSET:
            boundary_rows.append(summary)
    write_csv(
        analysis_dir / "misclassification_stability.csv",
        stability_rows,
        list(stability_rows[0]),
    )
    stable_wrong = [row for row in stability_rows if bool(row["stable_wrong_5_of_5"])]
    write_csv(
        analysis_dir / "stable_errors_5_of_5_seeds.csv",
        stable_wrong,
        list(stability_rows[0]),
    )
    write_csv(
        analysis_dir / "boundary_case_stability.csv",
        boundary_rows,
        list(boundary_rows[0]),
    )
    return stability_rows, boundary_rows


def transition_state(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_correct = bool(left["correct"])
    right_correct = bool(right["correct"])
    if left_correct and right_correct:
        return "both_correct"
    if not left_correct and not right_correct:
        return "both_wrong"
    if not left_correct and right_correct:
        return "fixed"
    return "regression"


def analyze_transitions(
    prediction_index: dict[tuple[str, str, int, str, int], dict[str, Any]],
    datasets: dict[str, list[baseline.Sample]],
    analysis_dir: Path,
) -> None:
    strategy_summary: list[dict[str, Any]] = []
    strategy_events: list[dict[str, Any]] = []
    for variant in TOKENIZER_VARIANTS:
        for left_strategy, right_strategy in itertools.combinations(STRATEGIES, 2):
            for dataset in PRIMARY_EVALUATIONS:
                counts: Counter[str] = Counter()
                for seed in SEEDS:
                    for row_number in range(1, len(datasets[dataset]) + 1):
                        left = prediction_index[
                            (left_strategy, variant, seed, dataset, row_number)
                        ]
                        right = prediction_index[
                            (right_strategy, variant, seed, dataset, row_number)
                        ]
                        state = transition_state(left, right)
                        counts[state] += 1
                        if state in {"fixed", "regression"}:
                            strategy_events.append(
                                {
                                    "tokenizer": variant,
                                    "source_strategy": left_strategy,
                                    "target_strategy": right_strategy,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "row": row_number,
                                    "transition": state,
                                    "text": left["text"],
                                    "expected_label": left["expected_label"],
                                    "source_prediction": left["predicted_label"],
                                    "target_prediction": right["predicted_label"],
                                }
                            )
                strategy_summary.append(
                    {
                        "tokenizer": variant,
                        "source_strategy": left_strategy,
                        "target_strategy": right_strategy,
                        "dataset": dataset,
                        **{state: counts[state] for state in ("both_correct", "both_wrong", "fixed", "regression")},
                        "net_fixed_minus_regressions": counts["fixed"] - counts["regression"],
                    }
                )
    write_csv(
        analysis_dir / "strategy_transition_summary.csv",
        strategy_summary,
        list(strategy_summary[0]),
    )
    write_csv(
        analysis_dir / "strategy_fixed_errors_and_regressions.csv",
        strategy_events,
        list(strategy_events[0]),
    )

    tokenizer_summary: list[dict[str, Any]] = []
    tokenizer_events: list[dict[str, Any]] = []
    word_variant = tokenizer_variant_by_kind("word")
    bpe_variant = tokenizer_variant_by_kind("bpe")
    for strategy in STRATEGIES:
        for dataset in PRIMARY_EVALUATIONS:
            counts: Counter[str] = Counter()
            for seed in SEEDS:
                for row_number in range(1, len(datasets[dataset]) + 1):
                    t1 = prediction_index[(strategy, word_variant, seed, dataset, row_number)]
                    t2 = prediction_index[(strategy, bpe_variant, seed, dataset, row_number)]
                    state = transition_state(t1, t2)
                    counts[state] += 1
                    if state in {"fixed", "regression"}:
                        tokenizer_events.append(
                            {
                                "strategy": strategy,
                                "dataset": dataset,
                                "seed": seed,
                                "row": row_number,
                                "transition_T1_to_T2": state,
                                "text": t1["text"],
                                "expected_label": t1["expected_label"],
                                "T1_prediction": t1["predicted_label"],
                                "T2_prediction": t2["predicted_label"],
                            }
                        )
            tokenizer_summary.append(
                {
                    "strategy": strategy,
                    "dataset": dataset,
                    **{state: counts[state] for state in ("both_correct", "both_wrong", "fixed", "regression")},
                    "T2_net_fixed_minus_regressions": counts["fixed"] - counts["regression"],
                }
            )
    write_csv(
        analysis_dir / "tokenizer_transition_summary.csv",
        tokenizer_summary,
        list(tokenizer_summary[0]),
    )
    write_csv(
        analysis_dir / "tokenizer_fixed_errors_and_regressions.csv",
        tokenizer_events,
        list(tokenizer_events[0]),
    )


def analyze_error_overlap(
    prediction_index: dict[tuple[str, str, int, str, int], dict[str, Any]],
    datasets: dict[str, list[baseline.Sample]],
    analysis_dir: Path,
) -> None:
    combos = list(itertools.product(STRATEGIES, TOKENIZER_VARIANTS))
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for dataset in PRIMARY_EVALUATIONS:
            for left, right in itertools.combinations(combos, 2):
                counts: Counter[str] = Counter()
                for row_number in range(1, len(datasets[dataset]) + 1):
                    left_row = prediction_index[(*left, seed, dataset, row_number)]
                    right_row = prediction_index[(*right, seed, dataset, row_number)]
                    if not left_row["correct"] and not right_row["correct"]:
                        counts["both_wrong"] += 1
                    elif not left_row["correct"]:
                        counts["only_left_wrong"] += 1
                    elif not right_row["correct"]:
                        counts["only_right_wrong"] += 1
                    else:
                        counts["both_correct"] += 1
                union_wrong = (
                    counts["both_wrong"]
                    + counts["only_left_wrong"]
                    + counts["only_right_wrong"]
                )
                rows.append(
                    {
                        "seed": seed,
                        "dataset": dataset,
                        "left_combination": combination_id(*left),
                        "right_combination": combination_id(*right),
                        "both_wrong": counts["both_wrong"],
                        "only_left_wrong": counts["only_left_wrong"],
                        "only_right_wrong": counts["only_right_wrong"],
                        "both_correct": counts["both_correct"],
                        "wrong_set_jaccard": (
                            counts["both_wrong"] / union_wrong if union_wrong else 1.0
                        ),
                    }
                )
    write_csv(analysis_dir / "error_overlap_per_seed.csv", rows, list(rows[0]))


def analyze_comparisons(
    metric_rows: Sequence[dict[str, Any]], analysis_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_index = {
        (
            str(row["strategy"]),
            str(row["tokenizer"]),
            int(row["seed"]),
            str(row["dataset"]),
        ): row
        for row in metric_rows
    }
    tokenizer_effect_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for dataset in PRIMARY_EVALUATIONS:
            for metric in ("accuracy", "macro_f1"):
                differences = [
                    float(metric_index[(strategy, "T1", seed, dataset)][metric])
                    - float(metric_index[(strategy, "T2", seed, dataset)][metric])
                    for seed in SEEDS
                ]
                tokenizer_effect_rows.append(
                    {
                        "strategy": strategy,
                        "dataset": dataset,
                        "metric": metric,
                        "direction": "T1_minus_T2",
                        "paired_difference_mean": float(np.mean(differences)),
                        "paired_difference_std_sample_ddof1": sample_std(differences),
                        "paired_difference_min": min(differences),
                        "paired_difference_max": max(differences),
                        "T1_better_seed_count": sum(value > 0 for value in differences),
                        "ties_seed_count": sum(value == 0 for value in differences),
                        "T2_better_seed_count": sum(value < 0 for value in differences),
                    }
                )
    write_csv(
        analysis_dir / "tokenizer_effect_by_strategy.csv",
        tokenizer_effect_rows,
        list(tokenizer_effect_rows[0]),
    )

    strategy_delta_rows: list[dict[str, Any]] = []
    for variant in TOKENIZER_VARIANTS:
        for source, target in itertools.combinations(STRATEGIES, 2):
            for dataset in PRIMARY_EVALUATIONS:
                for metric in ("accuracy", "macro_f1"):
                    differences = [
                        float(metric_index[(target, variant, seed, dataset)][metric])
                        - float(metric_index[(source, variant, seed, dataset)][metric])
                        for seed in SEEDS
                    ]
                    strategy_delta_rows.append(
                        {
                            "tokenizer": variant,
                            "source_strategy": source,
                            "target_strategy": target,
                            "dataset": dataset,
                            "metric": metric,
                            "direction": "target_minus_source",
                            "paired_difference_mean": float(np.mean(differences)),
                            "paired_difference_std_sample_ddof1": sample_std(differences),
                            "target_better_seed_count": sum(value > 0 for value in differences),
                            "ties_seed_count": sum(value == 0 for value in differences),
                            "source_better_seed_count": sum(value < 0 for value in differences),
                        }
                    )
    write_csv(
        analysis_dir / "strategy_pair_deltas.csv",
        strategy_delta_rows,
        list(strategy_delta_rows[0]),
    )

    tradeoff_seed_rows: list[dict[str, Any]] = []
    tradeoff_mean_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            for seed in SEEDS:
                semantic = metric_index[(strategy, variant, seed, "semantic_val")]
                asr = metric_index[(strategy, variant, seed, "asr_val")]
                tradeoff_seed_rows.append(
                    {
                        "strategy": strategy,
                        "tokenizer": variant,
                        "combination": combination_id(strategy, variant),
                        "seed": seed,
                        "semantic_accuracy": semantic["accuracy"],
                        "asr_accuracy": asr["accuracy"],
                        "semantic_minus_asr_accuracy": float(semantic["accuracy"])
                        - float(asr["accuracy"]),
                        "semantic_macro_f1": semantic["macro_f1"],
                        "asr_macro_f1": asr["macro_f1"],
                        "semantic_minus_asr_macro_f1": float(semantic["macro_f1"])
                        - float(asr["macro_f1"]),
                    }
                )
            group = [
                row
                for row in tradeoff_seed_rows
                if row["strategy"] == strategy and row["tokenizer"] == variant
            ]
            acc_gaps = [float(row["semantic_minus_asr_accuracy"]) for row in group]
            f1_gaps = [float(row["semantic_minus_asr_macro_f1"]) for row in group]
            tradeoff_mean_rows.append(
                {
                    "strategy": strategy,
                    "tokenizer": variant,
                    "combination": combination_id(strategy, variant),
                    "semantic_minus_asr_accuracy_mean": float(np.mean(acc_gaps)),
                    "semantic_minus_asr_accuracy_std_sample_ddof1": sample_std(acc_gaps),
                    "semantic_minus_asr_macro_f1_mean": float(np.mean(f1_gaps)),
                    "semantic_minus_asr_macro_f1_std_sample_ddof1": sample_std(f1_gaps),
                }
            )
    write_csv(
        analysis_dir / "semantic_asr_tradeoff_per_seed.csv",
        tradeoff_seed_rows,
        list(tradeoff_seed_rows[0]),
    )
    write_csv(
        analysis_dir / "semantic_asr_tradeoff_mean_std.csv",
        tradeoff_mean_rows,
        list(tradeoff_mean_rows[0]),
    )
    return tokenizer_effect_rows, tradeoff_mean_rows


def rank_combinations(
    metric_rows: Sequence[dict[str, Any]], analysis_dir: Path
) -> list[dict[str, Any]]:
    index = {
        (row["strategy"], row["tokenizer"], int(row["seed"]), row["dataset"]): row
        for row in metric_rows
    }
    ranking: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            per_seed_composites = [
                float(
                    np.mean(
                        [
                            float(index[(strategy, variant, seed, dataset)]["macro_f1"])
                            for dataset in DEVELOPMENT_EVALUATIONS
                        ]
                    )
                )
                for seed in SEEDS
            ]
            challenge_accuracies = [
                float(index[(strategy, variant, seed, "challenge_40")]["accuracy"])
                for seed in SEEDS
            ]
            ranking.append(
                {
                    "strategy": strategy,
                    "tokenizer": variant,
                    "combination": combination_id(strategy, variant),
                    "development_macro_f1_composite_mean": float(
                        np.mean(per_seed_composites)
                    ),
                    "development_macro_f1_composite_std_sample_ddof1": sample_std(
                        per_seed_composites
                    ),
                    "challenge_accuracy_mean": float(np.mean(challenge_accuracies)),
                    "challenge_accuracy_std_sample_ddof1": sample_std(challenge_accuracies),
                    "production_winner": False,
                }
            )
    ranking.sort(
        key=lambda row: (
            -float(row["development_macro_f1_composite_mean"]),
            -float(row["challenge_accuracy_mean"]),
            float(row["development_macro_f1_composite_std_sample_ddof1"]),
            str(row["combination"]),
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["development_rank"] = rank
    fields = ["development_rank", *[field for field in ranking[0] if field != "development_rank"]]
    write_csv(analysis_dir / "development_ranking.csv", ranking, fields)
    write_json(
        analysis_dir / "development_ranking.json",
        {"rule": RANKING_RULE, "ranking": ranking, "production_winner_selected": False},
    )
    return ranking


def select_representative_seeds(
    metric_rows: Sequence[dict[str, Any]], output_dir: Path, analysis_dir: Path
) -> list[dict[str, Any]]:
    index = {
        (row["strategy"], row["tokenizer"], int(row["seed"]), row["dataset"]): row
        for row in metric_rows
    }
    selection_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            vectors: dict[int, np.ndarray] = {}
            for seed in SEEDS:
                values: list[float] = []
                for dataset in DEVELOPMENT_EVALUATIONS:
                    row = index[(strategy, variant, seed, dataset)]
                    values.extend([float(row["accuracy"]), float(row["macro_f1"])])
                vectors[seed] = np.asarray(values, dtype=np.float64)
            mean_vector = np.mean(list(vectors.values()), axis=0)
            scored = []
            for seed, vector in vectors.items():
                differences = np.abs(vector - mean_vector)
                scored.append(
                    (
                        float(np.mean(differences)),
                        float(np.max(differences)),
                        seed,
                    )
                )
            chosen_score = min(scored)
            chosen_seed = chosen_score[2]
            for mean_abs, max_abs, seed in sorted(scored, key=lambda item: item[2]):
                row = {
                    "strategy": strategy,
                    "tokenizer": variant,
                    "combination": combination_id(strategy, variant),
                    "seed": seed,
                    "mean_absolute_distance_to_five_seed_mean_vector": mean_abs,
                    "maximum_absolute_component_distance": max_abs,
                    "selected": seed == chosen_seed,
                    "challenge_used_for_selection": False,
                }
                selection_rows.append(row)
                if seed == chosen_seed:
                    source = model_dir(output_dir, seed, strategy, variant)
                    selected.append(
                        {
                            **row,
                            "source_model_dir": str(source),
                            "source_tflite": str(source / "model_builtin.tflite"),
                            "source_tflite_sha256": sha256_file(
                                source / "model_builtin.tflite"
                            ),
                            "source_tokenizer_dir": str(source / "tokenizer"),
                        }
                    )
    write_csv(
        analysis_dir / "representative_seed_selection.csv",
        selection_rows,
        list(selection_rows[0]),
    )
    write_json(
        analysis_dir / "representative_models.json",
        {
            "selection_rule": REPRESENTATIVE_SEED_RULE,
            "models": selected,
            "production_winner_selected": False,
        },
    )

    selected_root = output_dir / "Ausgewaehlte Modelle"
    if selected_root.exists():
        manifest_path = selected_root / "representative_models.json"
        require(manifest_path.is_file(), f"Partial representative directory: {selected_root}")
        existing = read_json(manifest_path)
        require(existing["models"] == selected, "Representative selection changed on rerun")
    else:
        selected_root.mkdir(parents=True)
        for entry in selected:
            strategy = str(entry["strategy"])
            variant = str(entry["tokenizer"])
            source = Path(str(entry["source_model_dir"]))
            target = selected_root / MODEL_FOLDER_NAMES[(strategy, variant)]
            shutil.copytree(source, target)
            require(
                sha256_file(target / "model_builtin.tflite")
                == entry["source_tflite_sha256"],
                f"Representative TFLite copy mismatch: {target}",
            )
        write_json(
            selected_root / "representative_models.json",
            {
                "selection_rule": REPRESENTATIVE_SEED_RULE,
                "models": selected,
                "production_winner_selected": False,
            },
        )
    return selected


def percent(value: float) -> str:
    return f"{100.0 * value:.2f} %"


def metric_lookup(
    mean_rows: Sequence[dict[str, Any]], strategy: str, variant: str, dataset: str
) -> dict[str, Any]:
    return next(
        row
        for row in mean_rows
        if row["strategy"] == strategy
        and row["tokenizer"] == variant
        and row["dataset"] == dataset
    )


def write_report(
    analysis_dir: Path,
    metric_rows: Sequence[dict[str, Any]],
    mean_rows: Sequence[dict[str, Any]],
    recall_mean_rows: Sequence[dict[str, Any]],
    ranking: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
    boundary_rows: Sequence[dict[str, Any]],
    stability_rows: Sequence[dict[str, Any]],
    tokenizer_effect_rows: Sequence[dict[str, Any]],
    tradeoff_rows: Sequence[dict[str, Any]],
) -> None:
    lines = [
        "# M0-M3 × T1/T2 – vollständiger Fünf-Seed-Vergleich",
        "",
        "> **Einordnung:** Dies ist ausschließlich eine Development-Rangfolge. Es wurde kein Produktionsgewinner ausgewählt. Die stärksten Kandidaten müssen noch einen unabhängigen Blind-Holdout und reale Vosk-End-to-End-Tests bestehen.",
        "",
        "## Experimentprotokoll",
        "",
        f"Alle acht Kombinationen wurden mit denselben fünf Seeds `{list(SEEDS)}` vollständig neu trainiert. Die eingefrorene Datenbasis umfasst 2.918 Clean-Samples (einschließlich des 533-Sample-Hard-Negative-Patches) und 1.828 Vosk-/ASR-Samples. Tokenizer, Daten, Evaluationen, `max_len={MAX_LEN}` und BaselineCNN wurden während des Experiments nicht verändert.",
        "",
        "| Strategie | Phase 1 / einziges Training | Phase 2 | Lernrate |",
        "|---|---|---|---:|",
        "| M0 | Clean only | – | 1e-3 |",
        "| M1 | Joint Clean + Vosk ab Initialisierung | – | 1e-3 |",
        "| M2 | neuer M0-Clean-Checkpoint desselben Seeds/Tokenizers | Joint Clean + Vosk | 1e-4 |",
        "| M3 | neuer M0-Clean-Checkpoint desselben Seeds/Tokenizers | Vosk only | 1e-4 |",
        "",
        f"Jede Phase lief maximal {EPOCHS} Epochen mit Early-Stopping-Patience {PATIENCE}, Batchgröße {BATCH_SIZE}. M2/M3 übernehmen damit exakt die im Repository vorhandene reduzierte Fine-Tuning-Lernrate und bestehende Phasenlogik; es wurde kein alter Modellcheckpoint wiederverwendet.",
        "",
        "Wie in dieser vorhandenen Trainingslogik wurde der gemeinsame Pool aus Semantic-300, ASR-300 und Curated-60 (660 Sätze) für Early Stopping und Checkpoint-Auswahl verwendet. Diese drei Sets sind deshalb Development-Daten und ausdrücklich kein unabhängiger Test. Challenge-40 war weder Training noch Early Stopping zugänglich. Ein Audit fand keine normalisierten Satzüberschneidungen zwischen Training und den vier Evaluationssets.",
        "",
        "## Development-Rangfolge",
        "",
        "Primärschlüssel ist der gleich gewichtete Mittelwert der Macro-F1-Werte von Semantic-300, ASR-300 und Curated-60 über fünf Seeds. Challenge-40 dient nur als erster Tie-Breaker; sein Macro-F1 wird wegen der kleinen, nicht über alle zehn Wahrheitsklassen verteilten Stichprobe nicht in den Ranking-Score eingerechnet.",
        "",
        "| Rang | Kombination | Dev Macro-F1 | Seed-Std. | Challenge Accuracy |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row['development_rank']} | {row['combination']} | "
            f"{percent(float(row['development_macro_f1_composite_mean']))} | "
            f"{percent(float(row['development_macro_f1_composite_std_sample_ddof1']))} | "
            f"{percent(float(row['challenge_accuracy_mean']))} |"
        )

    lines.extend(
        [
            "",
            "## Ergebnisse je Seed",
            "",
            "| Kombination | Seed | Semantic Acc/F1 | ASR Acc/F1 | Curated Acc/F1 | Challenge Acc/F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    seed_metric_index = {
        (row["strategy"], row["tokenizer"], int(row["seed"]), row["dataset"]): row
        for row in metric_rows
    }
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            for seed in SEEDS:
                cells = []
                for dataset in PRIMARY_EVALUATIONS:
                    metric_row = seed_metric_index[(strategy, variant, seed, dataset)]
                    cells.append(
                        f"{100 * float(metric_row['accuracy']):.2f}/"
                        f"{100 * float(metric_row['macro_f1']):.2f} %"
                    )
                lines.append(
                    f"| {combination_id(strategy, variant)} | {seed} | "
                    + " | ".join(cells)
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Accuracy und Macro-F1: Mittelwert ± Stichproben-Standardabweichung",
            "",
            "| Kombination | Set | Accuracy | Macro-F1 |",
            "|---|---|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            for dataset in PRIMARY_EVALUATIONS:
                row = metric_lookup(mean_rows, strategy, variant, dataset)
                lines.append(
                    f"| {combination_id(strategy, variant)} | {DISPLAY_DATASETS[dataset]} | "
                    f"{percent(float(row['accuracy_mean']))} ± {percent(float(row['accuracy_std_sample_ddof1']))} | "
                    f"{percent(float(row['macro_f1_mean']))} ± {percent(float(row['macro_f1_std_sample_ddof1']))} |"
                )

    lines.extend(["", "## T1 Word vs. T2 BPE", ""])
    for strategy in STRATEGIES:
        deltas = [
            row
            for row in tokenizer_effect_rows
            if row["strategy"] == strategy
            and row["metric"] == "macro_f1"
            and row["dataset"] in DEVELOPMENT_EVALUATIONS
        ]
        mean_delta = float(np.mean([float(row["paired_difference_mean"]) for row in deltas]))
        direction = "T1" if mean_delta > 0 else "T2" if mean_delta < 0 else "keiner"
        lines.append(
            f"- **{strategy}:** über die drei balancierten Development-Sets liegt {direction} im Mittel um {abs(mean_delta) * 100:.2f} Prozentpunkte vorn. Vollständige gepaarte Seed-Differenzen: [`tokenizer_effect_by_strategy.csv`](tokenizer_effect_by_strategy.csv)."
        )

    tokenizer_effect_index = {
        (row["strategy"], row["dataset"], row["metric"]): row
        for row in tokenizer_effect_rows
    }
    ranking_index = {
        (row["strategy"], row["tokenizer"]): row for row in ranking
    }
    lines.extend(
        [
            "",
            "Positive Werte in der folgenden Interaktionstabelle bedeuten einen Vorteil für T2 gegenüber T1. Die Set-Spalten sind gepaarte Fünf-Seed-Differenzen der Macro-F1-Mittelwerte; Challenge verwendet wegen fehlender Wahrheitsklassen Accuracy.",
            "",
            "| Strategie | Semantic ΔT2−T1 | ASR ΔT2−T1 | Curated ΔT2−T1 | Dev-Score ΔT2−T1 | Challenge Acc ΔT2−T1 | Dev Seed-Std. T1 / T2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        set_deltas = {
            dataset: -float(
                tokenizer_effect_index[(strategy, dataset, "macro_f1")][
                    "paired_difference_mean"
                ]
            )
            for dataset in DEVELOPMENT_EVALUATIONS
        }
        challenge_delta = -float(
            tokenizer_effect_index[(strategy, "challenge_40", "accuracy")][
                "paired_difference_mean"
            ]
        )
        t1_rank = ranking_index[(strategy, "T1")]
        t2_rank = ranking_index[(strategy, "T2")]
        dev_delta = float(t2_rank["development_macro_f1_composite_mean"]) - float(
            t1_rank["development_macro_f1_composite_mean"]
        )
        lines.append(
            f"| {strategy} | {set_deltas['semantic_val'] * 100:+.2f} pp | "
            f"{set_deltas['asr_val'] * 100:+.2f} pp | "
            f"{set_deltas['complex_val'] * 100:+.2f} pp | "
            f"{dev_delta * 100:+.2f} pp | {challenge_delta * 100:+.2f} pp | "
            f"{float(t1_rank['development_macro_f1_composite_std_sample_ddof1']) * 100:.2f} / "
            f"{float(t2_rank['development_macro_f1_composite_std_sample_ddof1']) * 100:.2f} pp |"
        )

    lines.extend(["", "## Vergleich der vier Trainingsstrategien", ""])
    for variant in TOKENIZER_VARIANTS:
        tokenizer_ranking = sorted(
            [row for row in ranking if row["tokenizer"] == variant],
            key=lambda row: int(row["development_rank"]),
        )
        lines.append(
            f"- **{variant}:** "
            + " > ".join(
                f"{row['strategy']} ({100 * float(row['development_macro_f1_composite_mean']):.2f} %)"
                for row in tokenizer_ranking
            )
            + "."
        )
    lines.append(
        "Gepaarte Differenzen für jedes Strategiepaar, Set und jeden Tokenizer stehen in [`strategy_pair_deltas.csv`](strategy_pair_deltas.csv); die konkreten behobenen Fehler und Regressionen sind satzweise separat ausgewiesen."
    )
    mean_index = {
        (row["strategy"], row["tokenizer"], row["dataset"]): row
        for row in mean_rows
    }
    lines.extend(
        [
            "",
            "Die folgende Tabelle zeigt jede ASR-Trainingsstrategie relativ zur jeweiligen Clean-only-Baseline M0 desselben Tokenizers (Macro-F1; Challenge als Accuracy).",
            "",
            "| Tokenizer | Strategie vs. M0 | Semantic | ASR | Curated | Dev-Score | Challenge Acc |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in TOKENIZER_VARIANTS:
        base_rank = ranking_index[("M0", variant)]
        for strategy in ("M1", "M2", "M3"):
            deltas = {}
            for dataset in DEVELOPMENT_EVALUATIONS:
                deltas[dataset] = float(
                    mean_index[(strategy, variant, dataset)]["macro_f1_mean"]
                ) - float(mean_index[("M0", variant, dataset)]["macro_f1_mean"])
            score_delta = float(
                ranking_index[(strategy, variant)][
                    "development_macro_f1_composite_mean"
                ]
            ) - float(base_rank["development_macro_f1_composite_mean"])
            challenge_delta = float(
                mean_index[(strategy, variant, "challenge_40")]["accuracy_mean"]
            ) - float(mean_index[("M0", variant, "challenge_40")]["accuracy_mean"])
            lines.append(
                f"| {variant} | {strategy} | {deltas['semantic_val'] * 100:+.2f} pp | "
                f"{deltas['asr_val'] * 100:+.2f} pp | "
                f"{deltas['complex_val'] * 100:+.2f} pp | "
                f"{score_delta * 100:+.2f} pp | {challenge_delta * 100:+.2f} pp |"
            )

    lines.extend(["", "## Semantic-vs.-ASR-Trade-off", "", "| Kombination | Semantic minus ASR Accuracy | Semantic minus ASR Macro-F1 |", "|---|---:|---:|"])
    for row in tradeoff_rows:
        lines.append(
            f"| {row['combination']} | {float(row['semantic_minus_asr_accuracy_mean']) * 100:+.2f} pp ± {float(row['semantic_minus_asr_accuracy_std_sample_ddof1']) * 100:.2f} | "
            f"{float(row['semantic_minus_asr_macro_f1_mean']) * 100:+.2f} pp ± {float(row['semantic_minus_asr_macro_f1_std_sample_ddof1']) * 100:.2f} |"
        )

    lines.extend(["", "## Recall je Klasse", ""])
    for dataset in PRIMARY_EVALUATIONS:
        lines.extend(
            [
                f"### {DISPLAY_DATASETS[dataset]}",
                "",
                "| Kombination | " + " | ".join(baseline.LABELS) + " |",
                "|---|" + "---:|" * len(baseline.LABELS),
            ]
        )
        for strategy in STRATEGIES:
            for variant in TOKENIZER_VARIANTS:
                recalls = {
                    row["label"]: (
                        int(row["support_per_seed"]), float(row["recall_mean"])
                    )
                    for row in recall_mean_rows
                    if row["strategy"] == strategy
                    and row["tokenizer"] == variant
                    and row["dataset"] == dataset
                }
                lines.append(
                    f"| {combination_id(strategy, variant)} | "
                    + " | ".join(
                        "–"
                        if recalls[label][0] == 0
                        else f"{100 * recalls[label][1]:.1f}"
                        for label in baseline.LABELS
                    )
                    + " |"
                )
        lines.append("")

    lines.extend(
        [
            "`–` bedeutet, dass das jeweilige Set keine Ground-Truth-Samples dieser Klasse enthält. Stichproben-Standardabweichung, Minimum, Maximum und Support stehen vollständig in [`per_class_recall_mean_std.csv`](per_class_recall_mean_std.csv).",
            "",
        ]
    )

    stable_error_count = sum(
        bool(row["stable_wrong_5_of_5"])
        for row in stability_rows
        if row["dataset"] in PRIMARY_EVALUATIONS
    )
    boundary_stable_error_count = sum(
        bool(row["stable_wrong_5_of_5"])
        for row in stability_rows
        if row["dataset"] == "challenge_known_failure_subset_9"
    )
    lines.extend(
        [
            "## Fehlerstabilität, Überschneidungen und Regressionen",
            "",
            f"Über die vier geforderten Evaluationssets wurden {stable_error_count} Kombination/Satz-Fälle in allen fünf Seeds stabil falsch klassifiziert. Die Boundary-9-Teilmenge enthält zusätzlich {boundary_stable_error_count} solche Markierungen, die wegen ihrer Überschneidung mit Challenge-40 nicht nochmals in der Primärsumme gezählt werden. Satzweise Details stehen in [`misclassification_stability.csv`](misclassification_stability.csv) und [`stable_errors_5_of_5_seeds.csv`](stable_errors_5_of_5_seeds.csv). Paarweise Fehlerüberschneidungen stehen in [`error_overlap_per_seed.csv`](error_overlap_per_seed.csv). Behobene Fehler und Regressionen zwischen allen M0–M3-Paaren sind in [`strategy_fixed_errors_and_regressions.csv`](strategy_fixed_errors_and_regressions.csv), die T1/T2-Wechsel in [`tokenizer_fixed_errors_and_regressions.csv`](tokenizer_fixed_errors_and_regressions.csv) dokumentiert.",
            "",
            "| Kombination | Semantic stabil falsch | ASR stabil falsch | Curated stabil falsch | Challenge stabil falsch |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            counts = {
                dataset: sum(
                    bool(row["stable_wrong_5_of_5"])
                    for row in stability_rows
                    if row["strategy"] == strategy
                    and row["tokenizer"] == variant
                    and row["dataset"] == dataset
                )
                for dataset in PRIMARY_EVALUATIONS
            }
            lines.append(
                f"| {combination_id(strategy, variant)} | "
                f"{counts['semantic_val']} | {counts['asr_val']} | "
                f"{counts['complex_val']} | {counts['challenge_40']} |"
            )
    lines.extend(
        [
            "",
            "Über fünf Seeds summierte Confusion Matrices für jede Kombination und jedes Primärset liegen unter [`Confusion Matrices/`](<Confusion Matrices/>); zusätzlich enthält jeder einzelne Modellordner seine seed-spezifischen Matrizen.",
            "",
            "## Bekannte Boundary-Fälle",
            "",
            "Boundary-9 ist eine markierte Teilmenge von Challenge-40 und wird nicht als unabhängiger Test gezählt.",
            "",
            "| Kombination | stabil korrekt 5/5 | wechselhaft | stabil falsch 5/5 |",
            "|---|---:|---:|---:|",
        ]
    )
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            group = [
                row
                for row in boundary_rows
                if row["strategy"] == strategy and row["tokenizer"] == variant
            ]
            lines.append(
                f"| {combination_id(strategy, variant)} | "
                f"{sum(bool(row['stable_correct_5_of_5']) for row in group)} | "
                f"{sum(not bool(row['stable_correct_5_of_5']) and not bool(row['stable_wrong_5_of_5']) for row in group)} | "
                f"{sum(bool(row['stable_wrong_5_of_5']) for row in group)} |"
            )

    lines.extend(["", "### Boundary-Fälle nach Kombination", ""])
    for strategy in STRATEGIES:
        for variant in TOKENIZER_VARIANTS:
            group = [
                row
                for row in boundary_rows
                if row["strategy"] == strategy and row["tokenizer"] == variant
            ]
            stable_correct = [
                str(row["text"])
                for row in group
                if bool(row["stable_correct_5_of_5"])
            ]
            stable_wrong = [
                str(row["text"])
                for row in group
                if bool(row["stable_wrong_5_of_5"])
            ]
            variable = [
                f"{row['text']} ({row['correct_seed_count']}/5 korrekt)"
                for row in group
                if not bool(row["stable_correct_5_of_5"])
                and not bool(row["stable_wrong_5_of_5"])
            ]
            lines.append(f"- **{combination_id(strategy, variant)}**")
            lines.append(
                "  - stabil gelöst: "
                + ("; ".join(stable_correct) if stable_correct else "keiner")
            )
            lines.append(
                "  - stabil falsch: "
                + ("; ".join(stable_wrong) if stable_wrong else "keiner")
            )
            lines.append(
                "  - seed-abhängig: "
                + ("; ".join(variable) if variable else "keiner")
            )

    lines.extend(
        [
            "",
            "## TFLite und Ressourcen",
            "",
            "",
            "| Kombination | Parameter | TFLite-Größe | Host-Median | Top-1-Parität min. | Builtins-only |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    resource_rows = read_csv(analysis_dir / "resource_metrics_mean_std.csv")
    for row in sorted(resource_rows, key=lambda item: str(item["combination"])):
        lines.append(
            f"| {row['combination']} | {int(row['parameters']):,} | "
            f"{float(row['tflite_size_bytes_mean']) / 1024:.2f} KiB | "
            f"{float(row['tflite_median_ms_host_mean']):.5f} ± "
            f"{float(row['tflite_median_ms_host_std_sample_ddof1']):.5f} ms | "
            f"{100 * float(row['keras_tflite_top1_agreement_min']):.2f} % | "
            f"{'ja' if parse_bool(row['all_builtins_only']) else 'nein'} |"
        )
    lines.extend(
        [
            "",
            "## Repräsentative Seeds für die Desktop-App",
            "",
            "Die Auswahlregel wurde vor dem Training festgelegt: minimale mittlere absolute Distanz des sechs-dimensionalen Vektors aus Accuracy und Macro-F1 auf Semantic, ASR und Curated zum jeweiligen Fünf-Seed-Mittel. Challenge-40 ist ausgeschlossen; bei Gleichstand gelten maximale Komponentendistanz und danach die kleinere Seednummer.",
            "",
            "| Kombination | Seed | mittlere Distanz | maximale Distanz |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in sorted(selected, key=lambda item: (item["strategy"], item["tokenizer"])):
        lines.append(
            f"| {row['combination']} | {row['seed']} | "
            f"{float(row['mean_absolute_distance_to_five_seed_mean_vector']):.6f} | "
            f"{float(row['maximum_absolute_component_distance']):.6f} |"
        )

    deployment_path = analysis_dir / "desktop_app_deployment.json"
    if deployment_path.is_file():
        deployment = read_json(deployment_path)
        lines.extend(
            [
                "",
                "## Desktop-App-Integration",
                "",
                f"Die Desktop-App unter `{deployment['desktop_app_path']}` lädt bei jedem Durchlauf alle {deployment['active_model_count']} aktiven Modelle gleichzeitig: M0–M3 jeweils T1 Word und T2 BPE. Eine Modellauswahl gibt es nicht; alle {deployment['results_displayed_per_input']} Vorhersagen erscheinen gemeinsam in einer Tabelle. Die {deployment['log_group_count']} bisherigen M0–M3-Gruppen bleiben ausschließlich als Ordnerstruktur für die acht getrennten append-only TXT-Listen erhalten. Alte Modelle und alte Tokenizer sind nicht aktiv. Nach jeder Eingabe wird zuerst verpflichtend eine Soll-Kategorie gewählt und identisch in alle acht Listen geschrieben.",
                "",
                "| Aktive ID | repräsentativer Seed | Tokenizer | TFLite SHA-256 (kurz) |",
                "|---|---:|---|---|",
            ]
        )
        for model in deployment["models"]:
            lines.append(
                f"| {model['id']} | {model['seed']} | {model['tokenizer']} | "
                f"`{str(model['sha256'])[:16]}…` |"
            )
        verification = deployment["verification"]
        lines.extend(
            [
                "",
                f"Die App-Tests ({verification['unit_tests_passed']} Tests), der reale gemeinsame Acht-Modell-Durchlauf mit acht getrennten Logs, alle acht LiteRT-Smoke-Inferenzen, der vollständige Vosk-Modell-Ladetest sowie {verification['app_vs_training_tokenizer_encodings_checked']:,} App-/Training-Tokenizer-Encodings waren erfolgreich. Die vollständigen Hashes und Prüffakten stehen in [`desktop_app_deployment.json`](desktop_app_deployment.json). Eine reale Mikrofonaufnahme wurde nicht automatisiert.",
            ]
        )

    lines.extend(
        [
            "",
            "## Schlussfolgerung",
            "",
            f"Nach der vorab definierten Development-Metrik belegt **{ranking[0]['combination']}** Rang 1, gefolgt von **{ranking[1]['combination']}** und **{ranking[2]['combination']}**. Das ist eine Priorisierung für den nächsten Prüfpunkt, keine Produktionsentscheidung. Für eine belastbare Auswahl sind ein unangetasteter Blind-Holdout und reale End-to-End-Aufnahmen mit Vosk erforderlich.",
            "",
            "Die redaktionelle Interpretation der konkreten Befunde steht in [`MANUELLE_EVALUATION.md`](MANUELLE_EVALUATION.md).",
            "",
            "Die vollständigen Einzel-Seed-Metriken, Confusion Matrices, Wahrscheinlichkeiten, stabilen Fehler, Übergänge und Auswahlrechnungen liegen maschinenlesbar in diesem Analyseordner.",
            "",
        ]
    )
    (analysis_dir / "ABSCHLUSSBERICHT_M0_M3_T1_T2.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_tokenizer_diagnostics(
    tokenizers: dict[str, tokenizers_v1.Tokenizer],
    datasets: dict[str, list[baseline.Sample]],
    analysis_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for variant, tokenizer in tokenizers.items():
        copy_tokenizer_artifacts(
            TOKENIZER_SOURCE_DIRS[variant], analysis_dir / "tokenizers" / variant
        )
        payload[variant] = {
            "name": tokenizer.name,
            "vocabulary_size": len(tokenizer.vocabulary),
            "datasets": {},
        }
        for dataset in (
            "clean_train",
            "vosk_train",
            "final_train",
            *ALL_EVALUATIONS,
        ):
            statistics = tokenizers_v1.tokenizer_statistics(
                tokenizer, datasets[dataset]
            )
            payload[variant]["datasets"][dataset] = statistics
            rows.append(
                {
                    "tokenizer": variant,
                    "dataset": dataset,
                    "vocabulary_size": len(tokenizer.vocabulary),
                    **statistics,
                }
            )
    write_json(analysis_dir / "tokenizer_diagnostics.json", payload)
    write_csv(analysis_dir / "tokenizer_diagnostics.csv", rows, list(rows[0]))


def aggregate_all(
    output_dir: Path,
    datasets: dict[str, list[baseline.Sample]],
) -> dict[str, Any]:
    analysis_dir = output_dir / "Analysen"
    results = collect_results(output_dir)
    metric_rows, mean_rows, recall_mean_rows = aggregate_metrics(results, analysis_dir)
    sum_confusion_matrices(results, analysis_dir)
    prediction_rows, prediction_index = collect_predictions(output_dir, analysis_dir)
    stability_rows, boundary_rows = analyze_error_stability(
        prediction_rows, analysis_dir
    )
    analyze_transitions(prediction_index, datasets, analysis_dir)
    analyze_error_overlap(prediction_index, datasets, analysis_dir)
    tokenizer_effect_rows, tradeoff_rows = analyze_comparisons(
        metric_rows, analysis_dir
    )
    manual_evaluation_notes = ROOT / "MANUELLE_EVALUATION.md"
    if manual_evaluation_notes.is_file():
        shutil.copy2(manual_evaluation_notes, analysis_dir / manual_evaluation_notes.name)

    ranking = rank_combinations(metric_rows, analysis_dir)
    selected = select_representative_seeds(
        metric_rows, output_dir, analysis_dir
    )
    write_report(
        analysis_dir,
        metric_rows,
        mean_rows,
        recall_mean_rows,
        ranking,
        selected,
        boundary_rows,
        stability_rows,
        tokenizer_effect_rows,
        tradeoff_rows,
    )
    require(
        all(
            float(result["tflite"]["full_evaluation_parity"]["top1_agreement"])
            == 1.0
            for result in results
        ),
        "At least one model lacks full Keras/TFLite Top-1 parity",
    )
    require(
        all(bool(result["builtins_only_compatible"]) for result in results),
        "At least one model is not Builtins-only",
    )
    summary = {
        "status": "COMPLETE",
        "completed_models": len(results),
        "expected_models": 40,
        "combinations": 8,
        "seeds_per_combination": 5,
        "all_keras_tflite_top1_parity": True,
        "all_builtins_only": True,
        "tensor_contract": "int32[1,24] -> float32[1,10]",
        "development_rank_1": ranking[0]["combination"],
        "production_winner_selected": False,
        "representative_models": selected,
    }
    write_json(analysis_dir / "completion_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen M0-M3 x T1/T2 five-seed BaselineCNN experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "BaselineCNN" / "M0-M3 Multiseed",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit frozen inputs/tokenizers/architecture, but do not train.",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Rebuild aggregate analysis from 40 already completed model folders.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing preflight/output directory and skip complete models.",
    )
    parser.add_argument(
        "--verbose", type=int, choices=(0, 1, 2), default=2
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not (args.resume or args.analysis_only):
        print(
            f"ERROR: refusing to overwrite existing output directory: {output_dir}; "
            "use --resume only for this exact experiment",
            file=sys.stderr,
        )
        return 2
    if not output_dir.exists() and args.analysis_only:
        print(f"ERROR: analysis-only output does not exist: {output_dir}", file=sys.stderr)
        return 2

    try:
        datasets = load_datasets()
        output_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir = output_dir / "Analysen"
        frozen_files = verify_frozen_files(datasets)
        tokenizers, tokenizer_audit = load_tokenizers(datasets)
        data_audit = audit_data(datasets, output_dir)
        architecture = architecture_audit(tokenizers)
        write_tokenizer_diagnostics(tokenizers, datasets, analysis_dir)
        write_json(analysis_dir / "frozen_input_audit.json", frozen_files)
        write_json(analysis_dir / "frozen_tokenizer_audit.json", tokenizer_audit)
        write_json(analysis_dir / "data_audit.json", data_audit)
        write_json(analysis_dir / "architecture_audit.json", architecture)

        legacy_source = NLP_V1 / "train_intent_models_npu.py"
        run_config = {
            "status": "PREFLIGHT_COMPLETE" if args.preflight_only else "TRAINING",
            "experiment": "frozen_m0_m3_t1_t2_five_seed",
            "seeds": list(SEEDS),
            "seed_count": len(SEEDS),
            "strategies": {
                strategy: STRATEGY_DESCRIPTIONS[strategy]
                for strategy in STRATEGIES
            },
            "tokenizers": {
                "T1": "frozen deterministic Word-Level",
                "T2": "frozen deterministic BPE-2000",
            },
            "model_outputs_expected": 40,
            "max_len": MAX_LEN,
            "architecture": architecture,
            "batch_size": BATCH_SIZE,
            "epochs_max_each_phase": EPOCHS,
            "early_stopping_patience": PATIENCE,
            "learning_rates": LEARNING_RATES,
            "M1_M2_joint_mix": "all rows, deterministic upsampling to approximately 60% Clean / 40% Vosk",
            "M2_M3_phase1": "new M0 clean-only checkpoint from the same seed and tokenizer",
            "old_model_checkpoint_reuse": False,
            "legacy_M0_M3_logic_source": {
                "path": str(legacy_source),
                "sha256": sha256_file(legacy_source),
                "functions": ["make_domain_mix", "train_m0", "train_m1", "train_m2", "train_m3"],
            },
            "early_stopping_sets": list(DEVELOPMENT_EVALUATIONS),
            "challenge_used_for_training_or_early_stopping": False,
            "representative_seed_rule": REPRESENTATIVE_SEED_RULE,
            "development_ranking_rule": RANKING_RULE,
            "production_winner_selection_allowed": False,
            "production_winner_selected": False,
            "frozen_input_audit": frozen_files,
            "frozen_tokenizer_audit": tokenizer_audit,
            "data_audit": data_audit,
        }
        write_json(analysis_dir / "run_config.json", run_config)
        print(
            "PREFLIGHT OK: 4 strategies x 2 tokenizers x 5 seeds; "
            "frozen data/tokenizers and BaselineCNN verified.",
            flush=True,
        )
        if args.preflight_only:
            return 0
        if not args.analysis_only:
            train_all(output_dir, datasets, tokenizers, args.verbose)
        summary = aggregate_all(output_dir, datasets)
        run_config["status"] = "COMPLETE"
        run_config["completion_summary"] = summary
        write_json(analysis_dir / "run_config.json", run_config)
        print(
            f"COMPLETE: {summary['completed_models']} models; report: "
            f"{analysis_dir / 'ABSCHLUSSBERICHT_M0_M3_T1_T2.md'}",
            flush=True,
        )
        return 0
    except baseline.PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
