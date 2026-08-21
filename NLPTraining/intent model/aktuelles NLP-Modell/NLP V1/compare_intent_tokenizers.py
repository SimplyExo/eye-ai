#!/usr/bin/env python3
"""Prepare and run a leakage-safe T0/T1/T2 intent-tokenizer comparison.

Modes:
* provisional: technical test on the currently available training sources.
* final: requires an explicit final hard-negative patch and refuses any exact
  training/validation/challenge overlap.

No validation or challenge text is ever used to build either vocabulary.
Existing datasets and model artifacts are read-only; the output directory must
not exist before a run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

import diagnose_intent_model_architectures as diagnosis
import train_intent_models_npu as baseline


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_ID = 0
UNK_ID = 1
BPE_BOUNDARY = "▁"
DEFAULT_SEED = 20260810
DEFAULT_MAX_LEN = 24
DEFAULT_BPE_VOCAB_SIZE = 2000
DEFAULT_OUTPUT = "tokenizer_comparison_next"
CURRENT_RUN = "intent_models_current"
PARITY_EDGE_CASES = (
    "NICHT schneller, sondern nur die Stimme ändern!",
    "  Wie   viel Uhr ist es?  ",
    "ÄÖÜ äöü ß – 500 Hz; aber nicht 5 BPS.",
    "Wo finde ich das Autohaus?",
    "Ich möchte die geschriebenen Wortlaute hören.",
    "ÄRGER über ÖL, Übung und Straße.",
    "NICHT—nur–aber-sondern!",
    "Tempo auf 1.250,5 stellen?",
    "   mehrere     Leerzeichen   HIER   ",
    "\tTabs\tund   Leerzeichen\t",
    "Vollbreite：１２３ Hz",
    "„Stimme“ (WEIBLICH); bitte!",
    "E-Mail-Adresse ändern",
    "50% schneller + 2 BPS",
    "nur\u00a0nicht … SONDERN aber",
)


class Tokenizer(Protocol):
    name: str
    vocabulary: list[str]
    max_len: int

    def pieces(self, text: str) -> list[str]: ...

    def encode(self, text: str) -> np.ndarray: ...

    def write_artifacts(self, output_dir: Path) -> dict[str, Any]: ...


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vocabulary_checksum(vocabulary: Sequence[str]) -> str:
    payload = json.dumps(list(vocabulary), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class WordTokenizer:
    name: str
    vocabulary: list[str]
    max_len: int

    def __post_init__(self) -> None:
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise baseline.PipelineError(f"{self.name}: duplicate vocabulary entries")
        if self.vocabulary[0] != PAD_TOKEN:
            raise baseline.PipelineError(f"{self.name}: ID 0 must be {PAD_TOKEN}")
        if self.vocabulary[1] not in {UNK_TOKEN, baseline.OOV_TOKEN}:
            raise baseline.PipelineError(
                f"{self.name}: ID 1 must be {UNK_TOKEN} or historical {baseline.OOV_TOKEN}"
            )
        self.token_to_id = {
            token: index for index, token in enumerate(self.vocabulary)
        }

    @classmethod
    def from_training(
        cls,
        name: str,
        samples: Sequence[baseline.Sample],
        max_len: int,
    ) -> "WordTokenizer":
        counts = Counter(
            token
            for sample in samples
            for token in sample.normalized_text.split()
        )
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        return cls(name=name, vocabulary=[PAD_TOKEN, UNK_TOKEN, *ordered], max_len=max_len)

    def pieces(self, text: str) -> list[str]:
        return baseline.normalize_text(text).split()

    def ids_unpadded(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, UNK_ID) for token in self.pieces(text)]

    def encode(self, text: str) -> np.ndarray:
        ids = self.ids_unpadded(text)[: self.max_len]
        output = np.full(self.max_len, PAD_ID, dtype=np.int32)
        output[: len(ids)] = ids
        return output

    def write_artifacts(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "vocab.json", self.vocabulary)
        (output_dir / "vocab.txt").write_text(
            "\n".join(self.vocabulary) + "\n", encoding="utf-8"
        )
        config = {
            "version": 1,
            "tokenizer_type": "deterministic_word_level",
            "normalization": "shared_intent_nfkc_lower_punctuation_whitespace_v1",
            "split": "normalized whitespace",
            "padding": "post",
            "truncating": "post",
            "max_length": self.max_len,
            "reserved_tokens": {
                "PAD": {"token": self.vocabulary[PAD_ID], "id": PAD_ID},
                "UNK": {"token": self.vocabulary[UNK_ID], "id": UNK_ID},
            },
            "vocabulary_policy": "all normalized training tokens with frequency >= 1",
            "vocabulary_size": len(self.vocabulary),
            "vocabulary_checksum_sha256": vocabulary_checksum(self.vocabulary),
            "validation_or_challenge_used_for_vocabulary": False,
        }
        write_json(output_dir / "tokenizer_config.json", config)
        return tokenizer_artifact_sizes(output_dir)


@dataclass(frozen=True)
class BpeMerge:
    left: str
    right: str
    merged: str


@dataclass
class BpeTokenizer:
    name: str
    vocabulary: list[str]
    merges: list[BpeMerge]
    max_len: int
    requested_vocab_size: int

    def __post_init__(self) -> None:
        if self.vocabulary[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise baseline.PipelineError(
                f"{self.name}: reserved IDs must be {PAD_TOKEN}=0 and {UNK_TOKEN}=1"
            )
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise baseline.PipelineError(f"{self.name}: duplicate vocabulary entries")
        self.token_to_id = {
            token: index for index, token in enumerate(self.vocabulary)
        }
        self.merge_ranks = {
            (merge.left, merge.right): rank
            for rank, merge in enumerate(self.merges)
        }
        self.merge_outputs = {
            (merge.left, merge.right): merge.merged for merge in self.merges
        }

    @classmethod
    def train(
        cls,
        name: str,
        samples: Sequence[baseline.Sample],
        max_len: int,
        target_vocab_size: int,
    ) -> "BpeTokenizer":
        word_counts: Counter[str] = Counter(
            word
            for sample in samples
            for word in sample.normalized_text.split()
        )
        if not word_counts:
            raise baseline.PipelineError("Cannot train BPE on an empty corpus")
        characters = sorted({character for word in word_counts for character in word})
        vocabulary = [PAD_TOKEN, UNK_TOKEN, BPE_BOUNDARY, *characters]
        if target_vocab_size < len(vocabulary):
            raise baseline.PipelineError(
                f"BPE target {target_vocab_size} is below required base-symbol count "
                f"{len(vocabulary)}"
            )
        sequences: dict[str, tuple[str, ...]] = {
            word: tuple([BPE_BOUNDARY, *list(word)]) for word in word_counts
        }
        vocab_set = set(vocabulary)
        merges: list[BpeMerge] = []

        def merge_sequence(
            sequence: Sequence[str], left: str, right: str, merged: str
        ) -> tuple[str, ...]:
            result: list[str] = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and sequence[index] == left
                    and sequence[index + 1] == right
                ):
                    result.append(merged)
                    index += 2
                else:
                    result.append(sequence[index])
                    index += 1
            return tuple(result)

        while len(vocabulary) < target_vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for word, sequence in sequences.items():
                weight = word_counts[word]
                for index in range(len(sequence) - 1):
                    pair = (sequence[index], sequence[index + 1])
                    if pair[0] == UNK_TOKEN or pair[1] == UNK_TOKEN:
                        continue
                    if pair[0] + pair[1] in vocab_set:
                        continue
                    pair_counts[pair] += weight
            if not pair_counts:
                break
            best_count = max(pair_counts.values())
            best_pair = min(pair for pair, count in pair_counts.items() if count == best_count)
            merged = best_pair[0] + best_pair[1]
            vocabulary.append(merged)
            vocab_set.add(merged)
            merges.append(BpeMerge(best_pair[0], best_pair[1], merged))
            sequences = {
                word: merge_sequence(sequence, *best_pair, merged)
                for word, sequence in sequences.items()
            }
        return cls(
            name=name,
            vocabulary=vocabulary,
            merges=merges,
            max_len=max_len,
            requested_vocab_size=target_vocab_size,
        )

    def word_pieces(self, word: str) -> list[str]:
        pieces = [
            BPE_BOUNDARY,
            *[
                character if character in self.token_to_id else UNK_TOKEN
                for character in word
            ],
        ]
        while len(pieces) > 1:
            candidates = [
                (self.merge_ranks[(pieces[index], pieces[index + 1])], index)
                for index in range(len(pieces) - 1)
                if (pieces[index], pieces[index + 1]) in self.merge_ranks
            ]
            if not candidates:
                break
            selected_rank = min(rank for rank, _index in candidates)
            selected_merge = self.merges[selected_rank]
            selected_pair = (
                selected_merge.left,
                selected_merge.right,
            )
            merged = self.merge_outputs[selected_pair]
            result: list[str] = []
            index = 0
            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and pieces[index] == selected_pair[0]
                    and pieces[index + 1] == selected_pair[1]
                ):
                    result.append(merged)
                    index += 2
                else:
                    result.append(pieces[index])
                    index += 1
            pieces = result
        return pieces

    def pieces(self, text: str) -> list[str]:
        return [
            piece
            for word in baseline.normalize_text(text).split()
            for piece in self.word_pieces(word)
        ]

    def ids_unpadded(self, text: str) -> list[int]:
        return [self.token_to_id.get(piece, UNK_ID) for piece in self.pieces(text)]

    def encode(self, text: str) -> np.ndarray:
        ids = self.ids_unpadded(text)[: self.max_len]
        output = np.full(self.max_len, PAD_ID, dtype=np.int32)
        output[: len(ids)] = ids
        return output

    def write_artifacts(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "vocab.json", self.vocabulary)
        (output_dir / "vocab.txt").write_text(
            "\n".join(self.vocabulary) + "\n", encoding="utf-8"
        )
        merge_rows = [
            {
                "rank": rank,
                "left": merge.left,
                "right": merge.right,
                "merged": merge.merged,
            }
            for rank, merge in enumerate(self.merges)
        ]
        with (output_dir / "merges.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["rank", "left", "right", "merged"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(merge_rows)
        write_json(output_dir / "merges.json", merge_rows)
        config = {
            "version": 1,
            "tokenizer_type": "deterministic_word_boundary_bpe",
            "normalization": "shared_intent_nfkc_lower_punctuation_whitespace_v1",
            "word_boundary_symbol": BPE_BOUNDARY,
            "padding": "post",
            "truncating": "post",
            "max_length": self.max_len,
            "reserved_tokens": {
                "PAD": {"token": PAD_TOKEN, "id": PAD_ID},
                "UNK": {"token": UNK_TOKEN, "id": UNK_ID},
            },
            "requested_vocabulary_size": self.requested_vocab_size,
            "actual_vocabulary_size": len(self.vocabulary),
            "merge_count": len(self.merges),
            "vocabulary_checksum_sha256": vocabulary_checksum(self.vocabulary),
            "validation_or_challenge_used_for_vocabulary": False,
            "implementation_note": (
                "BPE merges are learned within normalized whitespace-delimited words; "
                "merges never cross word boundaries."
            ),
        }
        write_json(output_dir / "tokenizer_config.json", config)
        return tokenizer_artifact_sizes(output_dir)


def tokenizer_artifact_sizes(output_dir: Path) -> dict[str, Any]:
    files = {
        str(path.relative_to(output_dir)): path.stat().st_size
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    return {"files": files, "total_bytes": sum(files.values())}


def load_inputs(
    root: Path,
    hard_negative_patch: Path | None,
    challenge_holdout: Path | None,
) -> dict[str, Any]:
    current_dir = root / CURRENT_RUN
    datasets = diagnosis.load_all_inputs(root, current_dir)
    known_failure_templates = list(datasets["hard_cases_9"])
    if challenge_holdout is not None:
        replacement = baseline.load_dataset(
            challenge_holdout,
            source="challenge_holdout",
            domain="holdout",
        )
        if len(replacement) != 40:
            raise baseline.PipelineError(
                f"--challenge-holdout must contain exactly 40 rows; found {len(replacement)}"
            )
        datasets["challenge_40"] = replacement
    challenge_members: dict[tuple[str, str], list[baseline.Sample]] = defaultdict(list)
    for sample in datasets["challenge_40"]:
        challenge_members[(sample.normalized_text, sample.label)].append(sample)
    known_failure_subset: list[baseline.Sample] = []
    for template in known_failure_templates:
        matches = challenge_members.get((template.normalized_text, template.label), [])
        if len(matches) != 1:
            raise baseline.PipelineError(
                "Each known-failure template must map to exactly one Challenge-40 "
                f"row; {template.text!r} mapped to {len(matches)} row(s)."
            )
        known_failure_subset.append(matches[0])
    datasets["known_failure_templates_9"] = known_failure_templates
    datasets["challenge_known_failure_subset_9"] = known_failure_subset
    patch_samples: list[baseline.Sample] = []
    if hard_negative_patch is not None:
        patch_samples = baseline.load_dataset(
            hard_negative_patch,
            source="hard_negative_patch",
            domain="clean",
        )
    datasets["hard_negative_patch"] = patch_samples
    datasets["t0_train"] = list(datasets["all_training"])
    datasets["final_train"] = [*datasets["all_training"], *patch_samples]
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


def validate_comparison_data(
    datasets: dict[str, Any],
    *,
    mode: str,
    hard_negative_patch: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    if mode == "final" and hard_negative_patch is None:
        raise baseline.PipelineError(
            "Final mode requires --hard-negative-patch. The final T1-vs-T2 "
            "comparison is intentionally gated until that file exists."
        )
    patch_samples = datasets["hard_negative_patch"]
    if mode == "final" and not patch_samples:
        raise baseline.PipelineError(
            "Final mode requires a non-empty final hard-negative patch."
        )
    current_training = datasets["all_training"]
    final_training = datasets["final_train"]
    baseline.validate_dataset(final_training, "Tokenizer-comparison final training pool")
    challenge_report = baseline.validate_dataset(
        datasets["challenge_40"], "Challenge-40 holdout"
    )
    challenge_duplicates = challenge_report["duplicates"]
    if (
        challenge_duplicates["exact_duplicate_rows"]
        or challenge_duplicates["normalized_duplicate_rows"]
    ):
        raise baseline.PipelineError(
            "Challenge-40 must not contain exact or normalized duplicate rows."
        )

    subset_mapping_rows = [
        {
            "known_failure_template_line": template.line_no,
            "challenge_line": challenge_sample.line_no,
            "text": challenge_sample.text,
            "label": challenge_sample.label,
            "relationship": "member_of_challenge_40_not_independent_testset",
        }
        for template, challenge_sample in zip(
            datasets["known_failure_templates_9"],
            datasets["challenge_known_failure_subset_9"],
        )
    ]
    write_csv(
        output_dir / "data" / "known_failure_subset_mapping.csv",
        subset_mapping_rows,
        [
            "known_failure_template_line",
            "challenge_line",
            "text",
            "label",
            "relationship",
        ],
    )

    current_by_text = normalized_index(current_training, include_label=False)
    patch_duplicates: list[dict[str, Any]] = []
    for sample in patch_samples:
        for existing in current_by_text.get(sample.normalized_text, []):
            patch_duplicates.append(
                {
                    "patch_line": sample.line_no,
                    "patch_text": sample.text,
                    "patch_label": sample.label,
                    "existing_source": existing.source,
                    "existing_line": existing.line_no,
                    "existing_text": existing.text,
                    "existing_label": existing.label,
                }
            )
    if patch_duplicates:
        raise baseline.PipelineError(
            f"Hard-negative patch has {len(patch_duplicates)} normalized overlap(s) "
            "with current training; deduplicate the patch before comparison."
        )

    evaluation_sets = {
        name: datasets[name]
        for name in (
            "semantic_val",
            "asr_val",
            "complex_val",
            "challenge_40",
        )
    }
    overlap_rows: list[dict[str, Any]] = []
    final_by_text = normalized_index(final_training, include_label=False)
    for evaluation_name, samples in evaluation_sets.items():
        for sample in samples:
            for training_sample in final_by_text.get(sample.normalized_text, []):
                raw_exact_match = sample.text == training_sample.text
                overlap_rows.append(
                    {
                        "match_type": (
                            "exact_and_normalized"
                            if raw_exact_match
                            else "normalized_only"
                        ),
                        "evaluation_set": evaluation_name,
                        "evaluation_line": sample.line_no,
                        "evaluation_text": sample.text,
                        "evaluation_label": sample.label,
                        "training_source": training_sample.source,
                        "training_line": training_sample.line_no,
                        "training_text": training_sample.text,
                        "training_label": training_sample.label,
                        "introduced_by_patch": training_sample.source
                        == "hard_negative_patch",
                    }
                )
    write_csv(
        output_dir / "data" / "training_evaluation_overlap_audit.csv",
        overlap_rows,
        [
            "match_type",
            "evaluation_set",
            "evaluation_line",
            "evaluation_text",
            "evaluation_label",
            "training_source",
            "training_line",
            "training_text",
            "training_label",
            "introduced_by_patch",
        ],
    )
    exact_overlap_rows = [
        row for row in overlap_rows if row["match_type"] == "exact_and_normalized"
    ]
    patch_overlaps = [row for row in overlap_rows if row["introduced_by_patch"]]
    if patch_overlaps:
        raise baseline.PipelineError(
            f"Hard-negative patch leaks {len(patch_overlaps)} exact validation/"
            "challenge text(s); final comparison is forbidden."
        )
    if mode == "final" and overlap_rows:
        raise baseline.PipelineError(
            f"Final comparison requires a clean holdout, but {len(overlap_rows)} "
            "pre-existing normalized training/evaluation overlap(s) remain "
            f"({len(exact_overlap_rows)} also raw-exact). Resolve or replace the "
            "affected holdout rows first."
        )

    cross_evaluation_rows: list[dict[str, Any]] = []
    challenge_by_text = normalized_index(datasets["challenge_40"], include_label=False)
    for name in ("semantic_val", "asr_val", "complex_val"):
        for sample in datasets[name]:
            for challenge_sample in challenge_by_text.get(sample.normalized_text, []):
                cross_evaluation_rows.append(
                    {
                        "development_set": name,
                        "development_line": sample.line_no,
                        "challenge_line": challenge_sample.line_no,
                        "text": sample.text,
                    }
                )
    write_csv(
        output_dir / "data" / "development_challenge_overlap_audit.csv",
        cross_evaluation_rows,
        ["development_set", "development_line", "challenge_line", "text"],
    )
    return {
        "mode": mode,
        "hard_negative_patch": str(hard_negative_patch) if hard_negative_patch else None,
        "hard_negative_patch_samples": len(patch_samples),
        "t0_training_samples": len(datasets["t0_train"]),
        "t1_t2_training_samples": len(datasets["final_train"]),
        "exact_training_evaluation_overlaps": len(exact_overlap_rows),
        "normalized_training_evaluation_overlaps": len(overlap_rows),
        "patch_introduced_overlaps": len(patch_overlaps),
        "development_challenge_overlaps": len(cross_evaluation_rows),
        "challenge_internal_exact_duplicate_rows": challenge_duplicates[
            "exact_duplicate_rows"
        ],
        "challenge_internal_normalized_duplicate_rows": challenge_duplicates[
            "normalized_duplicate_rows"
        ],
        "known_failure_subset_samples": len(
            datasets["challenge_known_failure_subset_9"]
        ),
        "known_failure_subset_is_independent_testset": False,
        "final_comparison_allowed": mode == "final" and not overlap_rows,
        "provisional_reason": (
            None
            if mode == "final"
            else "Final hard-negative patch is not present; no tokenizer winner may be selected."
        ),
    }


def source_provenance(
    datasets: dict[str, Any],
    root: Path,
    hard_negative_patch: Path | None,
    challenge_holdout: Path | None,
) -> dict[str, Any]:
    source_paths = [
        root
        / "Trainingsdaten"
        / "01_Daten_mit_Widerspruechen_bereinigt_dedupliziert.txt",
        root
        / "Trainingsdaten"
        / "02_Saubere_Trainingsdaten_bereinigt_dedupliziert.txt",
        root
        / "Trainingsdaten"
        / "vosk_generation1_final_manuell_kuratiert.txt",
        root
        / "Trainingsdaten"
        / "vosk_generation2_streng_manuell_gefiltert.txt",
    ]
    if hard_negative_patch is not None:
        source_paths.append(hard_negative_patch)
    return {
        "training_sources": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in source_paths
        ],
        "sample_counts": {
            "clean": len(datasets["clean_train"]),
            "vosk": len(datasets["vosk_train"]),
            "hard_negative_patch": len(datasets["hard_negative_patch"]),
            "t0_train": len(datasets["t0_train"]),
            "t1_t2_train": len(datasets["final_train"]),
        },
        "vocabulary_sources": {
            "T0": "saved current training-only vocabulary",
            "T1": "T1/T2 training rows only",
            "T2": "T1/T2 training rows only",
        },
        "explicitly_excluded_from_all_vocabulary_training": [
            "semantic_val",
            "asr_val",
            "complex_val",
            "challenge_40",
            "challenge_known_failure_subset_9",
        ],
        "challenge_holdout_override": (
            {
                "path": str(challenge_holdout.resolve()),
                "bytes": challenge_holdout.stat().st_size,
                "sha256": sha256_file(challenge_holdout),
            }
            if challenge_holdout is not None
            else None
        ),
    }


def encode_samples(
    tokenizer: Tokenizer, samples: Sequence[baseline.Sample]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([tokenizer.encode(sample.text) for sample in samples]).astype(
        np.int32, copy=False
    )
    labels = np.asarray([sample.label_id for sample in samples], dtype=np.int32)
    return features, labels


def tokenizer_statistics(
    tokenizer: Tokenizer, samples: Sequence[baseline.Sample]
) -> dict[str, Any]:
    raw_words = [
        word for sample in samples for word in sample.normalized_text.split()
    ]
    encoded_pieces = [piece for sample in samples for piece in tokenizer.pieces(sample.text)]
    unk_pieces = sum(piece not in tokenizer.vocabulary for piece in encoded_pieces)
    # BPE emits the explicit [UNK] symbol before lookup; word tokenizers retain the
    # original unknown word, which is not in the vocabulary.
    unk_pieces += sum(piece == UNK_TOKEN for piece in encoded_pieces)
    unknown_word_count = 0
    if isinstance(tokenizer, WordTokenizer):
        unknown_word_count = sum(word not in tokenizer.token_to_id for word in raw_words)
    else:
        unknown_word_count = sum(
            UNK_TOKEN in tokenizer.word_pieces(word) for word in raw_words
        )
    lengths = [len(tokenizer.pieces(sample.text)) for sample in samples]
    sentences_with_unk = sum(
        any(piece == UNK_TOKEN or piece not in tokenizer.vocabulary for piece in tokenizer.pieces(sample.text))
        for sample in samples
    )
    truncated = sum(length > tokenizer.max_len for length in lengths)
    return {
        "sentences": len(samples),
        "raw_words": len(raw_words),
        "encoded_pieces": len(encoded_pieces),
        "unknown_words_or_character_sequences": unknown_word_count,
        "word_unknown_rate": unknown_word_count / len(raw_words) if raw_words else 0.0,
        "unk_pieces": unk_pieces,
        "unk_piece_rate": unk_pieces / len(encoded_pieces) if encoded_pieces else 0.0,
        "sentences_with_unk": sentences_with_unk,
        "sentence_unk_rate": sentences_with_unk / len(samples) if samples else 0.0,
        "mean_pieces_per_word": len(encoded_pieces) / len(raw_words) if raw_words else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_median": float(np.median(lengths)) if lengths else 0.0,
        "length_p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "length_p99": float(np.percentile(lengths, 99)) if lengths else 0.0,
        "length_maximum": max(lengths, default=0),
        "truncated_sentences": truncated,
        "truncation_rate": truncated / len(samples) if samples else 0.0,
    }


def write_tokenizer_diagnostics(
    tokenizers: dict[str, Tokenizer],
    datasets: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    set_names = (
        "t0_train",
        "final_train",
        "semantic_val",
        "asr_val",
        "complex_val",
        "challenge_40",
        "challenge_known_failure_subset_9",
    )
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    artifact_sizes: dict[str, Any] = {}
    for variant, tokenizer in tokenizers.items():
        tokenizer_dir = output_dir / "tokenizers" / variant
        artifact_sizes[variant] = tokenizer.write_artifacts(tokenizer_dir)
        summary[variant] = {
            "tokenizer_name": tokenizer.name,
            "vocabulary_size": len(tokenizer.vocabulary),
            "vocabulary_checksum_sha256": vocabulary_checksum(tokenizer.vocabulary),
            "datasets": {},
        }
        for set_name in set_names:
            statistics = tokenizer_statistics(tokenizer, datasets[set_name])
            summary[variant]["datasets"][set_name] = statistics
            rows.append(
                {
                    "variant": variant,
                    "dataset": set_name,
                    "vocabulary_size": len(tokenizer.vocabulary),
                    **statistics,
                }
            )
    write_json(output_dir / "tokenizer_diagnostics.json", summary)
    write_json(output_dir / "tokenizer_artifact_sizes.json", artifact_sizes)
    write_csv(
        output_dir / "tokenizer_diagnostics.csv",
        rows,
        list(rows[0]),
    )
    length_rows: list[dict[str, Any]] = []
    for variant in ("T1", "T2"):
        tokenizer = tokenizers[variant]
        for set_name in (
            "final_train",
            "semantic_val",
            "asr_val",
            "complex_val",
            "challenge_40",
        ):
            statistics = summary[variant]["datasets"][set_name]
            length_rows.append(
                {
                    "variant": variant,
                    "tokenizer": tokenizer.name,
                    "dataset": set_name,
                    "samples": statistics["sentences"],
                    "max_len": tokenizer.max_len,
                    "token_count_mean": statistics["length_mean"],
                    "token_count_median": statistics["length_median"],
                    "token_count_p95": statistics["length_p95"],
                    "token_count_p99": statistics["length_p99"],
                    "token_count_maximum": statistics["length_maximum"],
                    "truncated_samples": statistics["truncated_sentences"],
                    "truncation_rate": statistics["truncation_rate"],
                }
            )
    write_csv(
        output_dir / "sequence_length_comparison.csv",
        length_rows,
        list(length_rows[0]),
    )

    challenge_rows: list[dict[str, Any]] = []
    for variant, tokenizer in tokenizers.items():
        for index, sample in enumerate(datasets["challenge_40"], start=1):
            pieces = tokenizer.pieces(sample.text)
            ids = [
                getattr(tokenizer, "token_to_id").get(piece, UNK_ID)
                for piece in pieces
            ]
            challenge_rows.append(
                {
                    "variant": variant,
                    "challenge_line": index,
                    "text": sample.text,
                    "expected_label": sample.label,
                    "normalized_text": sample.normalized_text,
                    "pieces": json.dumps(pieces, ensure_ascii=False),
                    "piece_ids": json.dumps(ids),
                    "unk_pieces": sum(
                        piece == UNK_TOKEN or piece not in tokenizer.vocabulary
                        for piece in pieces
                    ),
                    "truncated_pieces": json.dumps(
                        pieces[tokenizer.max_len :], ensure_ascii=False
                    ),
                }
            )
    write_csv(
        output_dir / "challenge_tokenization.csv",
        challenge_rows,
        [
            "variant",
            "challenge_line",
            "text",
            "expected_label",
            "normalized_text",
            "pieces",
            "piece_ids",
            "unk_pieces",
            "truncated_pieces",
        ],
    )
    return {"statistics": summary, "artifact_sizes": artifact_sizes}


def benchmark_tokenizer(
    tokenizer: Tokenizer,
    texts: Sequence[str],
    *,
    warmup_runs: int = 100,
    measured_runs: int = 2000,
) -> dict[str, Any]:
    for index in range(warmup_runs):
        tokenizer.encode(texts[index % len(texts)])
    durations_ms: list[float] = []
    for index in range(measured_runs):
        start = time.perf_counter_ns()
        tokenizer.encode(texts[index % len(texts)])
        durations_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return {
        "environment": "host CPython; includes normalization and token lookup",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": float(np.mean(durations_ms)),
        "median_ms": float(np.median(durations_ms)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "android_latency_claim": False,
    }


def export_model(
    model: Any,
    vocab_size: int,
    max_len: int,
    verification_features: np.ndarray,
    model_dir: Path,
) -> dict[str, Any]:
    fixed_model = baseline.build_model(
        vocab_size,
        max_len,
        batch_size=1,
        name="tokenizer_comparison_fixed_batch",
    )
    fixed_model.set_weights(model.get_weights())
    model_path = model_dir / "model_builtin.tflite"
    baseline.convert_builtin_tflite(fixed_model, model_path)
    analyzer_text = baseline._analyze_tflite(model_path)
    (model_dir / "tflite_analyzer.txt").write_text(
        analyzer_text, encoding="utf-8"
    )
    keras_probabilities = np.asarray(
        model.predict(verification_features, verbose=0), dtype=np.float32
    )
    lite_probabilities = baseline._tflite_predict(model_path, verification_features)
    parity = baseline._parity_metrics(keras_probabilities, lite_probabilities)
    if parity["top1_agreement"] != 1.0 or parity["max_absolute_probability_difference"] > 1e-4:
        raise baseline.PipelineError(
            f"TFLite parity failed for {model_dir.name}: {parity}"
        )
    return {
        "success": True,
        "builtins_only": True,
        "fixed_input_shape": [1, max_len],
        "input_dtype": "int32",
        "model_size_bytes": model_path.stat().st_size,
        "parity": parity,
        "operators": diagnosis.tflite_operator_summary(model_path),
        "host_cpu_latency": diagnosis.benchmark_tflite(
            model_path, verification_features
        ),
    }


def train_and_evaluate(
    tokenizers: dict[str, Tokenizer],
    datasets: dict[str, Any],
    output_dir: Path,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    verbose: int,
) -> dict[str, Any]:
    tf = baseline.require_tensorflow()
    evaluation_names = (
        "semantic_val",
        "asr_val",
        "complex_val",
        "combined_val",
        "challenge_40",
        "challenge_known_failure_subset_9",
    )
    all_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for variant, tokenizer in tokenizers.items():
        tf.keras.backend.clear_session()
        baseline.set_global_determinism(seed)
        training_name = "t0_train" if variant == "T0" else "final_train"
        train_data = encode_samples(tokenizer, datasets[training_name])
        encoded_evaluations = {
            name: encode_samples(tokenizer, datasets[name])
            for name in evaluation_names
        }
        model_dir = output_dir / "models" / variant
        model = baseline.build_model(
            len(tokenizer.vocabulary),
            tokenizer.max_len,
            name=f"tokenizer_comparison_{variant}",
        )
        print(
            f"\n=== {variant}: {tokenizer.name}, vocab={len(tokenizer.vocabulary)}, "
            f"train={len(train_data[0])} ==="
        )
        baseline._train_phase(
            model,
            model_dir,
            train_data,
            encoded_evaluations["combined_val"],
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
            verbose=verbose,
        )
        result: dict[str, Any] = {
            "variant": variant,
            "tokenizer": tokenizer.name,
            "training_set": training_name,
            "training_samples": len(train_data[0]),
            "vocabulary_size": len(tokenizer.vocabulary),
            "parameter_count": int(model.count_params()),
            "training": diagnosis.history_summary(model_dir / "history.csv"),
            "evaluations": {},
        }
        for set_name in evaluation_names:
            probabilities = np.asarray(
                model.predict(
                    encoded_evaluations[set_name][0],
                    batch_size=batch_size,
                    verbose=0,
                ),
                dtype=np.float32,
            )
            metrics = diagnosis.metrics_from_probabilities(
                datasets[set_name], probabilities
            )
            result["evaluations"][set_name] = metrics
            diagnosis.write_confusion(
                model_dir / f"confusion_matrix_{set_name}.csv",
                metrics["confusion_matrix"],
            )
            for label in baseline.LABELS:
                recall_rows.append(
                    {
                        "variant": variant,
                        "dataset": set_name,
                        "label": label,
                        "recall": metrics["per_class"][label]["recall"],
                        "support": metrics["per_class"][label]["support"],
                    }
                )
            prediction_rows.extend(
                diagnosis.prediction_rows(
                    variant,
                    set_name,
                    datasets[set_name],
                    probabilities,
                )
            )
        verification_features = encoded_evaluations["combined_val"][0][:100]
        result["tflite"] = export_model(
            model,
            len(tokenizer.vocabulary),
            tokenizer.max_len,
            verification_features,
            model_dir,
        )
        result["tokenizer_host_latency"] = benchmark_tokenizer(
            tokenizer, [sample.text for sample in datasets["challenge_40"]]
        )
        write_json(model_dir / "metrics.json", result)
        all_results[variant] = result
        comparison_rows.append(
            {
                "variant": variant,
                "training_samples": result["training_samples"],
                "vocabulary_size": result["vocabulary_size"],
                "parameters": result["parameter_count"],
                "best_epoch": result["training"]["best_epoch_one_based"],
                "semantic_accuracy": result["evaluations"]["semantic_val"]["accuracy"],
                "semantic_macro_f1": result["evaluations"]["semantic_val"]["macro_f1_all_10_classes"],
                "asr_accuracy": result["evaluations"]["asr_val"]["accuracy"],
                "asr_macro_f1": result["evaluations"]["asr_val"]["macro_f1_all_10_classes"],
                "complex_accuracy": result["evaluations"]["complex_val"]["accuracy"],
                "complex_macro_f1": result["evaluations"]["complex_val"]["macro_f1_all_10_classes"],
                "challenge_accuracy": result["evaluations"]["challenge_40"]["accuracy"],
                "challenge_macro_f1_present": result["evaluations"]["challenge_40"]["macro_f1_present_truth_classes"],
                "challenge_known_failure_subset_correct": result["evaluations"][
                    "challenge_known_failure_subset_9"
                ]["correct_count"],
                "tflite_size_bytes": result["tflite"]["model_size_bytes"],
                "tokenizer_median_ms_host": result["tokenizer_host_latency"]["median_ms"],
                "tflite_median_ms_host": result["tflite"]["host_cpu_latency"]["median_ms"],
            }
        )
        del model
        tf.keras.backend.clear_session()
    write_json(output_dir / "model_metrics.json", all_results)
    write_csv(
        output_dir / "model_comparison.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )
    write_csv(
        output_dir / "per_class_recall.csv",
        recall_rows,
        ["variant", "dataset", "label", "recall", "support"],
    )
    write_csv(
        output_dir / "full_softmax_predictions.csv",
        prediction_rows,
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
    return all_results


def parity_test_texts(datasets: dict[str, Any]) -> list[str]:
    return [
        *(sample.text for sample in datasets["challenge_40"]),
        *PARITY_EDGE_CASES,
    ]


def write_parity_fixtures(
    tokenizers: dict[str, Tokenizer],
    datasets: dict[str, Any],
    output_dir: Path,
) -> None:
    texts = parity_test_texts(datasets)
    rows: list[dict[str, Any]] = []
    for index, text in enumerate(texts, start=1):
        row: dict[str, Any] = {
            "case": index,
            "text": text.replace("\t", " ").replace("\n", " "),
            "normalized": baseline.normalize_text(text),
        }
        for variant, tokenizer in tokenizers.items():
            row[f"{variant}_ids"] = ",".join(
                str(int(value)) for value in tokenizer.encode(text)
            )
            row[f"{variant}_pieces"] = json.dumps(
                tokenizer.pieces(text), ensure_ascii=False
            )
        rows.append(row)
    write_csv(
        output_dir / "android_parity_fixtures.tsv",
        rows,
        [
            "case",
            "text",
            "normalized",
            "T0_ids",
            "T0_pieces",
            "T1_ids",
            "T1_pieces",
            "T2_ids",
            "T2_pieces",
        ],
    )
    # Rewrite with a true tab delimiter for JVM tooling.
    with (output_dir / "android_parity_fixtures.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _single_cached_jar(pattern: str) -> Path:
    candidates = sorted(Path.home().glob(pattern))
    if not candidates:
        raise baseline.PipelineError(f"Required cached Kotlin/JVM jar not found: {pattern}")
    return candidates[-1]


def verify_kotlin_parity(
    tokenizers: dict[str, Tokenizer],
    datasets: dict[str, Any],
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    reference_dir = output_dir / "android_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        root / "android_reference" / "IntentTokenizer.kt",
        root / "android_reference" / "IntentModelInput.kt",
        root / "android_reference" / "TokenizerParityCli.kt",
    ]
    for source in source_files:
        shutil.copy2(source, reference_dir / source.name)
    integration_notes = root / "android_reference" / "ANDROID_INTEGRATION_NOTES.md"
    shutil.copy2(integration_notes, reference_dir / integration_notes.name)

    java = shutil.which("java")
    if java is None:
        raise baseline.PipelineError("Java runtime is required for Kotlin parity verification")
    compiler = _single_cached_jar(
        ".gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin/"
        "kotlin-compiler-embeddable/2.0.21/*/kotlin-compiler-embeddable-2.0.21.jar"
    )
    stdlib = _single_cached_jar(
        ".gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin/"
        "kotlin-stdlib/2.0.21/*/kotlin-stdlib-2.0.21.jar"
    )
    compiler_dependencies = [
        compiler,
        stdlib,
        _single_cached_jar(
            ".gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin/"
            "kotlin-script-runtime/2.0.21/*/kotlin-script-runtime-2.0.21.jar"
        ),
        _single_cached_jar(
            ".gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin/"
            "kotlin-reflect/2.0.21/*/kotlin-reflect-2.0.21.jar"
        ),
        _single_cached_jar(
            ".gradle/caches/modules-2/files-2.1/org.jetbrains.intellij.deps/"
            "trove4j/1.0.20200330/*/trove4j-1.0.20200330.jar"
        ),
        _single_cached_jar(
            ".gradle/caches/modules-2/files-2.1/org.jetbrains.kotlinx/"
            "kotlinx-coroutines-core-jvm/1.6.4/*/kotlinx-coroutines-core-jvm-1.6.4.jar"
        ),
        _single_cached_jar(
            ".gradle/caches/modules-2/files-2.1/org.jetbrains/annotations/"
            "13.0/*/annotations-13.0.jar"
        ),
    ]
    compiled_jar = reference_dir / "tokenizer-parity.jar"
    compile_command = [
        java,
        "-cp",
        os.pathsep.join(str(path) for path in compiler_dependencies),
        "org.jetbrains.kotlin.cli.jvm.K2JVMCompiler",
        "-no-stdlib",
        "-no-reflect",
        "-classpath",
        str(stdlib),
        "-jvm-target",
        "11",
        "-d",
        str(compiled_jar),
        *(str(path) for path in source_files),
    ]
    compile_result = subprocess.run(
        compile_command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    (reference_dir / "kotlin_compile.log").write_text(
        compile_result.stdout + compile_result.stderr, encoding="utf-8"
    )
    if compile_result.returncode != 0:
        raise baseline.PipelineError(
            f"Kotlin tokenizer compilation failed; see {reference_dir / 'kotlin_compile.log'}"
        )

    texts = parity_test_texts(datasets)
    variant_results: dict[str, Any] = {}
    runtime_classpath = os.pathsep.join([str(compiled_jar), str(stdlib)])
    for variant, tokenizer in tokenizers.items():
        tokenizer_dir = output_dir / "tokenizers" / variant
        mode = "bpe" if isinstance(tokenizer, BpeTokenizer) else "word"
        run_command = [
            java,
            "-cp",
            runtime_classpath,
            "com.algorithmic_alliance.eyeaiapp.nlp.tokenizer.TokenizerParityCliKt",
            mode,
            str(tokenizer_dir / "vocab.txt"),
            str(tokenizer.max_len),
        ]
        if mode == "bpe":
            run_command.append(str(tokenizer_dir / "merges.tsv"))
        run_result = subprocess.run(
            run_command,
            cwd=root,
            input="\n".join(texts) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if run_result.returncode != 0:
            raise baseline.PipelineError(
                f"Kotlin parity CLI failed for {variant}: {run_result.stderr}"
            )
        raw_kotlin_rows = run_result.stdout.splitlines()
        kotlin_rows = [
            tuple(row.split("\t", maxsplit=1)) if "\t" in row else (None, row)
            for row in raw_kotlin_rows
        ]
        expected_rows = [
            (
                baseline.normalize_text(text),
                ",".join(str(int(value)) for value in tokenizer.encode(text)),
            )
            for text in texts
        ]
        mismatches: list[dict[str, Any]] = []
        normalization_mismatches = 0
        id_mismatches = 0
        for index in range(max(len(expected_rows), len(kotlin_rows))):
            python_normalized, python_ids = (
                expected_rows[index] if index < len(expected_rows) else (None, None)
            )
            kotlin_normalized, kotlin_ids = (
                kotlin_rows[index] if index < len(kotlin_rows) else (None, None)
            )
            normalization_matches = python_normalized == kotlin_normalized
            ids_match = python_ids == kotlin_ids
            if not normalization_matches:
                normalization_mismatches += 1
            if not ids_match:
                id_mismatches += 1
            if not normalization_matches or not ids_match:
                mismatches.append(
                    {
                        "case": index + 1,
                        "text": texts[index] if index < len(texts) else None,
                        "python_normalized": python_normalized,
                        "kotlin_normalized": kotlin_normalized,
                        "python_ids": python_ids,
                        "kotlin_ids": kotlin_ids,
                    }
                )
        write_json(reference_dir / f"{variant}_parity_mismatches.json", mismatches)
        variant_results[variant] = {
            "cases": len(texts),
            "normalization_mismatches": normalization_mismatches,
            "id_mismatches": id_mismatches,
            "mismatches": len(mismatches),
            "exact_normalization_parity": normalization_mismatches == 0,
            "exact_id_parity": id_mismatches == 0,
        }
        if mismatches:
            raise baseline.PipelineError(
                f"Python/Kotlin tokenizer parity failed for {variant}: "
                f"{len(mismatches)} mismatch(es)"
            )
    summary = {
        "compiled": True,
        "compiler": str(compiler),
        "jvm_target": 11,
        "variants": variant_results,
        "all_variants_exact_normalization_parity": all(
            result["exact_normalization_parity"]
            for result in variant_results.values()
        ),
        "all_variants_exact_id_parity": all(
            result["exact_id_parity"] for result in variant_results.values()
        ),
    }
    write_json(reference_dir / "parity_summary.json", summary)
    return summary


def tokenizer_equivalence_report(
    tokenizers: dict[str, Tokenizer], datasets: dict[str, Any]
) -> dict[str, Any]:
    t0 = tokenizers["T0"]
    t1 = tokenizers["T1"]
    token_order_equal_except_unknown_name = (
        len(t0.vocabulary) == len(t1.vocabulary)
        and t0.vocabulary[0] == t1.vocabulary[0]
        and t0.vocabulary[2:] == t1.vocabulary[2:]
    )
    compared_samples = [
        *datasets["t0_train"],
        *datasets["semantic_val"],
        *datasets["asr_val"],
        *datasets["complex_val"],
        *datasets["challenge_40"],
    ]
    encoding_mismatches = sum(
        not np.array_equal(t0.encode(sample.text), t1.encode(sample.text))
        for sample in compared_samples
    )
    return {
        "token_order_equal_except_reserved_unknown_spelling": token_order_equal_except_unknown_name,
        "compared_samples": len(compared_samples),
        "encoding_mismatches": encoding_mismatches,
        "functionally_identical_on_compared_samples": encoding_mismatches == 0,
        "interpretation": (
            "With current data, T0 already contains every frequency>=1 training word; "
            "T1 differs only by [OOV] -> [UNK]."
            if encoding_mismatches == 0
            else "T1 differs because its final training pool contains additional patch tokens."
        ),
    }


def corpus_checksum(samples: Sequence[baseline.Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.normalized_text.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(sample.label.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def tokenizer_training_boundary_audit(
    tokenizers: dict[str, Tokenizer], datasets: dict[str, Any]
) -> dict[str, Any]:
    final_training = datasets["final_train"]
    training_words = {
        word for sample in final_training for word in sample.normalized_text.split()
    }
    evaluation_samples = [
        *datasets["semantic_val"],
        *datasets["asr_val"],
        *datasets["complex_val"],
        *datasets["challenge_40"],
    ]
    evaluation_words = {
        word for sample in evaluation_samples for word in sample.normalized_text.split()
    }
    evaluation_only_words = evaluation_words - training_words

    t1 = tokenizers["T1"]
    t1_words = set(t1.vocabulary[2:])
    t1_missing = sorted(training_words - t1_words)
    t1_extra = sorted(t1_words - training_words)
    evaluation_only_in_t1 = sorted(evaluation_only_words & t1_words)

    t2 = tokenizers["T2"]
    if not isinstance(t2, BpeTokenizer):
        raise baseline.PipelineError("T2 must be the BPE tokenizer")
    training_characters = sorted(
        {character for word in training_words for character in word}
    )
    expected_t2_vocabulary = [
        PAD_TOKEN,
        UNK_TOKEN,
        BPE_BOUNDARY,
        *training_characters,
        *(merge.merged for merge in t2.merges),
    ]
    available_symbols = {BPE_BOUNDARY, *training_characters}
    valid_merge_chain = True
    for merge in t2.merges:
        if (
            merge.left not in available_symbols
            or merge.right not in available_symbols
            or merge.merged != merge.left + merge.right
        ):
            valid_merge_chain = False
            break
        available_symbols.add(merge.merged)

    audit = {
        "training_corpus_samples": len(final_training),
        "training_corpus_checksum_sha256": corpus_checksum(final_training),
        "evaluation_samples_excluded_from_construction": len(evaluation_samples),
        "T1": {
            "reserved_tokens_correct": t1.vocabulary[:2]
            == [PAD_TOKEN, UNK_TOKEN],
            "training_unique_words": len(training_words),
            "non_reserved_vocabulary_tokens": len(t1.vocabulary[2:]),
            "missing_training_words": t1_missing,
            "extra_non_training_words": t1_extra,
            "evaluation_only_unique_words": len(evaluation_only_words),
            "evaluation_only_words_present_in_vocabulary": evaluation_only_in_t1,
            "exactly_all_frequency_ge_1_training_words": not t1_missing
            and not t1_extra,
        },
        "T2": {
            "reserved_tokens_correct": t2.vocabulary[:2]
            == [PAD_TOKEN, UNK_TOKEN],
            "base_characters_from_training_only": len(training_characters),
            "merge_count": len(t2.merges),
            "valid_training_derived_merge_chain": valid_merge_chain,
            "vocabulary_matches_training_base_plus_merges": t2.vocabulary
            == expected_t2_vocabulary,
            "evaluation_rows_passed_to_bpe_training": 0,
        },
    }
    if not (
        audit["T1"]["reserved_tokens_correct"]
        and audit["T1"]["exactly_all_frequency_ge_1_training_words"]
        and not evaluation_only_in_t1
        and audit["T2"]["reserved_tokens_correct"]
        and valid_merge_chain
        and audit["T2"]["vocabulary_matches_training_base_plus_merges"]
    ):
        raise baseline.PipelineError(
            "Tokenizer training-boundary audit failed; refusing comparison."
        )
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe T0/T1/T2 tokenizer comparison for the intent CNN."
    )
    parser.add_argument("--mode", choices=("provisional", "final"), default="provisional")
    parser.add_argument("--hard-negative-patch", type=Path, default=None)
    parser.add_argument(
        "--challenge-holdout",
        type=Path,
        default=None,
        help="Optional clean 40-row replacement holdout; never used for vocab/training.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--bpe-vocab-size", type=int, default=DEFAULT_BPE_VOCAB_SIZE)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    hard_negative_patch = (
        args.hard_negative_patch.resolve() if args.hard_negative_patch else None
    )
    challenge_holdout = (
        args.challenge_holdout.resolve() if args.challenge_holdout else None
    )
    if output_dir.exists():
        print(f"ERROR: refusing to overwrite existing output directory: {output_dir}", file=os.sys.stderr)
        return 2
    if args.mode == "final" and args.hard_negative_patch is None:
        print(
            "ERROR: --mode final requires --hard-negative-patch; the final "
            "comparison remains gated until that file exists.",
            file=os.sys.stderr,
        )
        return 2
    if args.mode == "final" and args.challenge_holdout is None:
        print(
            "ERROR: --mode final requires an explicit clean "
            "--challenge-holdout with exactly 40 rows.",
            file=os.sys.stderr,
        )
        return 2
    for option_name, path in (
        ("--hard-negative-patch", args.hard_negative_patch),
        ("--challenge-holdout", args.challenge_holdout),
    ):
        if path is not None and not path.is_file():
            print(f"ERROR: {option_name} is not a readable file: {path}", file=os.sys.stderr)
            return 2
    if args.bpe_vocab_size < 1500 or args.bpe_vocab_size > 3000:
        print("ERROR: --bpe-vocab-size must stay within the requested 1500-3000 range", file=os.sys.stderr)
        return 2
    if args.max_len <= 0:
        print("ERROR: --max-len must be positive", file=os.sys.stderr)
        return 2
    output_dir.mkdir(parents=True)
    try:
        datasets = load_inputs(root, hard_negative_patch, challenge_holdout)
        data_report = validate_comparison_data(
            datasets,
            mode=args.mode,
            hard_negative_patch=hard_negative_patch,
            output_dir=output_dir,
        )
        provenance = source_provenance(
            datasets, root, hard_negative_patch, challenge_holdout
        )
        write_json(output_dir / "data_report.json", data_report)
        write_json(output_dir / "vocabulary_provenance.json", provenance)

        current_vocabulary = json.loads(
            (root / CURRENT_RUN / "tokenizer" / "vocab.json").read_text(
                encoding="utf-8"
            )
        )
        t0 = WordTokenizer("current_word", current_vocabulary, args.max_len)
        t1 = WordTokenizer.from_training(
            "improved_word_all_frequency_ge_1",
            datasets["final_train"],
            args.max_len,
        )
        print(
            f"Training deterministic BPE vocab={args.bpe_vocab_size} from "
            f"{len(datasets['final_train'])} training-only samples..."
        )
        t2 = BpeTokenizer.train(
            "deterministic_bpe",
            datasets["final_train"],
            args.max_len,
            args.bpe_vocab_size,
        )
        tokenizers: dict[str, Tokenizer] = {"T0": t0, "T1": t1, "T2": t2}
        training_boundary_audit = tokenizer_training_boundary_audit(
            tokenizers, datasets
        )
        write_json(
            output_dir / "tokenizer_training_boundary_audit.json",
            training_boundary_audit,
        )
        diagnostics = write_tokenizer_diagnostics(tokenizers, datasets, output_dir)
        equivalence = tokenizer_equivalence_report(tokenizers, datasets)
        write_json(output_dir / "t0_t1_equivalence.json", equivalence)
        write_parity_fixtures(tokenizers, datasets, output_dir)
        kotlin_parity = verify_kotlin_parity(
            tokenizers, datasets, root, output_dir
        )

        run_config = {
            "status": (
                "PROVISIONAL"
                if args.mode == "provisional"
                else "FINAL_PREPARE_ONLY"
                if args.prepare_only
                else "FINAL"
            ),
            "winner_selection_allowed": args.mode == "final"
            and not args.prepare_only,
            "mode": args.mode,
            "seed": args.seed,
            "max_len": args.max_len,
            "bpe_vocab_size": args.bpe_vocab_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation": "same combined Semantic+ASR+Complex development validation",
            "architecture": diagnosis.ARCHITECTURE_DESCRIPTIONS["baseline_cnn"],
            "hard_negative_patch": str(hard_negative_patch) if hard_negative_patch else None,
            "challenge_holdout": str(challenge_holdout) if challenge_holdout else None,
            "prepare_only": args.prepare_only,
            "t0_t1_equivalence": equivalence,
            "tokenizer_training_boundary_audit": training_boundary_audit,
            "kotlin_parity": kotlin_parity,
            "tokenizer_diagnostics": diagnostics,
        }
        write_json(output_dir / "run_config.json", run_config)
        if not args.prepare_only:
            train_and_evaluate(
                tokenizers,
                datasets,
                output_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                verbose=args.verbose,
            )
        print(f"\nTokenizer comparison artifacts completed: {output_dir}")
        if args.mode == "provisional":
            print(
                "PROVISIONAL ONLY: no tokenizer winner may be selected before the "
                "final hard-negative patch and clean holdout are supplied."
            )
        elif args.prepare_only:
            print(
                "FINAL DATA-PATH PREPARATION ONLY: no models were trained and no "
                "tokenizer winner may be selected from this run."
            )
        return 0
    except baseline.PipelineError as error:
        (output_dir / "FAILED.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 2
    except Exception as error:
        (output_dir / "FAILED.txt").write_text(
            f"{type(error).__name__}: {error}\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
