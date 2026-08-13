#!/usr/bin/env python3
"""Verify frozen NLP V2 artifacts, tokenizer vectors, and LiteRT inference."""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError as error:  # pragma: no cover - environment setup guard
    raise SystemExit(
        "Missing ai-edge-litert. Install it in the verification environment first."
    ) from error


ASSET_ROOT = (
    Path(__file__).resolve().parents[1] / "app/src/main/assets/nlp-v2"
)
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
MODEL_HASHES = {
    "m0_t1_seed_20260812.tflite": "3776b1004d1d5c29bc77ab4021ab7cd63857cda95bb92ddb52c6a3156781a048",
    "m0_t2_seed_20260814.tflite": "288afe0a1461f2538a827282cea173b7df73d7bfeb531b7aa42e843a74f53417",
    "m1_t1_seed_20260814.tflite": "734041f301f53216a8de06466be3d35e05e166c7b8a3c95e5bc3d192a8e27e9b",
    "m1_t2_seed_20260812.tflite": "7f4328c0e57d112ea30320e147b5c1bd113e47dea2a84f4b4f7bee0fd79e3ed9",
    "m2_t1_seed_20260813.tflite": "414afe94c9de309a8969c8b66d98998ef8e3f78bcf491fba538be93db35886e8",
    "m2_t2_seed_20260814.tflite": "a7a5573b6521bcc77983d837931d17235555b64d2d282ba91f67bb87dd5ca21d",
    "m3_t1_seed_20260813.tflite": "0b312a3679c9932da93c2b54e1bab1552de96ea16ef21294cd31959b7f7c3905",
    "m3_t2_seed_20260810.tflite": "c4f1de24309535a8f7d7f96712695e40bb88399597c6d7a2236801205f3ac5b7",
}
KNOWN_UTTERANCES = [
    ("Lies mir bitte den Text vor", "TEXT_RECOGNITION"),
    ("Welche Gegenstände sind vor mir", "OBJECT_DETECTION"),
    ("Sprich bitte langsamer", "CHANGE_SPEECH_SPEED"),
    ("Wechsle zu einer anderen Stimme", "CHANGE_SPEAKER"),
    ("Erkläre mir die Relativitätstheorie", "REDIRECT_TO_LLM"),
    ("  ÖFFNE, die Einstellungen!  ", "OPEN_SETTINGS"),
    ("Stelle die Frequenz auf 800 Hertz", "SET_FREQUENCY"),
    ("Setze die BPS auf fünf", "SET_BPS"),
    ("Wie weit ist die Tür entfernt", "MEASURE_DISTANCE"),
    ("Brich den Vorgang ab", "ABORT"),
]
DIRECT_SETTINGS_UTTERANCES = [
    ("Öffne die Einstellungen.", "OPEN_SETTINGS"),
    ("Setze die Frequenz auf 700 Hertz.", "SET_FREQUENCY"),
    ("Ändere die Frequenz.", "SET_FREQUENCY"),
    ("Stell die Stimme auf männlich.", "CHANGE_SPEAKER"),
    ("Ändere die Stimme.", "CHANGE_SPEAKER"),
    ("Sprich schneller.", "CHANGE_SPEECH_SPEED"),
    ("Setze die Signalrate auf 4 BPS.", "SET_BPS"),
    ("Frequenz", "SET_FREQUENCY"),
    ("Abbrechen.", "ABORT"),
    ("Mach die Frequenz höher.", "SET_FREQUENCY"),
    ("Nimm die andere Stimme.", "CHANGE_SPEAKER"),
    ("Mach die Abstandssignale langsamer.", "SET_BPS"),
]
FROZEN_ENCODINGS = {
    "  ÖFFNE, die Einstellungen!  ": {
        "T1": [142, 2, 37],
        "T2": [524, 53, 234],
    },
    "Stelle die Frequenz auf 800 Hertz": {
        "T1": [419, 2, 34, 12, 973, 85],
        "T2": [1094, 53, 208, 96, 1014, 385, 391],
    },
    "Erkläre mir die Relativitätstheorie": {
        "T1": [1052, 11, 2, 1],
        "T2": [365, 87, 103, 53, 227, 63, 1870, 709, 1262, 47, 20, 17, 101, 48],
    },
    "Wie weit ist die Tür entfernt": {
        "T1": [10, 25, 4, 2, 46, 97],
        "T2": [86, 153, 85, 53, 259, 414],
    },
    "Brich den Vorgang ab": {
        "T1": [462, 9, 424, 217],
        "T2": [1191, 98, 1100, 127],
    },
}


def normalize(text: str) -> str:
    """Training normalization: NFKC, lower, punctuation/whitespace to space."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        " " if character.isspace() or unicodedata.category(character).startswith("P")
        else character
        for character in text
    )
    return " ".join(text.split())


class FrozenTokenizer:
    def __init__(self, family: str) -> None:
        self.family = family
        directory = ASSET_ROOT / "tokenizers" / family
        self.config = json.loads((directory / "tokenizer_config.json").read_text())
        self.vocabulary = json.loads((directory / "vocab.json").read_text())
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.vocabulary)
        }
        merges = (
            json.loads((directory / "merges.json").read_text())
            if family == "T2"
            else []
        )
        self.merge_by_pair = {
            (merge["left"], merge["right"]): (merge["rank"], merge["merged"])
            for merge in merges
        }
        self._validate(merges)

    def _validate(self, merges: list[dict[str, object]]) -> None:
        assert self.config["version"] == 1
        assert self.config["normalization"] == (
            "shared_intent_nfkc_lower_punctuation_whitespace_v1"
        )
        assert self.config["padding"] == self.config["truncating"] == "post"
        assert self.config["max_length"] == 24
        assert self.vocabulary[:2] == ["[PAD]", "[UNK]"]
        canonical_vocabulary = json.dumps(
            self.vocabulary, ensure_ascii=False, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(canonical_vocabulary).hexdigest() == self.config[
            "vocabulary_checksum_sha256"
        ]
        if self.family == "T1":
            assert self.config["tokenizer_type"] == "deterministic_word_level"
            assert self.config["split"] == "normalized whitespace"
            assert len(self.vocabulary) == self.config["vocabulary_size"]
            assert not merges
        else:
            assert self.config["tokenizer_type"] == "deterministic_word_boundary_bpe"
            assert self.config["word_boundary_symbol"] == "▁"
            assert len(self.vocabulary) == self.config["actual_vocabulary_size"]
            assert len(merges) == self.config["merge_count"]
            assert [merge["rank"] for merge in merges] == list(range(len(merges)))

    def encode(self, text: str) -> list[int]:
        tokens: list[str] = []
        words = normalize(text).split()
        if self.family == "T1":
            tokens = words
        else:
            for word in words:
                tokens.extend(self._encode_bpe_word(word))
        encoded = [self.token_to_id.get(token, 1) for token in tokens[:24]]
        return encoded + [0] * (24 - len(encoded))

    def _encode_bpe_word(self, word: str) -> list[str]:
        symbols = ["▁", *word]
        while len(symbols) > 1:
            candidates = [
                (self.merge_by_pair[pair][0], pair, self.merge_by_pair[pair][1])
                for pair in zip(symbols, symbols[1:])
                if pair in self.merge_by_pair
            ]
            if not candidates:
                break
            _, selected_pair, merged = min(candidates, key=lambda item: item[0])
            merged_symbols: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and (symbols[index], symbols[index + 1]) == selected_pair
                ):
                    merged_symbols.append(merged)
                    index += 2
                else:
                    merged_symbols.append(symbols[index])
                    index += 1
            symbols = merged_symbols
        return symbols


def verify_frozen_encodings(tokenizers: dict[str, FrozenTokenizer]) -> None:
    for text, family_vectors in FROZEN_ENCODINGS.items():
        for family, unpadded in family_vectors.items():
            expected = unpadded + [0] * (24 - len(unpadded))
            assert tokenizers[family].encode(text) == expected


def create_interpreter(model_path: Path) -> tuple[Interpreter, dict, dict]:
    interpreter = Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    assert len(inputs) == len(outputs) == 1
    assert tuple(inputs[0]["shape"]) == (1, 24)
    assert inputs[0]["dtype"] == np.int32
    assert tuple(outputs[0]["shape"]) == (1, 10)
    assert outputs[0]["dtype"] == np.float32
    return interpreter, inputs[0], outputs[0]


def infer(
    interpreter: Interpreter,
    input_tensor: dict,
    output_tensor: dict,
    encoded: list[int],
) -> np.ndarray:
    model_input = np.asarray([encoded], dtype=np.int32)
    assert model_input.shape == (1, 24)
    interpreter.set_tensor(input_tensor["index"], model_input)
    interpreter.invoke()
    probabilities = interpreter.get_tensor(output_tensor["index"])[0]
    assert probabilities.shape == (10,)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0) and np.all(probabilities <= 1.0)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-5)
    return probabilities


def verify_models(tokenizers: dict[str, FrozenTokenizer]) -> None:
    for filename, expected_hash in MODEL_HASHES.items():
        model_path = ASSET_ROOT / "models" / filename
        assert hashlib.sha256(model_path.read_bytes()).hexdigest() == expected_hash
        interpreter, input_tensor, output_tensor = create_interpreter(model_path)
        family = "T1" if "_t1_" in filename else "T2"
        infer(interpreter, input_tensor, output_tensor, tokenizers[family].encode(""))


def verify_default_pipeline(tokenizer: FrozenTokenizer) -> None:
    model_path = ASSET_ROOT / "models/m0_t1_seed_20260812.tflite"
    interpreter, input_tensor, output_tensor = create_interpreter(model_path)
    print("Default M0_T1 pipeline:")
    utterances = KNOWN_UTTERANCES + DIRECT_SETTINGS_UTTERANCES
    for original_text, expected_intent in utterances:
        probabilities = infer(
            interpreter,
            input_tensor,
            output_tensor,
            tokenizer.encode(original_text),
        )
        top_index = int(np.argmax(probabilities))
        actual_intent = LABELS[top_index]
        assert actual_intent == expected_intent, (original_text, actual_intent)
        print(f"  {actual_intent:22} {probabilities[top_index]:.6f}  {original_text!r}")


def main() -> int:
    labels_t1 = json.loads((ASSET_ROOT / "tokenizers/T1/labels.json").read_text())
    labels_t2 = json.loads((ASSET_ROOT / "tokenizers/T2/labels.json").read_text())
    assert labels_t1 == labels_t2 == LABELS

    tokenizers = {family: FrozenTokenizer(family) for family in ("T1", "T2")}
    verify_frozen_encodings(tokenizers)
    verify_models(tokenizers)
    verify_default_pipeline(tokenizers["T1"])
    formulation_count = len(KNOWN_UTTERANCES) + len(DIRECT_SETTINGS_UTTERANCES)
    print(
        f"Validated 8 models, 2 frozen tokenizers, and "
        f"{formulation_count} end-to-end formulations."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
