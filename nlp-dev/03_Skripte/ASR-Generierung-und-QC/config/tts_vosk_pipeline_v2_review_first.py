#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS -> Audio-Augmentation -> Vosk Pipeline (Generation 2 / Review-first)
für gelabelte EyeAI-Textdatensätze.

WICHTIGES PRINZIP
-----------------
Dieses Skript erzeugt Vosk-Transkripte und dokumentiert sie. Es entscheidet NICHT,
welche Transkripte später tatsächlich zum Modelltraining verwendet werden.

Verbesserungen gegenüber Generation 1:
- kein 35%-ASR-Cap
- kein accept/review/reject als Trainingsentscheidung
- kein finales Clean+Vosk-Trainingsset
- alle nichtleeren Vosk-Transkripte bleiben erhalten
- problematische Fälle werden nur mit Review-Flags markiert
- Piper dominiert; eSpeak standardmäßig nur 3% und niemals für
  SET_FREQUENCY/SET_BPS
- keine extremen/harten Augmentierungen mehr
- Severity standardmäßig 40% clean / 60% moderate
- mit --dataset wird jede Zeile einer einzelnen Datei genau einmal verarbeitet
- TTS-Normalisierung für Zahlen sowie Hz/BPS, damit TTS natürlicher spricht
- ursprünglicher Clean-Text bleibt unverändert; tts_text wird separat protokolliert

Quellenauswahl (wie bisher):
- organische/Widerspruchsdaten: 100%
- saubere Hauptdaten: 60% (zulässig 50-70%)
- alte gefilterte Daten: 25% (zulässig 20-30%)

Ausgaben:
- clean_originals.txt
- vosk_generated_all.txt
- vosk_generated_all_metadata.csv
- vosk_flagged_for_review.csv
- manifest_all_attempts.csv
- summary.json

Python-Pakete:
    pip install numpy scipy vosk piper-tts
Optional:
    sudo apt install espeak-ng
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
import unicodedata
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
except ImportError:
    Model = KaldiRecognizer = SetLogLevel = None

try:
    from piper import PiperVoice
    try:
        from piper import SynthesisConfig
    except ImportError:
        SynthesisConfig = None
except ImportError:
    PiperVoice = None
    SynthesisConfig = None


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

SOURCE_ORGANIC = "organic"
SOURCE_CLEAN = "clean"
SOURCE_OLD = "old"
SOURCE_DATASET = "dataset"

NEGATION_TOKENS = {
    "nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines",
    "nichts", "nie", "niemals",
}
CONTRAST_TOKENS = {
    "nur", "sondern", "statt", "aber", "doch", "außer", "ausser",
}

# Nur Flags für späteren menschlichen Review. Keine automatische Trainingsentscheidung.
ANCHORS = {
    "TEXT_RECOGNITION": {
        "lesen", "lies", "vorlesen", "text", "schrift", "geschrieben", "steht",
        "aufschrift", "buchstaben", "zettel", "schild", "etikett",
    },
    "OBJECT_DETECTION": {
        "objekt", "ding", "gegenstand", "erkenne", "erkennen", "finde", "finden",
        "wo", "hindernis", "tür", "tuer", "stuhl", "sehen", "siehst",
    },
    "CHANGE_SPEECH_SPEED": {
        "langsamer", "schneller", "tempo", "sprechtempo", "geschwindigkeit",
        "flotter", "zügiger", "zuegiger", "gemächlicher", "gemaechlicher",
    },
    "CHANGE_SPEAKER": {
        "stimme", "sprecher", "sprecherin", "frauenstimme", "männerstimme",
        "maennerstimme", "weiblich", "männlich", "maennlich",
    },
    "REDIRECT_TO_LLM": {
        "erklär", "erklaer", "erklären", "erklaeren", "warum", "wie",
        "was", "wer", "wieso", "bedeutet", "erzähl", "erzaehl",
    },
    "OPEN_SETTINGS": {
        "einstellungen", "settings", "menü", "menue", "optionen",
        "konfiguration", "konfig", "öffne", "oeffne",
    },
    "SET_FREQUENCY": {
        "frequenz", "hertz", "hz", "tonhöhe", "tonhoehe", "höher", "hoeher",
        "tiefer", "signalton", "piepton",
    },
    "SET_BPS": {
        "bps", "häufiger", "haeufiger", "seltener", "wiederholrate",
        "pulsrate", "taktung", "piepen", "pulse", "pro", "sekunde",
    },
    "MEASURE_DISTANCE": {
        "abstand", "entfernung", "distanz", "weit", "nah", "meter",
        "messen", "miss", "weg",
    },
    "ABORT": {
        "stopp", "stop", "abbrechen", "abbruch", "lass", "vergiss",
        "aufhören", "aufhoeren", "zurück", "zurueck", "nicht", "doch",
    },
}


@dataclass(frozen=True)
class Sample:
    base_id: str
    source: str
    line_no: int
    text: str
    label: str


@dataclass
class Attempt:
    attempt_id: str
    base_id: str
    source: str
    label: str
    base_text: str
    tts_text: str
    severity: str
    tts_backend: str = ""
    voice: str = ""
    tts_quality: str = ""
    transcript: str = ""
    similarity: float = 0.0
    token_recall: float = 0.0
    status: str = ""
    flags: str = ""
    audio_path: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Review-first TTS -> Audio-Augmentation -> Vosk Generation 2."
    )
    p.add_argument(
        "--dataset", type=Path, default=None,
        help=("Einzelne Datei im Format Text;LABEL. Wenn gesetzt, wird jede "
              "Zeile genau einmal verarbeitet und --organic/--clean/--old "
              "sowie deren Auswahlquoten werden ignoriert."),
    )
    p.add_argument("--organic", type=Path,
                   default=Path("01_Daten_mit_Widerspruechen_bereinigt_dedupliziert.txt"))
    p.add_argument("--clean", type=Path,
                   default=Path("02_Saubere_Trainingsdaten_bereinigt_dedupliziert.txt"))
    p.add_argument("--old", type=Path,
                   default=Path("03_Streng_gefilterte_alte_Daten_bereinigt_dedupliziert.txt"))
    p.add_argument("--vosk-model", type=Path, help="Pfad zum deutschen Vosk-Modell.")
    p.add_argument("--piper-voices-dir", type=Path, default=Path("piper_voices"),
                   help="Ordner mit *.onnx und passenden *.onnx.json Piper-Stimmen.")
    p.add_argument("--noise-dir", type=Path, default=None,
                   help="Optional: echte Hintergrundgeräusch-WAVs.")
    p.add_argument("--rir-dir", type=Path, default=None,
                   help="Optional: echte Raumimpulsantwort-WAVs.")
    p.add_argument("--output-dir", type=Path, default=Path("tts_vosk_output_v2"))
    p.add_argument("--seed", type=int, default=20260809)

    p.add_argument("--organic-ratio", type=float, default=1.00)
    p.add_argument("--clean-ratio", type=float, default=0.60)
    p.add_argument("--old-ratio", type=float, default=0.25)

    p.add_argument("--severity-clean", type=float, default=0.40)
    p.add_argument("--severity-moderate", type=float, default=0.60)

    p.add_argument("--espeak-weight", type=float, default=0.03,
                   help="eSpeak-Anteil bei geeigneten Samples. Default 3%%.")
    p.add_argument("--disable-espeak", action="store_true")
    p.add_argument("--no-tts-normalization", action="store_true",
                   help="Zahlen/Hz/BPS vor TTS nicht normalisieren.")
    p.add_argument("--keep-audio", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Nur zum Testen: maximale Zahl ausgewählter Basissamples.")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.organic_ratio <= 1.0):
        raise ValueError("--organic-ratio muss zwischen 0 und 1 liegen.")
    if not (0.50 <= args.clean_ratio <= 0.70):
        raise ValueError("--clean-ratio soll zwischen 0.50 und 0.70 liegen.")
    if not (0.20 <= args.old_ratio <= 0.30):
        raise ValueError("--old-ratio soll zwischen 0.20 und 0.30 liegen.")
    if not (0.0 <= args.espeak_weight <= 1.0):
        raise ValueError("--espeak-weight muss zwischen 0 und 1 liegen.")

    severity_sum = args.severity_clean + args.severity_moderate
    if min(args.severity_clean, args.severity_moderate) < 0:
        raise ValueError("Severity-Anteile dürfen nicht negativ sein.")
    if abs(severity_sum - 1.0) > 1e-6:
        raise ValueError("--severity-clean + --severity-moderate muss 1.0 ergeben.")

    input_paths = (args.dataset,) if args.dataset is not None else (
        args.organic, args.clean, args.old
    )
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Datensatz nicht gefunden: {path}")
    if not args.dry_run:
        if Model is None:
            raise RuntimeError("vosk ist nicht installiert: pip install vosk")
        if args.vosk_model is None or not args.vosk_model.exists():
            raise FileNotFoundError(
                "--vosk-model muss auf ein vorhandenes deutsches Vosk-Modell zeigen."
            )


def read_dataset(path: Path, source: str,
                 allowed_labels: Optional[set[str]] = None) -> list[Sample]:
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            if ";" not in raw:
                raise ValueError(f"{path}:{line_no}: kein ';' gefunden.")
            text, label = raw.rsplit(";", 1)
            text, label = text.strip(), label.strip()
            if not text or not label:
                raise ValueError(f"{path}:{line_no}: Text oder Label ist leer.")
            if allowed_labels is not None and label not in allowed_labels:
                raise ValueError(f"{path}:{line_no}: unbekanntes Label {label!r}")
            out.append(Sample(f"{source}:{line_no}", source, line_no, text, label))
    return out


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9äöüß]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize_text(text).split()


# ---------- Deutsche TTS-Normalisierung ----------

_ONES = {
    0: "null", 1: "eins", 2: "zwei", 3: "drei", 4: "vier", 5: "fünf",
    6: "sechs", 7: "sieben", 8: "acht", 9: "neun", 10: "zehn",
    11: "elf", 12: "zwölf", 13: "dreizehn", 14: "vierzehn",
    15: "fünfzehn", 16: "sechzehn", 17: "siebzehn", 18: "achtzehn",
    19: "neunzehn",
}
_TENS = {
    20: "zwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig",
    60: "sechzig", 70: "siebzig", 80: "achtzig", 90: "neunzig",
}


def german_int(n: int) -> str:
    """Ausreichend für typische Settingswerte; unterstützt 0..999999."""
    if n < 0:
        return "minus " + german_int(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return _TENS[tens]
        one_word = "ein" if ones == 1 else _ONES[ones]
        return one_word + "und" + _TENS[tens]
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        prefix = ("ein" if hundreds == 1 else _ONES[hundreds]) + "hundert"
        return prefix + (german_int(rest) if rest else "")
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        if thousands == 1:
            prefix = "eintausend"
        else:
            prefix = german_int(thousands) + "tausend"
        return prefix + (german_int(rest) if rest else "")
    return str(n)


def _speak_number_match(match: re.Match) -> str:
    raw = match.group(0)
    # Reine Ganzzahlen; große IDs/Telefonnummern werden absichtlich nicht > 6 Stellen expandiert.
    try:
        n = int(raw)
    except ValueError:
        return raw
    if 0 <= n < 1_000_000:
        return german_int(n)
    return raw


def normalize_for_tts(text: str) -> str:
    """
    Nur für die Synthese. base_text bleibt unverändert.
    Ziel: TTS soll Einheiten so aussprechen, wie ein deutscher Nutzer sie typischerweise sagt.
    """
    t = unicodedata.normalize("NFKC", text)

    # Einheiten zuerst ersetzen, damit z.B. 750 Hz -> siebenhundertfünfzig Hertz.
    t = re.sub(r"(?i)\bHz\b", "Hertz", t)
    t = re.sub(r"(?i)\bBPS\b", "Be Pe Es", t)

    # Dezimalzahlen zuerst, z.B. 6,5 -> "sechs komma fünf".
    def speak_decimal(m: re.Match) -> str:
        left = german_int(int(m.group(1)))
        right_digits = " ".join(german_int(int(ch)) for ch in m.group(2))
        return f"{left} komma {right_digits}"

    t = re.sub(r"(?<![\w])([0-9]{1,6})[,.]([0-9]{1,3})(?![\w])", speak_decimal, t)

    # Danach einfache Ganzzahlen verbalieren. Das verbessert insbesondere Hz/BPS-Kommandos.
    t = re.sub(r"(?<![\w])\d{1,6}(?![\w])", _speak_number_match, t)

    # TTS-freundliche Leerzeichen.
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------- Auswahl / Attempt-Plan ----------

def stratified_sample(samples: list[Sample], ratio: float,
                      rng: random.Random) -> list[Sample]:
    if ratio >= 0.999999:
        return list(samples)
    by_label: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)
    selected: list[Sample] = []
    for label in LABELS:
        group = list(by_label[label])
        rng.shuffle(group)
        n = max(1, round(len(group) * ratio)) if group else 0
        selected.extend(group[:n])
    rng.shuffle(selected)
    return selected


def build_attempt_plan(organic: list[Sample], clean: list[Sample],
                       old: list[Sample], rng: random.Random) -> list[tuple[Sample, int]]:
    # Für Vergleichbarkeit mit Generation 1 bleibt die Anzahl der Versuche gleich aufgebaut.
    plan: list[tuple[Sample, int]] = []
    for s in organic:
        plan.extend((s, i) for i in range(1, 4))
    for s in clean:
        n = 1 if rng.random() < 0.5 else 2
        plan.extend((s, i) for i in range(1, n + 1))
    for s in old:
        plan.append((s, 1))
    rng.shuffle(plan)
    return plan


def severity_schedule(n: int, clean_ratio: float, moderate_ratio: float,
                      rng: random.Random) -> list[str]:
    n_clean = round(n * clean_ratio)
    n_moderate = n - n_clean
    result = ["clean"] * n_clean + ["moderate"] * n_moderate
    rng.shuffle(result)
    return result


# ---------- Audio ----------
def dtype_to_float(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
    elif x.dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        peak = max(float(np.max(np.abs(x))), 1.0)
        y = x.astype(np.float32) / peak
    elif x.dtype == np.uint8:
        y = (x.astype(np.float32) - 128.0) / 128.0
    else:
        y = x.astype(np.float32)
        peak = max(float(np.max(np.abs(y))), 1.0)
        y /= peak
    return np.clip(y, -1.0, 1.0)


def read_wav_mono(path: Path, target_sr: int = 16000) -> np.ndarray:
    sr, x = wavfile.read(path)
    x = dtype_to_float(np.asarray(x))
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        frac = Fraction(target_sr, int(sr)).limit_denominator(1000)
        x = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def write_wav_mono(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak
    wavfile.write(path, sr, np.clip(x * 32767, -32768, 32767).astype(np.int16))


def normalize_peak(x: np.ndarray, peak: float = 0.92) -> np.ndarray:
    if len(x) == 0:
        return x
    p = float(np.max(np.abs(x)))
    if p < 1e-8:
        return x
    return (x / p * peak).astype(np.float32)


def change_speed(x: np.ndarray, factor: float) -> np.ndarray:
    if len(x) < 2 or abs(factor - 1.0) < 0.01:
        return x
    n = max(2, round(len(x) / factor))
    frac = Fraction(n, len(x)).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)


def discover_wavs(directory: Optional[Path]) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and p.suffix.lower() == ".wav")


def choose_segment(noise: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(noise) < n:
        noise = np.tile(noise, math.ceil(n / len(noise)))
    if len(noise) == n:
        return noise.astype(np.float32)
    start = rng.randint(0, len(noise) - n)
    return noise[start:start + n].astype(np.float32)


def synthetic_noise(n: int, sr: int, rng: random.Random) -> np.ndarray:
    white = np.array([rng.gauss(0.0, 1.0) for _ in range(n)], dtype=np.float32)
    low = lfilter([1.0], [1.0, -0.96], white).astype(np.float32)
    low /= max(float(np.std(low)), 1e-6)
    t = np.arange(n, dtype=np.float32) / sr
    hum = (np.sin(2 * np.pi * 50 * t)
           + 0.35 * np.sin(2 * np.pi * 100 * t)
           + 0.15 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    mix = 0.55 * white + 0.35 * low + 0.10 * hum
    return mix / max(float(np.std(mix)), 1e-6)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = math.sqrt(float(np.mean(signal ** 2)) + 1e-12)
    noise_rms = math.sqrt(float(np.mean(noise ** 2)) + 1e-12)
    if noise_rms < 1e-8:
        return signal
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    return (signal + noise * (target_noise_rms / noise_rms)).astype(np.float32)


def synthetic_rir(sr: int, rt60: float, rng: random.Random) -> np.ndarray:
    length = max(64, int(sr * rt60))
    t = np.arange(length, dtype=np.float32) / sr
    decay = np.exp(-6.91 * t / max(rt60, 0.05)).astype(np.float32)
    rir = np.zeros(length, dtype=np.float32)
    rir[0] = 1.0
    for _ in range(rng.randint(4, 10)):
        delay_ms = rng.uniform(8.0, min(75.0, rt60 * 450.0))
        idx = min(length - 1, int(delay_ms * sr / 1000.0))
        amp = rng.uniform(-0.35, 0.45) * math.exp(-3.8 * idx / length)
        rir[idx] += amp
    tail = np.array([rng.gauss(0.0, 1.0) for _ in range(length)], dtype=np.float32)
    rir += 0.035 * tail * decay
    return rir / math.sqrt(float(np.sum(rir ** 2)) + 1e-12)


def apply_reverb(x: np.ndarray, sr: int, rng: random.Random,
                 rt60_range: tuple[float, float], rir_paths: list[Path]) -> np.ndarray:
    if rir_paths and rng.random() < 0.70:
        rir = read_wav_mono(rng.choice(rir_paths), sr)
        if len(rir) > sr * 2:
            rir = rir[:sr * 2]
        rir = normalize_peak(rir, 1.0)
    else:
        rir = synthetic_rir(sr, rng.uniform(*rt60_range), rng)
    wet = fftconvolve(x, rir, mode="full")[:len(x)]
    return normalize_peak(np.asarray(wet, dtype=np.float32), 0.96)


def apply_bandlimit(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = sr / 2
    low = max(20.0, low_hz) / nyq
    high = min(nyq - 50.0, high_hz) / nyq
    if not (0 < low < high < 1):
        return x
    sos = butter(4, [low, high], btype="bandpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def augment_audio(x: np.ndarray, severity: str, sr: int, rng: random.Random,
                  noise_paths: list[Path], rir_paths: list[Path]) -> np.ndarray:
    x = normalize_peak(x, 0.90)

    if severity == "clean":
        return normalize_peak(x * rng.uniform(0.88, 1.04), 0.92)

    if severity == "moderate":
        # Generation 2: realistischer und etwas weniger aggressiv.
        if rng.random() < 0.55:
            x = apply_reverb(x, sr, rng, (0.12, 0.35), rir_paths)
        if noise_paths and rng.random() < 0.75:
            noise = choose_segment(read_wav_mono(rng.choice(noise_paths), sr), len(x), rng)
        else:
            noise = synthetic_noise(len(x), sr, rng)
        x = mix_at_snr(x, noise, rng.uniform(14.0, 28.0))
        if rng.random() < 0.20:
            x = apply_bandlimit(x, sr, 90.0, rng.uniform(5800.0, 7600.0))
        return normalize_peak(x * rng.uniform(0.80, 1.04), 0.94)

    raise ValueError(f"Unbekannter Severity-Wert: {severity}")


# ---------- TTS ----------
class PiperPool:
    def __init__(self, voice_dir: Path):
        self.voice_paths: list[Path] = []
        self.cache: dict[Path, object] = {}
        if voice_dir.exists():
            for onnx in sorted(voice_dir.rglob("*.onnx")):
                if Path(str(onnx) + ".json").exists():
                    self.voice_paths.append(onnx)

    def available(self) -> bool:
        return bool(self.voice_paths) and PiperVoice is not None

    def get_voice(self, path: Path):
        if path not in self.cache:
            self.cache[path] = PiperVoice.load(str(path))
        return self.cache[path]

    @staticmethod
    def infer_quality(path: Path) -> str:
        name = path.stem.lower()
        for q in ("x_low", "low", "medium", "high"):
            if q in name:
                return q
        return "unknown"

    def synthesize(self, text: str, output_path: Path,
                   rng: random.Random) -> tuple[str, str]:
        voice_path = rng.choice(self.voice_paths)
        voice = self.get_voice(voice_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(output_path), "wb") as wf:
            if SynthesisConfig is not None:
                syn = SynthesisConfig(
                    volume=rng.uniform(0.88, 1.04),
                    length_scale=rng.uniform(0.90, 1.12),
                    noise_scale=rng.uniform(0.58, 0.82),
                    noise_w_scale=rng.uniform(0.65, 0.92),
                    normalize_audio=True,
                )
                voice.synthesize_wav(text, wf, syn_config=syn)
            else:
                voice.synthesize_wav(text, wf)

        if SynthesisConfig is None:
            x = read_wav_mono(output_path, 16000)
            x = change_speed(x, rng.uniform(0.94, 1.08))
            write_wav_mono(output_path, x, 16000)

        return voice_path.stem, self.infer_quality(voice_path)


class EspeakEngine:
    def __init__(self):
        self.exe = shutil.which("espeak-ng")

    def available(self) -> bool:
        return self.exe is not None

    def synthesize(self, text: str, output_path: Path,
                   rng: random.Random) -> tuple[str, str]:
        if not self.exe:
            raise RuntimeError("espeak-ng ist nicht installiert.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Engerer Bereich als Generation 1; keine extremen Stimmen mehr.
        speed = rng.randint(145, 185)
        pitch = rng.randint(42, 58)
        amplitude = rng.randint(90, 115)
        subprocess.run([
            self.exe, "-v", "de", "-s", str(speed), "-p", str(pitch),
            "-a", str(amplitude), "-w", str(output_path), text,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"espeak-ng-de-s{speed}-p{pitch}", "low"


def choose_tts_backend(piper: PiperPool, espeak: EspeakEngine,
                       espeak_weight: float, disable_espeak: bool,
                       severity: str, label: str, rng: random.Random) -> str:
    have_piper = piper.available()
    have_espeak = espeak.available() and not disable_espeak
    if not have_piper and not have_espeak:
        raise RuntimeError("Keine TTS-Engine verfügbar.")

    # Generation 1 zeigte: eSpeak bei Hz/BPS ist zu zerstörerisch.
    if have_piper and label in {"SET_FREQUENCY", "SET_BPS"}:
        return "piper"

    if have_piper and have_espeak:
        return "espeak" if rng.random() < espeak_weight else "piper"
    return "piper" if have_piper else "espeak"


# ---------- Vosk / Review-Flags ----------
def transcribe_vosk(model, x: np.ndarray, sr: int = 16000) -> str:
    pcm = np.clip(x * 32767, -32768, 32767).astype(np.int16).tobytes()
    rec = KaldiRecognizer(model, sr)
    chunk_bytes = 8000
    for i in range(0, len(pcm), chunk_bytes):
        rec.AcceptWaveform(pcm[i:i + chunk_bytes])
    result = json.loads(rec.FinalResult())
    return str(result.get("text", "")).strip()


def multiset_recall(src: list[str], hyp: list[str]) -> float:
    if not src:
        return 1.0
    a, b = Counter(src), Counter(hyp)
    common = sum(min(a[k], b[k]) for k in a)
    return common / max(sum(a.values()), 1)


def review_flags(base_text: str, transcript: str, label: str) -> tuple[list[str], float, float]:
    src_norm = normalize_text(base_text)
    hyp_norm = normalize_text(transcript)
    if not hyp_norm:
        return ["empty_transcript"], 0.0, 0.0

    sim = SequenceMatcher(None, src_norm, hyp_norm).ratio()
    src_t, hyp_t = tokens(base_text), tokens(transcript)
    recall = multiset_recall(src_t, hyp_t)
    src_set, hyp_set = set(src_t), set(hyp_t)

    flags: list[str] = []
    if src_norm == hyp_norm:
        flags.append("identical_to_clean")
    elif sim >= 0.975:
        flags.append("near_duplicate_clean")

    if sim < 0.42 or recall < 0.30:
        flags.append("heavy_information_loss")
    if len(hyp_t) <= 1 and len(src_t) >= 3:
        flags.append("too_short")
    if src_set & NEGATION_TOKENS and not (hyp_set & NEGATION_TOKENS):
        flags.append("negation_lost")
    if src_set & CONTRAST_TOKENS and not (hyp_set & CONTRAST_TOKENS):
        flags.append("contrast_marker_lost")

    source_anchors = src_set & ANCHORS.get(label, set())
    hyp_anchors = hyp_set & ANCHORS.get(label, set())
    if source_anchors and not hyp_anchors:
        flags.append("all_label_anchors_lost")

    # Zusätzliche Diagnose speziell für die in Generation 1 problematischen Settings.
    base_low, hyp_low = base_text.lower(), transcript.lower()
    if label == "SET_FREQUENCY" and ("hz" in base_low or "hertz" in base_low):
        if not ("hz" in hyp_low or "hertz" in hyp_low or "herz" in hyp_low):
            flags.append("frequency_unit_lost")
    if label == "SET_BPS" and "bps" in base_low:
        if not any(x in hyp_low for x in ("bps", "b p s", "be pe es", "pro sekunde", "sekunde")):
            flags.append("bps_marker_lost")

    return flags, sim, recall


def add_global_transcript_flags(attempts: list[Attempt]) -> None:
    by_text: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        if a.status == "generated" and normalize_text(a.transcript):
            by_text[normalize_text(a.transcript)].append(a)

    for group in by_text.values():
        if len(group) > 1:
            for a in group:
                fs = set(filter(None, a.flags.split("|")))
                fs.add("duplicate_generated_transcript")
                a.flags = "|".join(sorted(fs))
        if len({a.label for a in group}) > 1:
            for a in group:
                fs = set(filter(None, a.flags.split("|")))
                fs.add("cross_label_transcript_conflict")
                a.flags = "|".join(sorted(fs))


def write_attempt_csv(path: Path, attempts: list[Attempt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(attempts[0]).keys()) if attempts else [
        "attempt_id", "base_id", "source", "label", "base_text", "tts_text",
        "severity", "tts_backend", "voice", "tts_quality", "transcript",
        "similarity", "token_recall", "status", "flags", "audio_path", "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in attempts:
            writer.writerow(asdict(a))


def write_text_label(path: Path, pairs: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text, label in pairs:
            f.write(f"{text};{label}\n")


def print_distribution(title: str, samples: Iterable[Sample]) -> None:
    samples = list(samples)
    c = Counter(s.label for s in samples)
    print(f"\n{title}: {len(samples)}")
    ordered_labels = [label for label in LABELS if label in c]
    ordered_labels.extend(sorted(set(c) - set(ordered_labels)))
    for label in ordered_labels:
        print(f"  {label:22s} {c[label]:4d}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.dataset is not None:
        all_clean = read_dataset(args.dataset, SOURCE_DATASET, allowed_labels=None)
        selected = list(all_clean)
        if args.max_samples is not None:
            rng.shuffle(selected)
            selected = selected[:args.max_samples]
        print_distribution("Einzeldatensatz gesamt", all_clean)
        print_distribution("Für TTS/Vosk gewählt (jeweils genau einmal)", selected)
        plan = [(sample, 1) for sample in selected]
        source_selection = {
            SOURCE_DATASET: {
                "path": str(args.dataset),
                "available": len(all_clean),
                "selected_for_tts": len(selected),
                "ratio": len(selected) / len(all_clean) if all_clean else 0.0,
                "attempts_per_sample": 1,
            }
        }
    else:
        allowed_labels = set(LABELS)
        organic_all = read_dataset(args.organic, SOURCE_ORGANIC, allowed_labels)
        clean_all = read_dataset(args.clean, SOURCE_CLEAN, allowed_labels)
        old_all = read_dataset(args.old, SOURCE_OLD, allowed_labels)
        all_clean = organic_all + clean_all + old_all

        # Globale Deduplizierung der Clean-Splits verifizieren.
        seen: dict[tuple[str, str], str] = {}
        duplicates = []
        for s in all_clean:
            key = (normalize_text(s.text), s.label)
            if key in seen:
                duplicates.append((seen[key], s.base_id, s.text, s.label))
            else:
                seen[key] = s.base_id
        if duplicates:
            raise RuntimeError(
                f"Die drei Eingabedateien enthalten noch {len(duplicates)} globale Duplikate."
            )

        organic_sel = stratified_sample(organic_all, args.organic_ratio, rng)
        clean_sel = stratified_sample(clean_all, args.clean_ratio, rng)
        old_sel = stratified_sample(old_all, args.old_ratio, rng)

        if args.max_samples is not None:
            merged = ([(s, SOURCE_ORGANIC) for s in organic_sel]
                      + [(s, SOURCE_CLEAN) for s in clean_sel]
                      + [(s, SOURCE_OLD) for s in old_sel])
            rng.shuffle(merged)
            keep = merged[:args.max_samples]
            organic_sel = [s for s, src in keep if src == SOURCE_ORGANIC]
            clean_sel = [s for s, src in keep if src == SOURCE_CLEAN]
            old_sel = [s for s, src in keep if src == SOURCE_OLD]

        print_distribution("Clean gesamt", all_clean)
        print_distribution("Für TTS/Vosk gewählt: organic", organic_sel)
        print_distribution("Für TTS/Vosk gewählt: clean", clean_sel)
        print_distribution("Für TTS/Vosk gewählt: old", old_sel)

        plan = build_attempt_plan(organic_sel, clean_sel, old_sel, rng)
        source_selection = {
            SOURCE_ORGANIC: {
                "available": len(organic_all), "selected_for_tts": len(organic_sel),
                "ratio": args.organic_ratio,
            },
            SOURCE_CLEAN: {
                "available": len(clean_all), "selected_for_tts": len(clean_sel),
                "ratio": args.clean_ratio,
            },
            SOURCE_OLD: {
                "available": len(old_all), "selected_for_tts": len(old_sel),
                "ratio": args.old_ratio,
            },
        }

    severities = severity_schedule(
        len(plan), args.severity_clean, args.severity_moderate, rng
    )
    print(f"\nGeplante Audio-Versuche: {len(plan)}")
    print("Severity:", dict(Counter(severities)))
    print("Review-first: keine automatische Trainingsauswahl, kein ASR-Cap.")

    if args.dry_run:
        print("\nDry-run beendet. Keine Audios oder Transkripte erzeugt.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    temp_dir = args.output_dir / "_tts_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    noise_paths = discover_wavs(args.noise_dir)
    rir_paths = discover_wavs(args.rir_dir)
    print(f"\nEchte Noise-WAVs: {len(noise_paths)}")
    print(f"Echte RIR-WAVs:   {len(rir_paths)}")
    if not noise_paths:
        print("Hinweis: synthetischer Noise-Fallback wird verwendet.")
    if not rir_paths:
        print("Hinweis: synthetischer Hall-Fallback wird verwendet.")

    piper = PiperPool(args.piper_voices_dir)
    espeak = EspeakEngine()
    print(f"Piper-Stimmen gefunden: {len(piper.voice_paths)}")
    for v in piper.voice_paths:
        print(f"  - {v.stem} ({piper.infer_quality(v)})")
    print(f"eSpeak-NG verfügbar: {espeak.available() and not args.disable_espeak}")
    print(f"eSpeak-Zielanteil (nur geeignete Fälle): {args.espeak_weight:.1%}")

    if not piper.available() and (not espeak.available() or args.disable_espeak):
        raise RuntimeError("Keine nutzbare TTS-Engine verfügbar.")

    SetLogLevel(-1)
    print("\nLade Vosk-Modell ...")
    vosk_model = Model(str(args.vosk_model))

    attempts: list[Attempt] = []
    for idx, ((sample, variant_no), severity) in enumerate(zip(plan, severities), 1):
        attempt_id = f"{sample.base_id}:v{variant_no}"
        tmp_tts = temp_dir / f"{idx:06d}_tts.wav"
        out_audio = audio_dir / f"{idx:06d}_{severity}.wav"
        tts_text = sample.text if args.no_tts_normalization else normalize_for_tts(sample.text)
        a = Attempt(
            attempt_id=attempt_id,
            base_id=sample.base_id,
            source=sample.source,
            label=sample.label,
            base_text=sample.text,
            tts_text=tts_text,
            severity=severity,
        )

        try:
            backend = choose_tts_backend(
                piper, espeak, args.espeak_weight, args.disable_espeak,
                severity, sample.label, rng
            )
            a.tts_backend = backend
            if backend == "piper":
                a.voice, a.tts_quality = piper.synthesize(tts_text, tmp_tts, rng)
            else:
                a.voice, a.tts_quality = espeak.synthesize(tts_text, tmp_tts, rng)

            x = read_wav_mono(tmp_tts, 16000)
            # Kleine globale Variation; enger als Generation 1.
            x = change_speed(x, rng.uniform(0.96, 1.04))
            x = augment_audio(x, severity, 16000, rng, noise_paths, rir_paths)

            if args.keep_audio:
                write_wav_mono(out_audio, x, 16000)
                a.audio_path = str(out_audio)

            a.transcript = transcribe_vosk(vosk_model, x, 16000)
            flags, sim, recall = review_flags(sample.text, a.transcript, sample.label)
            a.similarity = round(sim, 6)
            a.token_recall = round(recall, 6)
            if not normalize_text(a.transcript):
                a.status = "empty"
            else:
                a.status = "generated"
            a.flags = "|".join(flags)

        except Exception as exc:
            a.status = "error"
            a.error = f"{type(exc).__name__}: {exc}"
        finally:
            if tmp_tts.exists():
                try:
                    tmp_tts.unlink()
                except OSError:
                    pass

        attempts.append(a)
        if idx % 50 == 0 or idx == len(plan):
            counts = Counter(x.status for x in attempts)
            flagged = sum(bool(x.flags) for x in attempts if x.status == "generated")
            print(f"[{idx:5d}/{len(plan):5d}] generated={counts['generated']} "
                  f"empty={counts['empty']} error={counts['error']} flagged={flagged}")

    # Nur annotieren, nie löschen.
    add_global_transcript_flags(attempts)

    generated = [a for a in attempts if a.status == "generated"]
    flagged = [a for a in generated if a.flags]

    write_attempt_csv(args.output_dir / "manifest_all_attempts.csv", attempts)
    write_attempt_csv(args.output_dir / "vosk_generated_all_metadata.csv", generated)
    write_attempt_csv(args.output_dir / "vosk_flagged_for_review.csv", flagged)

    write_text_label(args.output_dir / "clean_originals.txt",
                     ((s.text, s.label) for s in all_clean))
    write_text_label(args.output_dir / "vosk_generated_all.txt",
                     ((a.transcript, a.label) for a in generated))

    summary = {
        "pipeline_version": "generation_2_review_first",
        "seed": args.seed,
        "clean_samples_total": len(all_clean),
        "source_selection": source_selection,
        "tts_attempts": len(plan),
        "severity_attempts": dict(Counter(severities)),
        "attempt_status": dict(Counter(a.status for a in attempts)),
        "generated_nonempty": len(generated),
        "flagged_for_review": len(flagged),
        "generated_by_source": dict(Counter(a.source for a in generated)),
        "generated_by_severity": dict(Counter(a.severity for a in generated)),
        "generated_by_label": dict(Counter(a.label for a in generated)),
        "tts_backend": dict(Counter(a.tts_backend for a in attempts if a.tts_backend)),
        "piper_voices": [str(v) for v in piper.voice_paths],
        "espeak_weight": args.espeak_weight,
        "tts_normalization": not args.no_tts_normalization,
        "real_noise_wavs": len(noise_paths),
        "real_rir_wavs": len(rir_paths),
        "note": "Keine harten Augmentierungen. Alle nichtleeren Vosk-Transkripte bleiben zunächst erhalten; Flags dienen dem anschließenden menschlichen Review.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print("\n=== FERTIG: GENERATION 2 / REVIEW-FIRST ===")
    print(f"Clean-Originale:         {len(all_clean)}")
    print(f"Audio-Versuche:          {len(plan)}")
    print(f"Vosk nichtleer:          {len(generated)}")
    print(f"Flagged für Review:      {len(flagged)}")
    print(f"Leer:                    {sum(a.status == 'empty' for a in attempts)}")
    print(f"Fehler:                  {sum(a.status == 'error' for a in attempts)}")
    print("Keine Vosk-Samples wurden semantisch verworfen oder automatisch ins Training gewählt.")
    print(f"Alle Vosk-Texte:         {args.output_dir / 'vosk_generated_all.txt'}")
    print(f"Metadaten:               {args.output_dir / 'vosk_generated_all_metadata.csv'}")
    print(f"Review-Flags:            {args.output_dir / 'vosk_flagged_for_review.csv'}")
    print(f"Manifest:                {args.output_dir / 'manifest_all_attempts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
