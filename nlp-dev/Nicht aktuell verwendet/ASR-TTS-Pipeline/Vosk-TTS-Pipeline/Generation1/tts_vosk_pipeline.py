#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS -> Audio-Augmentation -> Vosk Pipeline für den 10-Klassen-Intent-Classifier.

Konzept:
- Clean-Samples bleiben vollständig erhalten.
- Auswahl für TTS/Vosk nach Quelle:
    * Widersprüche/organisch: 100 %
    * Saubere Trainingsdaten: standardmäßig 60 % (erlaubt 50-70 %)
    * Alte gefilterte Daten: standardmäßig 25 % (erlaubt 20-30 %)
- TTS-Audio-Verteilung: 30 % clean, 50 % moderate, 20 % hard.
- Mehrere Piper-Stimmen + optional eSpeak-NG als bewusst andere/geringere Qualität.
- Vosk-Ausgaben werden nicht blind übernommen:
    * identische/nahezu identische Transkripte werden verworfen
    * semantisch riskante Fälle landen in review_vosk.csv
    * Cross-Label-Transkriptkonflikte werden ausgeschlossen
- Finales Dataset begrenzt standardmäßig ASR/Vosk-Samples auf ca. 35 %.

Python-Pakete:
    pip install numpy scipy vosk piper-tts
Optional unter Debian/Ubuntu/Zorin:
    sudo apt install espeak-ng

Empfohlen:
- mehrere deutsche Piper-Stimmen in --piper-voices-dir
- echte Hintergrundgeräusche als WAV-Dateien in --noise-dir
- echte Raumimpulsantworten als WAV-Dateien in --rir-dir

Ohne echte Noise/RIR-Dateien werden synthetische Näherungen erzeugt.
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

SEVERITY_RATIOS = {"clean": 0.30, "moderate": 0.50, "hard": 0.20}

NEGATION_TOKENS = {
    "nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines",
    "nichts", "nie", "niemals",
}
CONTRAST_TOKENS = {"nur", "sondern", "statt", "außer", "ausser"}

# Nur konservativer Plausibilitätscheck, kein Ersatz für echte Semantikprüfung.
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
        "aufhören", "aufhoeren", "zurück", "zurueck", "nicht",
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
    severity: str
    tts_backend: str = ""
    voice: str = ""
    tts_quality: str = ""
    transcript: str = ""
    similarity: float = 0.0
    token_recall: float = 0.0
    status: str = ""
    reason: str = ""
    audio_path: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kontrollierte TTS -> Noise/Reverb -> Vosk Augmentation."
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
    p.add_argument("--output-dir", type=Path, default=Path("tts_vosk_output"))
    p.add_argument("--seed", type=int, default=20260809)

    p.add_argument("--organic-ratio", type=float, default=1.00)
    p.add_argument("--clean-ratio", type=float, default=0.60)
    p.add_argument("--old-ratio", type=float, default=0.25)
    p.add_argument("--target-asr-share", type=float, default=0.35,
                   help="Vosk-Anteil am finalen Trainingsset; Default 35%%.")
    p.add_argument("--espeak-weight", type=float, default=0.15,
                   help="eSpeak-Anteil, wenn Piper und eSpeak verfügbar sind.")
    p.add_argument("--disable-espeak", action="store_true")
    p.add_argument("--keep-audio", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur Auswahl und Mengen planen, ohne TTS/Vosk.")
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
    if not (0.0 < args.target_asr_share < 0.5):
        raise ValueError("--target-asr-share sollte > 0 und < 0.5 sein.")
    if not (0.0 <= args.espeak_weight <= 1.0):
        raise ValueError("--espeak-weight muss zwischen 0 und 1 liegen.")
    for path in (args.organic, args.clean, args.old):
        if not path.exists():
            raise FileNotFoundError(f"Datensatz nicht gefunden: {path}")
    if not args.dry_run:
        if Model is None:
            raise RuntimeError("vosk ist nicht installiert: pip install vosk")
        if args.vosk_model is None or not args.vosk_model.exists():
            raise FileNotFoundError(
                "--vosk-model muss auf ein vorhandenes deutsches Vosk-Modell zeigen."
            )


def read_dataset(path: Path, source: str) -> list[Sample]:
    out: list[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            if ";" not in raw:
                raise ValueError(f"{path}:{line_no}: kein ';' gefunden.")
            text, label = raw.rsplit(";", 1)
            text, label = text.strip(), label.strip()
            if label not in LABELS:
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
    """
    Organic: 3 Audio-Versuche/Basissatz.
    Clean:   zufällig 1-2 Audio-Versuche/Basissatz.
    Old:     1 Audio-Versuch/Basissatz.

    Die spätere Filterung entscheidet, welche Vosk-Ausgaben wirklich trainiert werden.
    """
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


def severity_schedule(n: int, rng: random.Random) -> list[str]:
    n_clean = round(n * SEVERITY_RATIOS["clean"])
    n_moderate = round(n * SEVERITY_RATIOS["moderate"])
    n_hard = n - n_clean - n_moderate
    result = ["clean"] * n_clean + ["moderate"] * n_moderate + ["hard"] * n_hard
    rng.shuffle(result)
    return result


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
    """factor > 1: schneller/kürzer; factor < 1: langsamer/länger."""
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
    for _ in range(rng.randint(5, 14)):
        delay_ms = rng.uniform(8.0, min(90.0, rt60 * 500.0))
        idx = min(length - 1, int(delay_ms * sr / 1000.0))
        amp = rng.uniform(-0.45, 0.55) * math.exp(-3.5 * idx / length)
        rir[idx] += amp
    tail = np.array([rng.gauss(0.0, 1.0) for _ in range(length)], dtype=np.float32)
    rir += 0.045 * tail * decay
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


def random_dropout(x: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    y = x.copy()
    for _ in range(rng.randint(1, 2)):
        m = int(sr * rng.uniform(0.025, 0.080))
        if len(y) <= m + 2:
            continue
        start = rng.randint(0, len(y) - m)
        y[start:start + m] *= rng.uniform(0.05, 0.30)
    return y


def random_clipping(x: np.ndarray, rng: random.Random) -> np.ndarray:
    threshold = rng.uniform(0.35, 0.70)
    return np.clip(x, -threshold, threshold) / threshold


def augment_audio(x: np.ndarray, severity: str, sr: int, rng: random.Random,
                  noise_paths: list[Path], rir_paths: list[Path]) -> np.ndarray:
    x = normalize_peak(x, 0.90)

    if severity == "clean":
        # Bewusst kein Noise und kein Hall.
        return normalize_peak(x * rng.uniform(0.85, 1.05), 0.92)

    if severity == "moderate":
        if rng.random() < 0.65:
            x = apply_reverb(x, sr, rng, (0.15, 0.45), rir_paths)
        if noise_paths and rng.random() < 0.75:
            noise = choose_segment(read_wav_mono(rng.choice(noise_paths), sr), len(x), rng)
        else:
            noise = synthetic_noise(len(x), sr, rng)
        x = mix_at_snr(x, noise, rng.uniform(12.0, 25.0))
        if rng.random() < 0.30:
            x = apply_bandlimit(x, sr, 100.0, rng.uniform(5500.0, 7600.0))
        return normalize_peak(x * rng.uniform(0.75, 1.05), 0.94)

    if severity == "hard":
        if rng.random() < 0.90:
            x = apply_reverb(x, sr, rng, (0.40, 0.90), rir_paths)
        if noise_paths and rng.random() < 0.85:
            noise = choose_segment(read_wav_mono(rng.choice(noise_paths), sr), len(x), rng)
        else:
            noise = synthetic_noise(len(x), sr, rng)
        x = mix_at_snr(x, noise, rng.uniform(3.0, 12.0))
        if rng.random() < 0.45:
            x = apply_bandlimit(x, sr, 280.0, rng.uniform(3300.0, 4200.0))
        else:
            x = apply_bandlimit(x, sr, 160.0, rng.uniform(4300.0, 6000.0))
        if rng.random() < 0.30:
            x = random_clipping(x, rng)
        if rng.random() < 0.20:
            x = random_dropout(x, sr, rng)
        return normalize_peak(x * rng.uniform(0.45, 0.95), 0.94)

    raise ValueError(f"Unbekannter Severity-Wert: {severity}")


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
                    volume=rng.uniform(0.85, 1.05),
                    length_scale=rng.uniform(0.86, 1.18),
                    noise_scale=rng.uniform(0.55, 0.90),
                    noise_w_scale=rng.uniform(0.60, 1.00),
                    normalize_audio=True,
                )
                voice.synthesize_wav(text, wf, syn_config=syn)
            else:
                voice.synthesize_wav(text, wf)

        # Fallback/zusätzliche kleine Variation, falls API keine SynthesisConfig exportiert.
        if SynthesisConfig is None:
            x = read_wav_mono(output_path, 16000)
            x = change_speed(x, rng.uniform(0.90, 1.12))
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
        speed = rng.randint(135, 210)
        pitch = rng.randint(35, 65)
        amplitude = rng.randint(85, 130)
        subprocess.run([
            self.exe, "-v", "de", "-s", str(speed), "-p", str(pitch),
            "-a", str(amplitude), "-w", str(output_path), text,
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"espeak-ng-de-s{speed}-p{pitch}", "low"


def choose_tts_backend(piper: PiperPool, espeak: EspeakEngine,
                       espeak_weight: float, disable_espeak: bool,
                       rng: random.Random) -> str:
    have_piper = piper.available()
    have_espeak = espeak.available() and not disable_espeak
    if not have_piper and not have_espeak:
        raise RuntimeError(
            "Keine TTS-Engine verfügbar. Piper-Voices bereitstellen oder espeak-ng installieren."
        )
    if have_piper and have_espeak:
        return "espeak" if rng.random() < espeak_weight else "piper"
    return "piper" if have_piper else "espeak"


def transcribe_vosk(model, x: np.ndarray, sr: int = 16000) -> str:
    pcm = np.clip(x * 32767, -32768, 32767).astype(np.int16).tobytes()
    rec = KaldiRecognizer(model, sr)
    # 4000 Samples * 2 Bytes/sample.
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


def semantic_filter(base_text: str, transcript: str,
                    label: str) -> tuple[str, str, float, float]:
    """
    status:
      reject -> leer oder praktisch Duplikat, kein Zusatznutzen
      review -> Semantik könnte gekippt sein; manuell prüfen
      accept -> informative, konservativ plausible ASR-Variante
    """
    src_norm = normalize_text(base_text)
    hyp_norm = normalize_text(transcript)
    if not hyp_norm:
        return "reject", "empty_transcript", 0.0, 0.0
    if src_norm == hyp_norm:
        return "reject", "identical_to_clean", 1.0, 1.0

    sim = SequenceMatcher(None, src_norm, hyp_norm).ratio()
    src_t, hyp_t = tokens(base_text), tokens(transcript)
    recall = multiset_recall(src_t, hyp_t)

    if sim >= 0.975:
        return "reject", "near_duplicate_clean", sim, recall
    if sim < 0.42 or recall < 0.30:
        return "review", "too_much_information_lost", sim, recall

    src_set, hyp_set = set(src_t), set(hyp_t)
    if src_set & NEGATION_TOKENS and not (hyp_set & NEGATION_TOKENS):
        return "review", "negation_lost", sim, recall
    if src_set & CONTRAST_TOKENS and not (hyp_set & CONTRAST_TOKENS):
        return "review", "contrast_marker_lost", sim, recall

    source_anchors = src_set & ANCHORS.get(label, set())
    hyp_anchors = hyp_set & ANCHORS.get(label, set())
    if source_anchors and not hyp_anchors:
        return "review", "all_label_anchors_lost", sim, recall
    if len(hyp_t) <= 1 and len(src_t) >= 3:
        return "review", "too_short", sim, recall

    return "accept", "informative_asr_variant", sim, recall


def write_attempt_csv(path: Path, attempts: list[Attempt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(attempts[0]).keys()) if attempts else [
        "attempt_id", "base_id", "source", "label", "base_text", "severity",
        "tts_backend", "voice", "tts_quality", "transcript", "similarity",
        "token_recall", "status", "reason", "audio_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in attempts:
            writer.writerow(asdict(a))


def remove_cross_label_conflicts(attempts: list[Attempt]) -> None:
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for a in attempts:
        if a.status == "accept":
            labels_by_text[normalize_text(a.transcript)].add(a.label)
    conflicts = {t for t, labels in labels_by_text.items() if len(labels) > 1}
    for a in attempts:
        if a.status == "accept" and normalize_text(a.transcript) in conflicts:
            a.status = "review"
            a.reason = "cross_label_transcript_conflict"


def remove_generated_duplicates(attempts: list[Attempt]) -> None:
    groups: dict[tuple[str, str], list[Attempt]] = defaultdict(list)
    for a in attempts:
        if a.status == "accept":
            groups[(normalize_text(a.transcript), a.label)].append(a)
    severity_bonus = {"clean": 0.00, "moderate": 0.04, "hard": 0.02}
    for group in groups.values():
        if len(group) <= 1:
            continue
        def utility(a: Attempt) -> float:
            return (1.0 - abs(a.similarity - 0.76)) + severity_bonus.get(a.severity, 0.0)
        group.sort(key=utility, reverse=True)
        for duplicate in group[1:]:
            duplicate.status = "reject"
            duplicate.reason = "duplicate_generated_transcript"


def select_final_asr(attempts: list[Attempt], clean_samples: list[Sample],
                     target_share: float, rng: random.Random) -> list[Attempt]:
    """
    Begrenzung gegen harte Doppelgewichtung.
    Bei 3344 Clean und 35% gewünschtem finalen ASR-Anteil ergibt sich
    ungefähr 1800 zusätzliche ASR-Samples, sofern genug gute Kandidaten existieren.
    """
    accepted = [a for a in attempts if a.status == "accept"]
    if not accepted:
        return []
    desired_total = round(len(clean_samples) * target_share / (1 - target_share))
    desired_total = min(desired_total, len(accepted))

    clean_counts = Counter(s.label for s in clean_samples)
    total_clean = sum(clean_counts.values())
    quotas: dict[str, int] = {}
    remaining = desired_total
    for label in LABELS[:-1]:
        q = round(desired_total * clean_counts[label] / total_clean)
        quotas[label] = q
        remaining -= q
    quotas[LABELS[-1]] = max(0, remaining)

    source_bonus = {SOURCE_ORGANIC: 0.30, SOURCE_CLEAN: 0.15, SOURCE_OLD: 0.0}
    severity_bonus = {"clean": 0.0, "moderate": 0.06, "hard": 0.03}

    def utility(a: Attempt) -> float:
        sweet = 1.0 - abs(a.similarity - 0.75)
        return sweet + source_bonus.get(a.source, 0.0) + severity_bonus.get(a.severity, 0.0)

    by_label: dict[str, list[Attempt]] = defaultdict(list)
    for a in accepted:
        by_label[a.label].append(a)

    selected: list[Attempt] = []
    selected_ids: set[str] = set()
    for label in LABELS:
        group = by_label[label]
        rng.shuffle(group)
        group.sort(key=utility, reverse=True)
        quota = quotas[label]

        # Runde 1: maximal eine ASR-Variante je Basissatz.
        used_base: set[str] = set()
        label_count = 0
        for a in group:
            if label_count >= quota:
                break
            if a.base_id in used_base:
                continue
            selected.append(a)
            selected_ids.add(a.attempt_id)
            used_base.add(a.base_id)
            label_count += 1

        # Runde 2: zweite Varianten nur, falls die Quote sonst nicht gefüllt wird.
        if label_count < quota:
            for a in group:
                if label_count >= quota:
                    break
                if a.attempt_id in selected_ids:
                    continue
                selected.append(a)
                selected_ids.add(a.attempt_id)
                label_count += 1

    if len(selected) < desired_total:
        rest = [a for a in accepted if a.attempt_id not in selected_ids]
        rng.shuffle(rest)
        rest.sort(key=utility, reverse=True)
        selected.extend(rest[:desired_total - len(selected)])
    elif len(selected) > desired_total:
        rng.shuffle(selected)
        selected = selected[:desired_total]
    return selected


def write_text_label(path: Path, pairs: Iterable[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text, label in pairs:
            f.write(f"{text};{label}\n")


def print_distribution(title: str, samples: Iterable[Sample]) -> None:
    samples = list(samples)
    c = Counter(s.label for s in samples)
    print(f"\n{title}: {len(samples)}")
    for label in LABELS:
        print(f"  {label:22s} {c[label]:4d}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    organic_all = read_dataset(args.organic, SOURCE_ORGANIC)
    clean_all = read_dataset(args.clean, SOURCE_CLEAN)
    old_all = read_dataset(args.old, SOURCE_OLD)
    all_clean = organic_all + clean_all + old_all

    # Sicherstellen, dass wirklich die global deduplizierten Split-Dateien genutzt werden.
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
    severities = severity_schedule(len(plan), rng)
    print(f"\nGeplante Audio-Versuche: {len(plan)}")
    print("Severity:", dict(Counter(severities)))

    desired_asr = round(len(all_clean) * args.target_asr_share / (1 - args.target_asr_share))
    print(f"Ziel: ca. {args.target_asr_share:.0%} ASR im finalen Set "
          f"=> bis zu {desired_asr} Vosk-Samples zusätzlich zu {len(all_clean)} Clean-Samples.")

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
        a = Attempt(attempt_id, sample.base_id, sample.source, sample.label,
                    sample.text, severity)
        try:
            backend = choose_tts_backend(
                piper, espeak, args.espeak_weight, args.disable_espeak, rng
            )
            a.tts_backend = backend
            if backend == "piper":
                a.voice, a.tts_quality = piper.synthesize(sample.text, tmp_tts, rng)
            else:
                a.voice, a.tts_quality = espeak.synthesize(sample.text, tmp_tts, rng)

            x = read_wav_mono(tmp_tts, 16000)
            x = change_speed(x, rng.uniform(0.94, 1.06))
            x = augment_audio(x, severity, 16000, rng, noise_paths, rir_paths)

            if args.keep_audio:
                write_wav_mono(out_audio, x, 16000)
                a.audio_path = str(out_audio)

            a.transcript = transcribe_vosk(vosk_model, x, 16000)
            status, reason, sim, recall = semantic_filter(
                sample.text, a.transcript, sample.label
            )
            a.status, a.reason = status, reason
            a.similarity, a.token_recall = round(sim, 6), round(recall, 6)
        except Exception as exc:
            a.status = "error"
            a.reason = f"{type(exc).__name__}: {exc}"
        finally:
            if tmp_tts.exists():
                try:
                    tmp_tts.unlink()
                except OSError:
                    pass

        attempts.append(a)
        if idx % 50 == 0 or idx == len(plan):
            counts = Counter(x.status for x in attempts)
            print(f"[{idx:5d}/{len(plan):5d}] accept={counts['accept']} "
                  f"review={counts['review']} reject={counts['reject']} "
                  f"error={counts['error']}")

    remove_cross_label_conflicts(attempts)
    remove_generated_duplicates(attempts)

    selected_asr = select_final_asr(attempts, all_clean, args.target_asr_share, rng)
    selected_ids = {a.attempt_id for a in selected_asr}
    for a in attempts:
        if a.status == "accept" and a.attempt_id not in selected_ids:
            a.status = "accept_not_selected"
            a.reason = "asr_share_cap_or_stratified_selection"

    write_attempt_csv(args.output_dir / "manifest_all_attempts.csv", attempts)
    write_attempt_csv(args.output_dir / "review_vosk.csv",
                      [a for a in attempts if a.status == "review"])
    write_attempt_csv(args.output_dir / "selected_vosk_metadata.csv", selected_asr)

    write_text_label(args.output_dir / "clean_originals.txt",
                     ((s.text, s.label) for s in all_clean))
    write_text_label(args.output_dir / "accepted_vosk_selected.txt",
                     ((a.transcript, a.label) for a in selected_asr))

    final_pairs = [(s.text, s.label) for s in all_clean]
    final_pairs += [(a.transcript, a.label) for a in selected_asr]
    write_text_label(args.output_dir / "training_final_clean_plus_vosk.txt", final_pairs)

    final_n = len(final_pairs)
    actual_share = len(selected_asr) / final_n if final_n else 0.0
    summary = {
        "seed": args.seed,
        "clean_samples_total": len(all_clean),
        "source_selection": {
            SOURCE_ORGANIC: {"available": len(organic_all), "selected_for_tts": len(organic_sel),
                             "ratio": args.organic_ratio},
            SOURCE_CLEAN: {"available": len(clean_all), "selected_for_tts": len(clean_sel),
                           "ratio": args.clean_ratio},
            SOURCE_OLD: {"available": len(old_all), "selected_for_tts": len(old_sel),
                         "ratio": args.old_ratio},
        },
        "tts_attempts": len(plan),
        "severity_attempts": dict(Counter(severities)),
        "attempt_status_after_global_filter": dict(Counter(a.status for a in attempts)),
        "selected_vosk_samples": len(selected_asr),
        "target_asr_share": args.target_asr_share,
        "actual_asr_share": actual_share,
        "final_training_samples": final_n,
        "final_class_distribution": dict(Counter(label for _, label in final_pairs)),
        "selected_vosk_by_source": dict(Counter(a.source for a in selected_asr)),
        "selected_vosk_by_severity": dict(Counter(a.severity for a in selected_asr)),
        "selected_vosk_by_label": dict(Counter(a.label for a in selected_asr)),
        "piper_voices": [str(v) for v in piper.voice_paths],
        "real_noise_wavs": len(noise_paths),
        "real_rir_wavs": len(rir_paths),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    print("\n=== FERTIG ===")
    print(f"Clean:              {len(all_clean)}")
    print(f"Vosk ausgewählt:    {len(selected_asr)}")
    print(f"Final:              {final_n}")
    print(f"ASR-Anteil final:   {actual_share:.1%}")
    print(f"Review-Fälle:       {sum(a.status == 'review' for a in attempts)}")
    print(f"Finales Training:   {args.output_dir / 'training_final_clean_plus_vosk.txt'}")
    print(f"Review:             {args.output_dir / 'review_vosk.csv'}")
    print(f"Manifest:           {args.output_dir / 'manifest_all_attempts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
