# EyeAI Intent Classifier - Entwicklungsdokumentation

## Anforderungen

Der Intent-Classifier sollte zehn lokale EyeAI-Intents zuverlässig erkennen und dabei:
- sehr klein und Android-/TFLite-tauglich sein,
- geringe Latenz besitzen,
- vollständig lokal laufen,
- kurze/triviale Nutzeräußerungen ebenso wie längere Formulierungen verarbeiten,
- mit Vosk-ASR-Ausgaben robust umgehen,
- auf unbekannten bzw. schwierigen Formulierungen möglichst stabil generalisieren.

## Finale BaselineCNN-Architektur

Input: int32 [1,24]

Embedding(32)
→ Conv1D(32, kernel_size=3)
→ GlobalMaxPooling + GlobalMeanPooling
→ Dense(32, ReLU)
→ Dropout(0.15)
→ Dense(10, Softmax)

Tokenizer:
- T1: Word
- T2: BPE

Footprint:
- T1: ca. 80,234 Parameter / 317.97 KiB
- T2: ca. 69,514 Parameter / 276.10 KiB

Alle relevanten Sequenzen passten in max_len=24; keine relevante Truncation.

## Trainingsstrategien

- M0: Clean + Hard Negatives
- M1: Clean + Hard Negatives + Vosk gemeinsam von Beginn an
- M2: Phase 1 Clean + Hard Negatives; Phase 2 Fine-Tuning auf Clean + Hard Negatives + Vosk mit reduzierter Lernrate
- M3: Phase 1 Clean + Hard Negatives; Phase 2 Fine-Tuning ausschließlich auf Vosk mit reduzierter Lernrate

Es wurden 4 Strategien × 2 Tokenizer × 5 Seeds = 40 Modelle trainiert.

## Datenbasis

Bekannte dokumentierte Größen:
- Clean: 3,344 Samples
- Finaler Trainingspool für M0-M3: 4,746 Samples
- Semantic-300: Development
- ASR-300: Development
- Curated-60: Development
- Challenge-40: Stress-/Robustheitsset
- Known-Failure-9: Teilmenge des Challenge-Sets, daher kein unabhängiger Test

Semantic-300, ASR-300 und Curated-60 wurden für Development/Early Stopping verwendet.
Challenge-40 diente als separater Stress-Test.
Ein echter blinder Holdout und Human→Vosk-End-to-End-Test waren zu diesem Zeitpunkt noch offen.

## Ergebnisse

| Strategie | Tokenizer | Dev Macro-F1 | Challenge Accuracy |
|---|---:|---:|---:|
| M0 | T1 Word | 95.39% | 89.00% |
| M0 | T2 BPE | 94.58% | 81.00% |
| M1 | T1 Word | 95.39% | 87.50% |
| M1 | T2 BPE | 95.53% | 83.50% |
| M2 | T1 Word | 96.22% | 88.00% |
| M2 | T2 BPE | 96.38% | 82.50% |
| M3 | T1 Word | 95.39% | 87.50% |
| M3 | T2 BPE | 95.22% | 75.50% |

## Interpretation

M2 war die stärkste Trainingsstrategie auf dem Development-Set. M2_T2 erreichte mit 96.38% den höchsten Dev Macro-F1, dicht gefolgt von M2_T1 mit 96.22%.

Der Challenge-Test änderte jedoch die Rangfolge deutlich. T1/Word war über alle Strategien wesentlich robuster:
- M0_T1: 89.0%
- M2_T1: 88.0%
- M1_T1/M3_T1: 87.5%

T2/BPE zeigte dagegen teilweise erhebliche Einbrüche:
- M2_T2: 82.5%
- M3_T2: 75.5%

Damit zeigte sich, dass minimale Verbesserungen auf bekannten Development-Daten keine ausreichende Grundlage für die Produktionsauswahl waren.

## Verworfene Ansätze

### StrongCNN
Eine größere/stärkere CNN-Variante wurde getestet, aber verworfen, weil der zusätzliche Modellaufwand keine insgesamt bessere Qualität brachte.

### TinyGRU
Auch eine kleine GRU-Variante wurde getestet. Sie wurde verworfen, weil sie gegenüber der BaselineCNN insgesamt schlechter abschnitt und somit die zusätzliche sequentielle Modellkomplexität keinen praktischen Vorteil lieferte.

Damit blieb die kleine BaselineCNN bestehen.

## Warum die BaselineCNN sinnvoll blieb

Sie bot den besten Kompromiss aus:
- hoher Intent-Qualität,
- sehr kleiner Modellgröße,
- einfacher TFLite-/Android-Kompatibilität,
- niedriger Inferenzkomplexität,
- guter Robustheit,
- reproduzierbarer Multi-Seed-Evaluation.

Der Entwicklungsprozess folgte damit nicht dem Prinzip „größeres Modell = besser“, sondern dem Ziel, das kleinste Modell zu behalten, dessen zusätzliche Komplexität durch messbare Robustheit gerechtfertigt ist.

## Historischer Zwischenstand

Ein früher Clean-Ensemble-5-Fold-CV-Wert von ungefähr 93.5% wurde ebenfalls dokumentiert. Dieser Wert war jedoch ausdrücklich keine echte unabhängige Validation und sollte deshalb nicht direkt mit den späteren Dev-/Challenge-Ergebnissen gleichgesetzt werden.

## Aktuelle Schlussfolgerung

Der stärkste reine Development-Kandidat war M2_T2.
Der robustere Kandidat auf dem Challenge-Set war jedoch T1/Word, insbesondere M0_T1 bzw. M2_T1.

Für eine endgültige Produktionsentscheidung sollten deshalb weiterhin ein vollständig blinder Holdout und echte Human→Vosk-End-to-End-Daten maßgeblich sein.
