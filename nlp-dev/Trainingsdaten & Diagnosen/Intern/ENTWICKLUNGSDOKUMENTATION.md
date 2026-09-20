# EyeAI Intent-Classifier – Entwicklungs- und Evaluationsdokumentation

## Datenbasis
Diese Dokumentation verwendet ausschließlich Messwerte und Entwicklungsentscheidungen, die im bisherigen Projektverlauf explizit dokumentiert wurden. Fehlende Accuracy- oder Macro-F1-Werte wurden nicht ergänzt oder geschätzt.

## 1. Modellanforderungen
Ziel ist ein sehr kleiner deutscher 10-Klassen-Intent-Classifier für Android hinter Vosk-ASR.

Technische Anforderungen:
- sehr geringe Modellgröße und Latenz;
- Android-/TFLite-Tauglichkeit;
- möglichst TFLITE_BUILTINS-only und damit NPU-/Delegate-freundlich;
- robuste Klassifikation sehr kurzer und alltäglicher Formulierungen;
- Robustheit gegenüber realistischen Vosk-ASR-Fehlern;
- semantische Trennung eng benachbarter Intents;
- keine Entfernung semantisch wichtiger Wörter wie „nicht“, „nur“, „sondern“, „aber“;
- identische Tokenisierung und Normalisierung in Python und Android.

## 2. Architekturentwicklung
Die Baseline wurde bewusst klein gehalten:

Token-IDs → Embedding → Conv1D(kernel=3) → Global Max Pool + maskierter Global Mean Pool → Dense → Softmax(10).

Transformer wurden zunächst aufgrund der Zielgröße und mobilen Ausführung verworfen. Eine kleine GRU wurde später als Challenger getestet, war aber schwächer, größer und erzeugte einen komplexeren TFLite-Graphen. Daher wurde sie verworfen.

Ein stärkeres CNN gewann knapp auf normaler Development-Validation, verlor jedoch auf Challenge-/Hard-Cases gegen die Baseline. Deshalb blieb die Baseline zunächst Hauptkandidat, während das Strong CNN als späterer Challenger bestehen blieb.

## 3. Trainingsdatenentwicklung

### Clean-Daten
Nach globaler Deduplizierung lagen 3344 Clean-Samples vor.

### Warum ASR-Daten ergänzt wurden
Das Produkt sieht keine perfekten Texte, sondern Vosk-Transkripte. Deshalb wurden Clean-/Organic-Sätze über TTS gesprochen und wieder durch das produktionsnahe Vosk-Modell transkribiert.

### Generation 1
Gen1 erzeugte 3873 TTS/Vosk-Versuche. Ein automatischer Filter wählte zunächst 1281 Samples. Die manuelle Prüfung zeigte jedoch, dass dieser Filter semantisch beschädigte Transkripte zu oft akzeptierte.

Diagnostik:
- clean: Mean WER 26,6 %, Label-Match 91,4 %;
- moderate: Mean WER 40,8 %, Label-Match 80,0 %;
- hard: Mean WER 90,1 %, Label-Match 23,5 %.

Damit war die Hard-Augmentation zu aggressiv. Auch eSpeak erwies sich deutlich destruktiver als Piper.

Folgen:
- automatisches semantisches Filtering wurde verworfen;
- Gen1 wurde streng manuell geprüft;
- 1205 Samples bestanden zunächst die strenge Prüfung;
- nach Cross-Generation-Kuration blieben 280 Gen1-Samples als echter Zusatznutzen zu Gen2.

### Generation 2
Gen2 wurde „review first“ aufgebaut:
- keine automatische semantische Trainingsentscheidung;
- deutlich kleinerer Hard-Anteil;
- eSpeak stark reduziert;
- TTS-spezifische Normalisierung für Zahlen, Hertz und BPS;
- manuelle Prüfung auf menschliche Verständlichkeit und Intent-Erhalt.

Ergebnis:
- 1548 streng manuell akzeptierte Gen2-Samples;
- zusammen mit 280 komplementären Gen1-Samples: 1828 finale geprüfte Vosk-Samples.

## 4. Anforderungen an Trainingsdaten
Die Datenanforderungen wurden sukzessive verschärft:
1. kurze und triviale Alltagsäußerungen explizit abdecken;
2. Füllwörter, Selbstkorrekturen und Umgangssprache zulassen;
3. Negationen und Kontrastmarker erhalten;
4. Hard Negatives gezielt an realen Entscheidungsgrenzen erzeugen;
5. ASR-Varianten nur dann trainieren, wenn der Intent aus dem Transkript allein noch klar ist;
6. semantische ASR-Flips nicht als Classifier-Training verwenden;
7. Deduplizierung nicht nur exakt, sondern bei Cross-Generation-Kuration auch semantisch;
8. nicht zu viele Varianten derselben Basisformulierung;
9. REDIRECT_TO_LLM als breite Catch-all-Klasse abdecken;
10. OBJECT_DETECTION als visuelle Szene/Orientierung und nicht nur als Objektnamen verstehen.

## 5. Validation-Entwicklung
Ein einzelnes Validation-Set war nicht ausreichend. Deshalb entstanden mehrere getrennte Perspektiven:
- Semantic-Val: 300 manuell geschriebene semantische Beispiele;
- ASR-Val: 300 manuell geschriebene ASR-artige Beispiele;
- Complex-Val: älteres komplexeres Dataset;
- Challenge-/Regression-Set: 40 gezielte Grenzfälle.

Diese Sets haben unterschiedliche Rollen:
- Semantic-Val misst semantische Klassentrennung.
- ASR-Val misst Robustheit gegenüber Erkennungsfehlern.
- Complex-Val prüft ungewöhnlichere Strukturen.
- Challenge misst bekannte Shortcut- und Grenzfallprobleme.
- Ein späterer Human→Vosk speaker-disjoint Test soll die finale Produktionsqualität messen.

## 6. Frühe diagnostische Accuracy-Werte
Als Plausibilitätsprüfung der Daten wurden einfache Textklassifikatoren verwendet:
- Word TF-IDF + Logistic Regression: 91,9 % OOF Accuracy;
- Character 3–5 Gram: 93,0 %;
- Ensemble: 93,54 %.

Diese Werte dienten als Datensatzdiagnose und nicht als finales mobiles Modell.

## 7. Hard-Case-Diagnose
Auf neun später identifizierten natürlichen Problemfällen:
- M0: 3/9 korrekt;
- M1: 4/9 korrekt;
- M2: 4/9 korrekt;
- M3: 3/9 korrekt.

Das zeigte, dass normale Development-Metriken allein die reale semantische Robustheit nicht ausreichend abbildeten.

## 8. Tokenizer-Diagnose
Der aktuelle Word-Level-Tokenizer enthält bereits alle Trainingstokens mit frequency >= 1. Sein Problem ist deshalb weniger eine fehlerhafte Implementierung als die geschlossene Wortrepräsentation.

Gemessene Word-Level-UNK/OOV-Raten:
- Semantic-Val: 3,25 %;
- ASR-Val: 2,51 %;
- Complex-Val: 16,08 %;
- Challenge: 5,88 %;
- neun Hard Cases: ca. 10,81 %.

BPE mit ungefähr 2000 Tokens erreichte 0 % UNK, war in der provisorischen Klassifikationsgüte aber schwächer. Deshalb bleibt BPE Challenger und wird erst nach dem finalen Hard-Negative-Datenpatch fair erneut verglichen.

## 9. Warum bestimmte Ansätze verworfen wurden
- Automatisches Gen1-Filtering: semantisch zu permissiv.
- Sehr harte ASR-Augmentation: erzeugte häufig unrecoverable ASR failure statt sinnvoller Classifier-Robustheit.
- Hoher eSpeak-Anteil: schlechtere Intent-Erhaltung als Piper.
- GRU: schwächere Generalisierung, größer, komplexerer TFLite-Graph.
- Strong CNN: nicht verworfen, aber noch kein neuer Standard, weil Challenge-Generalisation schlechter war.
- BPE: ebenfalls nicht verworfen; 0 % OOV ist attraktiv, aber bisher geringere Klassifikationsgüte.

## 10. Aktueller methodischer Stand
Die Hauptursache wurde schrittweise von „Modell vielleicht zu klein“ zu einem kombinierten Problem präzisiert:
1. fehlende semantische Coverage;
2. Satzanfangs-/n-Gram-Shortcuts;
3. Word-Level-OOV als Verstärker;
4. erst danach mögliche Architekturgrenzen.

Der nächste zentrale Schritt ist daher die Erweiterung des kuratierten Hard-Negative-Pools von ungefähr 200 auf ungefähr 500 Samples. Danach sollen Word-Level vs. BPE und Baseline-CNN vs. Strong-CNN unter identischen Bedingungen erneut verglichen werden.
