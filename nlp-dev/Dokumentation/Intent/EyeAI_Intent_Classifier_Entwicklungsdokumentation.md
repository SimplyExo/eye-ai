**EyeAI Intent-Classifier – Entwicklungsdokumentation**  
**1. Ziel des Modells**  
Der Intent-Classifier soll kurze und längere deutsche Nutzeräußerungen einer von zehn Klassen zuordnen:  
- TEXT_RECOGNITION  
- OBJECT_DETECTION  
- CHANGE_SPEECH_SPEED  
- CHANGE_SPEAKER  
- REDIRECT_TO_LLM  
- OPEN_SETTINGS  
- SET_FREQUENCY  
- SET_BPS  
- MEASURE_DISTANCE  
- ABORT  
Die Aufgabe wurde bewusst als kleiner Closed-Set-Classifier formuliert. Das Modell soll auf Android lokal ausführbar sein, geringe Latenz und Speicheranforderungen haben, mit Vosk-ASR-Ausgaben umgehen können und reproduzierbar nach TFLite exportierbar sein. Für die Kernklassifikation sollte keine LLM-API nötig sein.  
**2. Technische Zielanforderungen**  
| | |  
|-|-|  
| **Anforderung** | **Konsequenz für die Entwicklung** |   
| Sehr kleines Modell | Kleine Embeddings, kleine CNN/GRU-Varianten; Ressourcenmessung wurde Teil der Evaluation |   
| Android/TFLite | Built-ins-only-Export, INT32-Eingabe [1,24], FLOAT32-Ausgabe [1,10] |   
| Geringe Latenz | Host-TFLite-Laufzeiten wurden vergleichend gemessen |   
| Robustheit gegen ASR | Separate ASR-Validation und Vosk-Trainingsdaten |   
| Kurze Alltagsbefehle + schwere Formulierungen | Mischung aus trivialen, natürlichen, kontrastiven und negierten Beispielen |   
| Keine Überanpassung an Einzelsätze | Hard-Negatives wurden unabhängig formuliert statt Testfehler wörtlich ins Training zu kopieren |   
| Vergleichbarkeit | Bei Tokenizer-/Architekturtests wurden Daten und Trainingsparameter eingefroren |   
| Seed-Robustheit | Nach ersten Einzelruns wurden 3- und 5-Seed-Vergleiche eingeführt |   
| Keine Leakage | Raw- und normalisierte Train/Eval-Overlaps, Duplikate und Labelkonflikte wurden geprüft |   
   
**3. Baseline-Architektur**  
Die erhaltene BaselineCNN besteht aus:  
Embedding(32) -> Conv1D(32, Kernel 3, ReLU) -> GlobalMaxPooling + GlobalAveragePooling -> Dense(32, ReLU) -> Dropout(0.15) -> Dense(10, Softmax)  
max_len = 24  
Ressourcen der Baseline:  
| | | | |  
|-|-|-|-|  
| **Tokenizer** | **Parameter** | **TFLite** | **Host-TFLite Median** |   
| T1 Word | 80.234 | 317,97 KiB | ca. 0,007–0,008 ms |   
| T2 BPE | 69.514 | 276,10 KiB | ca. 0,007–0,008 ms |   
   
Die gemessenen Host-Zeiten sind keine Android-Latenzen, zeigen aber den relativen Aufwand.  
**4. Entwicklung der Trainingsdaten**  
**Frühe Vosk-Generierungsphase**  
Frühe Generierungsphase:  
- Clean: 3.344  
- für Vosk ausgewählt: 1.281  
- damaliger Finalbestand: 4.625  
- ASR-Anteil: 27,7 %  
- Review-Fälle: 1.467  
Diese Zahlen gehören zur Generierungs-/Reviewphase und sind nicht identisch mit dem später finalen Trainingspool.  
**Eingefrorener Trainingspool für die Modellvergleiche**  
Der später verwendete Pool umfasste 4.746 Samples:  
| | |  
|-|-|  
| **Komponente** | **Samples** |   
| Clean-Quellen | 2.385 |   
| kuratierte Vosk/ASR-Generationen | 1.828 |   
| Hard-Negative-Patch | 533 |   
| **Gesamt** | **4.746** |   
   
Der Hard-Negative-Patch entstand aus beobachteten Entscheidungsgrenzen. Wichtig war, Fehlersätze nicht wörtlich in das Training zu übernehmen, sondern neue, linguistisch unabhängige Beispiele derselben Fehlerklasse zu erzeugen.  
Typische schwierige Grenzen waren unter anderem:  
- OBJECT_DETECTION vs. MEASURE_DISTANCE bei Negationen  
- SET_BPS vs. SET_FREQUENCY vs. CHANGE_SPEECH_SPEED  
- TEXT_RECOGNITION vs. OBJECT_DETECTION bei Formulierungen mit „steht“  
- OPEN_SETTINGS vs. direkte Settings-Aktion  
- REDIRECT_TO_LLM vs. lokale Aktion bei Wörtern wie „Frequenz“, „Abstand“ oder „Geschwindigkeit“  
**5. Entwicklung der Validation**  
Die Evaluation wurde sukzessive stärker getrennt.  
**Semantic-300**  
- 300 manuell kontrollierte semantische Sätze  
- exakt 30 pro Klasse  
- deckt saubere natürliche Sprache und Boundary-Fälle ab  
Zwischenzeitlich enthielt das Set 301 Samples. Eine Klassenkorrektur hatte einen zusätzlichen Satz erzeugt. Danach wurde ein redundanter Frequenzsatz entfernt, sodass wieder exakt 30 Samples pro Klasse vorhanden waren.  
**ASR-300**  
- 300 ASR/Vosk-nahe Sätze  
- 30 pro Klasse  
- soll Robustheit gegenüber Erkennungsfehlern und gesprochener Sprache messen  
**Curated-60**  
- 60 manuell kuratierte Kontrollbeispiele  
- 6 pro Klasse  
**Challenge-40**  
- 40 gezielt schwierige Fälle  
- bewusst nicht balanciert  
- außerhalb von Training und Early Stopping  
- dient als Boundary-/Generalisationstest  
**Known-Failure-9**  
- neun bekannte Regressionen  
- Teilmenge von Challenge-40  
- daher kein unabhängiger Test  
**Entwicklungs- vs. Testdaten**  
Semantic-300, ASR-300 und Curated-60 wurden für Early Stopping verwendet und sind damit Development-Daten. Challenge-40 ist ein externer, aber klein und unausgewogen. Für dia endgültige Produktionsentscheidung wurde deshalb später ein echter menschlicher Holdout als wichtig identifiziert.  
Ein Preflight prüfte schließlich:  
- 0 raw-exakte Train/Eval-Overlaps  
- 0 normalisierte Train/Eval-Overlaps  
- 0 interne Trainingsduplikate  
- 0 Labelkonflikte  
**6. Tokenizerexperiment: T1 Word vs. T2 BPE**  
**Einzelrun**  
Beide Varianten erreichten im gemeinsamen Devlopment-Split eine beste Validation-Accuracy von 95,15 %. T2 hatte im Einzelrun die niedrigere Validation-Loss.  
**5-Seed-Vergleich**  
| | | |  
|-|-|-|  
| **Set** | **T1 Accuracy** | **T2 Accuracy** |   
| Semantic-300 | 94,73 ± 1,46 % | 94,07 ± 1,64 % |   
| ASR-300 | 96,93 ± 1,19 % | 96,00 ± 0,33 % |   
| Curated-60 | 96,00 ± 1,90 % | 96,00 ± 2,79 % |   
| Challenge-40 | 85,50 ± 2,74 % | 84,50 ± 3,71 % |   
   
Ergebnis: Kein dramatischer Qualitätsunterschied. T1 war leicht stärker und hatte weniger Fehlerereignisse. T2 hatte 0 % UNK und ein kleineres neuronales Modell, benötigte aber ein deutlich größeres Tokenizer-Artefakt. Daher wurde der Tokenizer nicht nur anhand eines Einzelruns ausgewählt.  
**7. Architekturtests**  
**StrongCNN**  
StrongCNN verwendete parallele Conv1D-Kernel 2/3/5 und eine größere Dense-Schicht.  
T1:  
- Semantic: 94,11 -> 96,33 %  
- ASR: 96,78 -> 96,67 %  
- Curated: 97,22 -> 96,67 %  
- Challenge: 86,67 -> 83,33 %  
T2:  
- Semantic: 94,56 -> 94,67 %  
- ASR: 95,78 -> 96,89 %  
- Curated: 97,78 -> 97,78 %  
- Challenge: 86,67 -> 80,83 %  
Das Modell verbesserte einzelne Development-Werte, verschlechterte aber gerade Challenge-Generalisation und löste die stabilen Boundary-Fehler nicht systematisch. Gleichzeitig stiegen Modellgröße und Laufzeit. Deshalb wurde StrongCNN nicht weiterverfolgt.  
**TinyGRU**  
Das TinyGRU-Modell blieb hinter unrer Baseline:  
| | | |  
|-|-|-|  
| **Set** | **BaselineCNN T1** | **Masked TinyGRU T1** |   
| Semantic-300 | 94,11 % | 91,44 % |   
| ASR-300 | 96,78 % | 92,33 % |   
| Curated-60 | 97,22 % | 92,78 % |   
| Challenge-40 | 86,67 % | 80,00 % |   
   
Zusätzlich war das maskierte GRU im Host-TFLite-Benchmark rund 10,9-mal langsamer und größer. Entscheidender als die Latenz war jedoch die schlechtere Klassifikationsqualität. Der Ansatz wurde deshalb verworfen.  
**Warum BaselineCNN bestehen blieb**  
Die Baseline war:  
- klein,  
- sehr schnell,  
- TFLite-kompatibel,  
- stabil über Seeds,  
- auf allen Development-Sets stark,  
- in der Challenge robuster als die komplexeren Alternativen.  
Die Entwicklung zeigte damit, dass mehr Modellkomplexität die semantischen Boundary-Probleme nicht automatisch löst. Datenqualität und Trainingsstrategie wurden wichtiger als eine größere Architektur.  
**8. Trainingsstrategien M0–M3**  
Nach Auswahl der Baseline wurden vier Trainingsmethoden bei beiden Tokenizern über je fünf Seeds getestet: insgesamt 40 Modelle.  
- M0: Clean + Hard-Negatives  
- M1: Clean + Hard-Negatives + Vosk von Beginn an gemeinsam  
- M2: zuerst Clean + Hard-Negatives; danach Fine-Tuning mit Clean + Hard-Negatives + Vosk bei kleinerer Lernrate  
- M3: zuerst Clean + Hard-Negatives; danach Fine-Tuning nur auf Vosk  
**Development Macro-F1 / Challenge Accuracy**  
| | | |  
|-|-|-|  
| **Modell** | **Dev Macro-F1** | **Challenge-40 Accuracy** |   
| M0_T1 | 95,39 % | 89,00 % |   
| M0_T2 | 94,58 % | 81,00 % |   
| M1_T1 | 95,39 % | 87,50 % |   
| M1_T2 | 95,53 % | 83,50 % |   
| M2_T1 | 96,22 % | 88,00 % |   
| M2_T2 | 96,38 % | 82,50 % |   
| M3_T1 | 95,39 % | 87,50 % |   
| M3_T2 | 95,22 % | 75,50 % |   
   
M2 war auf den Development-Daten am stärksten. Gleichzeitig zeigte Challenge-40 einen deutlichen Vorteil der Word-Modelle. Deshalb wurde eine Produktionsentscheidung nicht allein aus Development-Scores abgeleitet.  
**9. Menschlicher Test**  
Später wurden acht repräsentative Deployment-Modelle auf 154 realen menschlichen Formulierungen getestet. Die Sätze enthalten sowohl Alltagsformulierungen als auch absichtlich schwierige, lange und teilweise ASR-artige Konstruktionen.  
| | |  
|-|-|  
| **Modell** | **Accuracy** |   
| M1_T1 | 70,78 % |   
| M1_T2 | 69,48 % |   
| M2_T1 | 65,58 % |   
| M0_T2 | 64,94 % |   
| M2_T2 | 63,64 % |   
| M3_T1 | 62,99 % |   
| M3_T2 | 62,99 % |   
| M0_T1 | 62,34 % |   
   
Dieser Test ist besonders interessant, weil M1_T1 hier vor M2 liegt. Das widerspricht nicht den Development-Ergebnissen, sondern zeigt, dass künstliche/manuell konstruierte Validation und tatsächlich menschliche Sprache unterschiedliche Aspekte der Generalisation messen können.  
Da in der Desktop-App pro Kombination nur ein repräsentativer Seed lag, ist dies zunächst ein Vergleich konkreter Deployment-Modelle und noch kein 5-Seed-Nachweis dafür, dass M1 grundsätzlich die beste Trainingsmethode ist.  
**10. Methodische Entwicklung**  
Die Entwicklung folgte zunehmend einem kontrollierten experimentellen Vorgehen:  
1. Datenbasis aufbauen und ASR-Augmentation erzeugen.  
2. Klassen- und Labeldefinitionen bereinigen.  
3. Hard-Negatives für beobachtete Entscheidungsgrenzen erzeugen.  
4. Train/Eval-Leakage entfernen und Validation balancieren.  
5. BaselineCNN einfrieren.  
6. Tokenizer isoliert vergleichen.  
7. Seed-Varianz durch 5-Seed-Tests messen.  
8. Stärkere CNN und rekurrente Alternativen isoliert testen.  
9. Nicht überzeugende Architekturvarianten verwerfen.  
10. Trainingsstrategie als 4 × 2 × 5-Faktorexperiment untersuchen.  
11. Repräsentative Modelle in eine Desktop-App integrieren.  
12. Echte menschliche/Vosk-Formulierungen als nächste Realitätsstufe testen.  
Ein wichtiger Grundsatz war, nicht jeweils das beste Seed-Ergebnis herauszugreifen. Für Aussagen wurden Mittelwert und Standardabweichung verwendet. einzelne repräsentative Seeds dienten nur der App-Integration.  
**11. Entwicklung der Datenanforderungen**  
Die Anforderungen an gute Trainingsdaten wurden mit den Tests präziser:  
- alle zehn Klassen ausreichend abdecken  
- nicht nur lange/komplizierte, sondern besonders viele kurze Alltagsbefehle enthalten  
- natürliche sprachliche Vielfalt statt Template-Paraphrasen  
- Vosk-typische Fehler realistisch abbilden  
- semantische Gegenbeispiele enthalten  
- Negation und Kontrast gezielt abdecken  
- benachbarte Settings-Klassen sauber trennen  
- keine wörtliche Übernahme von Challenge-/Testfehlern  
- Duplikate und Labelkonflikte vermeiden  
Für Validation/Test wurden zusätzlich gefordert:  
- klar getrennt vom Training  
- bei Haupt-Validation möglichst balanciert  
- separate saubere und ASR-nahe Sets  
- gezielte Challenge-Sets für Boundaries  
- Macro-F1 zusätzlich zu Accuracy  
- mehrere Seeds  
- langfristig echter speaker-disjointer Human-Holdout  
**13. Diagramme**  
- 01_tokenizer_multiseed_accuracy.png – T1 vs. T2 über fünf Seeds  
- 02_architekturvergleich_t1.png – BaselineCNN vs. StrongCNN vs. MaskedTinyGRU  
- 03_training_strategy_dev_f1.png – M0–M3 Development Macro-F1  
- 04_training_strategy_challenge.png – M0–M3 Challenge-Generalisation  
- 05_human_test_accuracy.png – 154 menschliche Äußerungen  
- 06_training_data_composition.png – Zusammensetzung des finalen 4.746er Trainingspools  
**14. Kernaussage**  
Der wesentliche Fortschritt bestand nicht darin, das Modell immer größer zu machen. Die stärksten Verbesserungen der Entwicklungsqualität entstanden durch:  
- klarere Intent-Definitionen,  
- bessere und realistischere Daten,  
- Hard-Negatives,  
- getrennte Semantic-/ASR-/Challenge-Evaluation,  
- Leakage-Kontrollen,  
- Multi-Seed-Auswertung,  
- systematische Tests von Tokenizer und Trainingsmethode,  
- und schließlich echte menschliche Sprachdaten.  
Die BaselineCNN blieb bestehen, weil komplexere Architekturen ihre Mehrkosten nicht durch robustere Generalisation rechtfertigten. Der nächste große Hebel ist deshalb eher ein hochwertiger Human/Vosk-Datensatz als eine weitere Vergrößerung der Architektur.  
