# Aktuelles REL2ABS-Modellpaket

* Dieser Ordner enthält nur das aktuell verwendete V6-Kontext-Gate für Objekte und die zugehörigen Trainingsskripte.
* Ein kleines neuronales Gate lernt für jedes Objekt, wie stark es der F1-Schätzung vertrauen sollte.
* Dafür nutzt das Gate vorhandene YOLO-Erkennungen, Informationen zur Begrenzungsbox, Zusammenfassungen der Bildszene und die Kamerahöhe.
* Das Gate verwendet 57 Eingabewerte, eine verborgene Schicht mit 16 ReLU-Einheiten und einen Ausgabewert für die Gewichtung der beiden Schätzungen.
* Die endgültige Tiefenschätzung liegt zwischen der visuellen Schätzung und der F1-Schätzung.
* Die Datei im Ordner model stammt aus dem aktuell verwendeten Asset-Ordner der EyeAI-App.
* Die Parameterdatei enthält kalibrierte Modelle für Kamerahöhen von 1,60 m, 1,70 m, 1,80 m, 1,90 m und 2,00 m.
* In der aktuellen App ist die Variante für 1,70 m als Standard eingestellt.
* Die Trainingsskripte liegen im Ordner training.
* Die Skripte erwarten weiterhin die ursprünglichen Rel2abs-Projektordner, die sind hier aber nicht vorhanden, logischerweise.
* Die verwendeten Daten und ihre Verarbeitung sind in TRAINING\_DATA\_REFERENCES.txt beschrieben.
* Ältere Modelle wie Z1, S2, Spline und B3 sind bewusst nicht enthalten.

