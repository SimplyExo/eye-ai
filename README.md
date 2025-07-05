# eye-ai
**Codename: Eye-Ai**
Bilderkennung einer Kamera, die Objekte in der Umgebung in Audio-Hinweise für den Benutzer umwandelt, damit dieser sich ohne Sicht bewegen kann.

**__Projekt-Plan:__**

**1. Schritt:**
Erkennung von Objekten im Raum ohne Klassifizierung jedoch mit Messung der Entfernung. Dann Umwandlung in Ton mit Richtung.
Wenn das Objekt nicht erkannt wird, wird stattdessen ein solider Block dahin gestellt, um zu verhindern, dass der Nutzer in etwas läuft.
Not-Aus: Wenn das Programm nicht mitkommt, wird der Nutzer gewarnt, er solle sich erstmal nicht weiter bewegen, bis das Programm aufgehohlt hat.

**2. Schritt:**
Klassifizierung von Objekten, möglicherweise Ausgabe per Sprache.
Möglichkeit der Ausgabe der Objekte im aktuellen Sichtfeld des Nutzers auf dessen Eingabe hin.

**3. Schritt**
Gesichtserkennung von bekannten Personen?
Nachtsicht?

## Requirements for building EyeAICore from source

- python3 (needed for xnnpack to generate it's microkernels)

## Building EyeAICore from source

> [!NOTE]
> Be careful with the thread count when building (specified by the -j flag), or it might just OOM your system :/

```bash
cd EyeAICore
cmake -B build
cmake --build build -j8
```

## Building EyeAICore from source using Docker container

Build docker image:

```bash
docker build -t eye-ai .
```

Now you have a docker image that contains a build of EyeAICore from source.

To run the Dataset Evaluation program, run the docker image in container:

```bash
docker run -it eye-ai
```

(when inside the container)
```bash
./EyeAICore/metric_depth/scripts/build_and_run_eval_dataset.sh <dataset_directory> <evaluation_dataset_directory>
```