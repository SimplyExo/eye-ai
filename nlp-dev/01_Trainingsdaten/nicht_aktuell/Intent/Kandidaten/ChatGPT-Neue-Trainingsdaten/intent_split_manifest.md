# EyeAI Intent Split-Manifest

Person 2 ist eine Kopie von Person 1; Person 5 ist eine Kopie von Person 4 und wird daher jeweils nicht doppelt gezählt.

- Neue Trainingsergänzungen: 24
- Neue Validation: 36
- Blind Core: 60 (6 pro Klasse)
- Blind Hard: 20

Es gibt zwischen den vier ausgewählten Splits keine normalisierten exakten Satzduplikate. Auch gegenüber dem bisherigen DATASET.train + DATASET.val wurden für diese Auswahl keine normalisierten exakten Duplikate übernommen.

## Train additions
- ABORT: 3
- MEASURE_DISTANCE: 3
- OBJECT_DETECTION: 3
- OPEN_SETTINGS: 2
- REDIRECT_TO_LLM: 5
- SET_BPS: 4
- TEXT_RECOGNITION: 4

## Validation
- ABORT: 2
- CHANGE_SPEAKER: 2
- CHANGE_SPEECH_SPEED: 5
- MEASURE_DISTANCE: 3
- OBJECT_DETECTION: 3
- OPEN_SETTINGS: 2
- REDIRECT_TO_LLM: 8
- SET_BPS: 3
- SET_FREQUENCY: 3
- TEXT_RECOGNITION: 5

## Blind Core
- ABORT: 6
- CHANGE_SPEAKER: 6
- CHANGE_SPEECH_SPEED: 6
- MEASURE_DISTANCE: 6
- OBJECT_DETECTION: 6
- OPEN_SETTINGS: 6
- REDIRECT_TO_LLM: 6
- SET_BPS: 6
- SET_FREQUENCY: 6
- TEXT_RECOGNITION: 6

## Blind Hard
- ABORT: 3
- CHANGE_SPEAKER: 1
- CHANGE_SPEECH_SPEED: 2
- OBJECT_DETECTION: 2
- OPEN_SETTINGS: 1
- REDIRECT_TO_LLM: 3
- SET_BPS: 4
- TEXT_RECOGNITION: 4
