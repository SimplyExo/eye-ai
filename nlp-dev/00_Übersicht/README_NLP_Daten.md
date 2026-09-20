# NLP-Daten - Übersicht

```text
01_Trainingsdaten/
├── aktuell/                 aktuelle Intent-Trainings-, Validierungs- und Evaluationsdaten
│   ├── Confirmation/        aktuelle Confirmation-Daten
│   ├── Intent-Training/     aktuelle Intent-Trainingsdaten (inkl. Development/Evaluation/Training/Provenance)
│   └── Settings-Parser/     aktuelle Settings-Parser-Daten (inkl. Provenienz)
├── legacy/                  alte Intent-, BiLSTM- und Validierungsdaten
│   ├── Intent/              alte Intent- und BiLSTM-Daten
│   └── Settings-Parser/     alte Settings-Parser-Daten
└── nicht_aktuell/           zurzeit nicht verwendete Daten
    ├── ASR-Generierung-und-QC/   ASR-Transkripte (generated), Quellen (sources) und Zwischenstände
    ├── ASR-TTS-Pipeline/         Piper-, TTS- und Vosk-Erzeugungsdaten
    └── Intent/                   Kandidaten, Rohdaten, Duplikate, Previews und ASR-Zwischenstände

02_Diagnosen/
├── ASR-Generierung-und-QC/  QC-Berichte, Manifests und Reviews zur ASR-Erzeugung
├── ASR-TTS-Pipeline/        Manifests und Reviews der Generation-1/2-Pipelines
├── Dokumentation/           Intent-Reports und Plots
│   └── Intent/              Intent-Classifier-Entwicklungsdokumentation
└── Intent/                  Intent-Analysen und Kandidaten (Review-Required, ChatGPT-Neue-Trainingsdaten)

03_Skripte/
├── ASR-Generierung-und-QC/  Skripte und Konfiguration zur ASR-Erzeugung/-Qualitätskontrolle
├── ASR-TTS-Pipeline/        Pipeline-Skripte der Generationen 1/2
└── Modelltraining/          Trainingsskripte (Intent-CNN etc.)
```
