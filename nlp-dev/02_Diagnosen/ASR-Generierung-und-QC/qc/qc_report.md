# ASR generation QC

Generation-2 TTS → synthetic noise/reverb augmentation → Vosk was run once
per source row. Empty outputs remain represented in the QC and are not silently
used for training/evaluation. Nonempty flagged outputs remain available as ASR
data; flags are diagnostic, not an automatic semantic filter.

| split | source rows | nonempty | OK | TRUNCATED | HEAVILY_CORRUPTED | EMPTY |
|---|---:|---:|---:|---:|---:|---:|
| train | 32 | 32 | 26 | 0 | 6 | 0 |
| validation | 66 | 65 | 53 | 0 | 12 | 1 |
| blind | 120 | 120 | 112 | 1 | 7 | 0 |

## Output files

- `train/asr_train_transcripts.txt` and `.gold` - row-preserving output
- `train/asr_train_transcripts.labeled.txt` - nonempty ASR rows for model use
- `train/qc.csv` - sample-level provenance and status
- `validation/asr_validation_transcripts.txt` and `.gold` - row-preserving output
- `validation/asr_validation_transcripts.labeled.txt` - nonempty ASR rows for model use
- `validation/qc.csv` - sample-level provenance and status
- `blind/asr_blind_transcripts.txt` and `.gold` - row-preserving output
- `blind/asr_blind_transcripts.labeled.txt` - nonempty ASR rows for model use
- `blind/qc.csv` - sample-level provenance and status
