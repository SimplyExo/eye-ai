# NLP V2 Android assets

This directory contains the eight representative BaselineCNN models from the
five-seed M0–M3 × T1/T2 experiment. Each model has a fixed `int32[1,24]` input
and a Float32 ten-class output.

The models share one tokenizer artifact per tokenizer family:

- `T1`: deterministic word tokenizer
- `T2`: deterministic word-boundary BPE tokenizer with 2,000 tokens

The representative seed for each combination is the run closest to that
combination's five-seed development mean. No production winner was selected by
the experiment. The app therefore exposes all eight models and uses `M0_T1`
as the same neutral comparison default as the desktop evaluation app.

Model selection is available under **Settings → Speech Recognition → NLP V2
Baseline Model**.

## Runtime contract

The production path loads exactly one selected model and its matching frozen
tokenizer. The default is `M0_T1` (`m0_t1_seed_20260812.tflite` with the T1
word tokenizer). Every tokenizer configuration is validated at load time,
including normalization, vocabulary checksum, reserved IDs, post-padding,
post-truncation, and `max_length = 24`.

The class-index order is fixed and shared by `labels.json`, `Intent`, and
`IntentResult.probabilities`:

1. `TEXT_RECOGNITION`
2. `OBJECT_DETECTION`
3. `CHANGE_SPEECH_SPEED`
4. `CHANGE_SPEAKER`
5. `REDIRECT_TO_LLM`
6. `OPEN_SETTINGS`
7. `SET_FREQUENCY`
8. `SET_BPS`
9. `MEASURE_DISTANCE`
10. `ABORT`

`NLPModel.classify` performs one inference and returns the top-1 intent,
confidence, unchanged Vosk text, and all ten probabilities. The StateMachine
keeps this unfiltered result while preserving its existing confidence and
settings routing behavior.

Run `python3 scripts/verify_nlp_v2.py` from `EyeAIApp` to validate the frozen
artifacts and LiteRT inference. Android tokenizer parity is covered by
`FrozenTokenizerParityTest` using vectors exported from the Python rules.
