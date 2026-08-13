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
