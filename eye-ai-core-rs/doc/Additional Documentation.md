# Additional Documentation

- [Eye AI Readme](additional_documentation::eye_ai_readme)
- [eye-ai-core-rs Readme](additional_documentation::eye_ai_core_rs_readme)
- [Google AI Studio](additional_documentation::google_ai_studio_readme)
- [OCR](additional_documentation::ocr_readme)
- [Spatial Audio](additional_documentation::spatial_audio_readme)
- [Speech Recognition](additional_documentation::speech_recognition_readme)
- [Statemachine](additional_documentation::state_machine_readme)
- [TTS-Engine](additional_documentation::tts_engine_readme)

#### (How to generate this documentation site? for developers only)

```bash
cargo doc --no-deps
```

The resulting static doc site files will be generated to `target/doc/`.

Simply copy this directory into the gh-pages repo.

Leave the root index.html file as is, dont remove it,
it automatically redirects to `eye_ai_core_rs/index.html`
