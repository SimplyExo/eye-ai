# TTS-Engine

We are using Androids built-in TTS Engine.
Speak API is implemented as usual and is documented in more detail in TextToSpeechInstance.kt

## TextToSpeechInstance.kt

### awaitSilence()

Used to create a consistent silence after speaking

### invokeOnFinishedWhenPlaybackStops()

calls awaitSilence() threadsafe.


