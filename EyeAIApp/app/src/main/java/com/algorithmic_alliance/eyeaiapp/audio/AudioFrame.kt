package com.algorithmic_alliance.eyeaiapp.audio

/** Canonical PCM representation for a future external audio source adapter. */
class AudioFrame(
    pcm16: ByteArray,
    val sampleRateHz: Int,
    val channelCount: Int,
    val timestampNanos: Long,
) {
    /** A defensive copy makes source-buffer ownership explicit. */
    val pcm16: ByteArray = pcm16.copyOf()

    init {
        require(sampleRateHz > 0) { "Audio sample rate must be positive" }
        require(channelCount > 0) { "Audio channel count must be positive" }
        require(pcm16.size % 2 == 0) { "PCM-16 audio must contain complete samples" }
    }
}

/**
 * Minimal source boundary for a future WebRTC audio adapter.
 * The current local Android microphone continues to be owned by Vosk's
 * SpeechService; no external/network implementation is installed here.
 */
fun interface AudioFrameSink {
    fun submit(frame: AudioFrame): Boolean
}
