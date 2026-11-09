package com.algorithmic_alliance.eyeaiapp.audio

/** Preference values are stable and match the non-translatable settings options. */
enum class AudioOutputMode(val preferenceValue: String) {
    STANDARD("Normal"), OBJECT_WATCH("Objekt-Watch"), R4("Szenenbeschreibung (R4)");

    fun spatialAudioEnabled(requested: Boolean): Boolean = this != R4 && requested

    companion object {
        fun fromStoredValue(value: String?): AudioOutputMode =
            entries.firstOrNull { it.preferenceValue == value } ?: STANDARD
    }
}
