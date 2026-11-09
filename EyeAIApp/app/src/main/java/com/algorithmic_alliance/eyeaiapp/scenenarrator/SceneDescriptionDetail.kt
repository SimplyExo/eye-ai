package com.algorithmic_alliance.eyeaiapp.scenenarrator

// Stable preference IDs. The previous long mode migrates to Normal.
enum class SceneDescriptionDetail {
    SHORT, STANDARD;
    companion object {
        fun fromStoredValue(value: String?): SceneDescriptionDetail =
            entries.firstOrNull {it.name.equals(value,ignoreCase=true)} ?: STANDARD
    }
}
