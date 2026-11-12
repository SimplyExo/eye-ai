package com.algorithmic_alliance.eyeaiapp.nlp

import com.algorithmic_alliance.eyeaiapp.audio.AudioOutputMode

/** Router order is independent of the ten legacy intent classes. */
object ModeRouterDecision {
    const val THRESHOLD = 0.3971867859363556f
    val LABELS = listOf("MODE_NORMAL", "MODE_OBJECT_WATCH", "MODE_SCENE_DESCRIPTION", "NO_MODE")
    fun select(probabilities: FloatArray, threshold: Float = THRESHOLD): AudioOutputMode? {
        require(probabilities.size == 4 && probabilities.all { it.isFinite() && it in 0f..1f })
        require(kotlin.math.abs(probabilities.sum() - 1f) <= 0.001f)
        val top = probabilities.indices.maxBy { probabilities[it] }
        if (top == 3 || probabilities[top] < threshold) return null
        return listOf(AudioOutputMode.STANDARD, AudioOutputMode.OBJECT_WATCH, AudioOutputMode.R4)[top]
    }
}
