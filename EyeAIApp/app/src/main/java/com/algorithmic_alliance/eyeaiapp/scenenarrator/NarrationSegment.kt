package com.algorithmic_alliance.eyeaiapp.scenenarrator

data class NarrationSegment(val text: String, val objectIds: List<Int>,
    val horizontalCenter: Float = .5f, val distanceMeters: Float? = null)
