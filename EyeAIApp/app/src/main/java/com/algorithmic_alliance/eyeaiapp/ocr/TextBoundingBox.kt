package com.algorithmic_alliance.eyeaiapp.ocr

import android.graphics.Rect

data class TextBoundingBox(
	val text: String,
	val width: Float,    // Von 0 bis 1!
	val height: Float,
	val x1: Float,
	val y1: Float,
	val x2: Float,
	val y2: Float,
	val bounding: Rect
)