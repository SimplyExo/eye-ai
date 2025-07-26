package com.algorithmic_alliance.eyeaiapp

import com.algorithmic_alliance.eyeaiapp.object_detection.BoundingBox
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import java.util.concurrent.atomic.AtomicReference

object AIModelData {
	val objectDetectionBoxes = AtomicReference<Array<BoundingBox>>()
	val ocrBoxes = AtomicReference<Array<TextBoundingBox>>()
	val depthEstimationData = AtomicReference<FloatArray>()
}