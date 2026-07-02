package com.algorithmic_alliance.eyeaiapp

import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicReference

object AIModelData {
	val detectedObjects = AtomicReference<Array<UniffiDetectedObject>?>()
	val ocrBoxes = AtomicReference<Array<TextBoundingBox>?>()
	val depthEstimationData = AtomicReference<NativeLib.NativeFloatBuffer>()
}