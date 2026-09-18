package com.algorithmic_alliance.eyeaiapp

import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import com.algorithmic_alliance.eyeaiapp.rel2abs.Rel2AbsFrameCache
import uniffi.NativeLib.UniffiDetectedObject
import java.util.concurrent.atomic.AtomicReference

object AIModelData {
	val detectedObjects = AtomicReference<Array<UniffiDetectedObject>?>()
	val ocrBoxes = AtomicReference<Array<TextBoundingBox>?>()
	val depthEstimationData = AtomicReference<NativeLib.NativeFloatBuffer>()
	val segmentationOutput = AtomicReference<NativeLib.NativeIntBuffer>()
	val segmentationClassImportances = AtomicReference<List<Float>>()
	/** Exact source-frame pairing for spoken object-distance queries. */
	val rel2AbsFrameCache = Rel2AbsFrameCache()
}
