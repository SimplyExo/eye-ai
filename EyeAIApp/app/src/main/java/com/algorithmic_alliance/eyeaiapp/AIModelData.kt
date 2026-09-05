package com.algorithmic_alliance.eyeaiapp

import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import com.algorithmic_alliance.eyeaiapp.camera.AnalysisResults
import java.util.concurrent.atomic.AtomicReference

object AIModelData {
	val analysisResults = AtomicReference(AnalysisResults())
	val ocrBoxes = AtomicReference<Array<TextBoundingBox>?>()
}
