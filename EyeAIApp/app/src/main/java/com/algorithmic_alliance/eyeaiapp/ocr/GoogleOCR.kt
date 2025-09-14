package com.algorithmic_alliance.eyeaiapp.ocr

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class GoogleOCR {
	private var ocrModel: TextRecognizer? = null

	// Hier speichern wir den erkannten Text + Position als String
	var lastResult: String = ""
		private set

	fun create() {
		if (ocrModel == null)
			ocrModel = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
	}

	suspend fun analyzeFrame(frame: Bitmap): List<TextBoundingBox> {
		if (ocrModel == null) {
			Log.e("OCR", "Failed to analyze frame, OCR Model not created yet!")
			return emptyList()
		}

		return suspendCoroutine { continuation ->
			val converted = InputImage.fromBitmap(frame, 0)
			ocrModel?.process(converted)
				?.addOnSuccessListener { visionText ->
					val tbb = ArrayList<TextBoundingBox>()
					val sb = StringBuilder()

					for (box in visionText.textBlocks) {
						val bounding = box.boundingBox!!
						val width = bounding.width().toFloat() / frame.width.toFloat()
						val height = bounding.height().toFloat() / frame.height.toFloat()

						val x1 = bounding.left.toFloat() / frame.width.toFloat()
						val y1 = bounding.top.toFloat() / frame.height.toFloat()
						val x2 = bounding.right.toFloat() / frame.width.toFloat()
						val y2 = bounding.bottom.toFloat() / frame.height.toFloat()

						// Liste für Overlay
						tbb.add(TextBoundingBox(box.text, width, height, x1, y1, x2, y2, bounding))

						// String-Ausgabe aufbauen
						sb.append("Text: \"${box.text}\" ")
						sb.append("(x1=$x1, y1=$y1, x2=$x2, y2=$y2, w=$width, h=$height)\n")
					}

					// Speichere den String global abrufbar
					lastResult = sb.toString().trim()

					continuation.resume(tbb)
				}
				?.addOnFailureListener { e ->
					continuation.resumeWithException(e)
				}
		}
	}
}
