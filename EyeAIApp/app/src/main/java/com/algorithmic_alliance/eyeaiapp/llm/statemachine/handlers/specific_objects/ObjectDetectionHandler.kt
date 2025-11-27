package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.AIModelData
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.llm.LLM

class ObjectDetectionHandler() {
	companion object {
		private const val DEPTH_WIDTH = 256
		private const val DEPTH_HEIGHT = 256

		data class DetectedObject(
			val label: String,
			val distance: Float,
			val height: Float,
			val width: Float,
			val x: Float,
			val y: Float
		)


		fun getGermanObjectLabelsForLLM(): List<String> {
			val objectDetectionBoxes = AIModelData.detectedObjects.get()

			if (objectDetectionBoxes.isNullOrEmpty()) {
				return emptyList()
			}

			return objectDetectionBoxes
				.mapNotNull { box ->
					val englishLabel = box.clsName
					if (englishLabel in LLM.Companion.knownObjectLabels) {
						TranslateEnglishToGerman.Companion.translateToGerman(englishLabel)
					} else null
				}
				.distinct()
		}

		fun handleGermanObjectQuery(germanQuery: String): ObjectDetectionResult {
			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "handleGermanObjectQuery strating")
			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Input query: '$germanQuery'")

			if (germanQuery.isBlank()) {
				Log.w(EyeAIApp.Companion.APP_LOG_TAG, "Query is blank - returning NoQueryProvided")
				return ObjectDetectionResult.NoQueryProvided
			}

			val objectDetectionBoxes = AIModelData.detectedObjects.get()
			val depthEstimationData = AIModelData.depthEstimationData.get()

			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Object detection boxes: ${objectDetectionBoxes?.size}")
			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Depth data size: ${depthEstimationData?.capacity()}")

			if (objectDetectionBoxes.isNullOrEmpty()) {
				Log.w(EyeAIApp.Companion.APP_LOG_TAG, "No object detection boxes - returning NoObjectsFound")
				return ObjectDetectionResult.NoObjectsFound
			}

			/*if (depthEstimationData.isEmpty()) {
				Log.w(EyeAIApp.Companion.APP_LOG_TAG, "Depth data is empty - returning DepthDataUnavailable")
				return ObjectDetectionResult.DepthDataUnavailable
			}*/

			if (depthEstimationData.capacity() != DEPTH_WIDTH * DEPTH_HEIGHT) {
				Log.w(EyeAIApp.Companion.APP_LOG_TAG, "Depth data has unexpected size: ${depthEstimationData.capacity()} (expected: ${DEPTH_WIDTH * DEPTH_HEIGHT})")
				return ObjectDetectionResult.DepthDataInvalid
			}

			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Processing ${objectDetectionBoxes.size} detected objects...")

			val detectedObjects = objectDetectionBoxes.mapNotNull { box ->
				val englishLabel = box.clsName
				Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Processing box with English label: '$englishLabel'")

				if (englishLabel in LLM.Companion.knownObjectLabels) {
					val germanLabel =
						TranslateEnglishToGerman.Companion.translateToGerman(englishLabel)
					Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Translated '$englishLabel' to '$germanLabel'")

					val depthX = (box.cx * (DEPTH_WIDTH - 1)).toInt().coerceIn(0, DEPTH_WIDTH - 1)
					val depthY = (box.cy * (DEPTH_HEIGHT - 1)).toInt().coerceIn(0, DEPTH_HEIGHT - 1)
					val depthIndex = depthY * DEPTH_WIDTH + depthX

					val distance = if (depthIndex < depthEstimationData.capacity()) {
						depthEstimationData[depthIndex]
					} else {
						Log.w(EyeAIApp.Companion.APP_LOG_TAG, "Depth-Index out of bounds")
						-1f
					}

					Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Created DetectedObject: label='$germanLabel', distance=$distance")
					DetectedObject(germanLabel, distance, box.h, box.w, box.cx, box.cy)
				} else {
					Log.d(EyeAIApp.Companion.APP_LOG_TAG, "English label '$englishLabel' not in knownObjectLabels - skipping")
					null
				}
			}

			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Total detected objects after filtering: ${detectedObjects.size}")
			detectedObjects.forEach { obj ->
				Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Detected object: ${obj.label}")
			}

			if (detectedObjects.isEmpty()) {
				Log.w(EyeAIApp.Companion.APP_LOG_TAG, "No known objects found - returning NoKnownObjectsFound")
				return ObjectDetectionResult.NoKnownObjectsFound
			}

			Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Searching for German query: '$germanQuery' (lowercase: '${germanQuery.lowercase()}')")

			val foundObject = detectedObjects.find { obj ->
				val objLabel = obj.label.lowercase()
				val queryLower = germanQuery.lowercase()

				val exactMatch = objLabel == queryLower
				val labelContainsQuery = objLabel.contains(queryLower)
				val queryContainsLabel = queryLower.contains(objLabel)

				Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Comparing '$objLabel' with '$queryLower': exact=$exactMatch, labelContains=$labelContainsQuery, queryContains=$queryContainsLabel")

				exactMatch || labelContainsQuery || queryContainsLabel
			}

			return if (foundObject != null) {
				Log.d(EyeAIApp.Companion.APP_LOG_TAG, "FOUND object: ${foundObject.label} at ${foundObject.distance}m - returning ObjectFound")
				ObjectDetectionResult.ObjectFound(foundObject)
			} else {
				val availableObjects = detectedObjects.map { it.label }.distinct().take(5)
				Log.d(EyeAIApp.Companion.APP_LOG_TAG, "Object '$germanQuery' NOT found. Available objects: $availableObjects - returning ObjectNotFound")
				ObjectDetectionResult.ObjectNotFound(availableObjects)
			}
		}
	}

	sealed class ObjectDetectionResult {
		object NoObjectsFound : ObjectDetectionResult()
		object NoQueryProvided : ObjectDetectionResult()
		object DepthDataUnavailable : ObjectDetectionResult()
		object DepthDataInvalid : ObjectDetectionResult()
		object NoKnownObjectsFound : ObjectDetectionResult()
		data class ObjectFound(val obj: DetectedObject) : ObjectDetectionResult()
		data class ObjectNotFound(val availableObjects: List<String>) : ObjectDetectionResult()
	}
}