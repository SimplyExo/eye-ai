package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.AIModelData.depthEstimationData
import com.algorithmic_alliance.eyeaiapp.AIModelData.objectDetectionBoxes
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.llm.LLM.Companion.knownObjectLabels

class ObjectDetectionHandler() {
	companion object {
		private const val DEPTH_WIDTH = 256
		private const val DEPTH_HEIGHT = 256
	}

	data class DetectedObject(
		val label: String,
		val distance: Float,
		val height: Float,
		val width: Float,
		val x: Float,
		val y: Float
	)

	fun handleObjectQuery(specificQuery: String): ObjectDetectionResult {
		if (specificQuery.isBlank()) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Objekterkennung ohne spezifische Anfrage aufgerufen")
			return ObjectDetectionResult.NoQueryProvided
		}

		val objectDetectionBoxes = objectDetectionBoxes.get()
		val depthEstimationData = depthEstimationData.get()

		Log.d(EyeAIApp.APP_LOG_TAG, "Searching for: '$specificQuery'")
		Log.d(EyeAIApp.APP_LOG_TAG, "Available objects: ${objectDetectionBoxes?.size}")
		Log.d(EyeAIApp.APP_LOG_TAG, "depth data: ${depthEstimationData?.size}")

		if (objectDetectionBoxes.isNullOrEmpty()) {
			return ObjectDetectionResult.NoObjectsFound
		}

		if (depthEstimationData.isEmpty()) {
			return ObjectDetectionResult.DepthDataUnavailable
		}

		if (depthEstimationData.size != DEPTH_WIDTH * DEPTH_HEIGHT) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Depth data has unexpected size: ${depthEstimationData.size}")
			return ObjectDetectionResult.DepthDataInvalid
		}


		val detectedObjects = objectDetectionBoxes.mapNotNull { box ->
			val label = box.clsName
			if (label in knownObjectLabels) {
				// Evaluating coordinates for getting the depth
				val depthX = (box.cx * (DEPTH_WIDTH - 1)).toInt().coerceIn(0, DEPTH_WIDTH - 1)
				val depthY = (box.cy * (DEPTH_HEIGHT - 1)).toInt().coerceIn(0, DEPTH_HEIGHT - 1)
				val depthIndex = depthY * DEPTH_WIDTH + depthX

				Log.d(EyeAIApp.APP_LOG_TAG, "Objekt '$label': Box(${box.cx}, ${box.cy}) -> Depth[$depthX, $depthY] = Index $depthIndex")

				val distance = if (depthIndex < depthEstimationData.size) {
					depthEstimationData[depthIndex]
				} else {
					Log.w(EyeAIApp.APP_LOG_TAG, "Depth-Index out of bounds")
					-1f
				}

				DetectedObject(label, distance, box.h, box.w, box.cx, box.cy)
			} else null
		}

		if (detectedObjects.isEmpty()) {
			return ObjectDetectionResult.NoKnownObjectsFound
		}

		val foundObject = detectedObjects.find { obj ->
			val objLabel = obj.label.lowercase()
			objLabel == specificQuery ||
				objLabel.contains(specificQuery) ||
				specificQuery.contains(objLabel)
		}

		return if (foundObject != null) {
			Log.d(EyeAIApp.APP_LOG_TAG, "object found: ${foundObject.label} at ${foundObject.distance}m")
			ObjectDetectionResult.ObjectFound(foundObject)
		} else {
			val availableObjects = detectedObjects.map { it.label }.distinct().take(5)
			Log.d(EyeAIApp.APP_LOG_TAG, "Object '$specificQuery' not found. Available objects: $availableObjects")
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
	data class ObjectFound(val obj: ObjectDetectionHandler.DetectedObject) : ObjectDetectionResult()
	data class ObjectNotFound(val availableObjects: List<String>) : ObjectDetectionResult()
}