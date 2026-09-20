package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects

import com.algorithmic_alliance.eyeaiapp.AIModelData
import com.algorithmic_alliance.eyeaiapp.rel2abs.MetricDistanceResolver
import uniffi.NativeLib.UniffiDetectedObject

class ObjectDetectionHandler {
	companion object {
		data class DetectedObject(
			val label: String,
			val distance: Float,
			val height: Float,
			val width: Float,
			val x: Float,
			val y: Float
		)

		private data class RecognizedObject(
			val label: String,
			val box: UniffiDetectedObject,
		)

		fun getGermanObjectLabels(): List<String> {
			val objectDetectionBoxes = AIModelData.detectedObjects.get()

			if (objectDetectionBoxes.isNullOrEmpty()) {
				return emptyList()
			}

			return objectDetectionBoxes.mapNotNull { box ->
				val englishLabel = box.clsName
				if (TranslateEnglishToGerman.isKnownEnglishLabel(englishLabel)) {
					TranslateEnglishToGerman.translateToGerman(englishLabel)
				} else null
			}.distinct()
		}

		fun handleGermanObjectQuery(germanQuery: String): ObjectDetectionResult {
			if (germanQuery.isBlank()) {
				return ObjectDetectionResult.NoQueryProvided
			}

			val paired = AIModelData.rel2AbsFrameCache.latestMatched()
			if (paired == null) {
				return if (AIModelData.detectedObjects.get().isNullOrEmpty()) {
					ObjectDetectionResult.NoObjectsFound
				} else {
					// A detection without a metric map from the same source frame is
					// intentionally not combined with a stale or asynchronous depth map.
					ObjectDetectionResult.DepthDataUnavailable
				}
			}

			if (paired.detectionFrame.detections.isEmpty()) return ObjectDetectionResult.NoObjectsFound
			val recognized = paired.detectionFrame.detections.mapNotNull { box ->
				box.clsName.takeIf(TranslateEnglishToGerman::isKnownEnglishLabel)?.let {
					RecognizedObject(TranslateEnglishToGerman.translateToGerman(it), box)
				}
			}
			if (recognized.isEmpty()) return ObjectDetectionResult.NoKnownObjectsFound

			val query = germanQuery.lowercase()
			val found = recognized.find { candidate ->
				val label = candidate.label.lowercase()
				label == query || label.contains(query) || query.contains(label)
			} ?: return ObjectDetectionResult.ObjectNotFound(
				recognized.map { it.label }.distinct().take(5),
			)

			return when (val result = MetricDistanceResolver.resolve(
				box = found.box,
				frame = paired.metricDepth,
				contextFeatures = paired.contextFeatures,
				detections = paired.detectionFrame.detections,
				segmentationContext = paired.segmentationContext,
			)) {
				is MetricDistanceResolver.Result.Available -> ObjectDetectionResult.ObjectFound(
					DetectedObject(
						label = found.label,
						distance = result.meters,
						height = found.box.h,
						width = found.box.w,
						x = found.box.cx,
						y = found.box.cy,
					),
				)

				MetricDistanceResolver.Result.Unavailable -> ObjectDetectionResult.DepthDataUnavailable
				MetricDistanceResolver.Result.Invalid -> ObjectDetectionResult.DepthDataInvalid
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
