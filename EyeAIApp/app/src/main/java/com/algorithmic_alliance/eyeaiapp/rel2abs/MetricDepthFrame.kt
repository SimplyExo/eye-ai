package com.algorithmic_alliance.eyeaiapp.rel2abs

import uniffi.NativeLib.UniffiDetectedObject

/** A single metric map produced from one logical source frame. */
data class MetricDepthFrame(
	val depthMeters: FloatArray,
	val width: Int,
	val height: Int,
	val sourceTimestampNanos: Long,
	val sourceWidth: Int,
	val sourceHeight: Int,
	val rotationDegrees: Int,
	val rel2absMode: Rel2AbsMode,
) {
	init {
		require(width > 0 && height > 0) { "Metric depth dimensions must be positive" }
		require(depthMeters.size == width * height) {
			"Metric depth size ${depthMeters.size} does not match ${width}x$height"
		}
		require(sourceWidth > 0 && sourceHeight > 0) { "Source dimensions must be positive" }
	}
}

/** Existing YOLO/ByteTrack output annotated with the source frame that produced it. */
data class DetectionFrame(
	val detections: Array<UniffiDetectedObject>,
	val sourceTimestampNanos: Long,
	val sourceWidth: Int,
	val sourceHeight: Int,
	val rotationDegrees: Int,
) {
	init {
		require(sourceWidth > 0 && sourceHeight > 0) { "Source dimensions must be positive" }
	}
}

data class MatchedMetricFrame(
	val metricDepth: MetricDepthFrame,
	val detectionFrame: DetectionFrame,
)
