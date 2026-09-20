package com.algorithmic_alliance.eyeaiapp.rel2abs

import uniffi.NativeLib.UniffiDetectedObject

/** Optional focal lengths in the coordinate system of the source bitmap. */
data class Rel2AbsCameraIntrinsics(
	val fxPx: Float,
	val fyPx: Float,
) {
	init {
		require(fxPx.isFinite() && fyPx.isFinite() && fxPx > 0f && fyPx > 0f) {
			"REL2ABS focal lengths must be finite and positive"
		}
	}
}

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
	val cameraIntrinsics: Rel2AbsCameraIntrinsics? = null,
	val neuralGateRunner: V6NeuralGateRunner? = null,
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
	val objectContextFeatures: FloatArray = Rel2AbsContextFeatures.objectFeatures(detections),
) {
	init {
		require(sourceWidth > 0 && sourceHeight > 0) { "Source dimensions must be positive" }
	}
}

/** Timestamped semantic summary produced by the existing segmentation model. */
data class SegmentationContextFrame(
	val segmentationFeatures: FloatArray,
	val sourceTimestampNanos: Long,
	val sourceWidth: Int,
	val sourceHeight: Int,
	val rotationDegrees: Int,
	val globalAreaFractions: FloatArray = FloatArray(Rel2AbsContextFeatures.SEMANTIC_CLASS_COUNT),
	val gridAreaFractions: FloatArray = FloatArray(Rel2AbsContextFeatures.SEGMENTATION_GRID_FEATURE_COUNT),
	val available: Boolean = false,
)

data class MatchedMetricFrame(
	val metricDepth: MetricDepthFrame,
	val detectionFrame: DetectionFrame,
	val contextFeatures: FloatArray = FloatArray(Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT),
	val segmentationContext: SegmentationContextFrame? = null,
)
