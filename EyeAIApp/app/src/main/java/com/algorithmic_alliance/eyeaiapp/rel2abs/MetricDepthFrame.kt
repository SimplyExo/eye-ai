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

	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as MetricDepthFrame

		if (width != other.width) return false
		if (height != other.height) return false
		if (sourceTimestampNanos != other.sourceTimestampNanos) return false
		if (sourceWidth != other.sourceWidth) return false
		if (sourceHeight != other.sourceHeight) return false
		if (rotationDegrees != other.rotationDegrees) return false
		if (!depthMeters.contentEquals(other.depthMeters)) return false
		if (rel2absMode != other.rel2absMode) return false
		if (cameraIntrinsics != other.cameraIntrinsics) return false
		if (neuralGateRunner != other.neuralGateRunner) return false

		return true
	}

	override fun hashCode(): Int {
		var result = width
		result = 31 * result + height
		result = 31 * result + sourceTimestampNanos.hashCode()
		result = 31 * result + sourceWidth
		result = 31 * result + sourceHeight
		result = 31 * result + rotationDegrees
		result = 31 * result + depthMeters.contentHashCode()
		result = 31 * result + rel2absMode.hashCode()
		result = 31 * result + (cameraIntrinsics?.hashCode() ?: 0)
		result = 31 * result + (neuralGateRunner?.hashCode() ?: 0)
		return result
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

	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as DetectionFrame

		if (sourceTimestampNanos != other.sourceTimestampNanos) return false
		if (sourceWidth != other.sourceWidth) return false
		if (sourceHeight != other.sourceHeight) return false
		if (rotationDegrees != other.rotationDegrees) return false
		if (!detections.contentEquals(other.detections)) return false
		if (!objectContextFeatures.contentEquals(other.objectContextFeatures)) return false

		return true
	}

	override fun hashCode(): Int {
		var result = sourceTimestampNanos.hashCode()
		result = 31 * result + sourceWidth
		result = 31 * result + sourceHeight
		result = 31 * result + rotationDegrees
		result = 31 * result + detections.contentHashCode()
		result = 31 * result + objectContextFeatures.contentHashCode()
		return result
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
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as SegmentationContextFrame

		if (sourceTimestampNanos != other.sourceTimestampNanos) return false
		if (sourceWidth != other.sourceWidth) return false
		if (sourceHeight != other.sourceHeight) return false
		if (rotationDegrees != other.rotationDegrees) return false
		if (available != other.available) return false
		if (!segmentationFeatures.contentEquals(other.segmentationFeatures)) return false
		if (!globalAreaFractions.contentEquals(other.globalAreaFractions)) return false
		if (!gridAreaFractions.contentEquals(other.gridAreaFractions)) return false

		return true
	}

	override fun hashCode(): Int {
		var result = sourceTimestampNanos.hashCode()
		result = 31 * result + sourceWidth
		result = 31 * result + sourceHeight
		result = 31 * result + rotationDegrees
		result = 31 * result + available.hashCode()
		result = 31 * result + segmentationFeatures.contentHashCode()
		result = 31 * result + globalAreaFractions.contentHashCode()
		result = 31 * result + gridAreaFractions.contentHashCode()
		return result
	}
}

data class MatchedMetricFrame(
	val metricDepth: MetricDepthFrame,
	val detectionFrame: DetectionFrame,
	val contextFeatures: FloatArray = FloatArray(Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT),
	val segmentationContext: SegmentationContextFrame? = null,
) {
	override fun equals(other: Any?): Boolean {
		if (this === other) return true
		if (javaClass != other?.javaClass) return false

		other as MatchedMetricFrame

		if (metricDepth != other.metricDepth) return false
		if (detectionFrame != other.detectionFrame) return false
		if (!contextFeatures.contentEquals(other.contextFeatures)) return false
		if (segmentationContext != other.segmentationContext) return false

		return true
	}

	override fun hashCode(): Int {
		var result = metricDepth.hashCode()
		result = 31 * result + detectionFrame.hashCode()
		result = 31 * result + contextFeatures.contentHashCode()
		result = 31 * result + (segmentationContext?.hashCode() ?: 0)
		return result
	}
}
