package com.algorithmic_alliance.eyeaiapp.rel2abs

import java.util.Arrays
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import uniffi.NativeLib.UniffiDetectedObject

/** Resolves an existing normalized YOLO box to the median current metric depth. */
object MetricDistanceResolver {
	const val MIN_VALID_ROI_PIXELS = 16

	sealed class Result {
		data class Available(val meters: Float, val validPixelCount: Int) : Result()
		object Unavailable : Result()
		object Invalid : Result()
	}

	fun resolve(
		box: UniffiDetectedObject,
		frame: MetricDepthFrame,
		contextFeatures: FloatArray = FloatArray(Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT),
		detections: Array<UniffiDetectedObject> = emptyArray(),
		segmentationContext: SegmentationContextFrame? = null,
	): Result {
		if (frame.depthMeters.size != frame.width * frame.height) return Result.Invalid
		if (!box.x1.isFinite() || !box.y1.isFinite() || !box.x2.isFinite() || !box.y2.isFinite()) {
			return Result.Invalid
		}

		val left = min(box.x1, box.x2).coerceIn(0f, 1f)
		val right = max(box.x1, box.x2).coerceIn(0f, 1f)
		val top = min(box.y1, box.y2).coerceIn(0f, 1f)
		val bottom = max(box.y1, box.y2).coerceIn(0f, 1f)
		if (right <= left || bottom <= top) return Result.Invalid

		val x0 = floor(left * frame.width).toInt().coerceIn(0, frame.width - 1)
		val y0 = floor(top * frame.height).toInt().coerceIn(0, frame.height - 1)
		val x1 = ceil(right * frame.width).toInt().coerceIn(x0 + 1, frame.width)
		val y1 = ceil(bottom * frame.height).toInt().coerceIn(y0 + 1, frame.height)
		val values = FloatArray((x1 - x0) * (y1 - y0))
		var count = 0
		for (y in y0 until y1) {
			for (x in x0 until x1) {
				val value = frame.depthMeters[y * frame.width + x]
				if (value.isFinite() && value > 0f) values[count++] = value
			}
		}
		if (count < MIN_VALID_ROI_PIXELS) return Result.Unavailable

		Arrays.sort(values, 0, count)
		val median = if (count % 2 == 0) {
			(values[count / 2 - 1] + values[count / 2]) / 2f
		} else {
			values[count / 2]
		}
		if (!median.isFinite() || median <= 0f) return Result.Invalid
		val neuralEstimate = frame.neuralGateRunner?.resolve(
			mode = frame.rel2absMode,
			box = box,
			visualMeters = median,
				frame = frame,
				contextFeatures = contextFeatures,
				detections = detections,
				segmentationContext = segmentationContext,
			)
		return Result.Available(neuralEstimate ?: median, count)
	}
}
