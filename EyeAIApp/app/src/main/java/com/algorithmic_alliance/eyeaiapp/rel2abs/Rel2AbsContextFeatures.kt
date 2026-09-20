package com.algorithmic_alliance.eyeaiapp.rel2abs

import com.algorithmic_alliance.eyeaiapp.NativeLib
import kotlin.math.ln
import uniffi.NativeLib.UniffiDetectedObject

/** Runtime counterpart of the frozen V6 frame-context feature contract. */
object Rel2AbsContextFeatures {
	const val OBJECT_FEATURE_COUNT = 9
	const val SEGMENTATION_FEATURE_COUNT = 11
	const val TOTAL_FEATURE_COUNT = OBJECT_FEATURE_COUNT + SEGMENTATION_FEATURE_COUNT
	const val SEMANTIC_CLASS_COUNT = 19
	const val SEGMENTATION_GRID_SIZE = 4
	const val SEGMENTATION_GRID_FEATURE_COUNT =
		SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE * SEMANTIC_CLASS_COUNT

	private val supportedClasses = setOf("person", "bicycle", "car")
	private const val ROAD = 0
	private const val SIDEWALK = 1
	private const val BUILDING = 2
	private const val VEGETATION = 8
	private const val SKY = 10
	private const val PERSON = 11
	private const val RIDER = 12
	private const val CAR = 13
	private const val TRUCK = 14
	private const val BUS = 15
	private const val TRAIN = 16
	private const val MOTORCYCLE = 17
	private const val BICYCLE = 18
	data class SegmentationFeatureContext(
		val globalAreaFractions: FloatArray,
		val gridAreaFractions: FloatArray,
		val available: Boolean,
	)

	fun objectFeatures(detections: Array<UniffiDetectedObject>): FloatArray {
		val areas = FloatArray(detections.size)
		val heights = FloatArray(detections.size)
		val confidences = FloatArray(detections.size)
		val classes = HashSet<String>()
		var supportedCount = 0
		var borderHits = 0
		for ((index, detection) in detections.withIndex()) {
			val left = minOf(detection.x1, detection.x2).coerceIn(0f, 1f)
			val right = maxOf(detection.x1, detection.x2).coerceIn(0f, 1f)
			val top = minOf(detection.y1, detection.y2).coerceIn(0f, 1f)
			val bottom = maxOf(detection.y1, detection.y2).coerceIn(0f, 1f)
			val width = (right - left).coerceAtLeast(0f)
			val height = (bottom - top).coerceAtLeast(0f)
			areas[index] = width * height
			heights[index] = height
			confidences[index] = confidenceOrZero(detection)
			val className = detection.clsName
			classes += className
			if (className in supportedClasses) supportedCount++
			if (left <= 0.01f || right >= 0.99f || top <= 0.01f || bottom >= 0.99f) borderHits++
		}

		val count = detections.size.toFloat()
		return floatArrayOf(
			kotlin.math.ln(1.0 + count.toDouble()).toFloat(),
			kotlin.math.ln(1.0 + supportedCount.toDouble()).toFloat(),
			areas.averageOrZero(),
			areas.maxOrNull() ?: 0f,
			heights.averageOrZero(),
			heights.maxOrNull() ?: 0f,
			confidences.averageOrZero(),
			classes.size.toFloat(),
			borderHits.toFloat() / count.coerceAtLeast(1f),
		)
	}

	/** Converts the existing 19-class semantic output into the frozen V6 summary. */
	fun segmentationFeatures(prediction: NativeLib.NativeIntBuffer): FloatArray {
		val areas = IntArray(SEMANTIC_CLASS_COUNT)
		val buffer = prediction.intBuffer
		for (index in 0 until buffer.capacity()) {
			val classIndex = buffer.get(index)
			if (classIndex in areas.indices) areas[classIndex]++
		}
		val total = areas.sum().toFloat()
		if (total <= 0f) return FloatArray(SEGMENTATION_FEATURE_COUNT)
		val fractions = FloatArray(areas.size) { areas[it] / total }
		val entropy = fractions.filter { it > 0f }.sumOf { value -> (-value * ln(value.toDouble())).toDouble() }.toFloat()
		return floatArrayOf(
			1f,
			areas.count { it > 0 }.toFloat(),
			fractions.maxOrNull() ?: 0f,
			entropy,
			areaFor(fractions, ROAD),
			areaFor(fractions, SIDEWALK),
			areaFor(fractions, BUILDING),
			areaFor(fractions, VEGETATION),
			areaFor(fractions, SKY),
			areaFor(fractions, PERSON) + areaFor(fractions, RIDER),
			areaFor(fractions, CAR) + areaFor(fractions, TRUCK) + areaFor(fractions, BUS) +
				areaFor(fractions, TRAIN) + areaFor(fractions, MOTORCYCLE) + areaFor(fractions, BICYCLE),
		)
	}

	/**
	 * Keeps the same global and 4x4 semantic-area contract used by the V6
	 * object-feature gate training pipeline.
	 */
	fun segmentationFeatureContext(
		prediction: NativeLib.NativeIntBuffer,
		width: Int,
		height: Int,
	): SegmentationFeatureContext {
		if (width <= 0 || height <= 0) {
			return SegmentationFeatureContext(
				FloatArray(SEMANTIC_CLASS_COUNT),
				FloatArray(SEGMENTATION_GRID_FEATURE_COUNT),
				false,
			)
		}

		val pixelCount = minOf(prediction.intBuffer.capacity(), width * height)
		if (pixelCount <= 0) {
			return SegmentationFeatureContext(
				FloatArray(SEMANTIC_CLASS_COUNT),
				FloatArray(SEGMENTATION_GRID_FEATURE_COUNT),
				false,
			)
		}

		val globalCounts = FloatArray(SEMANTIC_CLASS_COUNT)
		val gridCounts = FloatArray(SEGMENTATION_GRID_FEATURE_COUNT)
		val gridPixelCounts = IntArray(SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE)
		val buffer = prediction.intBuffer
		for (index in 0 until pixelCount) {
			val y = index / width
			val x = index - y * width
			val gridY = (y * SEGMENTATION_GRID_SIZE / height).coerceIn(0, SEGMENTATION_GRID_SIZE - 1)
			val gridX = (x * SEGMENTATION_GRID_SIZE / width).coerceIn(0, SEGMENTATION_GRID_SIZE - 1)
			val cell = gridY * SEGMENTATION_GRID_SIZE + gridX
			gridPixelCounts[cell]++
			val classIndex = buffer.get(index)
			if (classIndex in 0 until SEMANTIC_CLASS_COUNT) {
				globalCounts[classIndex]++
				gridCounts[cell * SEMANTIC_CLASS_COUNT + classIndex]++
			}
		}

		val denominator = pixelCount.toFloat()
		val global = FloatArray(SEMANTIC_CLASS_COUNT) { index -> globalCounts[index] / denominator }
		val grid = FloatArray(SEGMENTATION_GRID_FEATURE_COUNT)
		for (cell in gridPixelCounts.indices) {
			val cellDenominator = gridPixelCounts[cell].coerceAtLeast(1).toFloat()
			for (classIndex in 0 until SEMANTIC_CLASS_COUNT) {
				grid[cell * SEMANTIC_CLASS_COUNT + classIndex] =
					gridCounts[cell * SEMANTIC_CLASS_COUNT + classIndex] / cellDenominator
			}
		}
		return SegmentationFeatureContext(global, grid, global.sum() > 0f)
	}

	private fun areaFor(fractions: FloatArray, index: Int): Float = fractions.getOrElse(index) { 0f }

	private fun FloatArray.averageOrZero(): Float = if (isEmpty()) 0f else average().toFloat()

	fun confidenceOrZero(detection: UniffiDetectedObject): Float = detection.cnf.takeIf { it.isFinite() } ?: 0f
}
