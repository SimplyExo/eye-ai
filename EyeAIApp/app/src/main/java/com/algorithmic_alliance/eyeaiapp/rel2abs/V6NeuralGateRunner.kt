package com.algorithmic_alliance.eyeaiapp.rel2abs

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import uniffi.NativeLib.UniffiDetectedObject

/**
 * Small frozen V6 object gate. It is deliberately separate from the Z1/S2
 * pixel-map runner: V6 only chooses between the visual ROI estimate and the
 * deterministic F1 size anchor.
 */
class V6NeuralGateRunner private constructor(
	private val models: Map<String, GateModel>,
) {
	private data class GateModel(
		val mean: FloatArray,
		val scale: FloatArray,
		val weight1: Array<FloatArray>,
		val bias1: FloatArray,
		val weight2: FloatArray,
		val bias2: Float,
		val withContext: Boolean,
		val objectFeatureGate: Boolean,
		val cameraHeightFeature: Boolean,
	) {
		fun evaluate(features: FloatArray): Float {
			val expectedBaseDim = mean.size
			check(features.size == expectedBaseDim) {
				"V6 gate feature size ${features.size} != $expectedBaseDim"
			}
			val standardized = FloatArray(expectedBaseDim)
			for (index in features.indices) {
				standardized[index] = (features[index] - mean[index]) / scale[index].coerceAtLeast(1e-6f)
			}
			val hidden = FloatArray(weight1.size) { row ->
				var value = bias1[row]
				for (column in 0 until expectedBaseDim) value += weight1[row][column] * standardized[column]
				max(value, 0f)
			}
			var logit = bias2
			for (index in hidden.indices) logit += weight2[index] * hidden[index]
			return sigmoid(logit)
		}
	}

	private data class PhysicalPrior(
		val heightM: Float,
		val heightSigmaM: Float,
		val widthM: Float,
		val widthSigmaM: Float,
		val reliability: Float,
	)

	fun resolve(
		mode: Rel2AbsMode,
		box: UniffiDetectedObject,
		visualMeters: Float,
		frame: MetricDepthFrame,
		contextFeatures: FloatArray,
		detections: Array<UniffiDetectedObject> = emptyArray(),
		segmentationContext: SegmentationContextFrame? = null,
	): Float? {
		val modelId = mode.neuralGateId ?: return null
		val model = models[modelId] ?: return null
		if (!visualMeters.isFinite() || visualMeters <= 0f) return null
		val intrinsics = frame.cameraIntrinsics ?: return null
		val prior = if (model.objectFeatureGate) {
			objectFeaturePriors[box.clsName]
		} else {
			priors[box.clsName]
		} ?: return null
		val left = min(box.x1, box.x2).coerceIn(0f, 1f)
		val right = max(box.x1, box.x2).coerceIn(0f, 1f)
		val top = min(box.y1, box.y2).coerceIn(0f, 1f)
		val bottom = max(box.y1, box.y2).coerceIn(0f, 1f)
		val bboxWidth = right - left
		val bboxHeight = bottom - top
		if (bboxWidth <= 0f || bboxHeight <= 0f) return null

		val widthPixels = bboxWidth * frame.sourceWidth
		val heightPixels = bboxHeight * frame.sourceHeight
		if (widthPixels <= 0f || heightPixels <= 0f) return null
		val zHeight = intrinsics.fyPx * prior.heightM / heightPixels
		val zWidth = intrinsics.fxPx * prior.widthM / widthPixels
		if (!zHeight.isFinite() || !zWidth.isFinite() || zHeight <= 0f || zWidth <= 0f) return null
		val anchorMeters = exp((ln(zHeight.toDouble()) + ln(zWidth.toDouble())) / 2.0).toFloat()
		if (!anchorMeters.isFinite() || anchorMeters <= 0f) return null

		// The historical V6 gates use a visual-clamped F1 endpoint. The new
		// object-feature gate follows its trained contract and uses the raw F1
		// anchor, including when it is below the visual estimate.
		val sizeMeters = if (model.objectFeatureGate) anchorMeters else max(anchorMeters, visualMeters)
		val border = if (left <= 0.01f || right >= 0.99f || top <= 0.01f || bottom >= 0.99f) 1f else 0f
		val sigmaLog = sqrt(
			0.03f * 0.03f +
			0.5f * ((prior.heightSigmaM / prior.heightM) * (prior.heightSigmaM / prior.heightM) + (prior.widthSigmaM / prior.widthM) * (prior.widthSigmaM / prior.widthM)) +
			0.5f * ((2f / heightPixels).coerceAtMost(1f) * (2f / heightPixels).coerceAtMost(1f) + (2f / widthPixels).coerceAtMost(1f) * (2f / widthPixels).coerceAtMost(1f)) +
			border * 0.20f * 0.20f,
		)
		var reliability = prior.reliability * exp(-0.5f * border) / (1f + sigmaLog)
		val aspect = bboxWidth / bboxHeight
		val expectedAspect = prior.widthM / prior.heightM
		val aspectPenalty = min(1f, abs(ln((aspect / expectedAspect).coerceAtLeast(1e-6f).toDouble())).toFloat() / 3f)
		reliability *= 1f - 0.25f * aspectPenalty

		val base = floatArrayOf(
			ln(visualMeters.toDouble()).toFloat(),
			ln(sizeMeters.toDouble()).toFloat(),
			abs(ln((sizeMeters / visualMeters).coerceAtLeast(1e-6f).toDouble())).toFloat(),
			ln(bboxHeight.toDouble()).toFloat(),
			ln(bboxWidth.toDouble()).toFloat(),
			ln((bboxWidth * bboxHeight).toDouble()).toFloat(),
			ln(aspect.toDouble()).toFloat(),
			reliability,
			Rel2AbsContextFeatures.confidenceOrZero(box),
			border,
			0f,
			ln(intrinsics.fxPx.toDouble()).toFloat(),
			ln(intrinsics.fyPx.toDouble()).toFloat(),
			if (box.clsName == "person") 1f else 0f,
			if (box.clsName == "bicycle") 1f else 0f,
			if (box.clsName == "car") 1f else 0f,
		)
		val features = if (model.objectFeatureGate) {
			objectFeatureVector(
				visualMeters = visualMeters,
				anchorMeters = anchorMeters,
				box = box,
				frame = frame,
				prior = prior,
				widthPixels = widthPixels,
				heightPixels = heightPixels,
				bboxWidth = bboxWidth,
				bboxHeight = bboxHeight,
				left = left,
				top = top,
				right = right,
				bottom = bottom,
				border = border,
				sigmaLog = sigmaLog,
				reliability = reliability,
				detections = detections,
				segmentationContext = segmentationContext,
				cameraHeightM = if (model.cameraHeightFeature) {
					mode.cameraHeightPriorM ?: CAMERA_HEIGHT_REFERENCE_M
				} else {
					null
				},
			)
		} else if (model.withContext) {
			FloatArray(base.size + Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT).also {
				base.copyInto(it)
				contextFeatures.copyInto(
					it,
					destinationOffset = base.size,
					startIndex = 0,
					endIndex = min(contextFeatures.size, Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT),
				)
			}
		} else {
			base
		}
		val gate = model.evaluate(features)
		return exp((1f - gate) * ln(visualMeters.toDouble()) + gate * ln(sizeMeters.toDouble())).toFloat()
	}

	private fun objectFeatureVector(
		visualMeters: Float,
		anchorMeters: Float,
		box: UniffiDetectedObject,
		frame: MetricDepthFrame,
		prior: PhysicalPrior,
		widthPixels: Float,
		heightPixels: Float,
		bboxWidth: Float,
		bboxHeight: Float,
		left: Float,
		top: Float,
		right: Float,
		bottom: Float,
		border: Float,
		sigmaLog: Float,
		reliability: Float,
		detections: Array<UniffiDetectedObject>,
		segmentationContext: SegmentationContextFrame?,
		cameraHeightM: Float?,
	): FloatArray {
		val intrinsics = frame.cameraIntrinsics
			?: return FloatArray(OBJECT_FEATURE_INPUT_DIM)
		val signedDisagreement = ln((anchorMeters / visualMeters).coerceAtLeast(1e-6f).toDouble()).toFloat()
		val globalHeight = objectFeaturePriors.values.map { it.heightM }.average().toFloat()
		val globalWidth = objectFeaturePriors.values.map { it.widthM }.average().toFloat()
		val zGeneric = geometricMean(
			intrinsics.fyPx * globalHeight / heightPixels,
			intrinsics.fxPx * globalWidth / widthPixels,
		)
		val zGeometry = geometricMean(
			intrinsics.fyPx / heightPixels,
			intrinsics.fxPx / widthPixels,
		)

		val detectionContext = Rel2AbsContextFeatures.objectFeatures(detections)
		val widths = FloatArray(detections.size) { index ->
			val detection = detections[index]
			(max(detection.x1, detection.x2) - min(detection.x1, detection.x2)).coerceAtLeast(0f)
		}
		val meanWidth = widths.averageOrZero()
		val maxWidth = widths.maxOrZero()
		val targetConfidence = Rel2AbsContextFeatures.confidenceOrZero(box).coerceIn(0f, 1f)
		val targetClassSupported = if (box.clsName in SUPPORTED_DETECTION_CLASSES) 1f else 0f
		val segmentation = segmentationObjectFeatures(segmentationContext, box, left, top, right, bottom)

		val baseFeatures = floatArrayOf(
			boundedLog(visualMeters),
			boundedLog(anchorMeters),
			abs(signedDisagreement),
			signedDisagreement,
			boundedLog(bboxHeight),
			boundedLog(bboxWidth),
			boundedLog(bboxWidth * bboxHeight),
			boundedLog(bboxWidth / bboxHeight),
			((left + right) / 2f).coerceIn(0f, 1f),
			((top + bottom) / 2f).coerceIn(0f, 1f),
			bottom.coerceIn(0f, 1f),
			border,
			boundedLog(prior.heightM * intrinsics.fyPx / heightPixels),
			boundedLog(prior.widthM * intrinsics.fxPx / widthPixels),
			boundedLog(zGeneric),
			boundedLog(zGeometry),
			sigmaLog.coerceAtLeast(0f),
			reliability.coerceIn(0f, 1f),
			prior.reliability.coerceIn(0f, 1f),
			1f,
			detectionContext[0],
			detectionContext[1],
			detectionContext[2].coerceAtLeast(0f),
			detectionContext[3].coerceAtLeast(0f),
			meanWidth.coerceAtLeast(0f),
			maxWidth.coerceAtLeast(0f),
			detectionContext[4].coerceAtLeast(0f),
			detectionContext[5].coerceAtLeast(0f),
			detectionContext[6].coerceAtLeast(0f),
			detectionContext[7].coerceAtLeast(0f),
			detectionContext[8].coerceIn(0f, 1f),
			targetConfidence,
			targetClassSupported,
			*segmentation,
		)
		val calibrationHeight = cameraHeightM ?: return baseFeatures
		return FloatArray(baseFeatures.size + 1).also {
			baseFeatures.copyInto(it)
			it[baseFeatures.size] = calibrationHeight
		}
	}

	private fun segmentationObjectFeatures(
		context: SegmentationContextFrame?,
		box: UniffiDetectedObject,
		left: Float,
		top: Float,
		right: Float,
		bottom: Float,
	): FloatArray {
		val output = FloatArray(SEGMENTATION_OBJECT_FEATURE_COUNT)
		if (
			context == null ||
			!context.available ||
			context.globalAreaFractions.size != Rel2AbsContextFeatures.SEMANTIC_CLASS_COUNT ||
			context.gridAreaFractions.size != Rel2AbsContextFeatures.SEGMENTATION_GRID_FEATURE_COUNT
		) return output

		val global = context.globalAreaFractions.map { it.coerceIn(0f, 1f) }.toFloatArray()
		output[0] = 1f
		output[1] = global.count { it > 1e-6f }.toFloat()
		output[2] = global.maxOrZero()
		output[3] = normalizedEntropy(global)
		output[4] = classArea(global, ROAD)
		output[5] = classArea(global, SIDEWALK)
		output[6] = classArea(global, BUILDING)
		output[7] = classArea(global, VEGETATION)
		output[8] = classArea(global, SKY)
		output[9] = classArea(global, PERSON) + classArea(global, RIDER)
		output[10] = classArea(global, CAR) + classArea(global, TRUCK) + classArea(global, BUS) +
			classArea(global, TRAIN) + classArea(global, MOTORCYCLE) + classArea(global, BICYCLE)

		val x0 = left.coerceIn(0f, 1f)
		val x1 = right.coerceIn(0f, 1f)
		val y0 = top.coerceIn(0f, 1f)
		val y1 = bottom.coerceIn(0f, 1f)
		val bboxArea = ((x1 - x0) * (y1 - y0)).coerceAtLeast(1e-8f)
		val projected = FloatArray(Rel2AbsContextFeatures.SEMANTIC_CLASS_COUNT)
		for (gridY in 0 until Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE) {
			val cellY0 = gridY / Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE.toFloat()
			val cellY1 = (gridY + 1) / Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE.toFloat()
			val overlapY = overlap(y0, y1, cellY0, cellY1)
			if (overlapY <= 0f) continue
			for (gridX in 0 until Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE) {
				val cellX0 = gridX / Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE.toFloat()
				val cellX1 = (gridX + 1) / Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE.toFloat()
				val intersection = overlapY * overlap(x0, x1, cellX0, cellX1)
				if (intersection <= 0f) continue
				val cell = gridY * Rel2AbsContextFeatures.SEGMENTATION_GRID_SIZE + gridX
				val offset = cell * Rel2AbsContextFeatures.SEMANTIC_CLASS_COUNT
				for (classIndex in projected.indices) {
					projected[classIndex] += intersection / bboxArea * context.gridAreaFractions[offset + classIndex]
				}
			}
		}
		for (index in projected.indices) projected[index] = projected[index].coerceIn(0f, 1f)

		output[11] = 1f
		output[12] = projected.count { it > 1e-6f }.toFloat()
		output[13] = projected.maxOrZero()
		output[14] = normalizedEntropy(projected)
		val classIndex = SEMANTIC_LABELS.indexOf(box.clsName)
		output[15] = if (classIndex >= 0) 1f else 0f
		output[16] = if (classIndex >= 0) projected[classIndex] else 0f
		output[17] = classArea(projected, ROAD)
		output[18] = classArea(projected, BUILDING)
		output[19] = classArea(projected, VEGETATION)
		output[20] = classArea(projected, SKY)
		output[21] = classArea(projected, PERSON) + classArea(projected, RIDER)
		output[22] = classArea(projected, CAR) + classArea(projected, TRUCK) + classArea(projected, BUS) +
			classArea(projected, TRAIN) + classArea(projected, MOTORCYCLE) + classArea(projected, BICYCLE)
		return output
	}

	companion object {
		const val OBJECT_FEATURE_HEIGHT_ASSET_PATH = "rel2abs/v6_object_feature_gate_camera_height_160_200_parameters.json"
		private const val OBJECT_FEATURE_INPUT_DIM = 56
		private const val SEGMENTATION_OBJECT_FEATURE_COUNT = 23
		private const val CAMERA_HEIGHT_REFERENCE_M = 1.70f
		private const val CAMERA_HEIGHT_FEATURE_NAME = "camera_height_m"
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
		private val SEMANTIC_LABELS = listOf(
			"road", "sidewalk", "building", "wall", "fence", "pole",
			"traffic light", "traffic sign", "vegetation", "terrain", "sky",
			"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
		)
		private val SUPPORTED_DETECTION_CLASSES = setOf("person", "bicycle", "car")

		private val priors = mapOf(
			"person" to PhysicalPrior(1.7f, 0.2f, 0.5f, 0.15f, 0.65f),
			"bicycle" to PhysicalPrior(1.1f, 0.25f, 0.6f, 0.2f, 0.45f),
			"car" to PhysicalPrior(1.5f, 0.25f, 1.8f, 0.3f, 0.45f),
		)
		private val objectFeaturePriors = mapOf(
			"backpack" to PhysicalPrior(0.5f, 0.15f, 0.35f, 0.15f, 0.45f),
			"bicycle" to PhysicalPrior(1.1f, 0.25f, 0.6f, 0.2f, 0.45f),
			"bottle" to PhysicalPrior(0.25f, 0.1f, 0.08f, 0.04f, 0.45f),
			"bus" to PhysicalPrior(3.2f, 0.5f, 2.5f, 0.4f, 0.45f),
			"car" to PhysicalPrior(1.5f, 0.25f, 1.8f, 0.3f, 0.45f),
			"chair" to PhysicalPrior(0.9f, 0.2f, 0.55f, 0.2f, 0.45f),
			"motorcycle" to PhysicalPrior(1.2f, 0.25f, 0.75f, 0.25f, 0.45f),
			"person" to PhysicalPrior(1.7f, 0.2f, 0.5f, 0.15f, 0.65f),
			"table" to PhysicalPrior(0.75f, 0.2f, 1.2f, 0.4f, 0.45f),
			"truck" to PhysicalPrior(3.0f, 0.6f, 2.5f, 0.5f, 0.45f),
		)

		fun fromAssets(context: Context): V6NeuralGateRunner {
			val calibratedObjectText = context.assets.open(OBJECT_FEATURE_HEIGHT_ASSET_PATH).bufferedReader().use { it.readText() }
			val calibratedObjectRoot = JSONObject(calibratedObjectText)
			val models = mutableMapOf<String, GateModel>()
			val modelIds = calibratedObjectRoot.keys()
			while (modelIds.hasNext()) {
				val modelId = modelIds.next()
				models[modelId] = parseModel(calibratedObjectRoot.getJSONObject(modelId))
			}
			return V6NeuralGateRunner(models)
		}

		private fun parseModel(json: JSONObject): GateModel {
			val mean = readVector(json.getJSONArray("mean"))
			val scale = readVector(json.getJSONArray("scale"))
			val weight1 = readMatrix(json.getJSONArray("weight_1"))
			val bias1 = readVector(json.getJSONArray("bias_1"))
			val weight2 = readVector(json.getJSONArray("weight_2").getJSONArray(0))
			val bias2 = json.getJSONArray("bias_2").getDouble(0).toFloat()
			val featureNames = json.optJSONArray("feature_names")?.let { names ->
				buildSet {
					for (index in 0 until names.length()) add(names.getString(index))
				}
			} ?: emptySet()
			check(mean.size == scale.size && weight1.size == bias1.size && weight1.all { it.size == mean.size }) {
				"Malformed V6 gate parameter dimensions"
			}
			return GateModel(
				mean = mean,
				scale = scale,
				weight1 = weight1,
				bias1 = bias1,
				weight2 = weight2,
				bias2 = bias2,
				withContext = mean.size == 16 + Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT,
				objectFeatureGate = json.optString("format") == "rel2abs_v6_object_feature_gate_v2" || json.has("feature_set"),
				cameraHeightFeature = featureNames.contains(CAMERA_HEIGHT_FEATURE_NAME) ||
					json.optString("calibration_parameter") == "camera_height_m_1.70",
			)
		}

		private fun geometricMean(first: Float, second: Float): Float {
			return if (first.isFinite() && second.isFinite() && first > 0f && second > 0f) {
				exp((ln(first.toDouble()) + ln(second.toDouble())) / 2.0).toFloat()
			} else {
				0f
			}
		}

		private fun boundedLog(value: Float): Float =
			if (value.isFinite() && value > 0f) ln(value.toDouble()).toFloat() else 0f

		private fun overlap(firstStart: Float, firstEnd: Float, secondStart: Float, secondEnd: Float): Float =
			max(0f, min(firstEnd, secondEnd) - max(firstStart, secondStart))

		private fun normalizedEntropy(values: FloatArray): Float {
			var entropy = 0.0
			for (value in values) {
				if (value > 0f) entropy -= value.toDouble() * ln(value.toDouble())
			}
			return (entropy / ln(Rel2AbsContextFeatures.SEMANTIC_CLASS_COUNT.toDouble())).toFloat()
		}

		private fun classArea(values: FloatArray, classIndex: Int): Float =
			values.getOrElse(classIndex) { 0f }

		private fun FloatArray.averageOrZero(): Float = if (isEmpty()) 0f else average().toFloat()

		private fun FloatArray.maxOrZero(): Float = maxOrNull() ?: 0f

		private fun readVector(array: JSONArray): FloatArray = FloatArray(array.length()) { index -> array.getDouble(index).toFloat() }

		private fun readMatrix(array: JSONArray): Array<FloatArray> = Array(array.length()) { row ->
			readVector(array.getJSONArray(row))
		}

		private fun sigmoid(value: Float): Float {
			val bounded = value.coerceIn(-60f, 60f)
			return (1.0 / (1.0 + exp(-bounded.toDouble()))).toFloat()
		}
	}
}
