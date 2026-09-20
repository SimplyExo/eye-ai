package com.algorithmic_alliance.eyeaiapp.rel2abs

import android.content.Context
import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Frozen Z1 and S2 deployment contracts. The V6 neural-gate modes reuse the
 * same Z1 pixel map as their visual endpoint; their object-level F1 fusion is
 * applied later by [MetricDistanceResolver].
 */
class Rel2AbsRunner private constructor(
	private val context: Context,
	private val z1: Interpreter,
) : AutoCloseable {
	data class Output(val depthMeters: FloatArray, val mode: Rel2AbsMode)

	private var s2: Interpreter? = null
	private var closed = false

	init {
		validateZ1Contract(z1)
	}

	@Synchronized
	fun run(
		rgbFrame: Bitmap,
		rawRelativeDepth: NativeLib.NativeFloatBuffer,
		rawWidth: Int,
		rawHeight: Int,
		mode: Rel2AbsMode,
	): Output {
		check(!closed) { "REL2ABS runner is closed" }
		require(rawWidth == RAW_WIDTH && rawHeight == RAW_HEIGHT) {
			"Frozen Z1 requires ${RAW_WIDTH}x${RAW_HEIGHT} raw MiDaS depth, got ${rawWidth}x${rawHeight}"
		}
		require(rawRelativeDepth.floatBuffer.capacity() == rawWidth * rawHeight) {
			"Raw MiDaS buffer has unexpected capacity ${rawRelativeDepth.floatBuffer.capacity()}"
		}

		val raw = FloatArray(rawWidth * rawHeight) { index ->
			rawRelativeDepth.floatBuffer[index].takeIf { it.isFinite() } ?: 0f
		}
		val small = downsampleRawRelativeDepth(raw, rawWidth, rawHeight)
		val statistics = rawStatistics(raw)
		val z1Output = Array(1) { FloatArray(2) }
		z1.runForMultipleInputsOutputs(
			arrayOf(
				arrayOf(statistics),
				small,
				arrayOf(FloatArray(INTRINSICS_COUNT)),
				prepareRgb64(rgbFrame),
			),
			mutableMapOf<Int, Any>(0 to z1Output),
		)
		val z1Depth = decodeZ1(raw, z1Output[0][0], z1Output[0][1])
		if (mode == Rel2AbsMode.Z1 || mode.isNeuralGate) return Output(z1Depth, mode)

		val s2Features = s2Features(statistics, small)
		val s2Output = Array(1) { FloatArray(1) }
		val s2Interpreter = s2 ?: createS2Interpreter().also { s2 = it }
		s2Interpreter.run(s2Features, s2Output)
		val scale = exp(s2Output[0][0].toDouble()).toFloat().coerceIn(S2_SCALE_MIN, S2_SCALE_MAX)
		val corrected = FloatArray(z1Depth.size) { index -> z1Depth[index] * scale }
		return Output(corrected, mode)
	}

	@Synchronized
	override fun close() {
		if (closed) return
		closed = true
		z1.close()
		s2?.close()
		s2 = null
	}

	private fun prepareRgb64(source: Bitmap): Array<Array<Array<FloatArray>>> {
		val scaled = Bitmap.createScaledBitmap(source, RGB_WIDTH, RGB_HEIGHT, true)
		return try {
			val pixels = IntArray(RGB_WIDTH * RGB_HEIGHT)
			scaled.getPixels(pixels, 0, RGB_WIDTH, 0, 0, RGB_WIDTH, RGB_HEIGHT)
			Array(1) {
				Array(RGB_HEIGHT) { y ->
					Array(RGB_WIDTH) { x ->
						val pixel = pixels[y * RGB_WIDTH + x]
						floatArrayOf(
							((pixel shr 16) and 0xff) / 255f,
							((pixel shr 8) and 0xff) / 255f,
							(pixel and 0xff) / 255f,
						)
					}
				}
			}
		} finally {
			if (scaled !== source) scaled.recycle()
		}
	}

	/** Exact bilinear coordinate convention of torch.interpolate(..., align_corners=false). */
	private fun downsampleRawRelativeDepth(raw: FloatArray, inputWidth: Int, inputHeight: Int): Array<Array<Array<FloatArray>>> {
		return Array(1) {
			Array(SMALL_HEIGHT) { outputY ->
				val sourceY = (outputY + 0.5f) * inputHeight / SMALL_HEIGHT - 0.5f
				val yFloor = floor(sourceY).toInt()
				val y0 = yFloor.coerceIn(0, inputHeight - 1)
				val y1 = (yFloor + 1).coerceIn(0, inputHeight - 1)
				val wy = sourceY - yFloor
				Array(SMALL_WIDTH) { outputX ->
					val sourceX = (outputX + 0.5f) * inputWidth / SMALL_WIDTH - 0.5f
					val xFloor = floor(sourceX).toInt()
					val x0 = xFloor.coerceIn(0, inputWidth - 1)
					val x1 = (xFloor + 1).coerceIn(0, inputWidth - 1)
					val wx = sourceX - xFloor
					val top = raw[y0 * inputWidth + x0] * (1f - wx) + raw[y0 * inputWidth + x1] * wx
					val bottom = raw[y1 * inputWidth + x0] * (1f - wx) + raw[y1 * inputWidth + x1] * wx
					floatArrayOf(top * (1f - wy) + bottom * wy)
				}
			}
		}
	}

	private fun rawStatistics(raw: FloatArray): FloatArray {
		val valid = raw.filter { it.isFinite() && it > 0f }.sorted()
		if (valid.isEmpty()) return FloatArray(STATISTICS_COUNT)
		val n = valid.size
		val mean = valid.sum() / n
		val sampleStd = if (n > 1) {
			kotlin.math.sqrt(valid.sumOf { value -> (value - mean).toDouble() * (value - mean).toDouble() } / (n - 1)).toFloat()
		} else 0f
		return floatArrayOf(
			mean,
			sampleStd,
			valid.first(),
			valid.last(),
			valid[n / 10],
			valid[n / 2],
			valid[min(n - 1, 9 * n / 10)],
		)
	}

	private fun s2Features(statistics: FloatArray, small: Array<Array<Array<FloatArray>>>): Array<FloatArray> {
		val result = FloatArray(S2_FEATURE_COUNT)
		statistics.copyInto(result, 0)
		var featureIndex = STATISTICS_COUNT + INTRINSICS_COUNT
		for (gridY in 0 until 4) {
			for (gridX in 0 until 4) {
				var sum = 0f
				for (y in 0 until 8) for (x in 0 until 8) sum += small[0][gridY * 8 + y][gridX * 8 + x][0]
				result[featureIndex++] = sum / 64f
			}
		}
		return arrayOf(result)
	}

	private fun decodeZ1(rawRelativeDepth: FloatArray, rawM: Float, rawDelta: Float): FloatArray {
		val sigmoidM = sigmoid(rawM)
		val sigmoidDelta = sigmoid(rawDelta)
		val m = M_MIN + sigmoidM * (M_MAX - M_MIN)
		val delta = DELTA_MIN + sigmoidDelta * (DELTA_MAX - DELTA_MIN)
		val u0 = exp((m - delta / 2f).toDouble()).toFloat()
		val u1 = exp((m + delta / 2f).toDouble()).toFloat()
		return FloatArray(rawRelativeDepth.size) { index ->
			val r = rawRelativeDepth[index].takeIf { it.isFinite() } ?: 0f
			val t = ((r.coerceIn(R_LOW, R_HIGH) - R_LOW) / (R_HIGH - R_LOW)).coerceIn(0f, 1f)
			1f / max((1f - t) * u0 + t * u1, EPSILON)
		}
	}

	private fun sigmoid(value: Float): Float {
		val bounded = value.coerceIn(-60f, 60f)
		return (1.0 / (1.0 + exp(-bounded.toDouble()))).toFloat()
	}

	private fun createS2Interpreter(): Interpreter {
		val interpreter = Interpreter(mapAsset(context, S2_ASSET), Interpreter.Options().setNumThreads(1))
		interpreter.allocateTensors()
		val input = interpreter.getInputTensor(0)
		val output = interpreter.getOutputTensor(0)
		require(input.shape().contentEquals(intArrayOf(1, S2_FEATURE_COUNT)) && input.dataType() == DataType.FLOAT32) {
			"Unexpected frozen S2 input contract: ${input.shape().contentToString()} / ${input.dataType()}"
		}
		require(output.shape().contentEquals(intArrayOf(1, 1)) && output.dataType() == DataType.FLOAT32) {
			"Unexpected frozen S2 output contract: ${output.shape().contentToString()} / ${output.dataType()}"
		}
		return interpreter
	}

	private fun validateZ1Contract(interpreter: Interpreter) {
		interpreter.allocateTensors()
		val expectedShapes = arrayOf(
			intArrayOf(1, STATISTICS_COUNT),
			intArrayOf(1, SMALL_HEIGHT, SMALL_WIDTH, 1),
			intArrayOf(1, INTRINSICS_COUNT),
			intArrayOf(1, RGB_HEIGHT, RGB_WIDTH, 3),
		)
		require(interpreter.inputTensorCount == expectedShapes.size) {
			"Unexpected frozen Z1 input count: ${interpreter.inputTensorCount}"
		}
		expectedShapes.forEachIndexed { index, shape ->
			val tensor = interpreter.getInputTensor(index)
			require(tensor.shape().contentEquals(shape) && tensor.dataType() == DataType.FLOAT32) {
				"Unexpected frozen Z1 input $index: ${tensor.shape().contentToString()} / ${tensor.dataType()}"
			}
		}
		val output = interpreter.getOutputTensor(0)
		require(output.shape().contentEquals(intArrayOf(1, 2)) && output.dataType() == DataType.FLOAT32) {
			"Unexpected frozen Z1 output: ${output.shape().contentToString()} / ${output.dataType()}"
		}
	}

	companion object {
		const val Z1_ASSET = "rel2abs/rel2abs_z1_float32.tflite"
		const val S2_ASSET = "rel2abs/rel2abs_s2_seed42_float32.tflite"
		const val RAW_WIDTH = 256
		const val RAW_HEIGHT = 256
		private const val RGB_WIDTH = 64
		private const val RGB_HEIGHT = 64
		private const val SMALL_WIDTH = 32
		private const val SMALL_HEIGHT = 32
		private const val STATISTICS_COUNT = 7
		private const val INTRINSICS_COUNT = 5
		private const val S2_FEATURE_COUNT = 28
		private const val R_LOW = 121.375f
		private const val R_HIGH = 839.5f
		private const val M_MIN = -3.7245745095284133f
		private const val M_MAX = 0.202358881069104f
		private const val DELTA_MIN = 0f
		private const val DELTA_MAX = 3.1423029628820647f
		private const val EPSILON = 1e-6f
		private const val S2_SCALE_MIN = 0.25f
		private const val S2_SCALE_MAX = 4.0f

		private fun mapAsset(context: Context, assetPath: String): MappedByteBuffer =
			context.assets.openFd(assetPath).use { descriptor ->
				FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
					channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
				}
			}

		fun fromAssets(context: Context): Rel2AbsRunner = Rel2AbsRunner(
			context.applicationContext,
			Interpreter(mapAsset(context.applicationContext, Z1_ASSET), Interpreter.Options().setNumThreads(1)),
		)
	}
}
