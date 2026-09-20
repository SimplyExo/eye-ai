package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.rel2abs.Rel2AbsCameraIntrinsics
import java.util.concurrent.atomic.AtomicInteger

/**
 * A source-neutral frame envelope.
 *
 * The creator transfers its initial reference to [FrameAnalyzer.submitFrame].
 * Consumers must retain the frame before using it asynchronously and release
 * that reference when finished. The bitmap itself is deliberately not recycled
 * here because the depth and object workers may use it concurrently.
 */
class AnalysisFrame(
	val bitmap: Bitmap,
	val width: Int,
	val height: Int,
	/** Rotation of the source image that the source adapter applied. */
	val rotationDegrees: Int,
	val timestampNanos: Long,
	/** Optional Camera2 calibration scaled to this bitmap. */
	val cameraIntrinsics: Rel2AbsCameraIntrinsics? = null,
	private val onReleased: (() -> Unit)? = null,
) : AutoCloseable {
	private val references = AtomicInteger(1)

	init {
		require(width > 0 && height > 0) { "Frame dimensions must be positive" }
	}

	/** Acquires a consumer reference unless the frame has already been released. */
	fun tryRetain(): Boolean {
		while (true) {
			val current = references.get()
			if (current <= 0) return false
			if (references.compareAndSet(current, current + 1)) return true
		}
	}

	/** Releases one reference and invokes the source callback exactly once. */
	fun release() {
		val remaining = references.decrementAndGet()
		check(remaining >= 0) { "AnalysisFrame released more than once" }
		if (remaining == 0) onReleased?.invoke()
	}

	override fun close() = release()

	companion object {
		fun fromBitmap(
			bitmap: Bitmap,
			timestampNanos: Long = System.nanoTime(),
			rotationDegrees: Int = 0,
			onReleased: (() -> Unit)? = null,
		): AnalysisFrame = AnalysisFrame(
			bitmap = bitmap,
			width = bitmap.width,
			height = bitmap.height,
			rotationDegrees = rotationDegrees,
			timestampNanos = timestampNanos,
			onReleased = onReleased,
		)
	}
}
