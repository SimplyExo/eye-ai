package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.inference.throttling.AnalysisClock
import java.util.concurrent.atomic.AtomicInteger

enum class FramePixelFormat {
    RGBA_8888,
}

class AnalysisFrame(
    val bitmap: Bitmap,
    val pixelFormat: FramePixelFormat,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampNanos: Long,
    private val onReleased: (() -> Unit)? = null,
) : AutoCloseable {
    private val references = AtomicInteger(1)

    init {
        require(width > 0 && height > 0) { "Frame dimensions must be positive" }
    }

    fun tryRetain(): Boolean {
        while (true) {
            val current = references.get()
            if (current <= 0) return false
            if (references.compareAndSet(current, current + 1)) return true
        }
    }

    fun release() {
        val remaining = references.decrementAndGet()
        check(remaining >= 0) { "AnalysisFrame released more than once" }
        if (remaining == 0) onReleased?.invoke()
    }

    override fun close() = release()

    companion object {
        fun fromBitmap(
            bitmap: Bitmap,
            timestampNanos: Long = AnalysisClock.nowNanos(),
            rotationDegrees: Int = 0,
            onReleased: (() -> Unit)? = null,
        ): AnalysisFrame = AnalysisFrame(
            bitmap = bitmap,
            pixelFormat = FramePixelFormat.RGBA_8888,
            width = bitmap.width,
            height = bitmap.height,
            rotationDegrees = rotationDegrees,
            timestampNanos = timestampNanos,
            onReleased = onReleased,
        )
    }
}
