package com.algorithmic_alliance.eyeaiapp.worldstate

import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.rel2abs.DetectionFrame
import com.algorithmic_alliance.eyeaiapp.rel2abs.MetricDepthFrame
import com.algorithmic_alliance.eyeaiapp.rel2abs.SegmentationContextFrame
import java.util.Locale

// Converts perception results without running models or retaining images.
class WorldStatePipeline(val analyzer: WorldStateAnalyzer = WorldStateAnalyzer()) {
    fun objects(stamp: FrameStamp, frame: DetectionFrame) {
        if (!matches(stamp, frame.sourceTimestampNanos, frame.sourceWidth, frame.sourceHeight, frame.rotationDegrees)) return
        var rejected = 0
        val values = frame.detections.mapNotNull { d ->
            if (!d.cnf.isFinite() || d.cnf !in 0f..1f ||
                !listOf(d.x1, d.y1, d.x2, d.y2).all { it.isFinite() } || d.x2 <= d.x1 || d.y2 <= d.y1 ||
                d.clsName.isBlank()
            ) { rejected++
            return@mapNotNull null }
            val l = d.x1.coerceIn(0f, 1f)
            val t = d.y1.coerceIn(0f, 1f)
            val r = d.x2.coerceIn(0f, 1f)
            val b = d.y2.coerceIn(0f, 1f)
            if (r <= l || b <= t) { rejected++
            return@mapNotNull null }
            WorldObject(d.trackingId, d.cls, d.clsName.lowercase(Locale.ROOT), d.cnf, Box(l, t, r, b), stamp)
        }
        analyzer.publishObjects(stamp, values, rejected)
    }

    fun depth(
        stamp: FrameStamp,
        raw: NativeLib.NativeFloatBuffer,
        width: Int,
        height: Int,
        modelName: String,
        supportedRelativeModel: Boolean,
        metric: MetricDepthFrame?,
    ) {
        if (width <= 0 || height <= 0 || width.toLong() * height != raw.floatBuffer.capacity().toLong()) {
            analyzer.invalidateDepth(stamp, Availability.INVALID)
            return
        }
        val matched = metric?.takeIf {
            matches(stamp, it.sourceTimestampNanos, it.sourceWidth, it.sourceHeight, it.rotationDegrees) &&
                it.width == width && it.height == height
        }
        analyzer.publishDepth(stamp, width, height,
            raw.floatBuffer.takeIf { supportedRelativeModel }, matched?.depthMeters,
            modelName.takeIf { supportedRelativeModel }, matched?.let { "REL2ABS_${it.rel2absMode.name}_VISUAL" })
    }

    fun segmentation(stamp: FrameStamp, frame: SegmentationContextFrame) {
        if (!matches(stamp, frame.sourceTimestampNanos, frame.sourceWidth, frame.sourceHeight, frame.rotationDegrees)) return
        if (!frame.available) { analyzer.invalidateSegmentation(stamp, Availability.INVALID)
        return }
        if (frame.globalAreaFractions.size != 19 || frame.gridAreaFractions.size != 16 * 19 ||
            !frame.globalAreaFractions.all { it.isFinite() && it in 0f..1f } ||
            !frame.gridAreaFractions.all { it.isFinite() && it in 0f..1f } ||
            frame.globalAreaFractions.sum() > 1.001f ||
            (0..15).any { cell -> (0..18).sumOf { frame.gridAreaFractions[cell * 19 + it].toDouble() } > 1.001 }
        ) { analyzer.invalidateSegmentation(stamp, Availability.INVALID)
        return }
        analyzer.publishSegmentation(stamp, frame.globalAreaFractions.toList(),
            (0..15).map { cell -> (0..18).map { frame.gridAreaFractions[cell * 19 + it] } })
    }

    private fun matches(stamp: FrameStamp, timestamp: Long, width: Int, height: Int, rotation: Int) =
        stamp.sourceTimestampNanos == timestamp && stamp.width == width && stamp.height == height &&
            stamp.rotationDegrees == rotation
}
