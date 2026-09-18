package com.algorithmic_alliance.eyeaiapp.rel2abs

import java.util.TreeMap

/**
 * Small provenance cache, not a temporal filter.  It retains only enough
 * asynchronous outputs to pair one YOLO/ByteTrack result with the metric map
 * generated from the identical source frame.
 */
class Rel2AbsFrameCache {
	private val lock = Any()
	private val depths = TreeMap<Long, MetricDepthFrame>()
	private val detections = TreeMap<Long, DetectionFrame>()

	fun publishDepth(frame: MetricDepthFrame) = synchronized(lock) {
		depths[frame.sourceTimestampNanos] = frame
		trim(depths)
	}

	fun publishDetections(frame: DetectionFrame) = synchronized(lock) {
		detections[frame.sourceTimestampNanos] = frame
		trim(detections)
	}

	fun clear() = synchronized(lock) {
		depths.clear()
		detections.clear()
	}

	fun clearDepth() = synchronized(lock) { depths.clear() }

	fun clearDetections() = synchronized(lock) { detections.clear() }

	fun latestMatched(): MatchedMetricFrame? = synchronized(lock) {
		depths.descendingMap().entries.firstNotNullOfOrNull { (timestamp, depth) ->
			val detection = detections[timestamp] ?: return@firstNotNullOfOrNull null
			if (
				depth.sourceWidth != detection.sourceWidth ||
				depth.sourceHeight != detection.sourceHeight ||
				depth.rotationDegrees != detection.rotationDegrees
			) {
				return@firstNotNullOfOrNull null
			}
			MatchedMetricFrame(depth, detection)
		}
	}

	private fun <T> trim(values: TreeMap<Long, T>) {
		while (values.size > MAX_PENDING_FRAMES) values.pollFirstEntry()
	}

	private companion object {
		const val MAX_PENDING_FRAMES = 3
	}
}
