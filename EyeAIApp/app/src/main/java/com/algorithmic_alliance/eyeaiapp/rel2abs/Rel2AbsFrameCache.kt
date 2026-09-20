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
	private val segmentationContexts = TreeMap<Long, SegmentationContextFrame>()

	fun publishDepth(frame: MetricDepthFrame) = synchronized(lock) {
		depths[frame.sourceTimestampNanos] = frame
		trim(depths)
	}

	fun publishDetections(frame: DetectionFrame) = synchronized(lock) {
		detections[frame.sourceTimestampNanos] = frame
		trim(detections)
	}

	fun publishSegmentationContext(frame: SegmentationContextFrame) = synchronized(lock) {
		segmentationContexts[frame.sourceTimestampNanos] = frame
		trim(segmentationContexts)
	}

	fun clear() = synchronized(lock) {
		depths.clear()
		detections.clear()
		segmentationContexts.clear()
	}

	fun clearDepth() = synchronized(lock) { depths.clear() }

	fun clearDetections() = synchronized(lock) { detections.clear() }

	fun clearSegmentationContexts() = synchronized(lock) { segmentationContexts.clear() }

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
			val segmentation = segmentationContexts[timestamp]
			val context = FloatArray(Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT)
			detection.objectContextFeatures.copyInto(
				context,
				destinationOffset = 0,
				startIndex = 0,
				endIndex = minOf(detection.objectContextFeatures.size, Rel2AbsContextFeatures.OBJECT_FEATURE_COUNT),
			)
			if (
				segmentation != null &&
				segmentation.sourceWidth == detection.sourceWidth &&
				segmentation.sourceHeight == detection.sourceHeight &&
				segmentation.rotationDegrees == detection.rotationDegrees
			) {
				segmentation.segmentationFeatures.copyInto(
					context,
					destinationOffset = Rel2AbsContextFeatures.OBJECT_FEATURE_COUNT,
					startIndex = 0,
					endIndex = minOf(
						segmentation.segmentationFeatures.size,
						Rel2AbsContextFeatures.SEGMENTATION_FEATURE_COUNT,
					),
				)
			}
			MatchedMetricFrame(depth, detection, context, segmentation)
		}
	}

	private fun <T> trim(values: TreeMap<Long, T>) {
		while (values.size > MAX_PENDING_FRAMES) values.pollFirstEntry()
	}

	private companion object {
		const val MAX_PENDING_FRAMES = 3
	}
}
