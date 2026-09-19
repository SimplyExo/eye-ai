package com.algorithmic_alliance.eyeaiapp.rel2abs

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import uniffi.NativeLib.UniffiDetectedObject

class Rel2AbsFrameCacheTest {
	@Test
	fun `pairs depth and detections only from the identical source timestamp`() {
		val cache = Rel2AbsFrameCache()
		cache.publishDepth(depthFrame(timestamp = 101L))
		cache.publishDetections(detectionFrame(timestamp = 102L))
		assertNull(cache.latestMatched())

		cache.publishDetections(detectionFrame(timestamp = 101L))
		val matched = requireNotNull(cache.latestMatched())
		assertEquals(101L, matched.metricDepth.sourceTimestampNanos)
		assertEquals(101L, matched.detectionFrame.sourceTimestampNanos)
	}

	@Test
	fun `rejects matching timestamp when frame provenance differs`() {
		val cache = Rel2AbsFrameCache()
		cache.publishDepth(depthFrame(timestamp = 7L, sourceWidth = 1280))
		cache.publishDetections(detectionFrame(timestamp = 7L, sourceWidth = 640))

		assertNull(cache.latestMatched())
	}

	@Test
	fun `keeps only a bounded number of pending asynchronous frames`() {
		val cache = Rel2AbsFrameCache()
		for (timestamp in 1L..4L) {
			cache.publishDepth(depthFrame(timestamp))
			cache.publishDetections(detectionFrame(timestamp))
		}

		assertEquals(4L, requireNotNull(cache.latestMatched()).metricDepth.sourceTimestampNanos)
	}

	private fun depthFrame(
		timestamp: Long,
		sourceWidth: Int = 640,
	): MetricDepthFrame = MetricDepthFrame(
		depthMeters = FloatArray(4) { 2f },
		width = 2,
		height = 2,
		sourceTimestampNanos = timestamp,
		sourceWidth = sourceWidth,
		sourceHeight = 480,
		rotationDegrees = 0,
		rel2absMode = Rel2AbsMode.Z1,
	)

	private fun detectionFrame(
		timestamp: Long,
		sourceWidth: Int = 640,
	): DetectionFrame = DetectionFrame(
		detections = emptyArray<UniffiDetectedObject>(),
		sourceTimestampNanos = timestamp,
		sourceWidth = sourceWidth,
		sourceHeight = 480,
		rotationDegrees = 0,
	)
}
