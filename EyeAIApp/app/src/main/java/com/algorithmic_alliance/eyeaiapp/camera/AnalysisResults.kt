package com.algorithmic_alliance.eyeaiapp.camera

import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.inference.ObjectDetectionV1Policy
import uniffi.NativeLib.UniffiDetectedObject

data class AnalysisGeneration(
    val run: Long = 0,
    val objectDetection: Long = 0,
    val source: Long = 0,
    /** Geometry changes and stream gaps invalidate work without invalidating the source sink. */
    val content: Long = 0,
) {
    fun sameImageStream(other: AnalysisGeneration): Boolean =
        run == other.run && source == other.source && content == other.content
}

data class ObjectDetectionSnapshot(
    val objects: List<UniffiDetectedObject>,
    val frameArrivalNanos: Long,
    val inferenceStartedNanos: Long,
    /** A completed call, not proof of native success: the backend also maps some errors to []. */
    val completedNanos: Long,
    val sequence: Long,
    val generation: AnalysisGeneration,
)

data class DepthSnapshot(
    val prediction: NativeLib.NativeFloatBuffer,
    val width: Int,
    val height: Int,
    val frameArrivalNanos: Long,
    val completedNanos: Long,
    val generation: AnalysisGeneration,
)

/** Published as one atomic value, including the currently valid generation. Null means invalid. */
data class AnalysisResults(
    val generation: AnalysisGeneration = AnalysisGeneration(),
    val objects: ObjectDetectionSnapshot? = null,
    val depth: DepthSnapshot? = null,
) {
    fun freshObjects(now: Long): ObjectDetectionSnapshot? = objects?.takeIf {
        it.generation == generation && fresh(it.frameArrivalNanos, now) &&
            fresh(it.completedNanos, now)
    }

    fun freshDepth(now: Long): DepthSnapshot? = depth?.takeIf {
        it.generation.sameImageStream(generation) && fresh(it.frameArrivalNanos, now) &&
            fresh(it.completedNanos, now)
    }

    /** Bounding boxes can only sample a sufficiently nearby depth frame of the same stream. */
    fun alignedObjects(now: Long): List<UniffiDetectedObject> {
        val objects = freshObjects(now) ?: return emptyList()
        val depth = freshDepth(now) ?: return emptyList()
        val skew = maxOf(objects.frameArrivalNanos, depth.frameArrivalNanos) -
            minOf(objects.frameArrivalNanos, depth.frameArrivalNanos)
        return if (skew <= ObjectDetectionV1Policy.DEPTH_OD_MAX_SKEW_NANOS) objects.objects
        else emptyList()
    }

    private fun fresh(timestamp: Long, now: Long): Boolean =
        now >= timestamp && now - timestamp <= ObjectDetectionV1Policy.RESULT_TTL_NANOS
}
