package com.algorithmic_alliance.eyeaiapp.worldstate

import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import java.util.Collections

internal fun <T> frozen(values: Collection<T>): List<T> = Collections.unmodifiableList(ArrayList(values))

@Serializable enum class Availability { AVAILABLE, UNKNOWN, UNSUPPORTED, INVALID, STALE }
@Serializable enum class Evidence { OBSERVED, ESTIMATED, UNDETERMINED }
@Serializable enum class TrackStatus { CONFIRMED, NOT_OBSERVED }
@Serializable enum class ImageDirection { LEFT, CENTER, RIGHT }
@Serializable enum class WorldTask { DESCRIBE, FIND, NAVIGATE }
@Serializable enum class TimeOrigin { CAMERA_TIMESTAMP, LOCAL_RECEIPT }
@Serializable enum class DepthOrder { FIRST_NEARER, SECOND_NEARER, UNDETERMINED, CONFLICTING }

@Serializable
data class FrameStamp(
    val sessionId: Long,
    val frameId: Long,
    val sourceTimestampNanos: Long,
    val receivedAtNanos: Long,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timeOrigin: TimeOrigin,
)

@Serializable
data class SourceState(
    val status: Availability = Availability.UNKNOWN,
    val frame: FrameStamp? = null,
    val ageMillis: Long? = null,
)

@Serializable
data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    init {
        require(listOf(left, top, right, bottom).all { it.isFinite() && it in 0f..1f })
        require(right > left && bottom > top)
    }
    val centerX: Float = (left + right) * 0.5f
    val centerY: Float = (top + bottom) * 0.5f
}

@Serializable
data class DepthValue(
    val status: Availability = Availability.UNKNOWN,
    val value: Float? = null,
    val spread: Float? = null,
    val validFraction: Float? = null,
    val frame: FrameStamp? = null,
    val method: String? = null,
    val evidence: Evidence = Evidence.UNDETERMINED,
) {
    init {
        require(value == null || value.isFinite())
        require(spread == null || spread.isFinite() && spread >= 0f)
        require(validFraction == null || validFraction.isFinite() && validFraction in 0f..1f)
        require(status != Availability.AVAILABLE || value != null)
    }
}

@Serializable
data class SemanticSummary(
    val source: SourceState = SourceState(),
    val classFractions: List<Float> = emptyList(),
)

@Serializable
data class WorldObject(
    val trackId: Int,
    val classId: Int,
    val className: String,
    val confidence: Float,
    val box: Box,
    val observation: FrameStamp,
    val trackStatus: TrackStatus = TrackStatus.CONFIRMED,
    val freshness: SourceState = SourceState(),
    val relativeDepth: DepthValue = DepthValue(),
    val distanceMeters: DepthValue = DepthValue(),
    val semantics: SemanticSummary = SemanticSummary(),
    val motion: Availability = Availability.UNSUPPORTED,
) {
    init { require(confidence.isFinite() && confidence in 0f..1f) }
    val direction: ImageDirection = when {
        box.centerX < 0.35f -> ImageDirection.LEFT
        box.centerX > 0.65f -> ImageDirection.RIGHT
        else -> ImageDirection.CENTER
    }
}

@Serializable
data class SpatialRelation(
    val firstId: Int,
    val secondId: Int,
    val firstIsLeft: Boolean,
    val overlaps: Boolean,
    val nearInImage: Boolean,
    val depthOrder: DepthOrder,
    val possibleSightlineObstacleId: Int? = null,
    val imageEvidence: Evidence = Evidence.OBSERVED,
    val depthEvidence: Evidence = Evidence.UNDETERMINED,
)

@Serializable
data class LocalCell(
    val index: Int,
    val semantics: SemanticSummary = SemanticSummary(),
    val relativeDepth: DepthValue = DepthValue(),
    val distanceMeters: DepthValue = DepthValue(),
    val sidewalkFraction: Float? = null,
    val roadFraction: Float? = null,
    val obstacleFraction: Float? = null,
    val unknownFraction: Float? = null,
    val objectIds: List<Int> = emptyList(),
    val conflictingEvidence: Boolean = false,
    val traversability: Evidence = Evidence.UNDETERMINED,
    val physicalConnectivity: Availability = Availability.UNKNOWN,
    val imageNeighbors: List<Int> = emptyList(),
)

@Serializable
data class SceneFeatures(
    val classCounts: Map<String, Int> = emptyMap(),
    val occupiedCells: List<Int> = emptyList(),
)

@Serializable
data class WorldState(
    val schemaVersion: Int = 1,
    val sessionId: Long,
    val evaluatedAtNanos: Long,
    val latestFrame: FrameStamp? = null,
    val objects: List<WorldObject> = emptyList(),
    val detections: SourceState = SourceState(),
    val depth: SourceState = SourceState(),
    val segmentation: SemanticSummary = SemanticSummary(),
    val relativeDepth: DepthValue = DepthValue(),
    val distanceMeters: DepthValue = DepthValue(),
    val cells: List<LocalCell> = emptyList(),
    val relations: List<SpatialRelation> = emptyList(),
    val omittedRelations: Int = 0,
    val rejectedObjects: Int = 0,
    val features: SceneFeatures = SceneFeatures(),
    val cameraMotion: Availability = Availability.UNSUPPORTED,
    val userHeading: Availability = Availability.UNKNOWN,
    val outsideView: Availability = Availability.UNKNOWN,
)

@Serializable
data class TargetAnalysis(
    val trackId: Int,
    val status: Availability,
    val target: WorldObject? = null,
    val possibleObstacleIds: List<Int> = emptyList(),
    val cells: List<Int> = emptyList(),
    val afterBypassPosition: Availability = Availability.UNKNOWN,
    val observedSidewalkCells: List<Int> = emptyList(),
)

@Serializable
data class WorldModelInput(
    val schemaVersion: Int = 1,
    val task: WorldTask,
    val scene: WorldState,
    val target: TargetAnalysis? = null,
    val omittedObjects: Int = 0,
)

object WorldStateCodec {
    val json = Json { encodeDefaults = true; explicitNulls = true; allowSpecialFloatingPointValues = false }
    fun encode(input: WorldModelInput): String {
        require(input.schemaVersion == 1 && input.scene.schemaVersion == 1) { "Unsupported WorldState schema" }
        return json.encodeToString(input)
    }
    fun decode(value: String): WorldModelInput = json.decodeFromString<WorldModelInput>(value).also {
        require(it.schemaVersion == 1 && it.scene.schemaVersion == 1) { "Unsupported WorldState schema" }
    }
}
