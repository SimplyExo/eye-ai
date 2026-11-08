package com.algorithmic_alliance.eyeaiapp.worldstate

import kotlinx.serialization.Serializable
import kotlinx.serialization.SerialName
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import kotlin.math.floor

@Serializable enum class TinyAge { LE_100_MS, LE_500_MS, LE_1_S, LE_10_S, OLDER, INVALID }
@Serializable data class TinySource(@SerialName("a") val age: TinyAge, @SerialName("o") val origin: TimeOrigin)
@Serializable data class TinyChannel(@SerialName("t") val status: Availability, @SerialName("s") val source: Int? = null)
@Serializable data class TinyLimits(val objects: Int = 12, val relations: Int = 16) {
    init { require(objects in 1..16 && relations in 0..32) }
}
@Serializable data class TinyPolicy(val ttlNanos: List<Long>, val limits: TinyLimits)
// Values use decimeters, IQR and valid sample fraction are not calibrated confidence.
@Serializable data class TinyDepth(
    @SerialName("t") val status: Availability, @SerialName("s") val source: Int? = null,
    @SerialName("m") val method: Int? = null, @SerialName("v") val dm: Int? = null,
    @SerialName("q") val iqrDm: Int? = null, @SerialName("f") val validDecile: Int? = null,
    @SerialName("c") val clipped: Boolean = false,
)
// Top three area fractions in permille, remaining known and unknown classes are omitted.
@Serializable data class TinySemantics(
    @SerialName("t") val status: Availability, @SerialName("s") val source: Int? = null,
    @SerialName("k") val top: List<List<Int>> = emptyList(), @SerialName("r") val rest: Int? = null,
    @SerialName("u") val unknown: Int? = null,
)
@Serializable data class TinyObject(
    val id: Int, @SerialName("c") val cls: Int, @SerialName("b") val box: List<Int>,
    @SerialName("q") val confidenceDecile: Int, @SerialName("t") val track: TrackStatus,
    @SerialName("f") val freshness: Availability, @SerialName("s") val source: Int,
    @SerialName("d") val depth: TinyDepth, @SerialName("r") val relativeStatus: Availability,
)
@Serializable data class TinyCell(val id: Int, @SerialName("s") val semantics: TinySemantics,
    @SerialName("d") val depth: TinyDepth, @SerialName("x") val conflict: Boolean)
@Serializable data class TinyRelation(
    val a: Int, val b: Int, @SerialName("l") val left: Boolean, @SerialName("o") val overlap: Boolean,
    @SerialName("d") val depth: DepthOrder, @SerialName("s") val sightline: Int? = null,
)
@Serializable data class TinyLoss(
    val objects: Int, val relations: Int, val rejected: Int,
    val historicalObjects: Int, val clippedDepths: Int,
    val collapsedBoxes: Int,
)

// All versions are required, no timestamps, IDs, paths or labels.
@Serializable data class TinyWorldInput(
    val v: Int, val projection: Int, val taxonomy: String, val space: String,
    val task: WorldTask, val language: String, val navigationSupported: Boolean,
    val policy: TinyPolicy, val sources: List<TinySource>, val methods: List<String>,
    val channels: List<TinyChannel>, val objects: List<TinyObject>, val cells: List<TinyCell>,
    val context: TinySemantics, val relations: List<TinyRelation>,
    val counts: List<List<Int>>, val countScope: String, val loss: TinyLoss,
)

// Shared projection and serializer for runtime and training tools.
class TinyWorldProjection(
    private val config: WorldStateAnalyzer.Config = WorldStateAnalyzer.Config(),
    private val limits: TinyLimits = TinyLimits(),
) {
    fun project(state: WorldState, language: String = "de"): TinyWorldInput {
        require(language in listOf("de", "en"))
        WorldStateValidation.state(state)
        // Config is part of the contract: do not trust externally fabricated freshness.
        fun checkSource(source: SourceState, ttl: Long) {
            val f = source.frame ?: return
            val elapsed = state.evaluatedAtNanos - f.receivedAtNanos
            val expected = if (elapsed < 0) Availability.INVALID else if (elapsed > ttl) Availability.STALE else Availability.AVAILABLE
            require(source.status == expected) { "Source status does not match projection TTL configuration" }
        }
        checkSource(state.detections, config.objectFreshNanos)
        checkSource(state.depth, config.depthFreshNanos)
        checkSource(state.segmentation.source, config.segmentationFreshNanos)
        state.objects.forEach { o ->
            require(WorldTaxonomy.classes.getOrNull(o.classId) == o.className) { "Object taxonomy mismatch" }
            checkSource(o.freshness, config.objectFreshNanos)
            require(o.freshness.frame == o.observation)
            checkSource(o.semantics.source, config.segmentationFreshNanos)
        }
        // Geometry ordering avoids leaking global IDs and makes renumbering invariant.
        val canonical = compareBy<WorldObject> { it.classId }.thenBy { it.box.left }.thenBy { it.box.top }
            .thenBy { it.box.right }.thenBy { it.box.bottom }.thenByDescending { it.confidence }
            .thenBy { it.observation.receivedAtNanos }.thenBy { it.trackStatus.ordinal }
            .thenBy { it.freshness.status.ordinal }.thenBy { it.observation.frameId }.thenBy { it.trackId }
        val selected = state.objects.sortedWith(compareByDescending<WorldObject> { WorldStateValidation.current(it) }
            .then(canonical)).take(limits.objects).sortedWith(canonical)
        val ids = selected.mapIndexed { index, o -> o.trackId to index }.toMap()
        val frames = (selected.map { it.observation } + listOfNotNull(state.detections.frame, state.depth.frame,
            state.segmentation.source.frame) + state.cells.flatMap {
            listOfNotNull(it.semantics.source.frame, it.distanceMeters.frame)
        } + selected.flatMap { listOfNotNull(it.distanceMeters.frame) }).distinct()
            .sortedWith(compareBy<FrameStamp> { it.receivedAtNanos }.thenBy { it.frameId })
        fun sid(frame: FrameStamp?) = frame?.let { frames.indexOf(it).also { id -> require(id >= 0) } }
        fun age(f: FrameStamp): TinyAge = when (state.evaluatedAtNanos - f.receivedAtNanos) {
            in 0..100_000_000L -> TinyAge.LE_100_MS
            in 100_000_001L..500_000_000L -> TinyAge.LE_500_MS
            in 500_000_001L..1_000_000_000L -> TinyAge.LE_1_S
            in 1_000_000_001L..10_000_000_000L -> TinyAge.LE_10_S
            in 10_000_000_001L..Long.MAX_VALUE -> TinyAge.OLDER
            else -> TinyAge.INVALID
        }
        val methods = (selected.map { it.distanceMeters } + state.cells.map { it.distanceMeters })
            .mapNotNull { it.method }.distinct().sorted()
        fun depth(d: DepthValue): TinyDepth {
            val elapsed = d.frame?.let { state.evaluatedAtNanos - it.receivedAtNanos }
            val status = when {
                elapsed != null && elapsed < 0 -> Availability.INVALID
                elapsed != null && elapsed > config.depthFreshNanos -> Availability.STALE
                else -> d.status
            }
            val active = status == Availability.AVAILABLE
            fun dm(v: Float?) = v?.let { floor(it.toDouble() * 10).coerceIn(0.0, 2000.0).toInt() }
            return TinyDepth(status, sid(d.frame), d.method?.let(methods::indexOf),
                if (active) dm(d.value) else null, if (active) dm(d.spread) else null,
                if (active) d.validFraction?.let { (it * 10).toInt().coerceIn(0, 10) } else null,
                active && ((d.value ?: 0f) > 200f || (d.spread ?: 0f) > 200f))
        }
        fun semantics(s: SemanticSummary): TinySemantics {
            checkSource(s.source, config.segmentationFreshNanos)
            if (s.source.status != Availability.AVAILABLE) return TinySemantics(s.source.status, sid(s.source.frame))
            // Floor avoids rounding sum > 1000. Unknown includes sub-permille residue.
            val scale = maxOf(1.0, s.classFractions.sumOf { it.toDouble() })
            val q = s.classFractions.map { floor(it.toDouble() / scale * 1000).toInt() }
            val top = q.indices.filter { q[it] > 0 }.sortedWith(compareByDescending<Int> { q[it] }.thenBy { it }).take(3)
            return TinySemantics(s.source.status, sid(s.source.frame), top.map { listOf(it, q[it]) },
                q.sum() - top.sumOf { q[it] }, 1000 - q.sum())
        }
        val projectedObjects = selected.map { o ->
            TinyObject(ids.getValue(o.trackId), o.classId,
                listOf(o.box.left, o.box.top, o.box.right, o.box.bottom).map { (it * 1000).toInt() },
                (o.confidence * 10).toInt().coerceIn(0, 10), o.trackStatus, o.freshness.status,
                sid(o.observation)!!, depth(o.distanceMeters), o.relativeDepth.status)
        }
        val active = selected.filter(WorldStateValidation::current)
        val pairs = active.flatMapIndexed { i, a -> active.drop(i + 1).filter { it.observation == a.observation }
            .map { b -> WorldStateAnalyzer.relate(a, b) } }.sortedWith(compareBy<SpatialRelation> { ids.getValue(it.firstId) }.thenBy { ids.getValue(it.secondId) })
        val relations = pairs.take(limits.relations).map { r ->
            val a = ids.getValue(r.firstId)
            val b = ids.getValue(r.secondId)
            val ab = projectedObjects[a].box
            val bb = projectedObjects[b].box
            val overlap = minOf(ab[2], bb[2]) > maxOf(ab[0], bb[0]) && minOf(ab[3], bb[3]) > maxOf(ab[1], bb[1])
            TinyRelation(a, b, ab[0] + ab[2] < bb[0] + bb[2], overlap, r.depthOrder,
                if (overlap) r.possibleSightlineObstacleId?.let(ids::getValue) else null)
        }
        val totalPairs = state.objects.filter(WorldStateValidation::current).groupingBy { it.observation }.eachCount()
            .values.sumOf { it * (it - 1) / 2 }
        val cells = state.cells.sortedBy { it.index }.map { TinyCell(it.index, semantics(it.semantics), depth(it.distanceMeters), it.conflictingEvidence) }
        return TinyWorldInput(1, 1, WorldTaxonomy.VERSION, "upright_xyxy_permille", WorldTask.DESCRIBE, language, false,
            TinyPolicy(listOf(config.objectFreshNanos, config.depthFreshNanos, config.segmentationFreshNanos), limits),
            frames.map { TinySource(age(it), it.timeOrigin) }, methods,
            listOf(state.detections, state.depth, state.segmentation.source).map { TinyChannel(it.status, sid(it.frame)) },
            projectedObjects, cells, semantics(state.segmentation), relations,
            state.features.classCounts.entries.sortedBy { WorldTaxonomy.classes.indexOf(it.key) }.map {
                val cls = WorldTaxonomy.classes.indexOf(it.key)
 require(cls >= 0)
                listOf(cls, it.value)
            }, "all_current_observed_tracks",
            TinyLoss(state.objects.size - selected.size, totalPairs - relations.size, state.rejectedObjects,
                state.objects.count { !WorldStateValidation.current(it) },
                projectedObjects.count { it.depth.clipped } + cells.count { it.depth.clipped },
                projectedObjects.count { it.box[0] == it.box[2] || it.box[1] == it.box[3] }))
            .also(TinyWorldCodec::validate)
    }
}

object TinyWorldCodec {
    // One canonical compact wire representation, strict decoding, no platform-specific formatter.
    private val json = Json { encodeDefaults = false
 explicitNulls = false
    allowSpecialFloatingPointValues = false }
    fun encode(input: TinyWorldInput): String { validate(input)
    return json.encodeToString(input) }
    fun decode(value: String): TinyWorldInput = json.decodeFromString<TinyWorldInput>(value).also(::validate)

    fun validate(v: TinyWorldInput) {
        require(v.v == 1 && v.projection == 1 && v.taxonomy == WorldTaxonomy.VERSION && v.space == "upright_xyxy_permille")
        require(!v.navigationSupported && v.task == WorldTask.DESCRIBE && v.language in listOf("de", "en"))
        require(v.countScope == "all_current_observed_tracks" && v.policy.ttlNanos.size == 3 && v.policy.ttlNanos.all { it > 0 })
        require(v.objects.size <= v.policy.limits.objects && v.relations.size <= v.policy.limits.relations)
        require(v.objects.map { it.id } == v.objects.indices.toList() && v.methods.distinct() == v.methods &&
            v.methods.all { it.isNotBlank() && it.length <= 128 })
        fun ref(id: Int?) { require(id == null || id in v.sources.indices) }
        fun freshness(status: Availability, id: Int?, ttl: Long) {
            ref(id)
            if (status != Availability.AVAILABLE && status != Availability.STALE) return
            require(id != null)
            val band = v.sources[id].age
            val range = when (band) {
                TinyAge.LE_100_MS -> 0L..100_000_000L
                TinyAge.LE_500_MS -> 100_000_001L..500_000_000L
                TinyAge.LE_1_S -> 500_000_001L..1_000_000_000L
                TinyAge.LE_10_S -> 1_000_000_001L..10_000_000_000L
                TinyAge.OLDER -> 10_000_000_001L..Long.MAX_VALUE
                TinyAge.INVALID -> throw IllegalArgumentException("Invalid age cannot support fresh/stale evidence")
            }
            require(if (status == Availability.AVAILABLE) range.first <= ttl else range.last > ttl)
        }
        fun depth(d: TinyDepth) {
            ref(d.source)
            require(d.method == null || d.method in v.methods.indices)
            freshness(d.status, d.source, v.policy.ttlNanos[1])
            require(d.dm == null || d.dm in 0..2000)
            require(d.iqrDm == null || d.iqrDm in 0..2000)
            require(d.validDecile == null || d.validDecile in 0..10)
            if (d.status == Availability.AVAILABLE) require(d.dm != null && d.source != null && d.method != null)
            else require(d.dm == null && d.iqrDm == null && d.validDecile == null && !d.clipped)
        }
        fun semantics(s: TinySemantics) {
            ref(s.source)
            freshness(s.status, s.source, v.policy.ttlNanos[2])
            require(s.top.size <= 3 && s.top.all { it.size == 2 && it[0] in 0..18 && it[1] in 1..1000 })
            require(s.top.map { it[0] }.distinct().size == s.top.size)
            if (s.status == Availability.AVAILABLE) require(s.source != null && s.rest != null && s.unknown != null &&
                s.rest in 0..1000 && s.unknown in 0..1000 && s.top.sumOf { it[1] } + s.rest + s.unknown == 1000)
            else require(s.top.isEmpty() && s.rest == null && s.unknown == null)
        }
        require(v.channels.size == 3)
        v.channels.forEachIndexed { index, c -> freshness(c.status, c.source, v.policy.ttlNanos[index])
        ref(c.source) }
        v.objects.forEach { o ->
            require(o.cls in WorldTaxonomy.classes.indices && o.box.size == 4 && o.box.all { it in 0..1000 } &&
                o.box[0] <= o.box[2] && o.box[1] <= o.box[3] && o.confidenceDecile in 0..10)
            ref(o.source)
            depth(o.depth)
            freshness(o.freshness, o.source, v.policy.ttlNanos[0])
            freshness(o.relativeStatus, if (o.relativeStatus in listOf(Availability.AVAILABLE, Availability.STALE)) o.source else null, v.policy.ttlNanos[1])
            require(o.depth.source == null || o.depth.source == o.source)
            if (o.track == TrackStatus.CONFIRMED && o.freshness == Availability.AVAILABLE)
                require(v.channels[0].status == Availability.AVAILABLE && v.channels[0].source == o.source)
        }
        require(v.cells.isEmpty() || v.cells.map { it.id } == (0..15).toList())
        v.cells.forEach {
            semantics(it.semantics)
            depth(it.depth)
            require(it.semantics.source == v.channels[2].source && it.semantics.status == v.channels[2].status)
            require(it.depth.source == null || it.depth.source == v.channels[1].source)
            if (it.conflict) require(it.semantics.status == Availability.AVAILABLE)
        }
        semantics(v.context)
        require(v.context.source == v.channels[2].source && v.context.status == v.channels[2].status)
        require(v.relations.map { setOf(it.a, it.b) }.distinct().size == v.relations.size)
        v.relations.forEach { r ->
            require(r.a in v.objects.indices && r.b in v.objects.indices && r.a != r.b)
            val a = v.objects[r.a]
            val b = v.objects[r.b]
            require(a.track == TrackStatus.CONFIRMED && b.track == TrackStatus.CONFIRMED &&
                a.freshness == Availability.AVAILABLE && b.freshness == Availability.AVAILABLE && a.source == b.source)
            require(r.left == (a.box[0] + a.box[2] < b.box[0] + b.box[2]))
            require(r.overlap == (minOf(a.box[2], b.box[2]) > maxOf(a.box[0], b.box[0]) &&
                minOf(a.box[3], b.box[3]) > maxOf(a.box[1], b.box[1])))
            require(r.sightline == null || (r.overlap && when (r.depth) {
                DepthOrder.FIRST_NEARER -> r.sightline == r.a
                DepthOrder.SECOND_NEARER -> r.sightline == r.b
                else -> false
            }))
            val metric = a.depth.status == Availability.AVAILABLE && b.depth.status == Availability.AVAILABLE &&
                a.depth.method == b.depth.method
            val relative = a.relativeStatus == Availability.AVAILABLE && b.relativeStatus == Availability.AVAILABLE
            if (r.depth != DepthOrder.UNDETERMINED) require(metric || relative)
            if (r.depth == DepthOrder.CONFLICTING) require(metric && relative)
        }
        require(v.counts.all { it.size == 2 && it[0] in WorldTaxonomy.classes.indices && it[1] > 0 })
        require(v.counts.map { it[0] }.distinct().size == v.counts.size)
        require(listOf(v.loss.objects, v.loss.relations, v.loss.rejected, v.loss.historicalObjects, v.loss.clippedDepths).all { it >= 0 })
        require(v.loss.collapsedBoxes == v.objects.count { it.box[0] == it.box[2] || it.box[1] == it.box[3] })
        require(v.loss.clippedDepths == v.objects.count { it.depth.clipped } + v.cells.count { it.depth.clipped })
        val keptHistorical = v.objects.count { it.track != TrackStatus.CONFIRMED || it.freshness != Availability.AVAILABLE }
        require(v.loss.historicalObjects - keptHistorical in 0..v.loss.objects)
        val selectedCounts = v.objects.filter { it.track == TrackStatus.CONFIRMED && it.freshness == Availability.AVAILABLE }.groupingBy { it.cls }.eachCount()
        val allCounts = v.counts.associate { it[0] to it[1] }
        if (v.loss.objects == 0) require(allCounts == selectedCounts)
        else {
            require(selectedCounts.all { (cls, n) -> (allCounts[cls] ?: 0) >= n })
            require(allCounts.values.sum() - selectedCounts.values.sum() in 0..v.loss.objects)
        }
    }
}
