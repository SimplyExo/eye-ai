package com.algorithmic_alliance.eyeaiapp.worldstate

import java.nio.FloatBuffer
import java.util.TreeMap
import java.util.Collections
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class WorldStateAnalyzer(
    val config: Config = Config(),
    private val clock: () -> Long = System::nanoTime,
) {
    @kotlinx.serialization.Serializable
    data class Config(
        val objectFreshNanos: Long = 1_000_000_000L,
        val depthFreshNanos: Long = 500_000_000L,
        val segmentationFreshNanos: Long = 1_000_000_000L,
        val retainTracksNanos: Long = 10_000_000_000L,
        val maxExportObjects: Int = 64,
        val maxRelations: Int = 256,
    ) {
        init {
            require(objectFreshNanos > 0 && depthFreshNanos > 0 && segmentationFreshNanos > 0)
            require(retainTracksNanos >= objectFreshNanos && maxExportObjects > 0 && maxRelations > 0)
        }
    }

    private data class DepthFrame(
        val stamp: FrameStamp,
        val width: Int,
        val height: Int,
        val relative: FloatBuffer?,
        val meters: FloatArray?,
        val relativeMethod: String?,
        val metricMethod: String?,
        val cells: List<Pair<DepthValue, DepthValue>>,
        val global: Pair<DepthValue, DepthValue>,
    )
    private data class SegFrame(val stamp: FrameStamp, val global: List<Float>, val cells: List<List<Float>>)
    private data class Inputs(
        val session: Long,
        val latest: FrameStamp?,
        val detection: FrameStamp?,
        val objects: List<WorldObject>,
        val depths: Map<Long, DepthFrame>,
        val segments: Map<Long, SegFrame>,
        val depthStatus: Availability,
        val segmentStatus: Availability,
        val detectionStatus: Availability,
        val rejected: Int,
    )

    private val lock = Any()
    private var session = 0L
    private var sequence = 0L
    private var latest: FrameStamp? = null
    private var detection: FrameStamp? = null
    private val objects = LinkedHashMap<Int, WorldObject>()
    private val depths = TreeMap<Long, DepthFrame>()
    private val segments = TreeMap<Long, SegFrame>()
    private var depthStatus = Availability.UNKNOWN
    private var segmentStatus = Availability.UNKNOWN
    private var detectionStatus = Availability.UNKNOWN
    private var rejected = 0
    private var invalidDepthThrough = -1L
    private var invalidSegmentationThrough = -1L
    private var invalidObjectsThrough = -1L

    fun reset() = synchronized(lock) {
        session++
        latest = null
        detection = null
        objects.clear()
        depths.clear()
        segments.clear()
        depthStatus = Availability.UNKNOWN
        segmentStatus = Availability.UNKNOWN
        detectionStatus = Availability.UNKNOWN
        rejected = 0
        invalidDepthThrough = -1L
        invalidSegmentationThrough = -1L
        invalidObjectsThrough = -1L
    }

    fun acceptFrame(
        sourceTimestampNanos: Long,
        width: Int,
        height: Int,
        rotationDegrees: Int,
        origin: TimeOrigin,
        receivedAtNanos: Long = clock(),
    ): FrameStamp = synchronized(lock) {
        require(width > 0 && height > 0 && rotationDegrees in listOf(0, 90, 180, 270))
        latest?.let {
            if (it.width != width || it.height != height || it.rotationDegrees != rotationDegrees ||
                it.timeOrigin != origin || sourceTimestampNanos < it.sourceTimestampNanos
            ) reset()
        }
        FrameStamp(session, ++sequence, sourceTimestampNanos, receivedAtNanos, width, height,
            rotationDegrees, origin).also { latest = it }
    }

    fun isCurrent(stamp: FrameStamp): Boolean = synchronized(lock) { stamp.sessionId == session }

    fun publishObjects(stamp: FrameStamp, values: List<WorldObject>, rejectedCount: Int = 0) = synchronized(lock) {
        if (!isCurrent(stamp) || stamp.frameId <= (detection?.frameId ?: -1) || stamp.frameId <= invalidObjectsThrough) return@synchronized
        require(values.all { it.observation == stamp })
        detection = stamp
        detectionStatus = Availability.AVAILABLE
        rejected = rejectedCount
        val ids = values.mapTo(HashSet()) { it.trackId }
        objects.replaceAll { id, old -> if (id in ids) old else old.copy(trackStatus = TrackStatus.NOT_OBSERVED) }
        values.forEach { objects[it.trackId] = it }
        objects.entries.removeAll { stamp.receivedAtNanos - it.value.observation.receivedAtNanos > config.retainTracksNanos }
    }

    // Buffers must be owned by this inference result and remain unmodified after publication.
    fun publishDepth(
        stamp: FrameStamp,
        width: Int,
        height: Int,
        relative: FloatBuffer?,
        meters: FloatArray?,
        relativeMethod: String?,
        metricMethod: String?,
    ) {
        require(width > 0 && height > 0 && width.toLong() * height <= Int.MAX_VALUE)
        require(relative == null || relative.capacity() == width * height)
        require(meters == null || meters.size == width * height)
        if (!isCurrent(stamp)) return
        val raw = relative?.asReadOnlyBuffer()?.apply { clear() }
        fun summarize(box: Box) = sample(stamp, width, height, box, raw, null, relativeMethod) to
            sample(stamp, width, height, box, null, meters, metricMethod)
        val result = DepthFrame(stamp, width, height, raw, meters, relativeMethod, metricMethod,
            (0..15).map { summarize(cellBox(it)) }, summarize(Box(0f, 0f, 1f, 1f)))
        synchronized(lock) {
            if (!isCurrent(stamp) || stamp.frameId <= invalidDepthThrough) return
            depths[stamp.frameId] = result
            trim(depths)
            depthStatus = Availability.AVAILABLE
        }
    }

    fun publishSegmentation(stamp: FrameStamp, global: List<Float>, cells: List<List<Float>>) {
        require(global.size == 19 && cells.size == 16 && cells.all { it.size == 19 })
        require((listOf(global) + cells).all { row ->
            row.all { it.isFinite() && it in 0f..1f } && row.sum() <= 1.001f
        })
        val result = SegFrame(stamp, frozen(global), frozen(cells.map(::frozen)))
        synchronized(lock) {
            if (!isCurrent(stamp) || stamp.frameId <= invalidSegmentationThrough) return
            segments[stamp.frameId] = result
            trim(segments)
            segmentStatus = Availability.AVAILABLE
        }
    }

    fun invalidateDepth(stamp: FrameStamp? = null, status: Availability = Availability.UNKNOWN) = synchronized(lock) {
        if (stamp != null && !isCurrent(stamp)) return@synchronized
        if (stamp != null && stamp.frameId < invalidDepthThrough) return@synchronized
        if (stamp != null && stamp.frameId < (depths.lastEntry()?.key ?: -1)) return@synchronized
        invalidDepthThrough = max(invalidDepthThrough, stamp?.frameId ?: latest?.frameId ?: -1L)
        depths.clear()
        depthStatus = status
    }

    fun invalidateSegmentation(stamp: FrameStamp? = null, status: Availability = Availability.UNKNOWN) = synchronized(lock) {
        if (stamp != null && !isCurrent(stamp)) return@synchronized
        if (stamp != null && stamp.frameId < invalidSegmentationThrough) return@synchronized
        if (stamp != null && stamp.frameId < (segments.lastEntry()?.key ?: -1)) return@synchronized
        invalidSegmentationThrough = max(invalidSegmentationThrough, stamp?.frameId ?: latest?.frameId ?: -1L)
        segments.clear()
        segmentStatus = status
    }

    fun invalidateObjects(stamp: FrameStamp? = null, status: Availability = Availability.UNKNOWN) = synchronized(lock) {
        if (stamp != null && !isCurrent(stamp)) return@synchronized
        if (stamp != null && stamp.frameId < invalidObjectsThrough) return@synchronized
        if (stamp != null && stamp.frameId < (detection?.frameId ?: -1)) return@synchronized
        invalidObjectsThrough = max(invalidObjectsThrough, stamp?.frameId ?: latest?.frameId ?: -1L)
        objects.clear()
        detection = null
        detectionStatus = status
    }

    fun snapshot(nowNanos: Long = clock()): WorldState {
        val input = synchronized(lock) {
            Inputs(session, latest, detection, objects.values.toList(), depths.toMap(), segments.toMap(),
                depthStatus, segmentStatus, detectionStatus, rejected)
        }
        val currentObjects = input.objects.filter {
            nowNanos - it.observation.receivedAtNanos <= config.retainTracksNanos
        }.sortedBy { it.trackId }.map { obj ->
            val d = input.depths[obj.observation.frameId]
            val s = input.segments[obj.observation.frameId]
            val sem = s?.let {
                // Context of the image cell containing the center, not object-mask semantics.
                val index = (obj.box.centerY * 4).toInt().coerceIn(0, 3) * 4 +
                    (obj.box.centerX * 4).toInt().coerceIn(0, 3)
                SemanticSummary(source(it.stamp, config.segmentationFreshNanos, nowNanos), it.cells[index])
            } ?: SemanticSummary(SourceState(input.segmentStatus.takeUnless { it == Availability.AVAILABLE } ?: Availability.UNKNOWN))
            obj.copy(
                freshness = source(obj.observation, config.objectFreshNanos, nowNanos),
                relativeDepth = d?.let {
                    age(sample(it.stamp, it.width, it.height, obj.box, it.relative, null, it.relativeMethod), nowNanos)
                } ?: DepthValue(status = missing(input.depthStatus)),
                distanceMeters = d?.let {
                    age(sample(it.stamp, it.width, it.height, obj.box, null, it.meters, it.metricMethod), nowNanos)
                } ?: DepthValue(status = missing(input.depthStatus)),
                semantics = sem,
            )
        }
        val latestDepth = input.depths.maxByOrNull { it.key }?.value
        val latestSeg = input.segments.maxByOrNull { it.key }?.value
        val cells = (0..15).map { index ->
            val sem = latestSeg?.let {
                SemanticSummary(source(it.stamp, config.segmentationFreshNanos, nowNanos), it.cells[index])
            } ?: SemanticSummary(SourceState(input.segmentStatus))
            val activeSem = sem.classFractions.takeIf { sem.source.status == Availability.AVAILABLE }
            val box = cellBox(index)
            val ids = currentObjects.filter {
                it.trackStatus == TrackStatus.CONFIRMED && it.freshness.status == Availability.AVAILABLE && overlap(it.box, box)
            }.map { it.trackId }
            val obstacle = activeSem?.let { f -> OBSTACLE_CLASSES.sumOf { f[it].toDouble() }.toFloat().coerceIn(0f, 1f) }
            val ground = activeSem?.get(1)
            LocalCell(index, sem,
                latestDepth?.cells?.get(index)?.first?.let { age(it, nowNanos) } ?: DepthValue(status = missing(input.depthStatus)),
                latestDepth?.cells?.get(index)?.second?.let { age(it, nowNanos) } ?: DepthValue(status = missing(input.depthStatus)),
                ground, activeSem?.get(0), obstacle,
                activeSem?.let { (1f - it.sum()).coerceIn(0f, 1f) }, frozen(ids),
                // Only call it a conflict when both sources describe exactly this frame.
                ground != null && ground >= 0.5f && ids.any { id ->
                    currentObjects.first { it.trackId == id }.observation == latestSeg?.stamp
                },
                if (ground != null && ground >= 0.5f) Evidence.ESTIMATED else Evidence.UNDETERMINED,
                imageNeighbors = frozen(listOf(index - 4, index + 4, index - 1, index + 1).filter {
                    it in 0..15 && abs(it % 4 - index % 4) + abs(it / 4 - index / 4) == 1
                }.sorted()))
        }
        val relations = ArrayList<SpatialRelation>()
        val activeCount = currentObjects.filter(::current).groupingBy { it.observation }.eachCount()
        var omitted = activeCount.values.sumOf { it * (it - 1) / 2 }
        val relationObjects = currentObjects.filter(::current).sortedWith(
            compareByDescending<WorldObject> { it.confidence }.thenBy { it.trackId }).take(config.maxExportObjects)
        // Keep output bounded, target analysis computes its own complete target relations.
        for (i in relationObjects.indices) for (j in i + 1 until relationObjects.size) {
            val a = relationObjects[i]
            val b = relationObjects[j]
            if (!current(a) || !current(b) || a.observation != b.observation) continue
            if (relations.size < config.maxRelations) { relations += relate(a, b)
            omitted-- }
        }
        val counts = currentObjects.filter(::current).groupingBy { it.className }.eachCount().toSortedMap()
        return WorldState(sessionId = input.session, evaluatedAtNanos = nowNanos, latestFrame = input.latest,
            objects = frozen(currentObjects),
            detections = input.detection?.let { source(it, config.objectFreshNanos, nowNanos) } ?: SourceState(input.detectionStatus),
            depth = latestDepth?.let { source(it.stamp, config.depthFreshNanos, nowNanos) } ?: SourceState(input.depthStatus),
            segmentation = latestSeg?.let {
                SemanticSummary(source(it.stamp, config.segmentationFreshNanos, nowNanos), it.global)
            } ?: SemanticSummary(SourceState(input.segmentStatus)),
            relativeDepth = latestDepth?.global?.first?.let { age(it, nowNanos) } ?: DepthValue(status = missing(input.depthStatus)),
            distanceMeters = latestDepth?.global?.second?.let { age(it, nowNanos) } ?: DepthValue(status = missing(input.depthStatus)),
            cells = frozen(cells), relations = frozen(relations), omittedRelations = omitted, rejectedObjects = input.rejected,
            features = SceneFeatures(Collections.unmodifiableMap(counts), frozen(cells.filter { it.objectIds.isNotEmpty() }.map { it.index })))
    }

    private fun age(value: DepthValue, now: Long): DepthValue {
        val stamp = value.frame ?: return value
        val freshness = source(stamp, config.depthFreshNanos, now).status
        return if (freshness == Availability.STALE || freshness == Availability.INVALID)
            value.copy(status = freshness) else value
    }

    private fun source(stamp: FrameStamp, ttl: Long, now: Long): SourceState {
        val elapsed = now - stamp.receivedAtNanos
        return SourceState(if (elapsed < 0) Availability.INVALID else if (elapsed > ttl) Availability.STALE
            else Availability.AVAILABLE, stamp, if (elapsed < 0) null else elapsed / 1_000_000)
    }

    private fun missing(status: Availability) = if (status == Availability.AVAILABLE) Availability.UNKNOWN else status

    companion object {
        private val OBSTACLE_CLASSES = intArrayOf(2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18)
        private fun <T> trim(map: TreeMap<Long, T>) { while (map.size > 3) map.pollFirstEntry() }
        private fun current(obj: WorldObject) = obj.trackStatus == TrackStatus.CONFIRMED && obj.freshness.status == Availability.AVAILABLE
        fun cellBox(index: Int): Box {
            require(index in 0..15)
            val x = index % 4 * 0.25f
            val y = index / 4 * 0.25f
            return Box(x, y, x + 0.25f, y + 0.25f)
        }
        private fun overlap(a: Box, b: Box) = min(a.right, b.right) > max(a.left, b.left) &&
            min(a.bottom, b.bottom) > max(a.top, b.top)

        private fun order(a: DepthValue, b: DepthValue, inverse: Boolean): DepthOrder {
            if (a.status != Availability.AVAILABLE || b.status != Availability.AVAILABLE || a.frame != b.frame ||
                a.method != b.method) return DepthOrder.UNDETERMINED
            val av = a.value ?: return DepthOrder.UNDETERMINED
            val bv = b.value ?: return DepthOrder.UNDETERMINED
            val margin = max(max(abs(av), abs(bv)) * 0.1f, max(a.spread ?: 0f, b.spread ?: 0f))
            if (abs(av.toDouble() - bv) <= margin) return DepthOrder.UNDETERMINED
            return if ((av < bv) != inverse) DepthOrder.FIRST_NEARER else DepthOrder.SECOND_NEARER
        }

        fun relate(a: WorldObject, b: WorldObject): SpatialRelation {
            val metric = order(a.distanceMeters, b.distanceMeters, false)
            val relative = order(a.relativeDepth, b.relativeDepth, true)
            val depth = if (metric != DepthOrder.UNDETERMINED && relative != DepthOrder.UNDETERMINED && metric != relative)
                DepthOrder.CONFLICTING else if (metric != DepthOrder.UNDETERMINED) metric else relative
            val overlaps = overlap(a.box, b.box)
            val obstacle = if (!overlaps) null else when (depth) {
                DepthOrder.FIRST_NEARER -> a.trackId
                DepthOrder.SECOND_NEARER -> b.trackId
                else -> null
            }
            val dx = a.box.centerX - b.box.centerX
            val dy = a.box.centerY - b.box.centerY
            return SpatialRelation(a.trackId, b.trackId, a.box.centerX < b.box.centerX, overlaps,
                dx * dx + dy * dy < 0.01f, depth, obstacle,
                depthEvidence = if (depth == DepthOrder.FIRST_NEARER || depth == DepthOrder.SECOND_NEARER)
                    Evidence.ESTIMATED else Evidence.UNDETERMINED)
        }

        // At most 81 values per ROI, IQR is not calibrated uncertainty.
        private fun sample(
            stamp: FrameStamp, width: Int, height: Int, box: Box,
            relative: FloatBuffer?, meters: FloatArray?, method: String?,
        ): DepthValue {
            if (relative == null && meters == null || method == null) return DepthValue(Availability.UNSUPPORTED)
            val x0 = (box.left * width).toInt().coerceIn(0, width - 1)
            val y0 = (box.top * height).toInt().coerceIn(0, height - 1)
            val x1 = kotlin.math.ceil(box.right * width).toInt().coerceIn(x0 + 1, width)
            val y1 = kotlin.math.ceil(box.bottom * height).toInt().coerceIn(y0 + 1, height)
            val nx = min(9, x1 - x0)
            val ny = min(9, y1 - y0)
            val values = FloatArray(nx * ny)
            var count = 0
            for (y in 0 until ny) for (x in 0 until nx) {
                val px = x0 + ((x + 0.5f) * (x1 - x0) / nx).toInt().coerceAtMost(x1 - x0 - 1)
                val py = y0 + ((y + 0.5f) * (y1 - y0) / ny).toInt().coerceAtMost(y1 - y0 - 1)
                val v = relative?.get(py * width + px) ?: meters!![py * width + px]
                if (v.isFinite() && (meters == null || v > 0f)) values[count++] = v
            }
            val fraction = count.toFloat() / values.size
            if (values.size < 16) return DepthValue(Availability.UNKNOWN,
                validFraction = fraction, frame = stamp, method = method)
            if (count < 16 || fraction < 0.5f) return DepthValue(Availability.INVALID,
                validFraction = fraction, frame = stamp, method = method)
            java.util.Arrays.sort(values, 0, count)
            val median = if (count % 2 == 1) values[count / 2] else
                (values[count / 2 - 1].toDouble() * 0.5 + values[count / 2].toDouble() * 0.5).toFloat()
            val spread = (values[(count * 3 / 4).coerceAtMost(count - 1)].toDouble() - values[count / 4]).toFloat()
            if (!spread.isFinite()) return DepthValue(Availability.INVALID, frame = stamp, method = method)
            return DepthValue(Availability.AVAILABLE, median, spread, fraction, stamp, method, Evidence.ESTIMATED)
        }
    }
}
