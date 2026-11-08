package com.algorithmic_alliance.eyeaiapp.worldstate

import kotlin.math.abs

// Checks semantics at serialization boundaries.
object WorldStateValidation {
    internal fun current(o: WorldObject) = o.trackStatus == TrackStatus.CONFIRMED &&
        o.freshness.status == Availability.AVAILABLE

    fun input(input: WorldModelInput) {
        require(input.schemaVersion == 1 && input.scene.schemaVersion == 1) { "Unsupported WorldState schema" }
        require(input.omittedObjects >= 0)
        state(input.scene, input.omittedObjects)
        input.target?.let { t ->
            require(t.omittedObstacleCount in 0..input.omittedObjects && t.afterBypassPosition == Availability.UNKNOWN)
            val obj = input.scene.objects.singleOrNull { it.trackId == t.trackId }
            require(t.target == obj)
            if (obj != null) {
                geometry(t.target!!)
                require(t.status == if (current(obj)) Availability.AVAILABLE else
                    if (obj.freshness.status == Availability.STALE) Availability.STALE else Availability.UNKNOWN)
            } else require(t.status == Availability.UNKNOWN)
            require(t.possibleObstacleIds.distinct() == t.possibleObstacleIds)
            require(t.possibleObstacleIds.all { id ->
                val other = input.scene.objects.singleOrNull { it.trackId == id }
                obj != null && other != null && current(obj) && current(other) &&
                    obj.observation == other.observation &&
                    WorldStateAnalyzer.relate(obj, other).possibleSightlineObstacleId == id
            })
            require(t.cells == input.scene.cells.filter { t.trackId in it.objectIds }.map { it.index })
            if (obj != null) require(t.observedSidewalkCells == input.scene.cells.filter {
                it.semantics.source.status == Availability.AVAILABLE && (it.sidewalkFraction ?: 0f) >= 0.5f
            }.map { it.index })
            else require(t.observedSidewalkCells.isEmpty() && t.possibleObstacleIds.isEmpty() && t.omittedObstacleCount == 0)
        }
    }

    fun geometry(o: WorldObject) {
        val b = o.box
        require(b.centerX == (b.left + b.right) * 0.5f && b.centerY == (b.top + b.bottom) * 0.5f) {
            "Inconsistent serialized box center"
        }
        val direction = when {
            b.centerX < 0.35f -> ImageDirection.LEFT
            b.centerX > 0.65f -> ImageDirection.RIGHT
            else -> ImageDirection.CENTER
        }
        require(o.direction == direction) { "Inconsistent serialized image direction" }
    }

    fun state(s: WorldState, omittedObjects: Int = 0) {
        require(s.schemaVersion == 1 && s.omittedRelations >= 0 && s.rejectedObjects >= 0)
        val frames = HashMap<Long, FrameStamp>()
        fun frame(f: FrameStamp) {
            require(f.sessionId == s.sessionId && f.width > 0 && f.height > 0 && f.frameId > 0 &&
                f.rotationDegrees in listOf(0, 90, 180, 270))
            require(frames.put(f.frameId, f)?.let { it == f } != false) { "Conflicting frame identity" }
            s.latestFrame?.let { latest ->
                require(f.frameId <= latest.frameId && f.width == latest.width && f.height == latest.height &&
                    f.rotationDegrees == latest.rotationDegrees && f.timeOrigin == latest.timeOrigin)
            }
        }
        fun source(v: SourceState) {
            v.frame?.let(::frame)
            require(v.ageMillis == null || v.ageMillis >= 0)
            if (v.frame == null) require(v.ageMillis == null)
            if (v.status == Availability.AVAILABLE || v.status == Availability.STALE) {
                val f = requireNotNull(v.frame)
                val age = s.evaluatedAtNanos - f.receivedAtNanos
                require(age >= 0 && v.ageMillis == age / 1_000_000)
            }
        }
        fun semantic(v: SemanticSummary) {
            source(v.source)
            require(v.classFractions.isEmpty() || v.classFractions.size == 19)
            if (v.classFractions.isNotEmpty()) require(v.source.frame != null)
            require(v.classFractions.all { it.isFinite() && it in 0f..1f } && v.classFractions.sum() <= 1.001f)
            if (v.source.status == Availability.AVAILABLE || v.source.status == Availability.STALE)
                require(v.classFractions.size == 19)
        }
        fun depth(v: DepthValue, metric: Boolean) {
            v.frame?.let(::frame)
            if (v.value != null) require((!metric || v.value > 0f) && v.frame != null && !v.method.isNullOrBlank() &&
                v.evidence == Evidence.ESTIMATED)
            if (v.status == Availability.AVAILABLE || v.status == Availability.STALE) {
                require(v.frame != null && !v.method.isNullOrBlank())
                if (v.value != null) require(v.evidence == Evidence.ESTIMATED)
            }
        }
        s.latestFrame?.let(::frame)
        source(s.detections)
 source(s.depth)
        semantic(s.segmentation)
        depth(s.relativeDepth, false)
        depth(s.distanceMeters, true)
        require(s.objects.map { it.trackId }.distinct().size == s.objects.size)
        val byId = s.objects.associateBy { it.trackId }
        s.objects.forEach { o ->
            geometry(o)
 frame(o.observation)
            source(o.freshness)
            require(o.classId >= 0 && o.className.isNotBlank() && o.className == o.className.lowercase(java.util.Locale.ROOT))
            require(o.freshness.frame == null || o.freshness.frame == o.observation)
            if (current(o)) require(s.detections.status == Availability.AVAILABLE && o.observation == s.detections.frame)
            depth(o.relativeDepth, false)
 depth(o.distanceMeters, true)
            semantic(o.semantics)
            require(listOf(o.relativeDepth.frame, o.distanceMeters.frame, o.semantics.source.frame)
                .all { it == null || it == o.observation }) { "Cross-frame object fusion" }
        }
        require(s.cells.isEmpty() || s.cells.map { it.index }.sorted() == (0..15).toList())
        if (s.latestFrame != null) require(s.cells.size == 16)
        s.cells.forEach { c ->
            semantic(c.semantics)
 depth(c.relativeDepth, false)
            depth(c.distanceMeters, true)
            require(c.semantics.source == s.segmentation.source && c.relativeDepth.frame == s.relativeDepth.frame &&
                c.distanceMeters.frame == s.distanceMeters.frame) { "Inconsistent scene cell provenance" }
            require(c.objectIds.distinct() == c.objectIds && c.objectIds.all { it in byId && current(byId.getValue(it)) })
            val expectedIds = s.objects.filter { current(it) && overlap(it.box, WorldStateAnalyzer.cellBox(c.index)) }.map { it.trackId }.toSet()
            require(c.objectIds.toSet() == expectedIds)
            val active = c.semantics.source.status == Availability.AVAILABLE
            val f = c.semantics.classFractions
            fun fraction(actual: Float?, expected: Float?) {
                require(if (actual == null || expected == null) actual == expected else abs(actual - expected) < 0.00001f)
            }
            fraction(c.sidewalkFraction, if (active) f[1] else null)
            fraction(c.roadFraction, if (active) f[0] else null)
            fraction(c.unknownFraction, if (active) (1f - f.sum()).coerceIn(0f, 1f) else null)
            val obstacleClasses = listOf(2,3,4,5,6,7,8,11,12,13,14,15,16,17,18)
            fraction(c.obstacleFraction, if (active) obstacleClasses.sumOf { f[it].toDouble() }.toFloat().coerceIn(0f, 1f) else null)
            val ground = active && f[1] >= 0.5f
            require(c.traversability == if (ground) Evidence.ESTIMATED else Evidence.UNDETERMINED)
            require(c.physicalConnectivity == Availability.UNKNOWN)
            require(c.conflictingEvidence == (ground && c.objectIds.any {
                byId.getValue(it).observation == c.semantics.source.frame
            }))
            require(c.imageNeighbors.sorted() == listOf(c.index - 4, c.index + 4, c.index - 1, c.index + 1)
                .filter { it in 0..15 && abs(it % 4 - c.index % 4) + abs(it / 4 - c.index / 4) == 1 }.sorted())
        }
        require(s.relations.map { setOf(it.firstId, it.secondId) }.distinct().size == s.relations.size)
        s.relations.forEach { r ->
            val a = requireNotNull(byId[r.firstId])
            val b = requireNotNull(byId[r.secondId])
            require(a.trackId != b.trackId && current(a) && current(b) && a.observation == b.observation)
            require(r == WorldStateAnalyzer.relate(a, b)) { "Inconsistent derived relation" }
        }
        val knownPairs = s.objects.filter(::current).groupingBy { it.observation }.eachCount()
            .values.sumOf { it * (it - 1) / 2 }
        if (omittedObjects == 0) require(s.omittedRelations == knownPairs - s.relations.size)
        else require(s.omittedRelations >= knownPairs - s.relations.size)
        val counts = s.objects.filter(::current).groupingBy { it.className }.eachCount()
        require(s.features.classCounts.all { it.key.isNotBlank() && it.value > 0 })
        if (omittedObjects == 0) require(s.features.classCounts == counts)
        else {
            require(counts.all { (name, n) -> (s.features.classCounts[name] ?: 0) >= n })
            require(s.features.classCounts.values.sum() - counts.values.sum() in 0..omittedObjects)
        }
        require(s.features.occupiedCells.sorted() == s.features.occupiedCells.distinct().sorted() &&
            s.features.occupiedCells.all { it in 0..15 })
        if (omittedObjects == 0) require(s.features.occupiedCells.toSet() == s.cells.filter { it.objectIds.isNotEmpty() }.map { it.index }.toSet())
    }

    private fun overlap(a: Box, b: Box) = minOf(a.right, b.right) > maxOf(a.left, b.left) &&
        minOf(a.bottom, b.bottom) > maxOf(a.top, b.top)
}
