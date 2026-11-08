package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*

// Narration-only temporal admission, other app consumers are unchanged.
class StableNarrationEvidence(
    private val minimumObservations: Int = 4,
    private val minimumDurationNanos: Long = 750_000_000L,
    private val maximumGapNanos: Long = 1_500_000_000L,
    private val minimumConfidence: Float = .6f,
) {
    private data class Track(val cls: Int,val first: Long,var last: Long,var frame: Long,var hits: Int,var box: Box)
    private val tracks=mutableMapOf<Int,Track>()
    private var session: Long?=null
    var pendingCount=0
    private set
    var transientCount=0
    private set
    fun reset() { tracks.clear()
session=null
pendingCount=0
    transientCount=0 }
    fun update(state: WorldState): List<WorldObject> {
        if(session!=state.sessionId) { reset()
        session=state.sessionId }
        val now=state.evaluatedAtNanos
        tracks.entries.removeAll {
            val expired=now-it.value.last>maximumGapNanos || now<it.value.last
            if(expired && (it.value.hits<minimumObservations||it.value.last-it.value.first<minimumDurationNanos))transientCount++
            expired
        }
        val current=state.objects.filter { WorldStateValidation.current(it) && it.observation==state.detections.frame && it.confidence>=minimumConfidence }
        val currentIds=current.map {it.trackId}.toSet()
        for(o in current) {
            val time=o.observation.receivedAtNanos
            var t=tracks[o.trackId]
            if(t==null) {
                val matches=tracks.filter {(id,track)->id !in currentIds&&track.cls==o.classId&&
                    com.algorithmic_alliance.eyeaiapp.context.ContextFeatures.iou(track.box,o.box)>=.50}
                val match=matches.entries.singleOrNull()?.takeIf {entry->current.count {it.classId==o.classId&&
                    com.algorithmic_alliance.eyeaiapp.context.ContextFeatures.iou(entry.value.box,it.box)>=.50}==1}
                if(match!=null) {t=match.value
tracks.remove(match.key)
                tracks[o.trackId]=t}
            }
            if(t==null || t.cls!=o.classId || time<t.last || time-t.last>maximumGapNanos) {
                t=Track(o.classId,time,time,o.observation.frameId,1,o.box)
                tracks[o.trackId]=t
            } else if(t.frame!=o.observation.frameId) { t.hits++
t.last=time
t.frame=o.observation.frameId
            t.box=o.box }
        }
        val admitted=current.filter { o->tracks.getValue(o.trackId).let { it.hits>=minimumObservations && it.last-it.first>=minimumDurationNanos } }
        pendingCount=current.size-admitted.size
        return admitted
    }

    companion object {
        // Filtering objects must also rebuild every object-dependent invariant.
        internal fun subset(state: WorldState,objects: List<WorldObject>): WorldState {
            val current=objects.filter(WorldStateValidation::current)
            val ids=current.map {it.trackId}.toSet()
            val cells=state.cells.map {cell->
                val members=cell.objectIds.filter {it in ids}
                cell.copy(objectIds=members,conflictingEvidence=cell.semantics.source.status==Availability.AVAILABLE &&
                    cell.semantics.classFractions.getOrNull(1)?.let {it>=.5f}==true &&
                    current.any {it.trackId in members&&it.observation==cell.semantics.source.frame})
            }
            val relations=state.relations.filter {it.firstId in ids&&it.secondId in ids}
            val pairs=current.groupingBy {it.observation}.eachCount().values.sumOf {it*(it-1)/2}
            return state.copy(objects=objects,relations=relations,omittedRelations=pairs-relations.size,cells=cells,
                features=state.features.copy(classCounts=current.groupingBy {it.className}.eachCount(),
                    occupiedCells=cells.filter {it.objectIds.isNotEmpty()}.map {it.index}))
        }
    }
}
