package com.algorithmic_alliance.eyeaiapp.context

import android.graphics.Bitmap
import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlin.math.ln

object ContextFeatures {
    data class Visual(val values: List<Float>,val usable: Boolean,val quality: Double=1.0)
    // Small, exposure-centred luma + chromaticity descriptor, no additional model.
    fun visual(bitmap: Bitmap): Visual {
        val pixels=IntArray(32*24)
        for(y in 0 until 24)for(x in 0 until 32)pixels[y*32+x]=bitmap.getPixel(((x+.5)*bitmap.width/32).toInt().coerceAtMost(bitmap.width-1),((y+.5)*bitmap.height/24).toInt().coerceAtMost(bitmap.height-1))
        return visualSamples(pixels)
    }
    fun visualSamples(pixels: IntArray): Visual {
        require(pixels.size==32*24)
        val luma=FloatArray(pixels.size)
val colors=FloatArray(32)
        val counts=IntArray(16)
        for(y in 0 until 24)for(x in 0 until 32) {
            val c=pixels[y*32+x]
val r=(c shr 16)and 255
val g=(c shr 8)and 255
            val b=c and 255
            luma[y*32+x]=(77*r+150*g+29*b)/65280f
            val cell=y/6*4+x/8
            val sum=(r+g+b).coerceAtLeast(1).toFloat()
            colors[cell*2]+=r/sum
colors[cell*2+1]+=g/sum
            counts[cell]++
        }
        val mean=luma.average().toFloat()
        val variance=luma.sumOf {(it-mean).toDouble()*(it-mean)}/luma.size
        for(i in luma.indices)luma[i]=(luma[i]-mean)*.5f+.5f
        for(i in colors.indices)colors[i]/=counts[i/2].coerceAtLeast(1)
        val usable=!((mean<.035f||mean>.985f)&&variance<.001)
        val clipped=pixels.count {c->val r=(c shr 16)and 255
val g=(c shr 8)and 255
val b=c and 255
        (r<5&&g<5&&b<5)||(r>250&&g>250&&b>250)}.toDouble()/pixels.size
        // Keep uniform coloured surfaces usable, reject clipped, featureless occlusion.
        val quality=if(!usable)0.0 else (1-.75*clipped).coerceIn(.25,1.0)
        return Visual(luma.toList()+colors.toList(),usable,quality)
    }
    fun iou(a: Box,b: Box): Double {
        val intersection=(minOf(a.right,b.right)-maxOf(a.left,b.left)).coerceAtLeast(0f)*
            (minOf(a.bottom,b.bottom)-maxOf(a.top,b.top)).coerceAtLeast(0f)
        val union=(a.right-a.left)*(a.bottom-a.top)+(b.right-b.left)*(b.bottom-b.top)-intersection
        return if(union>0)intersection.toDouble()/union else 0.0
    }
    fun observe(stamp: FrameStamp,now: Long,raw: WorldState,stable: List<WorldObject>,visual: Visual?,motion: Double? = null,history: ContextFeatureHistory?=null,pending: Int=0,transient: Int=0,legacy: Boolean=false): ContextObservation {
        val fresh=raw.detections.status==Availability.AVAILABLE
        val unconfirmed=raw.objects.any {WorldStateValidation.current(it)} && stable.isEmpty()
        val admitted=stable.filter {WorldStateValidation.current(it)&&it.observation==raw.detections.frame}
        val validObjects=fresh&&!unconfirmed&&(admitted.isNotEmpty()||visual?.usable!=false)
        val objects=if(validObjects)admitted.groupBy {it.className}.mapValues {(_,v)->v.sumOf {it.confidence.toDouble()}.toFloat()}else null
        val layout=if(validObjects)buildMap<String,Float> {
            for(o in admitted) {
                val cell=(o.box.centerY*4).toInt().coerceIn(0,3)*4+(o.box.centerX*4).toInt().coerceIn(0,3)
                val key="${o.className}:$cell"
                put(key,(get(key)?:0f)+o.confidence)
            }
        }else null
        val relative=raw.cells.map {it.relativeDepth.takeIf {d->d.status==Availability.AVAILABLE}?.value}
        val finite=relative.filterNotNull().filter {it.isFinite()}.sorted()
        val low=finite.getOrNull(finite.size/10)
        val high=finite.getOrNull(finite.size*9/10)
        val rel=if(finite.size>=8&&low!=null&&high!=null&&high-low>1e-6f)relative.map {v->v?.let {((it-low)/(high-low)).coerceIn(0f,1f)}}else List(16){null}
        val metric=raw.cells.map {it.distanceMeters.takeIf {d->d.status==Availability.AVAILABLE}?.value?.takeIf {v->v>0&&v.isFinite()}?.let {v->(ln(1.0+v)/ln(51.0)).toFloat().coerceIn(0f,1f)}}
        val depth=(rel+metric).takeIf {it.count {v->v!=null}>=8}
        val global=raw.segmentation.takeIf {it.source.status==Availability.AVAILABLE}?.classFractions?.takeIf {it.size==19}
        val grid=raw.cells.takeIf {cells->cells.all {it.semantics.source.status==Availability.AVAILABLE&&it.semantics.classFractions.size==19}}?.flatMap {it.semantics.classFractions}
        val semantics=global?.let {if(legacy||grid==null)it else it+grid}
        val inventory=history?.update(raw,stable)
        val selected=inventory ?: admitted.map {ContextEntity(it.trackId,it.className,it.confidence,it.box)}
        val classes=inventory?.groupingBy {it.cls}?.eachCount()?.mapValues {it.value.toFloat()} ?: objects
        val places=inventory?.groupBy {o->"${o.cls}:${(o.box.centerY*4).toInt().coerceIn(0,3)*4+(o.box.centerX*4).toInt().coerceIn(0,3)}"}?.mapValues {it.value.size.toFloat()} ?: layout
        val objectQuality=if(history!=null&&inventory==null)0.0 else if(classes==null)0.0 else if(selected.isEmpty())1.0 else selected.map {it.confidence.toDouble()}.average()
        val quality=mapOf("objects" to objectQuality,"visual" to ((visual?.quality?:0.0)*(1-.30*(motion?.takeIf {it.isFinite()}?.coerceIn(0.0,1.0)?:0.0))),
            "depth" to (if(depth==null)0.0 else maxOf(rel.count {it!=null},metric.count {it!=null})/16.0),"semantics" to (if(semantics!=null)1.0 else 0.0))
        return ContextObservation(stamp,now,if(history==null)objects else inventory?.let {classes},if(history==null)layout else inventory?.let {places},visual?.values,depth,semantics,visual?.usable?:true,
            mapOf("objects" to raw.detections.frame,"depth" to raw.depth.frame,"segmentation" to raw.segmentation.source.frame),motion,
            if(legacy)"context-features-1" else "context-features-2",inventory,quality,ContextHistoryStats(pending,transient,inventory?.size?:admitted.size))
    }
}

// Trigger-only inventory, held entries do not become facts or refresh source TTLs.
class ContextFeatureHistory {
    private data class Entry(var entity: ContextEntity,var missingAt: Long?=null,var misses: Int=0)
    private val entries=linkedMapOf<Int,Entry>()
    private var session: Long?=null
    private var lastFrame: FrameStamp?=null
    private var firstHealthy: Long?=null
    private var healthyFrames=0
    fun reset() {entries.clear()
session=null
lastFrame=null
firstHealthy=null
    healthyFrames=0}
    fun update(raw: WorldState,stable: List<WorldObject>): List<ContextEntity>? {
        if(session!=raw.sessionId) {reset()
        session=raw.sessionId}
        if(raw.detections.status!=Availability.AVAILABLE)return null
        val frame=raw.detections.frame ?: return null
        if(lastFrame!=frame) {
            if(lastFrame?.let {frame.receivedAtNanos-it.receivedAtNanos>1_500_000_000L}==true) {
                entries.clear()
healthyFrames=0
                firstHealthy=null
            }
            lastFrame=frame
            if(firstHealthy==null)firstHealthy=frame.receivedAtNanos
            healthyFrames++
            val visible=raw.objects.filter {WorldStateValidation.current(it)&&it.confidence>=.6f}
            val admitted=stable.filter(WorldStateValidation::current)
            val seen=mutableSetOf<Int>()
            for(o in visible) {
                val exact=entries[o.trackId]?.takeIf {it.entity.cls==o.className&&ContextFeatures.iou(it.entity.box,o.box)>=.10}
                val overlaps=entries.filter {(id,e)->id !in seen&&e.entity.cls==o.className&&ContextFeatures.iou(e.entity.box,o.box)>=.50}
                val previous=exact ?: overlaps.values.singleOrNull()?.takeIf {e->visible.count {it.className==o.className&&ContextFeatures.iou(e.entity.box,it.box)>=.50}==1}
                if(previous!=null) {
                    entries.entries.removeAll {it.value===previous}
                    previous.entity=ContextEntity(o.trackId,o.className,o.confidence,o.box)
                    previous.missingAt=null
previous.misses=0
entries[o.trackId]=previous
                    seen+=o.trackId
                }else if(admitted.any {it.trackId==o.trackId}) {
                    entries[o.trackId]=Entry(ContextEntity(o.trackId,o.className,o.confidence,o.box))
                    seen+=o.trackId
                }
            }
            entries.entries.removeAll {(id,e)->
                if(id in seen)false else {
                    if(e.missingAt==null)e.missingAt=frame.receivedAtNanos
                    e.misses++
                    e.misses>=4&&frame.receivedAtNanos-e.missingAt!!>=750_000_000L
                }
            }
        }
        // Distinguish healthy empty detections from candidates awaiting admission.
        if(entries.isEmpty()&&(healthyFrames<4||frame.receivedAtNanos-(firstHealthy?:frame.receivedAtNanos)<750_000_000L||
            raw.objects.any {WorldStateValidation.current(it)&&it.confidence>=.6f}))return null
        return entries.values.map {it.entity}
    }
}
