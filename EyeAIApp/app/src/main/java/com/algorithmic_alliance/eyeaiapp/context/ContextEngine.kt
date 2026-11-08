package com.algorithmic_alliance.eyeaiapp.context

import com.algorithmic_alliance.eyeaiapp.worldstate.FrameStamp
import com.algorithmic_alliance.eyeaiapp.worldstate.Box
import kotlinx.serialization.Serializable
import kotlin.math.abs

@Serializable data class ContextEntity(val id: Int,val cls: String,val confidence: Float,val box: Box)
@Serializable data class ContextHistoryStats(val pending: Int=0,val transient: Int=0,val confirmed: Int=0)
@Serializable data class ContextObservation(
    val frame: FrameStamp, val atNanos: Long,
    val objects: Map<String,Float>? = null, val layout: Map<String,Float>? = null,
    val visual: List<Float>? = null, val depth: List<Float?>? = null,
    val semantics: List<Float>? = null, val visualUsable: Boolean = true,
    val channelFrames: Map<String,FrameStamp?> = emptyMap(),
    val motion: Double? = null,
    val featureVersion: String = "context-features-1",
    val entities: List<ContextEntity>? = null,
    val quality: Map<String,Double> = emptyMap(),
    val history: ContextHistoryStats = ContextHistoryStats(),
)
@Serializable data class ContextScores(val objects: Double? = null,val layout: Double? = null,
    val visual: Double? = null,val depth: Double? = null,val semantics: Double? = null,val combined: Double = 0.0,
    val reliability: Map<String,Double> = emptyMap(),val weights: Map<String,Double> = emptyMap(),
    val coverage: Double=0.0,val support: Int=0,val additions: Int=0,val removals: Int=0)
@Serializable data class ContextDecision(val episodeId: Long,val reason: String,val observation: ContextObservation,
    val scores: ContextScores,val policyVersion: String = "context-v1-unvalidated")
@Serializable data class ContextPolicyConfig(
    val settleNanos: Long = 1_200_000_000L,val visualSettleNanos: Long = 2_000_000_000L,
    val minimumIntervalNanos: Long = 2_000_000_000L,val semanticThreshold: Double = .6,
    val combinedThreshold: Double = .45,val visualThreshold: Double = .65,
    val generation: Int = 1,
    val objectWeight: Double=.45,val imageWeight: Double=.25,val depthWeight: Double=.15,val segmentationWeight: Double=.15,
    val minimumCoverage: Double=.50,val supportThreshold: Double=.20,val resetThreshold: Double=.25,
    val maximumImageWeight: Double=.35,
    val collectionNanos: Long=2_000_000_000L,val minimumCollectionNanos: Long=750_000_000L,val quietNanos: Long=500_000_000L,
) { init { require(settleNanos>=0 && visualSettleNanos>=settleNanos && minimumIntervalNanos>=0)
    require(listOf(semanticThreshold,combinedThreshold,visualThreshold,minimumCoverage,supportThreshold,resetThreshold,maximumImageWeight).all {it.isFinite()&&it in 0.0..1.0})
    require(generation in 1..2 && (generation==1||resetThreshold<combinedThreshold))
    require(listOf(objectWeight,imageWeight,depthWeight,segmentationWeight).all {it.isFinite()&&it>0})
    require(generation==1||maximumImageWeight<combinedThreshold)
    require(collectionNanos>=minimumCollectionNanos && minimumCollectionNanos>=0 && quietNanos>=0) } }

@Serializable data class ContextEngineState(val baseline: ContextObservation?,val latest: ContextObservation?,
    val lastDecisionAt: Long?,val nextEpisode: Long,val window: List<ContextObservation>,val pending: Boolean,
    val spoken: ContextObservation?=null,val completed: Boolean=false)

// Pure, replayable policy. Optional channels never become zero-valued observations.
class ContextEngine(val config: ContextPolicyConfig = ContextPolicyConfig()) {
    private var baseline: ContextObservation? = null
    private var spoken: ContextObservation? = null
    private var completed=false
    private var session: Triple<Long,Int,Int>? = null
    private var rotation: Int? = null
    private var lastTime: Long? = null
    private var lastDecisionAt: Long? = null
    private val window=ArrayDeque<ContextObservation>()
    // Observations are immutable, identity keys avoid hashing their feature vectors.
    private val agreements=java.util.IdentityHashMap<ContextObservation,Boolean>()
    private var agreementReference: ContextObservation?=null
    private var agreementBaseline: ContextObservation?=null
    private fun clearWindow() { window.clear()
agreements.clear()
agreementReference=null
    agreementBaseline=null }
    private var nextEpisode=0L
    var latest: ContextObservation? = null
    private set
    var scores=ContextScores()
    private set
    var reason="WARMUP"
    private set
    var pending=false
    private set
    fun reset() { baseline=null
spoken=null
completed=false
session=null
rotation=null
lastTime=null
lastDecisionAt=null
clearWindow()
latest=null
pending=false
    reason="RESET" }
    fun snapshot()=ContextEngineState(baseline,latest,lastDecisionAt,nextEpisode,window.toList(),pending,spoken,completed)
    fun restore(state: ContextEngineState) {
        reset()
spoken=state.spoken
completed=state.completed
baseline=state.baseline
latest=state.latest
lastDecisionAt=state.lastDecisionAt
        nextEpisode=state.nextEpisode
        window.addAll(state.window)
        pending=state.pending
        state.latest?.let {session=Triple(it.frame.sessionId,it.frame.width,it.frame.height)
rotation=it.frame.rotationDegrees
        lastTime=it.atNanos}
    }
    fun observe(o: ContextObservation): ContextDecision? {
        val shape=Triple(o.frame.sessionId,o.frame.width,o.frame.height)
        if(session!=shape || rotation!=o.frame.rotationDegrees || lastTime?.let {o.atNanos<it}==true) {
            reset()
session=shape
            rotation=o.frame.rotationDegrees
        }
        // Missing camera observations do not prove that the view stayed stable.
        if(lastTime?.let {o.atNanos-it>1_000_000_000L}==true)clearWindow()
        lastTime=o.atNanos
        latest=o
        if(config.generation==2 && o.featureVersion!="context-features-2") {
            clearWindow()
scores=ContextScores()
reason="INCOMPATIBLE_FEATURE_VERSION"
            return null
        }
        scores=baseline?.let { if(config.generation==2)fused(it,o,config)else distance(it,o) } ?: ContextScores()
        val b=baseline
        val strong=if(config.generation==2)when {
            b==null && (o.objects?.isNotEmpty()==true || o.visualUsable&&o.visual!=null) -> "INITIAL"
            scores.coverage<config.minimumCoverage -> null
            scores.support<2 -> null
            scores.combined>=config.combinedThreshold -> "FUSED_CHANGE"
            else -> null
        }else when {
            b==null && (o.objects?.isNotEmpty()==true || o.visualUsable && (o.objects!=null || o.visual!=null)) -> "INITIAL"
            (scores.objects ?: 0.0)>=config.semanticThreshold && (o.objects?.isNotEmpty()==true || o.visualUsable) -> "OBJECT_CHANGE"
            (o.visualUsable || o.objects?.isNotEmpty()==true) && scores.combined>=config.combinedThreshold && listOf(scores.objects,scores.layout,scores.visual,scores.depth,scores.semantics).count {(it ?: 0.0)>=.25}>=2 -> "MULTIMODAL_CHANGE"
            o.visualUsable && (scores.visual ?: 0.0)>=config.visualThreshold -> "VISUAL_CHANGE"
            else -> null
        }
        if(strong==null && config.generation==2) {
            reason=when {
                scores.coverage<config.minimumCoverage -> "INSUFFICIENT_COVERAGE"
                scores.combined<=config.resetThreshold -> "UNCHANGED"
                scores.support<2 -> "INSUFFICIENT_SUPPORT"
                else -> "BELOW_THRESHOLD"
            }
            if(scores.combined<=config.resetThreshold || scores.coverage<config.minimumCoverage) {
                pending=false
                clearWindow()
            }else if(pending) {
                if(window.lastOrNull()?.frame!=o.frame)window.addLast(o)
                while(window.size>32)window.removeFirst()
            }
            return null
        }
        if(strong==null) {
            // Do not expire a transition while optional channels or the speech interval are pending.
            if(listOf(scores.objects,scores.layout,scores.visual,scores.depth,scores.semantics).all {it==null}) {clearWindow()
reason="NO_USABLE_CHANNELS"
            return null}
            if(scores.combined<=.15 && (scores.objects ?: 0.0)<=.15) { pending=false
clearWindow()
            reason="UNCHANGED" }
            else {clearWindow()
            reason=if(pending) "WAITING_CONFIRMATION" else "BELOW_THRESHOLD"}
            return null
        }
        pending=true
        if(window.lastOrNull()?.frame!=o.frame) window.addLast(o)
        val hold=if(strong=="VISUAL_CHANGE" || strong=="INITIAL" && o.objects==null)config.visualSettleNanos else config.settleNanos
        // A fixed reference prevents slow-pan drift. A single detector outlier
        // must not restart the whole timer: require 80% agreement instead.
        while(window.size>32)window.removeFirst()
        val reference=window.first()
        if(agreementReference!==reference || agreementBaseline!==b) {
            agreements.clear()
agreementReference=reference
            agreementBaseline=b
        }
        val agreeing=window.count {candidate->agreements.getOrPut(candidate) {
            stabilityDistance(reference,candidate)<=.20 && (config.generation==1||b==null||fused(b,candidate,config).let {
                it.combined>=config.combinedThreshold&&it.coverage>=config.minimumCoverage&&it.support>=2
            })
        }}
        if(window.size>=4 && agreeing*2<=window.size) {clearWindow()
window.addLast(o)
reason="SETTLING"
        return null}
        if(agreeing<4 || agreeing*5<window.size*4 || stabilityDistance(reference,o)>.20 || o.atNanos-reference.atNanos<hold) { reason="SETTLING"
        return null }
        if(lastDecisionAt?.let { o.atNanos-it<config.minimumIntervalNanos }==true) { reason="MINIMUM_INTERVAL"
        return null }
        nextEpisode++
lastDecisionAt=o.atNanos
baseline=o
clearWindow()
pending=false
        reason=strong
        return ContextDecision(nextEpisode,strong,o,scores,if(config.generation==2)"context-v2-unvalidated" else "context-v1-unvalidated")
    }
    // Last confirmed immutable context, not a mutable image buffer at generation completion.
    fun matches(reference: ContextObservation): Boolean {
        val now=latest ?: return false
        if(now.frame.sessionId!=reference.frame.sessionId || now.frame.width!=reference.frame.width || now.frame.height!=reference.frame.height || now.frame.rotationDegrees!=reference.frame.rotationDegrees)return false
        val d=distance(reference,now)
        if(config.generation==2) {
            // Admission of additional objects is expected during the collection window.
            // Content discontinuity or loss of the old class inventory invalidates old text.
            val old=reference.objects.orEmpty()
            val current=now.objects
            val retained=if(current==null||old.isEmpty())1.0 else old.keys.sumOf {minOf(old[it]?:0f,current[it]?:0f).toDouble()}/old.values.sum().coerceAtLeast(1f)
            return (d.visual?:0.0)<config.visualThreshold && !((d.depth?:0.0)>.65&&(d.semantics?:0.0)>.40) && retained>=.40
        }
        return (d.objects ?: 0.0)<config.semanticThreshold && d.combined<config.combinedThreshold && (d.visual ?: 0.0)<config.visualThreshold
    }
    fun speechStarted(reference: ContextObservation) {
        spoken=reference
baseline=reference
completed=false
clearWindow()
        pending=false
    }
    fun speechFinished() { completed=true }
    private fun stabilityDistance(a: ContextObservation,b: ContextObservation): Double {
        if(config.generation==1)return candidateDistance(a,b)
        val d=distance(a,b)
        // Sensor motion never becomes positive scene-change evidence.
        return maxOf(d.visual?:0.0,d.depth?:0.0,d.semantics?:0.0)
    }
    companion object {
        fun mapDistance(a: Map<String,Float>,b: Map<String,Float>): Double {
            val keys=a.keys+b.keys
            val union=keys.sumOf {maxOf(a[it]?:0f,b[it]?:0f).toDouble()}
            return if(union<=0)0.0 else (1-keys.sumOf {minOf(a[it]?:0f,b[it]?:0f).toDouble()}/union).coerceIn(0.0,1.0)
        }
        private fun vector(a: List<Float?>,b: List<Float?>,start: Int=0,end: Int=a.size): Double? {
            if(a.size!=b.size)return null
            var sum=0.0
            var count=0
            for(i in start until end) {
                val x=a[i]
                val y=b[i]
                if(x!=null&&y!=null&&x.isFinite()&&y.isFinite()) {sum+=abs(x-y).toDouble()
                count++}
            }
            return if(count>=4)(sum/count).coerceIn(0.0,1.0)else null
        }
        fun distance(a: ContextObservation,b: ContextObservation): ContextScores {
            val ob=if(a.objects!=null&&b.objects!=null)mapDistance(a.objects,b.objects)else null
            val lo=if(a.layout!=null&&b.layout!=null)mapDistance(a.layout,b.layout)else null
            val vi=if(a.visualUsable&&b.visualUsable&&a.visual!=null&&b.visual!=null)if(a.visual.size==800&&b.visual.size==800) {
                maxOf((vector(a.visual,b.visual,0,768)?:0.0)*3,
                    (vector(a.visual,b.visual,768,800)?:0.0)*2).coerceAtMost(1.0)
            } else vector(a.visual,b.visual)?.let {(it*3).coerceAtMost(1.0)}else null
            val de=if(a.depth!=null&&b.depth!=null)vector(a.depth,b.depth)else null
            val se=if(a.semantics!=null&&b.semantics!=null)vector(a.semantics,b.semantics)?.let {(it*9.5).coerceAtMost(1.0)}else null
            val values=listOf(ob to .35,lo to .25,vi to .20,de to .15,se to .05).filter {it.first!=null}
            val total=values.sumOf {it.second}
            val score=if(total>0)values.sumOf {it.first!!*it.second}/total else 0.0
            return ContextScores(ob,lo,vi,de,se,score)
        }
        fun fused(a: ContextObservation,b: ContextObservation,c: ContextPolicyConfig): ContextScores {
            val legacy=distance(a,b)
            var additions=0
            var removals=0
            val inventory=if(a.entities!=null&&b.entities!=null) {
                val unmatched=b.entities.toMutableList()
                for(old in a.entities) {
                    val same=unmatched.singleOrNull {it.id==old.id&&it.cls==old.cls&&ContextFeatures.iou(it.box,old.box)>=.10}
                    val overlaps=unmatched.filter {it.cls==old.cls&&ContextFeatures.iou(it.box,old.box)>=.50}
                    val match=same ?: overlaps.singleOrNull()?.takeIf {candidate->a.entities.count {it.cls==candidate.cls&&ContextFeatures.iou(it.box,candidate.box)>=.50}==1}
                    if(match==null)removals++ else unmatched.remove(match)
                }
                additions=unmatched.size
                val denominator=maxOf(1,a.entities.size,b.entities.size)
                ((additions+removals).toDouble()/denominator).coerceIn(0.0,1.0)
            }else null
            // Related YOLO signals count as one family, not separate corroboration.
            val ob=legacy.objects?.let { counts->
                val layout=legacy.layout ?: counts
                if(inventory==null).65*counts+.35*layout else .50*inventory+.30*counts+.20*layout
            }
            val novelties=mapOf("objects" to ob,"visual" to legacy.visual,"depth" to legacy.depth,"semantics" to legacy.semantics)
            val rawBase=mapOf("objects" to c.objectWeight,"visual" to c.imageWeight,"depth" to c.depthWeight,"semantics" to c.segmentationWeight)
            val baseTotal=rawBase.values.sum()
            val q=novelties.mapValues {(key,value)->if(value==null)0.0 else minOf(a.quality[key]?:0.0,b.quality[key]?:0.0).coerceIn(0.0,1.0)}
            val effective=rawBase.mapValues {(key,value)->value/baseTotal*q.getValue(key)}
            val coverage=effective.values.sum()
            val weights=effective.mapValues {if(coverage>0)it.value/coverage else 0.0}.toMutableMap()
            val image=weights.getValue("visual")
            if(image>c.maximumImageWeight) {
                val others=1-image
                weights["visual"]=c.maximumImageWeight
                for(key in weights.keys.filter {it!="visual"})weights[key]=if(others>1e-9)weights.getValue(key)*(1-c.maximumImageWeight)/others else 0.0
            }
            val score=novelties.entries.sumOf {(key,value)->(value?:0.0)*weights.getValue(key)}
            val support=novelties.count {(key,value)->value!=null&&value>=c.supportThreshold&&q.getValue(key)>=.25}
            return legacy.copy(objects=ob,combined=score,reliability=q,weights=weights,coverage=coverage,support=support,additions=additions,removals=removals)
        }
        private fun candidateDistance(a: ContextObservation,b: ContextObservation): Double {
            val d=distance(a,b)
            return maxOf(d.visual?:0.0,((d.objects?:0.0)+(d.layout?:0.0))*.5)
        }
    }
}
