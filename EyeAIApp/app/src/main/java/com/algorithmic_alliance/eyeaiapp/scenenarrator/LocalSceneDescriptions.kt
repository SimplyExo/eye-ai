package com.algorithmic_alliance.eyeaiapp.scenenarrator

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.context.*
import com.algorithmic_alliance.eyeaiapp.rel2abs.*
import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*
import java.util.concurrent.Executors

// Context observation runs independently of capture, R4 and speech.
class LocalSceneDescriptions(context: Context,settleNanos: Long?=null,private val clock: ()->Long=System::nanoTime,collectionNanos: Long?=null): AutoCloseable {
    data class State(val enabled: Boolean=false,val busy: Boolean=false,val text: String="",
        val segments: List<NarrationSegment> = emptyList(),
        val model: String="Deterministische Beschreibung + R4",val score: Double=0.0,val latencyMs: Long?=null,
        val sourceFrameId: Long?=null,val error: String?=null,val announcementId: Long=0,val phase: String="Aus",
        val r4Diagnostics: List<R4RawDiagnostic> = emptyList(),val experimentalR4Enabled: Boolean=true,
        val descriptionDetail: SceneDescriptionDetail=SceneDescriptionDetail.STANDARD,val episodeId: Long=0,
        val contextReason: String="WARMUP",val contextScores: ContextScores=ContextScores(),val shadow: Boolean=false,
        val captureChannels: Map<String,String> = emptyMap(),val collecting: Boolean=false,
        val collectionRemainingMs: Long=0,val objectHistory: ContextHistoryStats=ContextHistoryStats())
    private val appContext=context.applicationContext
    val recorder=ContextRecorder(appContext)
    val observing get() = mutable.value.enabled || recorder.state.value.active
    private val mutable=MutableStateFlow(State())
    val state=mutable.asStateFlow()
    private val lock=Any()
private val executor=Executors.newSingleThreadExecutor()
    private val dispatcher=executor.asCoroutineDispatcher()
    private val scope=CoroutineScope(SupervisorJob()+dispatcher)
    private val policy=ContextEngine(runCatching {
        val root=Json.parseToJsonElement(appContext.assets.open("context-policy.json").bufferedReader().use {it.readText()}).jsonObject
        require(root.getValue("schema").jsonPrimitive.content in setOf("eyeai-context-policy-1","eyeai-context-policy-2"))
        Json.decodeFromJsonElement<ContextPolicyConfig>(root.getValue("config"))
    }.getOrElse {ContextPolicyConfig(generation=2)}.let {config->
        val wait=collectionNanos ?: if(settleNanos==0L)0L else config.collectionNanos
        config.copy(settleNanos=settleNanos?:config.settleNanos,visualSettleNanos=maxOf(settleNanos?:config.settleNanos,config.visualSettleNanos),
            collectionNanos=wait,minimumCollectionNanos=minOf(wait,config.minimumCollectionNanos),quietNanos=if(wait==0L)0 else config.quietNanos)
    })
    private val featureHistory=ContextFeatureHistory()
    private var collecting=false
private var collectionStarted=0L
    private var lastAdmissionChange=0L
    private var lastAdmissionSet=emptySet<Int>()
    private val evidence=StableNarrationEvidence()
    private var model: LocalNarratorModel?=null
    private var closed=false
private var revision=0L
private var sequence=0L
    private var busy=false
    private var useR4=true
private var detail=SceneDescriptionDetail.STANDARD
    private var shape: List<Long>?=null
    private var lastCamera=Long.MIN_VALUE
    private var lastCapture=Long.MIN_VALUE
    private var pending: ContextSnapshot?=null
    private var unspoken: ContextDecision?=null
    private var lastVisual: ContextFeatures.Visual?=null
    private var admittedIds=emptySet<Int>()
    private var backendConfiguration: String?=null
    private var lastSpeechStart: Long?=null
    private var textReference: ContextObservation?=null
private var textCreated=0L
    private var spokenId=0L
    private val json=Json {encodeDefaults=true
    explicitNulls=true}
    private fun resetPolicy() {
        policy.reset()
featureHistory.reset()
collecting=false
        lastAdmissionSet=emptySet()
        recorder.event("POLICY_STATE") { json.encodeToJsonElement(policy.snapshot()) }
    }
    fun recordConfiguration()=synchronized(lock) {
        recorder.event("POLICY_CONFIG") { json.encodeToJsonElement(policy.config) }
        recorder.event("POLICY_STATE") { json.encodeToJsonElement(policy.snapshot()) }
    }
    fun markContext(label: String)=synchronized(lock) {
        recorder.event("MANUAL_LABEL",frame=policy.latest?.frame) { buildJsonObject {put("label",label)
        put("episode",mutable.value.episodeId)} }
    }
    fun setEnabled(value: Boolean)=synchronized(lock) {
        if(closed||mutable.value.enabled==value)return@synchronized
        invalidate()
evidence.reset()
resetPolicy()
        shape=null
        mutable.value=State(enabled=value,busy=busy,experimentalR4Enabled=useR4,descriptionDetail=detail,shadow=mutable.value.shadow,
            phase=if(value)"Umgebung wird beobachtet …" else "Aus")
    }
    fun stop()=setEnabled(false)
    private fun invalidate() {revision++
pending=null
unspoken=null
textReference=null
spokenId=0
lastCapture=Long.MIN_VALUE
collecting=false
    mutable.update {it.copy(collecting=false,collectionRemainingMs=0)}}
    fun setExperimentalR4Enabled(value: Boolean)=synchronized(lock) {
        if(useR4==value)return@synchronized
        useR4=value
invalidate()
        resetPolicy()
        mutable.update {it.copy(experimentalR4Enabled=value,text="",announcementId=0,episodeId=0)}
    }
    fun setDetail(value: SceneDescriptionDetail)=synchronized(lock) {
        if(detail==value)return@synchronized
        detail=value
invalidate()
        resetPolicy()
        mutable.update {it.copy(descriptionDetail=value,text="",announcementId=0,episodeId=0)}
    }
    fun setShadow(value: Boolean)=synchronized(lock) {
        if(mutable.value.shadow==value)return@synchronized
        invalidate()
resetPolicy()
        mutable.update {it.copy(shadow=value,announcementId=0,episodeId=0,text="")}
    }
    fun setBackendConfiguration(key: String)=synchronized(lock) {
        if(backendConfiguration==null) {backendConfiguration=key
        return@synchronized}
        if(backendConfiguration==key)return@synchronized
        backendConfiguration=key
invalidate()
resetPolicy()
        evidence.reset()
        mutable.update {it.copy(text="",announcementId=0,episodeId=0,phase="Backend geändert · Kontext wird neu geprüft")}
        recorder.event("CONFIGURATION_RESET") { JsonPrimitive(key) }
    }
    // Called on raw source frames, not only when a model happens to finish.
    fun onCameraFrame(bitmap: Bitmap,stamp: FrameStamp,snapshot: ()->WorldState,motion: Double?=null): ContextSnapshot?=synchronized(lock) {
        if(closed||!observing)return@synchronized null
        val geometry=listOf(stamp.sessionId,stamp.width.toLong(),stamp.height.toLong(),stamp.rotationDegrees.toLong())
        if(shape!=geometry) {invalidate()
resetPolicy()
evidence.reset()
shape=geometry
        lastCamera=Long.MIN_VALUE
            mutable.update {it.copy(text="",announcementId=0,episodeId=0,sourceFrameId=null,phase="Umgebung wird beobachtet …")} }
        val now=clock()
        if(lastCamera!=Long.MIN_VALUE&&now-lastCamera in 0..199_999_999L)return@synchronized null
        lastCamera=now
        val world=snapshot()
        val stable=evidence.update(world)
        admittedIds=stable.filter {WorldStateValidation.current(it)}.map {it.trackId}.toSet()
        val visual=ContextFeatures.visual(bitmap)
        lastVisual=visual
        if(admittedIds!=lastAdmissionSet||evidence.pendingCount>0) {lastAdmissionChange=now
        lastAdmissionSet=admittedIds}
        val observation=ContextFeatures.observe(stamp,now,world,stable,visual,motion,if(policy.config.generation==2)featureHistory else null,evidence.pendingCount,evidence.transientCount,legacy=policy.config.generation==1)
        val decision=policy.observe(observation)
        recorder.observation(observation,policy.scores,policy.reason)
        recorder.image(stamp,bitmap)
        mutable.update {it.copy(score=policy.scores.combined,contextReason=policy.reason,contextScores=policy.scores,objectHistory=observation.history)}
        if(decision!=null) {
            revision++
unspoken=decision
pending=null
spokenId=0
collecting=true
collectionStarted=now
            lastAdmissionChange=now
            mutable.update {it.copy(episodeId=decision.episodeId,text="",announcementId=0,phase="Neuer Kontext bestätigt · Objekte werden gesammelt …",collecting=true)}
            recorder.event("CONTEXT_CONFIRMED",frame=stamp) { json.encodeToJsonElement(decision) }
            if(policy.config.collectionNanos==0L) {collecting=false
            return@synchronized capture(decision,observation,bitmap)}
        }
        if(collecting) {
            val elapsed=now-collectionStarted
            val quiet=now-lastAdmissionChange>=policy.config.quietNanos&&evidence.pendingCount==0
            if(elapsed>=policy.config.collectionNanos || elapsed>=policy.config.minimumCollectionNanos&&quiet) {
                collecting=false
                recorder.event("COLLECTION_FINISHED",frame=stamp) { buildJsonObject {put("episode",mutable.value.episodeId)
put("elapsedNanos",elapsed)
                put("deadline",elapsed>=policy.config.collectionNanos)} }
            }
            mutable.update {it.copy(collecting=collecting,collectionRemainingMs=((policy.config.collectionNanos-elapsed).coerceAtLeast(0)/1_000_000),
                phase=if(collecting)"Neue Umgebung · Objekte werden bestätigt …" else "Aufnahme wird gebunden …")}
        }
        if(!collecting)tryNarrate(now)
        val episode=unspoken
        if(episode!=null&&!collecting&&!busy&&policy.matches(episode.observation)&&admittedIds.isNotEmpty()&&
            (pending==null || pending!!.timedOut(now) || pending!!.narrationIssued && now-textCreated>=1_000_000_000L)&&
            (lastCapture==Long.MIN_VALUE||now-lastCapture>=1_000_000_000L)) {
            return@synchronized capture(episode,observation,bitmap)
        }
        null
    }
    private fun capture(decision: ContextDecision,o: ContextObservation,bitmap: Bitmap): ContextSnapshot {
        pending?.let {recorder.event("CAPTURE_SUPERSEDED",frame=it.frame) { buildJsonObject {put("episode",it.episodeId)} }}
        val capture=ContextSnapshot(decision,o.frame,o)
pending=capture
lastCapture=clock()
        mutable.update {it.copy(collecting=false,collectionRemainingMs=0)}
        recorder.image(o.frame,bitmap,true)
        recorder.event("CAPTURE_REQUESTED",frame=o.frame) { buildJsonObject {
            put("episode",decision.episodeId)
put("windowStartNanos",o.atNanos-10_000_000_000L)
            put("windowEndNanos",o.atNanos+5_000_000_000L)
        } }
        scope.launch {delay(2000L)
        synchronized(lock) {
            if(pending===capture) {capture.finishTimeout()
tryNarrate(clock())
            recorder.event("CAPTURE_STATUS",frame=capture.frame) { json.encodeToJsonElement(capture.statuses()) }}
        }}
        return capture
    }
    fun capturedObjects(stamp: FrameStamp,value: DetectionFrame)=synchronized(lock) {
        pending?.takeIf {it.frame==stamp}?.objects(value)
        tryNarrate(clock())
    }
    fun capturedDepth(stamp: FrameStamp,raw: FloatArray?,meters: FloatArray?,width: Int,height: Int,relativeMethod: String?,metricMethod: String?,metric: MetricDepthFrame?=null)=synchronized(lock) {
        pending?.takeIf {it.frame==stamp}?.depth(raw,meters,width,height,relativeMethod,metricMethod,metric)
        tryNarrate(clock())
    }
    fun capturedSegmentation(stamp: FrameStamp,value: SegmentationContextFrame)=synchronized(lock) {
        pending?.takeIf {it.frame==stamp}?.segmentation(value)
        tryNarrate(clock())
    }
    fun capturedStatus(stamp: FrameStamp,channel: String,status: String)=synchronized(lock) {
        pending?.takeIf {it.frame==stamp}?.channel(channel,status)
        tryNarrate(clock())
    }
    // Model callbacks maintain temporal evidence, triggers do not depend on callback order.
    fun onFrame(bitmap: Bitmap,stamp: FrameStamp,snapshot: ()->WorldState)=synchronized(lock) {
        if(closed||!observing)return@synchronized
        val stable=evidence.update(snapshot())
        admittedIds=stable.filter {WorldStateValidation.current(it)}.map {it.trackId}.toSet()
        tryNarrate(clock())
    }
    private fun tryNarrate(now: Long) {
        if(!observing)return
        val capture=pending ?: return
        if(capture.timedOut(now))capture.finishTimeout()
        mutable.update {it.copy(captureChannels=capture.statuses())}
        if(recorder.state.value.active&&!capture.fusionRecorded&&capture.statuses()["objects"]!="PENDING"&&capture.statuses()["depth"]!="PENDING"&&(capture.statuses()["segmentation"]!="PENDING"||capture.timedOut(now))) {
            capture.fusionRecorded=true
            for((id,values) in capture.fusionTrace())recorder.event("REL2ABS_OBJECT_DIAGNOSTIC",buildJsonObject {
                put("track",id)
                put("used_for_narration",false)
                put("values",buildJsonObject {values.forEach {(k,v)->put(k,if(v.isFinite())JsonPrimitive(v)else JsonNull)}})
            },capture.frame)
        }
        if(closed||collecting||busy||capture.narrationIssued||!capture.ready(now)||!policy.matches(capture.reference))return
        val input=try {
            val world=capture.world(now)
            val objects=world.objects.filter {WorldStateValidation.current(it)&&it.trackId in admittedIds}
            if(objects.isEmpty()) {mutable.update {it.copy(phase="Warte auf bestätigte Umgebung …",contextReason="NO_SPEAKABLE_EVIDENCE")}
            return}
            val stable=StableNarrationEvidence.subset(world,objects)
            TinyWorldProjection().project(stable)
        }catch(cancelled: CancellationException) {throw cancelled}
        catch(error: Exception) {
            // This runs on camera/model callbacks too, before the generator's coroutine catch.
            // Reject the invalid snapshot once, the next matching camera frame can recapture it.
            capture.narrationIssued=true
pending=null
lastCapture=now
            textReference=null
            val message="${error.javaClass.simpleName}: ${error.message.orEmpty()}"
            mutable.update {it.copy(text="",announcementId=0,error=message,phase="Warte auf eine aktuelle Umgebung …")}
            Log.w(EyeAIApp.APP_LOG_TAG,"Invalid narration snapshot rejected: episode=${capture.episodeId}, frame=${capture.frame.frameId}",error)
            recorder.event("NARRATION_INPUT_REJECTED",frame=capture.frame) { buildJsonObject {put("error",message)
            put("episode",capture.episodeId)} }
            return
        }
        val requestRevision=revision
val requestDetail=detail
        val requestR4=useR4
        capture.narrationIssued=true
busy=true
        mutable.update {it.copy(busy=true,phase="Orientierung wird erstellt …")}
        if(recorder.state.value.active) {
            recorder.event("NARRATION_INPUT",Json.parseToJsonElement(TinyWorldCodec.encode(input)),capture.frame)
        }
        scope.launch {
            val started=clock()
            try {
                val runner=model ?: LocalNarratorModel(appContext).also {model=it}
                val trace: ((String,Int,Int,FloatArray)->Unit)?=if(recorder.state.value.active) {name,w,h,values->recorder.floats(capture.frame,name,w,h,values)}else null
                val result=runner.generate(input,requestDetail.name,requestR4,trace)
                if(trace!=null)recorder.event("R4_FULL_DIAGNOSTICS",json.encodeToJsonElement(result.r4AllDiagnostics),capture.frame)
                synchronized(lock) {
                    if(!closed&&revision==requestRevision&&pending===capture&&policy.matches(capture.reference)) {
                        textCreated=clock()
                        textReference=capture.reference
                        recorder.event("NARRATION_RESULT",frame=capture.frame) { buildJsonObject {put("episode",capture.episodeId)
put("text",result.text)
put("r4Executed",result.r4ModelExecuted)
                        put("error",result.error)} }
                        if(mutable.value.enabled&&!mutable.value.shadow&&result.text.isNotBlank()) {
                            sequence++
                            mutable.update {it.copy(text=result.text,segments=result.segments,announcementId=sequence,sourceFrameId=capture.frame.frameId,
                                model=result.modelName,latencyMs=(clock()-started)/1_000_000,error=result.error,r4Diagnostics=result.r4Diagnostics,
                                phase="Orientierung bereit · wartet auf Sprachstart")}
                        }else {if(mutable.value.shadow)unspoken=null
                        mutable.update {it.copy(text=result.text,segments=result.segments,sourceFrameId=capture.frame.frameId,r4Diagnostics=result.r4Diagnostics,
                            phase=if(it.shadow)"Schattenanalyse · keine Ansage" else "Keine formulierbare Evidenz")}}
                    }else recorder.event("STALE_NARRATION_REJECTED",frame=capture.frame)
                }
            }catch(e:Exception) {synchronized(lock) {if(revision==requestRevision)mutable.update {it.copy(error=e.message,phase="Orientierung derzeit nicht verfügbar")}}}
            finally {synchronized(lock) {busy=false
            mutable.update {it.copy(busy=false)}}}
        }
    }
    fun isCurrentAnnouncement(id: Long): Boolean=synchronized(lock) {
        !closed&&mutable.value.enabled&&!mutable.value.shadow&&id>0&&mutable.value.announcementId==id&&
            (spokenId==id || textReference?.let {policy.matches(it)&&policy.latest?.visualUsable!=false&&clock()-it.frame.receivedAtNanos<1_000_000_000L}==true)
    }
    fun speechStartAllowed(): Boolean=synchronized(lock) {lastSpeechStart?.let {clock()-it>=policy.config.minimumIntervalNanos}?:true}
    fun speechStarted(id: Long)=synchronized(lock) {
        if(!isCurrentAnnouncement(id))return@synchronized
        spokenId=id
unspoken=null
        lastSpeechStart=clock()
        textReference?.let {reference->
            policy.speechStarted(reference)
            recorder.event("SPEECH_REFERENCE") { json.encodeToJsonElement(reference) }
        }
        recorder.event("SPEECH_STARTED") { buildJsonObject {put("announcement",id)
        put("episode",mutable.value.episodeId)} }
        mutable.update {it.copy(phase="Orientierung wird vorgelesen …")}
    }
    fun speechCancelled(id: Long)=synchronized(lock) {
        if(mutable.value.announcementId==id) {
            policy.speechFinished()
            mutable.update {it.copy(phase="Orientierung abgebrochen")}
        }
    }
    fun speechFinished(id: Long)=synchronized(lock) {
        recorder.event("SPEECH_FINISHED") { buildJsonObject {put("announcement",id)} }
        if(mutable.value.announcementId==id) {
            policy.speechFinished()
            recorder.event("SPEECH_REFERENCE_FINISHED")
            mutable.update {it.copy(phase="Orientierung abgeschlossen · wartet auf neue Ansicht")}
        }
    }
    override fun close() {
        synchronized(lock) {if(closed)return
closed=true
invalidate()
        mutable.value=State()}
        executor.execute {model?.close()
model=null
scope.cancel()
dispatcher.close()
recorder.close()}
        executor.shutdown()
    }
}
