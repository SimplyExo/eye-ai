package com.algorithmic_alliance.eyeaiapp.context

import com.algorithmic_alliance.eyeaiapp.rel2abs.*
import com.algorithmic_alliance.eyeaiapp.worldstate.*
import java.nio.FloatBuffer

// One retained source frame, no nearest-timestamp joins and no freshness rebasing.
class ContextSnapshot(val decision: ContextDecision,val frame: FrameStamp,val reference: ContextObservation) {
    private var evaluation=frame.receivedAtNanos
    private val pipeline=WorldStatePipeline(WorldStateAnalyzer(clock={evaluation}))
    private val local=pipeline.analyzer.acceptFrame(frame.sourceTimestampNanos,frame.width,frame.height,frame.rotationDegrees,frame.timeOrigin)
    private val states=linkedMapOf("objects" to "PENDING","depth" to "PENDING","segmentation" to "PENDING")
    private var metricFrame: MetricDepthFrame?=null
    private var detectionFrame: DetectionFrame?=null
    private var segmentationFrame: SegmentationContextFrame?=null
    var fusionRecorded=false
    var narrationIssued=false
    val episodeId get()=decision.episodeId
    fun channel(name: String,status: String) {states[name]=if(states[name]?.startsWith("TIMEOUT")==true||states[name]?.startsWith("LATE_")==true) "LATE_$status" else status}
    fun statuses(): Map<String,String> = states.toMap()
    private fun matches(time: Long,width: Int,height: Int,rotation: Int)=time==frame.sourceTimestampNanos&&width==frame.width&&height==frame.height&&rotation==frame.rotationDegrees
    fun objects(value: DetectionFrame) {
        if(!matches(value.sourceTimestampNanos,value.sourceWidth,value.sourceHeight,value.rotationDegrees)) {channel("objects","SOURCE_MISMATCH")
        return}
        detectionFrame=value
pipeline.objects(local,value)
        channel("objects","AVAILABLE")
    }
    fun depth(raw: FloatArray?,meters: FloatArray?,width: Int,height: Int,relativeMethod: String?,metricMethod: String?,metric: MetricDepthFrame?=null) {
        if(metric!=null&&!matches(metric.sourceTimestampNanos,metric.sourceWidth,metric.sourceHeight,metric.rotationDegrees)) {channel("depth","SOURCE_MISMATCH")
        return}
        metricFrame=metric?.copy(depthMeters=meters ?: metric.depthMeters.copyOf())
        pipeline.analyzer.publishDepth(local,width,height,raw?.let(FloatBuffer::wrap),meters,relativeMethod,metricMethod)
        channel("depth","AVAILABLE")
    }
    fun segmentation(value: SegmentationContextFrame) {
        if(!matches(value.sourceTimestampNanos,value.sourceWidth,value.sourceHeight,value.rotationDegrees)) {channel("segmentation","SOURCE_MISMATCH")
        return}
        segmentationFrame=value
pipeline.segmentation(local,value)
        channel("segmentation",if(value.available)"AVAILABLE" else "INVALID")
    }
    fun ready(now: Long): Boolean = states["objects"]!="PENDING" && (states["depth"]!="PENDING" || now-frame.receivedAtNanos>=350_000_000L)
    fun world(now: Long): WorldState {evaluation=now
    return pipeline.analyzer.snapshot(now)}
    fun timedOut(now: Long): Boolean = now-frame.receivedAtNanos>=2_000_000_000L
    fun fusionTrace(): List<Pair<Int,Map<String,Float>>> {
        val metric=metricFrame ?: return emptyList()
        val detection=detectionFrame ?: return emptyList()
        val features=FloatArray(Rel2AbsContextFeatures.TOTAL_FEATURE_COUNT)
        detection.objectContextFeatures.copyInto(features,endIndex=minOf(detection.objectContextFeatures.size,Rel2AbsContextFeatures.OBJECT_FEATURE_COUNT))
        segmentationFrame?.segmentationFeatures?.let {it.copyInto(features,destinationOffset=Rel2AbsContextFeatures.OBJECT_FEATURE_COUNT,endIndex=minOf(it.size,Rel2AbsContextFeatures.SEGMENTATION_FEATURE_COUNT))}
        return detection.detections.map {o->
            val trace=linkedMapOf<String,Float>()
            val result=MetricDistanceResolver.resolve(o,metric,features,detection.detections,segmentationFrame) {trace.putAll(it)}
            trace["resultAvailable"]=if(result is MetricDistanceResolver.Result.Available)1f else 0f
            if(result is MetricDistanceResolver.Result.Available)trace["resolvedMeters"]=result.meters
            o.trackingId to trace.toMap()
        }
    }
    fun finishTimeout() {states.keys.toList().forEach {if(states[it]=="PENDING")states[it]="TIMEOUT"}}
}
