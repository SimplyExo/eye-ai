package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlinx.serialization.json.*
import kotlin.math.*

object SceneGraphV2 {
    val kinds = listOf("IMAGE_LEFT", "IMAGE_ABOVE", "IMAGE_OVERLAP", "ESTIMATED_NEARER", "POSSIBLE_OCCLUSION")
    data class Edge(val kind: String, val a: Int, val b: Int)
    fun direction(box: List<Int>): Int { val x=box[0]+box[2]
return when {x<698->0
x in 702..1298->1
x>1302->2
    else->3} }
    fun derive(v: TinyWorldInput): JsonObject {
        TinyWorldCodec.validate(v)
        require(v.task != WorldTask.NAVIGATE && !v.navigationSupported)
        val active=v.objects.filter {v.channels[0].status==Availability.AVAILABLE && it.track==TrackStatus.CONFIRMED && it.freshness==Availability.AVAILABLE}
        val byid=active.associateBy {it.id}
        val edges=mutableListOf<Edge>()
        fun edge(k:String,a:Int,b:Int) {edges+=Edge(k,a,b)}
        for(i in active.indices) for(j in i+1 until active.size) {
            val a=active[i]
val b=active[j]
val ab=a.box
            val bb=b.box
            if(ab[2]+25<=bb[0]) edge("IMAGE_LEFT",a.id,b.id)
            if(bb[2]+25<=ab[0]) edge("IMAGE_LEFT",b.id,a.id)
            if(ab[3]+25<=bb[1]) edge("IMAGE_ABOVE",a.id,b.id)
            if(bb[3]+25<=ab[1]) edge("IMAGE_ABOVE",b.id,a.id)
            val intersection=max(0,min(ab[2],bb[2])-max(ab[0],bb[0]))*max(0,min(ab[3],bb[3])-max(ab[1],bb[1]))
            val smaller=min((ab[2]-ab[0])*(ab[3]-ab[1]),(bb[2]-bb[0])*(bb[3]-bb[1]))
            if(smaller>0 && intersection*5>=smaller) edge("IMAGE_OVERLAP",a.id,b.id)
        }
        for(r in v.relations) {
            if(r.a !in byid || r.b !in byid || r.depth !in listOf(DepthOrder.FIRST_NEARER,DepthOrder.SECOND_NEARER)) continue
            val (a,b)=if(r.depth==DepthOrder.FIRST_NEARER) r.a to r.b else r.b to r.a
            if(v.channels[1].status!=Availability.AVAILABLE || byid.getValue(a).source!=byid.getValue(b).source) continue
            edge("ESTIMATED_NEARER",a,b)
            val overlap=edges.any {it.kind=="IMAGE_OVERLAP" && setOf(it.a,it.b)==setOf(a,b)}
            if(overlap && r.sightline==a) edge("POSSIBLE_OCCLUSION",a,b)
        }
        val aligned=v.channels[0].status==Availability.AVAILABLE && v.channels[2].status==Availability.AVAILABLE && v.channels[0].source==v.channels[2].source
        val mismatch=v.channels[0].status==Availability.AVAILABLE && v.channels[2].status==Availability.AVAILABLE && v.channels[0].source!=v.channels[2].source
        val dominant=v.cells.filter {aligned && it.semantics.status==Availability.AVAILABLE && !it.conflict &&
            it.semantics.top.isNotEmpty() && it.semantics.top[0][1]>=550 && (it.semantics.unknown?:0)<=250}
            .associate {it.id to it.semantics.top[0][0]}
        val unseen=dominant.keys.toMutableSet()
        val areas=mutableListOf<JsonObject>()
        while(unseen.isNotEmpty()) {
            val seed=unseen.min()
val cls=dominant.getValue(seed)
val todo=mutableListOf(seed)
val component=mutableListOf<Int>()
            unseen.remove(seed)
            while(todo.isNotEmpty()) {val n=todo.removeAt(todo.lastIndex)
            component+=n
                val neighbors=unseen.filter {dominant[it]==cls && abs(it/4-n/4)+abs(it%4-n%4)==1}.sorted()
                for(m in neighbors) {unseen.remove(m)
                todo+=m}
            }
            if(component.size<2) continue
            val cells=component.sorted()
            val band=if(cells.max()/4<2) 0 else if(cells.min()/4>=2) 2 else 1
            areas+=buildJsonObject {put("class",cls)
put("cells",JsonArray(cells.map(::JsonPrimitive)))
            put("band",band)}
        }
        return buildJsonObject {
            put("version","scene-graph-2.1")
            put("groups",buildJsonArray {
                if(v.channels[0].status==Availability.AVAILABLE) for(c in v.counts.sortedBy {it[0]}) {
                    val members=active.filter {it.cls==c[0]}
                    add(buildJsonObject {put("class",c[0])
put("count",c[1])
                    put("members",JsonArray(members.map {JsonPrimitive(it.id)}))
                        put("directions",JsonArray(members.map {direction(it.box)}.distinct().sorted().map(::JsonPrimitive)))
                        put("complete",c[1]==members.size)})
                }
            })
            put("edges",JsonArray(edges.sortedWith(compareBy<Edge> {kinds.indexOf(it.kind)}.thenBy {it.a}.thenBy {it.b}).map {
                buildJsonObject {put("kind",it.kind)
put("a",it.a)
                put("b",it.b)} }))
            put("areas",JsonArray(areas))
            put("quality",buildJsonObject {put("detector",v.channels[0].status.name)
            put("depth",v.channels[1].status.name)
                put("segmentation",v.channels[2].status.name)
put("omitted_objects",v.loss.objects)
                put("omitted_relations",v.loss.relations)
                put("source_mismatch",mismatch)
                put("depth_conflicts",v.relations.count {it.depth==DepthOrder.CONFLICTING})
                put("historical_objects",v.loss.historicalObjects)})
            put("capabilities",buildJsonObject {put("physical_motion","UNSUPPORTED")
            put("camera_motion","UNSUPPORTED")
                put("physical_connectivity","UNKNOWN")
                put("outside_view","UNKNOWN")})
        }
    }
}

