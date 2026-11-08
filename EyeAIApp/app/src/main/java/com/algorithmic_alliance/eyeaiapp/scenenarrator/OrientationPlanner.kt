package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlinx.serialization.json.*
import kotlin.math.abs
import kotlin.math.max

// Shared spatial analysis for both speaking styles, no inferred motion.
object OrientationPlanner {
    enum class Side { LEFT, RIGHT }
    enum class RangeOrder { NEARER, FARTHER }
    data class Item(val ids: List<Int>,val cls: Int,val count: Int,val direction: Int,val complete: Boolean,val priority: Boolean)
    data class Edge(val a: Int,val b: Int,val type: String,val fact: Int)
    data class Placement(val side: Side?,val depth: RangeOrder?,val beside: Boolean=false,val behind: Boolean=false)
    data class Plan(val anchor: Int?,val indices: List<Int>,val items: List<Item>,val placements: Map<Int,Placement>,
        val edges: List<Edge>,val ranges: Map<Int,Int>,val rangeId: Int?)
    private fun JsonObject.s(k: String)=getValue(k).jsonPrimitive.content
    private fun JsonObject.i(k: String)=getValue(k).jsonPrimitive.int
    private fun args(f: JsonObject)=f.getValue("args").jsonObject

    private fun reliableRange(input: TinyWorldInput,o: TinyObject): Int? {
        val d=o.depth
val dm=d.dm ?: return null
        val spread=d.iqrDm ?: return null
        if(input.channels[1].status!=Availability.AVAILABLE || d.status!=Availability.AVAILABLE || d.source!=o.source ||
            input.channels[1].source!=o.source || d.method==null || d.clipped || dm<=0 || (d.validDecile?:0)<7 ||
            spread>max(3,dm/4) || input.relations.any {it.depth==DepthOrder.CONFLICTING&&o.id in listOf(it.a,it.b)})return null
        return dm
    }
    private fun side(input: TinyWorldInput,anchor: Int,other: Int): Side? {
        val a=input.objects[anchor]
        val b=input.objects[other]
        if(a.source!=b.source)return null
        return when {b.box[2]+25<=a.box[0]->Side.LEFT
a.box[2]+25<=b.box[0]->Side.RIGHT
        else->null}
    }
    fun plan(input: TinyWorldInput,facts: List<JsonObject>,detail: String,referenceable: (Int)->Boolean): Plan {
        require(detail in setOf("SHORT","STANDARD"))
        val active=input.objects.filter {input.channels[0].status==Availability.AVAILABLE && it.track==TrackStatus.CONFIRMED &&
            it.freshness==Availability.AVAILABLE && it.confidenceDecile>=6 && it.box[0]<it.box[2]&&it.box[1]<it.box[3]}
        val ids=active.map {it.id}.toSet()
        val ranges=active.mapNotNull {o->reliableRange(input,o)?.let {o.id to it}}.toMap()
        fun priority(id: Int): Boolean {
            val o=input.objects[id]
            val dm=ranges[id] ?: return false
            // Potential nearby obstacles, not a claim that a collision or free passage is known.
            return dm<=10 || dm<=20 && o.box[0]<650&&o.box[2]>350
        }
        val edges=facts.mapIndexedNotNull {i,f->
            if(f.s("kind")!="RELATION" || args(f).s("relation") !in R4RelationTypes.outputIndex)null else {
                val a=args(f).getValue("a").jsonObject.i("id")
                val b=args(f).getValue("b").jsonObject.i("id")
                if(a in ids&&b in ids)Edge(a,b,args(f).s("relation"),i)else null
            }
        }
        fun score(id: Int): Double {
            val o=input.objects[id]
            val b=o.box
            val area=(b[2]-b[0]).toDouble()*(b[3]-b[1])/1_000_000
            val central=1-abs(b[0]+b[2]-1000)/1000.0
            val landmark=when(o.cls) {0,56,57,59,60->.3
            else->0.0}
            return 1.2*area+.9*central+o.confidenceDecile*.03+landmark+
                (ranges[id]?.let {.4/(1+it/30.0)}?:0.0)+edges.count {id==it.a||id==it.b}.coerceAtMost(3)*.1
        }
        val ranked=compareByDescending<Int> {priority(it)}.thenBy {if(priority(it))ranges[it]?:Int.MAX_VALUE else 0}
            .thenByDescending {score(it)}.thenBy {it}
        val groups=facts.indices.filter {facts[it].s("kind")=="GROUP"}
        val items=groups.flatMap {i->
            val f=facts[i]
val a=args(f)
            val members=f.getValue("records").jsonArray.map {it.jsonPrimitive.int}.filter {it in ids}
            val complete=a.getValue("complete").jsonPrimitive.boolean
            if(members.isEmpty())emptyList() else if(!complete)listOf(Item(members,a.i("class"),a.i("count"),3,false,members.any(::priority)))
            else members.groupBy {SceneGraphV2.direction(input.objects[it].box)}.values.map {sameSector->
                Item(sameSector,a.i("class"),sameSector.size,SceneGraphV2.direction(input.objects[sameSector.first()].box),true,sameSector.any(::priority))
            }
        }
        // A repeated class in one sector is a group, not a uniquely named landmark.
        val candidates=items.filter {it.complete&&it.ids.size==1&&referenceable(it.ids.single())}.map {it.ids.single()}
        val anchor=candidates.sortedWith(ranked).firstOrNull()?.takeIf {id->items.none {it.priority}||priority(id)}
        fun itemScore(item: Item)=item.ids.maxOf(::score)
        val ordered=items.sortedWith(compareByDescending<Item> {it.priority}.thenBy {if(it.priority)it.ids.mapNotNull(ranges::get).minOrNull()?:Int.MAX_VALUE else 0}
            .thenByDescending(::itemScore).thenBy {it.ids.first()})
        val selected=linkedSetOf<Item>()
        // Keep every observed nearby obstacle, even when this exceeds the ordinary inventory budget.
        selected+=ordered.filter {it.priority}
        anchor?.let {a->selected+=items.first {a in it.ids}}
        for(e in edges.filter {it.a==anchor||it.b==anchor}.sortedBy {it.fact}) {
            for(id in listOf(e.a,e.b))if(selected.size<5)items.firstOrNull {id in it.ids}?.let(selected::add)
        }
        for(item in ordered)if(selected.size<5)selected+=item
        val selectedIds=selected.flatMap {it.ids}.toSet()
        val selectedEdges=edges.filter {it.a in selectedIds&&it.b in selectedIds}.take(4)
        val placements=linkedMapOf<Int,Placement>()
        val placeable=selected.filter {it.complete&&it.ids.size==1}.map {it.ids.single()}.toSet()
        if(anchor!=null)for(id in placeable-anchor) {
            val a=input.objects[anchor]
            val b=input.objects[id]
            val relation=input.relations.singleOrNull {setOf(it.a,it.b)==setOf(anchor,id)}
            val depth=relation?.takeIf {r->input.channels[1].status==Availability.AVAILABLE&&input.channels[1].source==a.source&&
                a.source==b.source&&(a.relativeStatus==Availability.AVAILABLE&&b.relativeStatus==Availability.AVAILABLE || anchor in ranges&&id in ranges&&a.depth.method==b.depth.method)&&
                r.depth in setOf(DepthOrder.FIRST_NEARER,DepthOrder.SECOND_NEARER)}?.let {r->
                val nearer=if(r.depth==DepthOrder.FIRST_NEARER)r.a else r.b
                // Contradictory reliable metric values suppress an otherwise asserted order.
                val da=ranges[anchor]
                val db=ranges[id]
                if(da!=null&&db!=null&&a.depth.method==b.depth.method&&(if(nearer==anchor)da>=db else db>=da))null
                else if(nearer==id)RangeOrder.NEARER else RangeOrder.FARTHER
            }
            val lateral=side(input,anchor,id)
            val beside=selectedEdges.any {it.type=="BESIDE"&&setOf(it.a,it.b)==setOf(anchor,id)}
            // "Behind" requires a shared sightline, metric support, depth order AND a bound occlusion edge.
            val behind=depth==RangeOrder.FARTHER&&lateral==null&&anchor in ranges&&id in ranges&&a.depth.method==b.depth.method&&relation?.overlap==true&&relation.sightline==anchor&&
                selectedEdges.any {it.type=="OCCLUDES"&&it.a==anchor&&it.b==id}
            placements[id]=Placement(lateral,depth,beside,behind)
        }
        val sorted=selected.sortedWith(compareBy<Item> {if(anchor in it.ids)0 else if(it.priority)1 else 2}
            .thenBy {if(it.priority)it.ids.mapNotNull(ranges::get).minOrNull()?:Int.MAX_VALUE else 0}
            .thenByDescending(::itemScore).thenBy {it.ids.first()})
        val rangeId=anchor?.takeIf {it in ranges} ?: sorted.firstOrNull {it.complete&&it.ids.size==1&&it.ids.single() in ranges}?.ids?.single()
        val classes=selected.map {it.cls}.toSet()
        val indices=groups.filter {args(facts[it]).i("class") in classes}+selectedEdges.map {it.fact}+
            facts.indices.filter {facts[it].s("kind")=="RELATION"&&args(facts[it]).s("relation")=="ESTIMATED_NEARER"&&
                facts[it].getValue("records").jsonArray.all {it.jsonPrimitive.int in selectedIds}}
        return Plan(anchor,indices.distinct(),sorted,placements,selectedEdges,ranges,rangeId)
    }
}
