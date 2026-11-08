package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlinx.serialization.json.*
import java.security.MessageDigest
import kotlin.math.*

// Independent Kotlin evidence projection, without simulator or language targets.
object TinyWorldEvidence {
    private fun JsonObject.i(k:String)=getValue(k).jsonPrimitive.int
    private fun JsonObject.s(k:String)=getValue(k).jsonPrimitive.content
    private fun JsonObject.b(k:String)=getValue(k).jsonPrimitive.boolean
    private fun JsonObject.a(k:String)=getValue(k).jsonArray
    private fun ints(v:Iterable<Int>)=JsonArray(v.map(::JsonPrimitive))
    private fun canonical(v:JsonElement):String=when(v) {
        is JsonObject -> v.keys.sorted().joinToString(",","{","}") { JsonPrimitive(it).toString()+":"+canonical(v.getValue(it)) }
        is JsonArray -> v.joinToString(",","[","]") {canonical(it)}
        else -> v.toString()
    }
    private fun hash(v:JsonElement):String {
        val bytes=MessageDigest.getInstance("SHA-256").digest(canonical(v).toByteArray())
val hex="0123456789abcdef"
        val out=StringBuilder(20)
        for(i in 0..9){val b=bytes[i].toInt() and 255
out.append(hex[b ushr 4])
out.append(hex[b and 15])}
        return out.toString()
    }
    private fun direction(o:JsonObject):Int {val b=o.a("b")
val x=b[0].jsonPrimitive.int+b[2].jsonPrimitive.int
    return if(x<698)0 else if(x in 702..1298)1 else if(x>1302)2 else 3}
    private fun active(o:JsonObject)=o.s("t")=="CONFIRMED" && o.s("f")=="AVAILABLE"
    private fun reliable(f:JsonObject,o:JsonObject):Boolean {
        val d=o.getValue("d").jsonObject
val n=d["v"]?.jsonPrimitive?.intOrNull?:0
        val q=d["q"]?.jsonPrimitive?.intOrNull
        return d.s("t")=="AVAILABLE" && !(d["c"]?.jsonPrimitive?.booleanOrNull?:false) && n>0 && (d["f"]?.jsonPrimitive?.intOrNull?:0)>=7 && q!=null && q<=max(3,n/4) && f.a("relations").none {val r=it.jsonObject
        r.s("d")=="CONFLICTING" && o.i("id") in listOf(r.i("a"),r.i("b"))}
    }
    private fun descriptor(f:JsonObject,id:Int):JsonObject? {
        val o=f.a("objects")[id].jsonObject
val cls=o.i("c")
        val peers=f.a("objects").map {it.jsonObject}.filter {it.i("c")==cls && active(it)}
        val count=f.a("counts").firstOrNull {it.jsonArray[0].jsonPrimitive.int==cls}?.jsonArray?.get(1)?.jsonPrimitive?.int
        val region=direction(o)
        if(count==1)return buildJsonObject {put("class",cls)
put("region",3)
        put("id",id)}
        if(region!=3 && peers.size==count && peers.count {direction(it)==region}==1)return buildJsonObject {put("class",cls)
put("region",region)
        put("id",id)}
        return null
    }
    fun facts(v:TinyWorldInput):JsonArray {
        val f=Json.parseToJsonElement(TinyWorldCodec.encode(v)).jsonObject
val g=SceneGraphV2.derive(v)
val out=mutableListOf<JsonObject>()
        val objects=f.a("objects").map {it.jsonObject}
        fun add(key:String,kind:String,args:JsonObject,records:List<Int> = emptyList(),required:Boolean=false,region:Int=3,weight:Double=1.0) {
            out+=buildJsonObject {put("key",key)
put("kind",kind)
put("args",args)
put("records",ints(records))
put("required",required)
put("region",region)
put("weight",weight)
            put("id",hash(JsonArray(listOf(JsonPrimitive(key),args))))}
        }
        val q=g.getValue("quality").jsonObject
        if(q.s("detector")!="AVAILABLE") {
            add("detector","STATUS",buildJsonObject {put("detector",q.s("detector"))},required=true,weight=12.0)
            return JsonArray(out)
        }
        val groups=g.a("groups").map {it.jsonObject}
        val edges=g.a("edges").map {it.jsonObject}
        if(groups.isEmpty())add("empty","STATUS",buildJsonObject {put("empty",true)},required=true,weight=10.0)
        for(group in groups) {
            val cls=group.i("class")
val count=group.i("count")
val members=group.a("members").map {it.jsonPrimitive.int}
val dirs=group.a("directions").map {it.jsonPrimitive.int}
            val region=if(group.b("complete") && dirs.size==1)dirs[0] else 3
            val area=members.maxOfOrNull {i->val b=objects[i].a("b")
            (b[2].jsonPrimitive.int-b[0].jsonPrimitive.int)*(b[3].jsonPrimitive.int-b[1].jsonPrimitive.int)}?:0
            val weight=3.0+min(1.0,area/150000.0)+min(count,4)*.1
            add("group:$cls","GROUP",buildJsonObject {put("class",cls)
put("count",count)
put("direction",region)
            put("complete",group.b("complete"))},members,region=region,weight=weight)
            if(group.b("complete") && dirs.size>1 && 3 !in dirs)add("spread:$cls","DISTRIBUTION",buildJsonObject {put("class",cls)
            put("directions",ints(dirs))},members,weight=2.0)
        }
        for(e in edges) {
            val a=descriptor(f,e.i("a"))?:continue
            val b=descriptor(f,e.i("b"))?:continue
            val kind=e.s("kind")
val ai=e.i("a")
            val bi=e.i("b")
            add("edge:$kind:$ai:$bi","RELATION",buildJsonObject {put("relation",kind)
put("a",a)
            put("b",b)},listOf(ai,bi),region=direction(objects[ai]),weight=2.6)
        }
        for(o in objects) {
            if(!active(o) || !reliable(f,o))continue
            val desc=descriptor(f,o.i("id"))?:continue
            val id=o.i("id")
            add("distance:$id","DISTANCE",buildJsonObject {put("object",desc)
            put("dm",o.getValue("d").jsonObject.i("v"))},listOf(id),region=direction(o),weight=2.8)
        }
        for(x in g.a("areas")) {
            val a=x.jsonObject
val cls=a.i("class")
val band=a.i("band")
            val key="area:$cls:$band"
            if(out.none {it.s("key")==key})add(key,"AREA",buildJsonObject {put("class",cls)
            put("band",band)},weight=1.5+a.a("cells").size/16.0)
        }
        for((key,field) in listOf("omitted" to "omitted_objects","source_mismatch" to "source_mismatch","conflict" to "depth_conflicts","historical" to "historical_objects")) {
            val present=if(field=="source_mismatch")q.b(field) else q.i(field)>0
            if(present)add(key,"QUALITY",buildJsonObject {put("quality",key)},required=key in listOf("omitted","source_mismatch"),weight=if(key in listOf("omitted","source_mismatch"))5.0 else 2.0)
        }
        if(q.s("depth")!="AVAILABLE" && groups.isNotEmpty())add("distance_missing","QUALITY",buildJsonObject {put("quality","distance_missing")},weight=2.0)
        return JsonArray(out.sortedBy {it.s("key")})
    }
}
