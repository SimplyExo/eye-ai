package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlinx.serialization.json.*

// German output from the shared spatial plan.
internal class OrientationRenderer(language: JsonObject) {
    private val nouns=language.getValue("nouns").jsonArray.map {it.jsonObject}
    private val numbers=listOf("null","ein","zwei","drei","vier","fünf","sechs","sieben","acht","neun","zehn","elf","zwölf")
    private fun JsonObject.s(k: String)=getValue(k).jsonPrimitive.content
    private fun cap(s: String)=s.replaceFirstChar {it.uppercaseChar()}
    private fun number(n: Int)=numbers.getOrNull(n)?:n.toString()
    private fun location(direction: Int,short: Boolean)=if(short)listOf("links vor dir","voraus","rechts vor dir","in Blickrichtung")[direction]
        else listOf("Links vor dir","Vor dir","Rechts vor dir","In deiner Blickrichtung")[direction]
    private fun range(dm: Int,short: Boolean): String {
        val value=if(dm%10==0)number(dm/10) else "${dm/10},${dm%10}"
        return "etwa ${if(dm==10&&!short)"einen" else value} Meter"+(if(short)"" else " entfernt")
    }
    fun render(input: TinyWorldInput,plan: OrientationPlanner.Plan,detail: String): String {
        val short=detail=="SHORT"
        val introduced=linkedSetOf<Int>()
val mentionedClasses=mutableSetOf<Int>()
val lines=mutableListOf<String>()
        val consumed=mutableSetOf<Int>()
        var rangeUsed=false
        val depthSpoken=mutableSetOf<Int>()
        fun noun(id: Int,case: String="nom",definite: Boolean=false,position: Boolean=false): String {
            val o=input.objects[id]
val n=nouns[o.cls]
val gender=n.s("gender")
            val mass=n.getValue("mass").jsonPrimitive.boolean
            val article=if(mass||short&&case=="nom")"" else if(definite)when(case) {
                "dat"->if(gender=="F")"der" else "dem"
"acc"->when(gender){"M"->"den"
"F"->"die"
                else->"das"}
                else->when(gender){"M"->"der"
"F"->"die"
                else->"das"}
            } else when(case) {"dat"->if(gender=="F")"einer" else "einem"
"acc"->when(gender){"M"->"einen"
"F"->"eine"
else->"ein"}
            else->if(gender=="F")"eine" else "ein"}
            val further=if(!definite&&!mass&&o.cls in mentionedClasses)when(case) {
                "dat"->"weiteren"
"acc"->when(gender){"M"->"weiteren"
"F"->"weitere"
                else->"weiteres"}
                else->when(gender){"M"->"weiterer"
"F"->"weitere"
                else->"weiteres"}
            }else ""
            val text=n.s(if(case=="dat"||case=="acc"&&gender=="M")"dative" else "singular")
            return listOf(article,further,text,if(position)location(SceneGraphV2.direction(o.box),short).lowercase()else "").filter {it.isNotEmpty()}.joinToString(" ")
        }
        fun itemName(item: OrientationPlanner.Item): String {
            if(item.ids.size==1&&item.count==1)return noun(item.ids.single())
            val n=nouns[item.cls]
            val further=if(item.cls in mentionedClasses)" weitere" else ""
            // Retain an incomplete class count without assigning retained members' positions/ranges to all of it.
            return if(n.getValue("mass").jsonPrimitive.boolean)n.s("singular") else number(item.count)+further+" "+n.s("plural")
        }
        fun introduce(item: OrientationPlanner.Item) {introduced+=item.ids
        mentionedClasses+=item.cls}
        fun distance(item: OrientationPlanner.Item): String {
            if(rangeUsed||plan.rangeId !in item.ids||item.ids.size!=1||!item.complete)return ""
            val dm=plan.ranges[plan.rangeId]?:return ""
            rangeUsed=true
            return ", "+range(dm,short)
        }
        fun direct(item: OrientationPlanner.Item): String {
            val name=itemName(item)
            val distance=distance(item)
            val line=if(short)"${cap(name)} ${location(item.direction,true)}$distance."
                else "${location(item.direction,false)} ${if(item.count==1||nouns[item.cls].getValue("mass").jsonPrimitive.boolean)"ist" else "sind"} $name$distance."
            introduce(item)
            return line
        }
        val anchor=plan.anchor
        val anchorItem=anchor?.let {a->plan.items.single {a in it.ids}}
        if(anchorItem!=null)lines+=direct(anchorItem)
        // Do not bury another nearby obstacle beneath secondary relations or an inventory.
        for(item in plan.items.filter {it.priority&&it!=anchorItem})lines+=direct(item)

        fun reference(id: Int,case: String="nom"): String {
            val repeated=input.counts.firstOrNull {it[0]==input.objects[id].cls}?.get(1)?.let {it>1}==true
            return noun(id,case,definite=id in introduced,position=repeated)
        }
        // Physical predicates share their subject. Their direction and grammatical cases never change with style.
        for(e in plan.edges.filter {it.type=="OCCLUDES"&&it.a==anchor&&plan.placements[it.b]?.behind==true}) {
            if(introduced!=setOf(anchor)||e.b in introduced)continue
            val item=plan.items.first {e.b in it.ids}
val name=itemName(item)
            val d=distance(item)
            lines+=if(short)"${cap(name)} dahinter$d, teilweise verdeckt." else "Dahinter ist $name$d, teilweise verdeckt."
            introduce(item)
consumed+=e.fact
            depthSpoken+=e.b
        }
        val physical=plan.edges.filterNot {it.fact in consumed}.filter {it.type !in setOf("BESIDE","INSIDE") || it.type=="INSIDE"&&it.b!=anchor}

        for((subject,edges) in physical.groupBy {it.a}) {
            for(chunk in edges.chunked(2)) {
                val subjectText=reference(subject)
                val clauses=chunk.map {e->
                    val other=input.objects[e.b]
                    val sameSector=SceneGraphV2.direction(other.box)==SceneGraphV2.direction(input.objects[e.a].box)
                    fun obj(case: String): String=noun(e.b,case,definite=e.b in introduced,position=!sameSector&&e.b !in introduced)
                    when(e.type) {
                        "SUPPORTED_BY"->if(short)"gestützt von ${obj("dat")}" else "wird von ${obj("dat")} gestützt"
                        "CARRIES"->"trägt ${obj("acc")}"
                        "HOLDS"->"hält ${obj("acc")}"
                        "WEARS"->"trägt ${obj("acc")} am Körper"
                        "OCCLUDES"->"verdeckt ${obj("acc")} aus deiner Sicht teilweise"
                        "INSIDE"->if(short)"in ${obj("dat")}" else "ist in ${obj("dat")}"
                        else->error("Unsupported physical relation")
                    }
                }
                val placement=plan.placements[subject]
                val explainDepth=subject !in introduced&&introduced==setOf(anchor)&&placement?.depth!=null
                if(explainDepth) {
                    val order=if(placement.depth==OrientationPlanner.RangeOrder.NEARER)"näher" else "weiter entfernt"
                    val side=placement.side?.let {if(it==OrientationPlanner.Side.LEFT)"links" else "rechts"}
                    val prefix=side?.let {"$it davon"}
                    val name=noun(subject)
                    lines+=if(short)"${cap(name)} ${side?:location(SceneGraphV2.direction(input.objects[subject].box),true)}, $order; ${clauses.joinToString(" und ")}."
                        else if(prefix!=null)"${cap(prefix)} wirkt $name $order und ${clauses.joinToString(" und ")}."
                        else "${cap(name)} wirkt $order als ${reference(anchor!!)} und ${clauses.joinToString(" und ")}."
                    depthSpoken+=subject
                }else lines+="${cap(subjectText)} ${clauses.joinToString(" und ")}."
                introduced+=subject
                mentionedClasses+=input.objects[subject].cls
                for(e in chunk) {introduced+=e.b
mentionedClasses+=input.objects[e.b].cls
                consumed+=e.fact}
            }
        }
        // The container must be the only introduced referent for a bare "darin".
        for(e in plan.edges.filter {it.type=="INSIDE"&&it.b==anchor&&it.a !in introduced}) {
            val item=plan.items.first {e.a in it.ids}
            val prefix=if(introduced==setOf(anchor))"Darin" else "In "+reference(e.b,"dat")
            val name=itemName(item)
            val d=distance(item)
            lines+=if(short)"${cap(name)} ${prefix.replaceFirstChar {it.lowercaseChar()}}$d." else "$prefix ist $name$d."
            introduce(item)
            consumed+=e.fact
        }
        val remaining=plan.items.filter {item->item.ids.any {it !in introduced}}
        // Deictics are confined to one parallel sentence immediately following its single explicit anchor.
        val canUseAnchor=anchor!=null&&introduced==setOf(anchor)
        val candidates=if(canUseAnchor)remaining.filter {it.complete&&it.ids.size==1&&plan.placements[it.ids.single()]?.let {p->
            p.side!=null||p.depth!=null||p.beside||p.behind}==true}.sortedWith(compareBy<OrientationPlanner.Item> {
                when(plan.placements[it.ids.single()]?.side){OrientationPlanner.Side.LEFT->0
OrientationPlanner.Side.RIGHT->1
                else->2}
            }.thenBy {it.ids.single()}) else emptyList()
        fun identity(item: OrientationPlanner.Item)=item.cls to plan.placements[item.ids.single()]
        val relative=candidates.filter {item->candidates.count {identity(it)==identity(item)}==1}.take(2)
        val names=mutableMapOf<Int,String>()
        val clauses=relative.map {item->
            val id=item.ids.single()
val p=plan.placements.getValue(id)
val name=itemName(item)
names[id]=name
            val d=distance(item)
            val side=p.side?.let {if(it==OrientationPlanner.Side.LEFT)"links" else "rechts"}
            val order=p.depth?.let {if(it==OrientationPlanner.RangeOrder.NEARER)"näher" else "weiter entfernt"}
            val prefix=when {p.behind->"dahinter"
side!=null->side+if(p.beside)" daneben" else " davon"
p.beside->"daneben"
            else->null}
            val text=if(prefix==null) {
                val target=reference(anchor!!)
                if(short)"$name $order als $target$d" else "$name wirkt $order als $target$d"
            }else if(short) {
                // In the parallel anchor clause, bare left/right remains tied to that one anchor.
                val loc=if(p.side!=null&&!p.beside)side!! else prefix
                "$name $loc"+(if(order!=null&&!p.behind)", $order" else "")+d+if(p.behind)", teilweise verdeckt" else ""
            }else if(order!=null&&!p.behind)"$prefix wirkt $name $order$d"
            else "$prefix ist $name$d"+if(p.behind)", teilweise verdeckt" else ""
            if(p.depth!=null)depthSpoken+=id
            introduce(item)
            text
        }
        if(clauses.isNotEmpty()) {
            val placements=relative.map {plan.placements.getValue(it.ids.single())}
            val shared=placements.size==2&&placements[0]==placements[1]&&placements[0].depth==null&&!placements[0].behind
            if(shared) {
                val p=placements[0]
                val side=p.side?.let {if(it==OrientationPlanner.Side.LEFT)"links" else "rechts"}
                val loc=if(side==null)"daneben" else side+if(p.beside)" daneben" else " davon"
                val objects=relative.joinToString(" und ") {names.getValue(it.ids.single())}
                lines+=if(short)"${cap(objects)} ${if(p.beside)loc else side}." else "${cap(loc)} sind $objects."
            }else lines+=cap(clauses.joinToString(", "))+"."
        }
        for(item in remaining.filter {it !in relative})lines+=direct(item)
        // Adjacency between already named, non-anchor objects must still have explicit referents.
        for(e in plan.edges.filter {it.type=="BESIDE"&&it.fact !in consumed}) {
            if(e.a==anchor&&relative.any {e.b in it.ids} || e.b==anchor&&relative.any {e.a in it.ids})continue
            val a=reference(e.a)
            val b=reference(e.b,"dat")
            lines+=if(short)"${cap(a)} neben $b." else "${cap(a)} ist neben $b."
        }
        for((id,p) in plan.placements) {
            if(p.depth==null||id in depthSpoken)continue
            val order=if(p.depth==OrientationPlanner.RangeOrder.NEARER)"näher" else "weiter entfernt"
            val name=reference(id)
            val target=reference(anchor!!)
            lines+=if(short)"${cap(name)} $order als $target." else "${cap(name)} wirkt $order als $target."
        }
        return lines.joinToString(" ").replace("von dem ","vom ").replace("in dem ","im ").replace("In dem ","Im ")
    }
}
