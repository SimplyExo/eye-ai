package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Item
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Placement
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Plan
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.RangeOrder
import com.algorithmic_alliance.eyeaiapp.scenenarrator.OrientationPlanner.Side
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyWorldInput
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

internal class OrientationRenderer(language: JsonObject) {
    private val nouns = language.getValue("nouns").jsonArray.map { it.jsonObject }
    private val numbers = listOf("null", "ein", "zwei", "drei", "vier", "fünf", "sechs",
        "sieben", "acht", "neun", "zehn", "elf", "zwölf")

    private enum class Case { NOMINATIVE, ACCUSATIVE, DATIVE }
    private enum class Kind {
        HOLDS, CARRIES, WEARS, OCCLUDES, SUPPORTED_BY, INSIDE, BESIDE, PLACEMENT;

        val hasActiveVerb get() = this in setOf(HOLDS, CARRIES, WEARS, OCCLUDES)
    }

    private data class Link(val a: Item, val b: Item, val kind: Kind, val placement: Placement? = null) {
        fun touches(item: Item) = item == a || item == b
        fun other(item: Item) = if (item == a) b else a
    }

    // A reference points into the previous sentence, without introducing its object again.
    private data class Sentence(
        val focus: Item,
        val additions: List<Link> = emptyList(),
        val reference: Link? = null,
        val useBehind: Boolean = false,
        val sameLocation: Item? = null,
    ) {
        val introductions: List<Item> get() = listOfNotNull(focus, sameLocation) + additions.map { it.other(focus) }

        fun hasClearLocation(antecedent: Item): Boolean =
            focus == antecedent && reference?.kind?.hasActiveVerb != true && sameLocation == null &&
                additions.all { it.kind in setOf(Kind.SUPPORTED_BY, Kind.INSIDE) && it.b == antecedent }
    }

    fun render(input: TinyWorldInput, plan: Plan, detail: String): String =
        renderSegments(input, plan, detail).joinToString(" ") { it.text }

    fun renderSegments(input: TinyWorldInput, plan: Plan, detail: String): List<NarrationSegment> =
        planSentences(plan).map { sentence ->
            val focus = sentence.focus
            val located = listOfNotNull(focus, sentence.sameLocation)
            val locatedIds = located.flatMap { it.ids }
            val complete = located.all { it.complete }
            val center = if (!complete) .5f else
                locatedIds.sumOf { input.objects[it].box.let { box -> box[0] + box[2] } } /
                    (2000f * locatedIds.size)
            val distance = locatedIds.singleOrNull()?.takeIf { complete }?.let(plan.ranges::get)
            NarrationSegment(
                wording(sentence, plan, short = detail == "SHORT"),
                sentence.introductions.flatMap { it.ids }, center, distance?.div(10f),
            )
        }

    private fun planSentences(plan: Plan): List<Sentence> {
        val singletons = plan.items.filter { it.complete && it.count == 1 }.associateBy { it.ids.single() }
        val anchor = singletons[plan.anchor]
        val links = plan.edges.sortedBy { it.fact }.mapNotNull { edge ->
            val a = singletons[edge.a] ?: return@mapNotNull null
            val b = singletons[edge.b] ?: return@mapNotNull null
            if (a == b) return@mapNotNull null
            val relative = when (anchor) {
                a -> plan.placements[edge.b]
                b -> plan.placements[edge.a]?.reversed()
                else -> null
            }
            Link(a, b, Kind.valueOf(edge.type), relative)
        }
        val placements = plan.placements.mapNotNull { (id, placement) ->
            val item = singletons[id] ?: return@mapNotNull null
            if (anchor == null || (placement.side == null && placement.depth == null)) return@mapNotNull null
            Link(anchor, item, Kind.PLACEMENT, placement)
        }
        val allLinks = links + placements
        val remaining = plan.items.toMutableList()
        val sentences = mutableListOf<Sentence>()
        while (remaining.isNotEmpty()) {
            // Reorder within the current priority tier to keep a clear antecedent adjacent.
            val first = remaining.first()
            val tier = remaining.takeWhile { it.priority == first.priority }
            val previous = sentences.lastOrNull()
            val candidates = if (first.priority) listOf(first) else tier
            val continuation = if (previous == null) null else allLinks.firstNotNullOfOrNull { link ->
                candidates.firstOrNull { link.touches(it) && canContinue(previous, link, it) }
                    ?.let { item ->
                        Sentence(item, reference = link, useBehind = canSayBehind(previous, link, item))
                    }
            }
            val focus = continuation?.focus ?: first
            // Introduce one physical relation in the same sentence. Coordinate two targets
            // of the same predicate when that does not delay another nearby obstacle.
            val availableTargets = if (focus.priority && tier.size > 1) emptyList() else remaining
            val eligible = links.filter { link ->
                val target = link.other(focus)
                link.touches(focus) && link.kind !in setOf(Kind.BESIDE, Kind.PLACEMENT) &&
                    target in availableTargets && plan.rangeId !in target.ids &&
                    !(link.kind == Kind.OCCLUDES && link.a == focus && link.placement?.behind == true)
            }
            val additions = if (continuation != null) emptyList() else eligible.map { primary ->
                coordinatedLinks(focus, primary, eligible)
            }.maxByOrNull { group ->
                val introduction = Sentence(focus, group)
                val next = remaining - introduction.introductions.toSet()
                val canFollow = allLinks.any { link ->
                    next.any { link.touches(it) && canContinue(introduction, link, it) }
                }
                group.size + if (canFollow) 1 else 0
            }.orEmpty()
            val sameLocation = if (continuation == null && additions.isEmpty() &&
                canShareLocation(focus, plan, allLinks)
            ) {
                tier.drop(1).firstOrNull { it.direction == focus.direction && canShareLocation(it, plan, allLinks) }
            } else null
            val sentence = continuation ?: Sentence(focus, additions, sameLocation = sameLocation)
            sentences += sentence
            remaining.removeAll(sentence.introductions)
        }
        return sentences
    }

    private fun coordinatedLinks(focus: Item, primary: Link, eligible: List<Link>): List<Link> {
        val canCoordinate = focus == primary.a && primary.kind in setOf(Kind.HOLDS, Kind.CARRIES, Kind.WEARS) ||
            focus == primary.b && primary.kind in setOf(Kind.SUPPORTED_BY, Kind.INSIDE)
        if (!canCoordinate) return listOf(primary)
        return eligible.filter { it.kind == primary.kind && (it.a == focus) == (primary.a == focus) }
            .distinctBy { it.other(focus) }.take(2)
    }

    private fun canShareLocation(item: Item, plan: Plan, links: List<Link>): Boolean =
        item.complete && !item.priority && plan.rangeId !in item.ids && links.none { it.touches(item) }

    private fun Placement.reversed() = Placement(
        side = when (side) {
            Side.LEFT -> Side.RIGHT
            Side.RIGHT -> Side.LEFT
            null -> null
        },
        depth = when (depth) {
            RangeOrder.NEARER -> RangeOrder.FARTHER
            RangeOrder.FARTHER -> RangeOrder.NEARER
            null -> null
        },
    )

    private fun canContinue(previous: Sentence, link: Link, item: Item): Boolean {
        val antecedent = link.other(item)
        if (antecedent !in previous.introductions) return false
        val priorItems = previous.introductions + listOfNotNull(previous.reference?.other(previous.focus))
        if (canSayBehind(previous, link, item)) return true
        if (link.kind.hasActiveVerb || link.kind in setOf(Kind.INSIDE, Kind.SUPPORTED_BY) && item == link.b) {
            val case = if (item == link.b) Case.NOMINATIVE else Case.ACCUSATIVE
            val reference = pronoun(antecedent, case)
            // Only use a pronoun that distinguishes its antecedent from the other recent objects.
            return priorItems.count { pronoun(it, case) == reference } == 1
        }
        if (!previous.hasClearLocation(antecedent)) return false
        return when (link.kind) {
            Kind.BESIDE, Kind.PLACEMENT -> true
            Kind.INSIDE, Kind.SUPPORTED_BY -> item == link.a
            else -> false
        }
    }

    private fun canSayBehind(previous: Sentence, link: Link, item: Item): Boolean =
        link.kind == Kind.OCCLUDES && item == link.b && link.placement?.behind == true &&
            previous.hasClearLocation(link.a)

    private fun wording(sentence: Sentence, plan: Plan, short: Boolean): String {
        val focus = sentence.focus
        val link = sentence.reference
        val antecedent = link?.other(focus)
        val distance = focus.ids.singleOrNull()?.takeIf { focus.complete && it == plan.rangeId }
            ?.let(plan.ranges::get)?.let { range(it, short) }
        val distanceSuffix = distance?.let { ", $it" }.orEmpty()
        val position = location(focus.direction, short)
        val name = noun(focus)
        val behind = sentence.useBehind
        val text = when {
            link != null && antecedent != null && link.kind.hasActiveVerb && !behind -> {
                if (link.a == antecedent) {
                    val target = withPosition(noun(focus, Case.ACCUSATIVE), focus, antecedent)
                    "${pronoun(antecedent, Case.NOMINATIVE)} ${activeVerb(link.kind)} $target${activeEnding(link.kind)}$distanceSuffix"
                } else {
                    val target = pronoun(antecedent, Case.ACCUSATIVE)
                    "$position ${activeVerb(link.kind)} $target $name${activeEnding(link.kind)}$distanceSuffix"
                }
            }
            link != null && antecedent != null && link.b == focus &&
                link.kind in setOf(Kind.SUPPORTED_BY, Kind.INSIDE) -> {
                val preposition = if (link.kind == Kind.SUPPORTED_BY) "auf" else "in"
                val target = withPosition(noun(focus, Case.DATIVE), focus, antecedent)
                "${pronoun(antecedent, Case.NOMINATIVE)} ist $preposition $target$distanceSuffix"
            }
            link != null -> {
                val placement = if (focus == link.b) link.placement else link.placement?.reversed()
                val side = placement?.side?.let { if (it == Side.LEFT) "links" else "rechts" }
                val depth = if (link.kind != Kind.OCCLUDES) when (placement?.depth) {
                    RangeOrder.NEARER -> "näher bei dir"
                    RangeOrder.FARTHER -> "weiter von dir entfernt"
                    null -> null
                } else null
                val relativePosition = when (link.kind) {
                    Kind.BESIDE -> side?.let { "$it daneben" } ?: "daneben"
                    Kind.INSIDE -> "darin"
                    Kind.SUPPORTED_BY -> "darauf"
                    Kind.OCCLUDES -> "dahinter"
                    Kind.PLACEMENT -> side?.let { "$it davon" }
                    else -> error("Unsupported continuation")
                }
                val where = listOfNotNull(relativePosition, depth).joinToString(" und ")
                val description = if (short) "$where $name" else "$where ist $name"
                description + distanceSuffix + if (behind) ", teilweise verdeckt" else ""
            }
            sentence.additions.isNotEmpty() -> {
                val where = distance?.let { "$position, $it," } ?: position
                describeRelated(sentence, where, short)
            }
            sentence.sameLocation != null ->
                "$position ${if (short) "" else "sind "}$name und ${noun(sentence.sameLocation)}"
            short -> "$position $name$distanceSuffix"
            else -> "$position ${if (focus.count == 1 || mass(focus)) "ist" else "sind"} $name$distanceSuffix"
        }
        return text.replaceFirstChar { it.uppercaseChar() } + "."
    }

    private fun describeRelated(sentence: Sentence, position: String, short: Boolean): String {
        val focus = sentence.focus
        val link = sentence.additions.first()
        val others = sentence.additions.map { it.other(focus) }
        val name = noun(focus)
        fun targets(case: Case) = others.joinToString(" und ") { withPosition(noun(it, case), it, focus) }

        return when {
            link.kind == Kind.INSIDE && focus == link.b ->
                "$position ${if (short) "" else "ist "}$name mit ${targets(Case.DATIVE)} darin"
            link.kind == Kind.SUPPORTED_BY && focus == link.b ->
                "$position ${if (short) "" else "ist "}$name mit ${targets(Case.DATIVE)} darauf"
            link.kind == Kind.INSIDE ->
                "$position ${if (short) "" else "ist "}$name in ${targets(Case.DATIVE)}"
            link.kind == Kind.SUPPORTED_BY ->
                "$position ${if (short) "" else "befindet sich "}$name auf ${targets(Case.DATIVE)}"
            focus == link.a && short ->
                "$position $name, ${relativePronoun(focus)} ${targets(Case.ACCUSATIVE)}${activeEnding(link.kind)} ${activeVerb(link.kind)}"
            focus == link.a ->
                "$position ${activeVerb(link.kind)} $name ${targets(Case.ACCUSATIVE)}${activeEnding(link.kind)}"
            else -> {
                val passive = when (link.kind) {
                    Kind.HOLDS -> "gehalten"
                    Kind.CARRIES -> "getragen"
                    Kind.WEARS -> "am Körper getragen"
                    Kind.OCCLUDES -> "teilweise verdeckt"
                    else -> error("Unsupported passive relation")
                }
                if (short) "$position $name, $passive von ${targets(Case.DATIVE)}"
                else "$position wird $name von ${targets(Case.DATIVE)} $passive"
            }
        }
    }

    private fun withPosition(name: String, item: Item, focus: Item): String =
        if (item.direction == focus.direction) name else "$name ${location(item.direction)}"

    private fun location(direction: Int, short: Boolean = false) = when (direction) {
        0 -> "links vor dir"
        1 -> if (short) "voraus" else "vor dir"
        2 -> "rechts vor dir"
        else -> "in Blickrichtung"
    }

    private fun activeVerb(kind: Kind) = when (kind) {
        Kind.HOLDS -> "hält"
        Kind.CARRIES, Kind.WEARS -> "trägt"
        Kind.OCCLUDES -> "verdeckt"
        else -> error("Unsupported active relation")
    }

    private fun activeEnding(kind: Kind) = when (kind) {
        Kind.WEARS -> " am Körper"
        Kind.OCCLUDES -> " aus deiner Sicht teilweise"
        else -> ""
    }

    private fun pronoun(item: Item, case: Case) = if (item.count != 1) "sie" else when (gender(item)) {
        "M" -> if (case == Case.ACCUSATIVE) "ihn" else "er"
        "F" -> "sie"
        else -> "es"
    }

    private fun relativePronoun(item: Item) = when (gender(item)) {
        "M" -> "der"
        "F" -> "die"
        else -> "das"
    }

    private fun gender(item: Item) = nouns[item.cls].getValue("gender").jsonPrimitive.content
    private fun mass(item: Item) = nouns[item.cls].getValue("mass").jsonPrimitive.boolean
    private fun number(n: Int) = numbers.getOrNull(n) ?: n.toString()

    private fun noun(item: Item, case: Case = Case.NOMINATIVE): String {
        val n = nouns[item.cls]
        fun word(key: String) = n.getValue(key).jsonPrimitive.content
        if (mass(item)) return word("singular")
        if (item.count != 1) return "${number(item.count)} ${word("plural")}"
        val gender = gender(item)
        val article = when (case) {
            Case.NOMINATIVE -> if (gender == "F") "eine" else "ein"
            Case.ACCUSATIVE -> when (gender) {
                "M" -> "einen"
                "F" -> "eine"
                else -> "ein"
            }
            Case.DATIVE -> if (gender == "F") "einer" else "einem"
        }
        val form = if (case == Case.DATIVE || case == Case.ACCUSATIVE && gender == "M") "dative" else "singular"
        return "$article ${word(form)}"
    }

    private fun range(dm: Int, short: Boolean): String {
        val value = when {
            dm == 10 -> if (short) "ein" else "einen"
            dm % 10 == 0 -> number(dm / 10)
            else -> "${dm / 10},${dm % 10}"
        }
        return "etwa $value Meter" + if (short) "" else " entfernt"
    }
}
