package com.algorithmic_alliance.eyeaiapp.scenenarrator

import com.algorithmic_alliance.eyeaiapp.worldstate.*
import kotlinx.serialization.json.*

// Evidence-based spatial orientation for DESCRIBE.
class LocalNarrative(language: JsonObject) {
    private val renderer = OrientationRenderer(language)
    data class Prepared(val input: TinyWorldInput, val facts: List<JsonObject>, val indices: List<Int>,
        val spatialPlan: OrientationPlanner.Plan, val r4Candidates: List<R4RelationCandidate> = emptyList(),
        internal val anchorId: Int? = null, internal val detail: String = "STANDARD")
    private fun JsonObject.s(k: String) = getValue(k).jsonPrimitive.content
    private fun JsonObject.i(k: String) = getValue(k).jsonPrimitive.int
    private fun args(f: JsonObject) = f.getValue("args").jsonObject
    private fun ints(values: Iterable<Int>) = JsonArray(values.map(::JsonPrimitive))
    private fun descriptor(input: TinyWorldInput, id: Int): JsonObject? {
        val o = input.objects.getOrNull(id) ?: return null
        val peers = input.objects.filter { it.cls == o.cls && it.track == TrackStatus.CONFIRMED && it.freshness == Availability.AVAILABLE }
        val count = input.counts.firstOrNull { it[0] == o.cls }?.get(1)
        val region = SceneGraphV2.direction(o.box)
        if (count == 1) return buildJsonObject { put("class", o.cls)
 put("region", 3)
        put("id", id) }
        if (region != 3 && peers.size == count && peers.count { SceneGraphV2.direction(it.box) == region } == 1)
            return buildJsonObject { put("class", o.cls)
 put("region", region)
            put("id", id) }
        return null
    }
    private fun qualifiedR4(input: TinyWorldInput, candidates: List<R4RelationCandidate>): List<R4RelationCandidate> {
        if (candidates.isEmpty()) return emptyList()
        val snapshot = R4Tensorizer.snapshot(input)
        val valid = candidates.filter { c ->
            val evidencePolicyPassed = when (c.status) {
                "qualified" -> c.score >= c.threshold && c.precision >= .8 && c.bootstrapLower95 >= .7 &&
                    c.statements >= 30 && c.images >= 20
                R4ProvisionalSpeechPolicy.STATUS, R4ProvisionalSpeechPolicy.UNVALIDATED_STATUS -> R4ProvisionalSpeechPolicy.accepts(c)
                else -> false
            }
            if (!evidencePolicyPassed || c.projectionId != R4Tensorizer.PROJECTION_ID || c.snapshotSha256 != snapshot ||
                !R4RelationTypes.enabledInApp(c.relation)) return@filter false
            val a = input.objects.getOrNull(c.subjectId) ?: return@filter false
            val b = input.objects.getOrNull(c.objectId) ?: return@filter false
            if (a.cls + 1 != c.subjectClass || b.cls + 1 != c.objectClass || a.source != c.sourceId || b.source != c.sourceId ||
                a.track != TrackStatus.CONFIRMED || b.track != TrackStatus.CONFIRMED ||
                a.freshness != Availability.AVAILABLE || b.freshness != Availability.AVAILABLE ||
                a.confidenceDecile < 5 || b.confidenceDecile < 5) return@filter false
            val ageA = input.sources[a.source].age
            val ageB = input.sources[b.source].age
            ageA in setOf(TinyAge.LE_100_MS, TinyAge.LE_500_MS) && ageB in setOf(TinyAge.LE_100_MS, TinyAge.LE_500_MS)
        }
        return R4RelationTypes.normalize(valid)
    }

    private fun r4Fact(input: TinyWorldInput, c: R4RelationCandidate): JsonObject {
        val a = descriptor(input, c.subjectId) ?: error("R4 subject is ambiguous")
        val b = descriptor(input, c.objectId) ?: error("R4 object is ambiguous")
        val region = a.i("region")
        val identity = "r4:${c.snapshotSha256}:${c.subjectId}:${c.objectId}:${c.relation}"
        return buildJsonObject {
            put("key", identity)
            put("kind", "RELATION")
            put("args", buildJsonObject {
                put("relation", c.relation)
 put("a", a)
                put("b", b)
                put("snapshot_sha256", c.snapshotSha256)
                put("projection_id", c.projectionId)
                put("source_id", c.sourceId)
 put("score", c.score)
                put("threshold", c.threshold)
                put("release_status", c.status)
            })
            put("records", ints(listOf(c.subjectId, c.objectId)))
            put("required", false)
            put("region", region)
 put("weight", 4.0 + c.score)
            put("id", identity)
        }
    }

    fun prepareDeterministic(input: TinyWorldInput, detail: String = "STANDARD",
        r4Relations: List<R4RelationCandidate> = emptyList()): Prepared {
        TinyWorldCodec.validate(input)
        val baseFacts = TinyWorldEvidence.facts(input).map { it.jsonObject }
        val admittedR4 = qualifiedR4(input, r4Relations)
            .filter { descriptor(input, it.subjectId) != null && descriptor(input, it.objectId) != null }
            .take((256 - baseFacts.size).coerceAtMost(64).coerceAtLeast(0))
        val facts = baseFacts + admittedR4.map { r4Fact(input, it) }
        require(facts.size <= 256)
        val plan = OrientationPlanner.plan(input, facts, detail) { descriptor(input, it) != null }
        return Prepared(input, facts, plan.indices, plan, admittedR4, plan.anchor, detail)
    }

    private fun verifyEvidence(p: Prepared): Boolean =
        OrientationPlanner.plan(p.input, p.facts, p.detail) { descriptor(p.input, it) != null }.let {
            it == p.spatialPlan && it.anchor == p.anchorId && it.indices == p.indices
        } &&
            p.facts == TinyWorldEvidence.facts(p.input).map { it.jsonObject } + p.r4Candidates.map { r4Fact(p.input, it) } &&
            qualifiedR4(p.input, p.r4Candidates) == p.r4Candidates &&
            p.indices.all { it in p.facts.indices } &&
            p.indices.filter { p.facts[it].s("kind") == "RELATION" }.all {
                args(p.facts[it]).s("relation") in SceneGraphV2.kinds || args(p.facts[it]).s("relation") in R4RelationTypes.outputIndex
            }

    internal fun renderBoth(p: Prepared): Pair<String, List<NarrationSegment>> {
        require(verifyEvidence(p)) { "Narration facts no longer match their snapshot" }
        val segments = renderer.renderSegments(p.input, p.spatialPlan, p.detail)
        return segments.joinToString(" ") { it.text } to segments
    }
    fun render(p: Prepared): String {
        require(verifyEvidence(p)) { "Narration facts no longer match their snapshot" }
        return renderer.render(p.input, p.spatialPlan, p.detail)
    }
}
