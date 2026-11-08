package com.algorithmic_alliance.eyeaiapp.scenenarrator

import kotlinx.serialization.json.*

// Raw model output that did not meet either the release or provisional speech policy.
@kotlinx.serialization.Serializable
data class R4RawDiagnostic(
    val snapshotSha256: String, val subjectId: Int, val objectId: Int,
    val subjectClass: Int, val objectClass: Int, val relation: String,
    val score: Double, val calibrated: Boolean, val reason: String,
)

// All seven R4 heads share this snapshot and typed-fact order.
internal object R4RelationTypes {
    val outputIndex=linkedMapOf("BESIDE" to 0,"SUPPORTED_BY" to 1,"INSIDE" to 2,
        "CARRIES" to 3,"HOLDS" to 4,"WEARS" to 5,"OCCLUDES" to 6)
    val disabledInApp=setOf("INSIDE")
    fun enabledInApp(relation: String)=relation in outputIndex&&relation !in disabledInApp
    val factKinds=mapOf("SUPPORTED_BY" to 12,"CARRIES" to 13,"HOLDS" to 14,"WEARS" to 15,
        "BESIDE" to 16,"INSIDE" to 17,"OCCLUDES" to 18)
    fun normalize(candidates: List<R4RelationCandidate>): List<R4RelationCandidate> {
        val unique=candidates.filter {enabledInApp(it.relation)}.groupBy {
            if(it.relation=="BESIDE") Triple(minOf(it.subjectId,it.objectId),maxOf(it.subjectId,it.objectId),it.relation)
            else Triple(it.subjectId,it.objectId,it.relation)
        }.values.map { it.maxBy { c->c.score } }
        val specificPairs=unique.filter {it.relation!="BESIDE"}.map {setOf(it.subjectId,it.objectId)}.toSet()
        val selected=unique.filter {it.relation!="BESIDE" || setOf(it.subjectId,it.objectId) !in specificPairs}
            .groupBy {it.subjectId to it.objectId}.values.map {it.maxBy {c->c.score}}
        return selected.filter {c-> c.relation=="BESIDE" || selected.none {
            it.relation==c.relation && it.subjectId==c.objectId && it.objectId==c.subjectId
        }}.sortedWith(compareByDescending<R4RelationCandidate> {it.score}
            .thenBy {it.subjectId}.thenBy {it.objectId}.thenBy {it.relation})
    }
}

internal object R4ProvisionalSpeechPolicy {
    const val STATUS="provisional"
    const val UNVALIDATED_STATUS="provisional-unvalidated"
    fun threshold(relation: String): Double = when(relation) {
        "HOLDS" -> .74
        "SUPPORTED_BY" -> .52
        in R4RelationTypes.outputIndex.keys -> .90
        else -> error("Unsupported provisional relation")
    }
    fun accepts(c: R4RelationCandidate): Boolean {
        if(!R4RelationTypes.enabledInApp(c.relation) || !c.score.isFinite() ||
            kotlin.math.abs(c.threshold-threshold(c.relation))>1e-9 || c.score<c.threshold) return false
        return when(c.status) {
            STATUS -> c.relation in setOf("HOLDS","SUPPORTED_BY") && c.precision>=.8 &&
                c.bootstrapLower95>=.7 && c.statements>=(if(c.relation=="HOLDS")30 else 20) && c.images>=20
            UNVALIDATED_STATUS -> c.relation !in setOf("HOLDS","SUPPORTED_BY") &&
                c.precision==0.0 && c.bootstrapLower95==0.0 && c.statements==0 && c.images==0
            else -> false
        }
    }
}

internal object R4GateContract {
    private val relationNames=setOf("SUPPORTED_BY","CARRIES","HOLDS","WEARS")
    private val sha256=Regex("[0-9a-f]{64}")
    private const val releaseInputScope="confirmed-current-objects-only; depth-unknown; segmentation-unknown; describe-camera-forward"
    fun validate(gate: JsonObject, projectionId: String, modelSha256: String) {
        val schema=gate.getValue("schema").jsonPrimitive.content
        require(schema=="android-r4-relation-gate-2")
        require(gate.getValue("projection_id").jsonPrimitive.content==projectionId)
        for(key in listOf("model_sha256","probability_fit_groups_sha256","threshold_selection_groups_sha256","freeze_sha256"))
            require(sha256.matches(gate.getValue(key).jsonPrimitive.content)) { "Invalid R4 gate provenance: $key" }
        require(gate.getValue("model_sha256").jsonPrimitive.content==modelSha256)
        require(gate.getValue("data_scope").jsonPrimitive.content.isNotBlank())
        require(gate.getValue("probability_fit_groups_sha256")!=gate.getValue("threshold_selection_groups_sha256"))
        require(gate.getValue("age_bands_ms").jsonArray.map { it.jsonPrimitive.int }==listOf(100,500))
        val rows=gate.getValue("relations").jsonObject
        require(rows.keys==relationNames)
        val hasEnabled=rows.values.any { it.jsonObject["enabled"]?.jsonPrimitive?.booleanOrNull==true }
        if(hasEnabled) {
            require(sha256.matches(gate.getValue("independent_blind_audit_sha256").jsonPrimitive.content))
            require(gate.getValue("input_scope").jsonPrimitive.content==releaseInputScope)
            val trace=gate.getValue("spoken_trace_attestation").jsonObject
            val builder=gate.getValue("snapshot_builder_attestation").jsonObject
            require(trace.getValue("schema").jsonPrimitive.content=="r4-app-v2-kotlin-trace-1" &&
                sha256.matches(trace.getValue("source_sha256").jsonPrimitive.content) &&
                trace.getValue("input_scope").jsonPrimitive.content==releaseInputScope)
            require(builder.getValue("scope").jsonPrimitive.content=="android-worldstate-projection-1" &&
                builder.getValue("input_scope").jsonPrimitive.content==releaseInputScope &&
                sha256.matches(builder.getValue("source_sha256").jsonPrimitive.content))
        }
        for(row0 in rows.values) {
            val row=row0.jsonObject
            if(row["enabled"]?.jsonPrimitive?.booleanOrNull!=true) continue
            val selection=row.getValue("selection").jsonObject
            fun validateMetrics(metrics: JsonObject) {
                require(metrics.getValue("statements").jsonPrimitive.int>=30)
                require(metrics.getValue("images").jsonPrimitive.int>=20)
                val precision=metrics.getValue("precision").jsonPrimitive.double
                val lower=metrics.getValue("bootstrap_lower_95").jsonPrimitive.double
                require(precision.isFinite() && precision in .8..1.0)
                require(lower.isFinite() && lower in .7..1.0)
                require(metrics.getValue("closed_annotations").jsonPrimitive.boolean)
                require(metrics.getValue("unknown_spoken").jsonPrimitive.int==0)
            }
            validateMetrics(selection)
            require(selection.getValue("independent_groups").jsonPrimitive.boolean)
            require(selection.getValue("age_robust").jsonPrimitive.boolean)
            val age=selection.getValue("age_band_metrics").jsonObject
            require(age.keys==setOf("100","500"))
            age.values.forEach { validateMetrics(it.jsonObject) }
            val details=selection.getValue("detail_metrics").jsonObject
            // Compatibility with frozen pre-migration R4 evaluation metadata only.
            val activeDetails=SceneDescriptionDetail.entries.map {it.name}.toSet()
            require(details.keys.containsAll(activeDetails)&&details.keys.all {it in activeDetails||it=="DETAILED"})
            details.values.forEach { detail ->
                val bands=detail.jsonObject
                require(bands.keys==setOf("100","500"))
                bands.values.forEach { validateMetrics(it.jsonObject) }
            }
            require(selection.getValue("threshold").jsonPrimitive.double in 0.0..1.0)
            val params=row.getValue("probability").jsonObject
            require(params.getValue("scale").jsonPrimitive.double.let { it.isFinite() && it>0.0 })
            require(params.getValue("bias").jsonPrimitive.double.isFinite())
            require(row.getValue("class_pairs").jsonArray.all { pair ->
                pair.jsonArray.size==2 && pair.jsonArray.all { it.jsonPrimitive.int in 1..80 }
            })
        }
    }

    fun validateProvisionalPolicy(policy: JsonObject,gate: JsonObject,modelSha256: String,gateSha256: String) {
        require(policy.getValue("schema").jsonPrimitive.content=="android-r4-provisional-speech-policy-2")
        require(policy.getValue("model_sha256").jsonPrimitive.content==modelSha256)
        require(policy.getValue("source_gate_sha256").jsonPrimitive.content==gateSha256)
        require(policy.getValue("projection_id").jsonPrimitive.content==R4Tensorizer.PROJECTION_ID)
        val rows=policy.getValue("relations").jsonObject
        require(rows.keys==R4RelationTypes.outputIndex.keys)
        for((relation,value) in rows) {
            val row=value.jsonObject
            require(row.getValue("threshold").jsonPrimitive.double==R4ProvisionalSpeechPolicy.threshold(relation))
            val known=relation in setOf("HOLDS","SUPPORTED_BY")
            require(row.getValue("status").jsonPrimitive.content==if(known) R4ProvisionalSpeechPolicy.STATUS else R4ProvisionalSpeechPolicy.UNVALIDATED_STATUS)
            val original=gate.getValue("relations").jsonObject[relation] as? JsonObject
            val fit=row["probability_fit"] as? JsonObject
            if(fit!=null) {
                require(original!=null)
                val expected=original.getValue("probability").jsonObject
                for(key in listOf("scale","bias")) require(fit.getValue(key)==expected.getValue(key))
                require(row.getValue("score_type").jsonPrimitive.content=="affine_calibrated")
            } else require(row.getValue("score_type").jsonPrimitive.content=="raw_sigmoid")
            if(original!=null) require(row.getValue("allowed_class_pairs")==original.getValue("class_pairs"))
            else {
                require(row.getValue("class_pair_policy").jsonPrimitive.content=="current_coco_pairs")
                require(row.getValue("allowed_class_pairs").jsonArray.isEmpty())
            }
            val metrics=row["known_annotation_metrics"] as? JsonObject
            if(!known) { require(metrics==null)
            continue }
            require(metrics!=null)
            val details=metrics.getValue("detail_age_strata").jsonObject
            // Compatibility with frozen pre-migration R4 evaluation metadata only.
            val activeDetails=SceneDescriptionDetail.entries.map {it.name}.toSet()
            require(details.keys.containsAll(activeDetails)&&details.keys.all {it in activeDetails||it=="DETAILED"})
            for(detail in details.values) {
                require(detail.jsonObject.keys==setOf("100","500"))
                for(m in detail.jsonObject.values.map {it.jsonObject}) {
                    require(m.getValue("statements").jsonPrimitive.int>=(if(relation=="HOLDS")30 else 20))
                    require(m.getValue("images").jsonPrimitive.int>=20)
                    require(m.getValue("precision").jsonPrimitive.double in .8..1.0)
                    require(m.getValue("bootstrap_lower_95").jsonPrimitive.double in .7..1.0)
                    require(m.getValue("unknown_spoken").jsonPrimitive.int>0)
                }
            }
        }
    }

}

// Snapshot-bound release record for the R4 adapter, C2 planner and verifier.
data class R4RelationCandidate(
    val snapshotSha256: String,
    val projectionId: String,
    val subjectId: Int,
    val objectId: Int,
    val subjectClass: Int,
    val objectClass: Int,
    val sourceId: Int,
    val relation: String,
    val score: Double,
    val threshold: Double,
    val precision: Double,
    val bootstrapLower95: Double,
    val statements: Int,
    val images: Int,
    val status: String = "qualified",
)
