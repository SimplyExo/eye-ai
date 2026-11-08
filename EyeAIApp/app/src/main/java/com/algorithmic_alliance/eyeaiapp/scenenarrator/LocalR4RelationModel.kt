package com.algorithmic_alliance.eyeaiapp.scenenarrator

import android.content.Context
import com.algorithmic_alliance.eyeaiapp.worldstate.Availability
import kotlinx.serialization.json.*
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.exp

// Offline R4 inference with separate checksums for released and provisional speech policies.
internal class LocalR4RelationModel(context: Context) : AutoCloseable {
    data class Result(val executed: Boolean, val candidates: List<R4RelationCandidate>,
        val scoredPairs: Int, val projectionId: String, val snapshotSha256: String,
        val diagnostics: List<R4RawDiagnostic> = emptyList(),val allDiagnostics: List<R4RawDiagnostic> = emptyList())

    private val assets = context.applicationContext.assets
    private fun bytes(name: String) = assets.open("scene-narrator/$name").use { it.readBytes() }
    private val manifest = Json.parseToJsonElement(bytes("r4-manifest.json").toString(Charsets.UTF_8)).jsonObject
    private val gate = Json.parseToJsonElement(bytes("r4-gate.json").toString(Charsets.UTF_8)).jsonObject
    private val experimentalPolicyFile = manifest["experimental_policy_file"]?.jsonPrimitive?.content
    private val experimentalPolicy = experimentalPolicyFile?.let {
        Json.parseToJsonElement(bytes(it).toString(Charsets.UTF_8)).jsonObject
    }
    private var interpreter: Interpreter? = null
    private var modelBytes: ByteBuffer? = null
    private val modelFile: String
    val modelPrecision: String
    private val relations = R4RelationTypes.outputIndex
    private val provisionalPolicies: Map<String,JsonObject> = experimentalPolicy?.get("relations")
        ?.jsonObject?.filterKeys(R4RelationTypes::enabledInApp)?.mapValues {it.value.jsonObject}.orEmpty()
    val provisionalRelationNames: Set<String>
        get() = provisionalPolicies.keys.filter { relation ->
            (gate.getValue("relations").jsonObject[relation] as? JsonObject)?.get("enabled")?.jsonPrimitive?.booleanOrNull != true
        }.toSet()

    init {
        val schema=manifest.getValue("schema").jsonPrimitive.content
        require(schema == "android-r4-lite-2")
        manifest["app_disabled_relations"]?.let {require(it.jsonArray.map {value->value.jsonPrimitive.content}.toSet()==R4RelationTypes.disabledInApp)}
        require(manifest.getValue("projection_id").jsonPrimitive.content == R4Tensorizer.PROJECTION_ID)
        modelFile=manifest.getValue("model_file").jsonPrimitive.content
        modelPrecision=manifest.getValue("model_precision").jsonPrimitive.content.uppercase()
        require(modelPrecision in setOf("FP32","FP16","INT8"))
        require(modelFile=="r4-model.tflite")
        val files=manifest.getValue("files").jsonObject
        require(experimentalPolicyFile=="r4-experimental-policy.json")
        val expectedFiles=setOf(modelFile,"r4-gate.json") + listOfNotNull(experimentalPolicyFile)
        require(files.keys==expectedFiles)
        for((name,hash) in files) {
            val actual=MessageDigest.getInstance("SHA-256").digest(bytes(name))
                .joinToString("") { "%02x".format(it.toInt() and 255) }
            require(actual==hash.jsonPrimitive.content) { "Packaged R4 checksum mismatch: $name" }
        }
        val gateSchema=gate.getValue("schema").jsonPrimitive.content
        require(gateSchema=="android-r4-relation-gate-2")
        R4GateContract.validate(gate,R4Tensorizer.PROJECTION_ID,files.getValue(modelFile).jsonPrimitive.content)
        experimentalPolicy?.let {
            val gateSha=MessageDigest.getInstance("SHA-256").digest(bytes("r4-gate.json"))
                .joinToString("") { value -> "%02x".format(value.toInt() and 255) }
            R4GateContract.validateProvisionalPolicy(it,gate,files.getValue(modelFile).jsonPrimitive.content,gateSha)
        }
    }

    private fun load(): Interpreter {
        interpreter?.let { return it }
        val data = bytes(modelFile)
        val storage = ByteBuffer.allocateDirect(data.size).order(ByteOrder.nativeOrder()).put(data).apply { rewind() }
        val value = Interpreter(storage, Interpreter.Options().setNumThreads(2))
        value.allocateTensors()
        modelBytes = storage
        interpreter = value
        return value
    }

    private fun direct(bytes: Int) = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder())

    fun infer(input: com.algorithmic_alliance.eyeaiapp.worldstate.TinyWorldInput, trace: ((String,Int,Int,FloatArray)->Unit)? = null): Result {
        val tensor = R4Tensorizer.encode(input)
        trace?.let {sink->
            sink("r4-cats",R4Tensorizer.NODES,1,tensor.cats.map {it.toFloat()}.toFloatArray())
            sink("r4-nums",R4Tensorizer.NODES,tensor.nums.size/R4Tensorizer.NODES,tensor.nums)
            sink("r4-pairs",R4Tensorizer.NODES,tensor.pairs.size/R4Tensorizer.NODES,tensor.pairs)
            sink("r4-valid",R4Tensorizer.NODES,1,tensor.valid.map {it.toFloat()}.toFloatArray())
        }
        val cats = direct(tensor.cats.size * 4).also { b -> tensor.cats.forEach(b::putInt)
        b.rewind() }
        val nums = direct(tensor.nums.size * 4).also { b -> tensor.nums.forEach(b::putFloat)
        b.rewind() }
        val pairs = direct(tensor.pairs.size * 4).also { b -> tensor.pairs.forEach(b::putFloat)
        b.rewind() }
        val valid = direct(tensor.valid.size).also { b -> tensor.valid.forEach(b::put)
        b.rewind() }
        val model = load()
        val key = model.signatureKeys.single()
        val outputs = model.getSignatureOutputs(key).associateWith {
            direct(model.getOutputTensorFromSignature(it, key).numBytes())
        }
        model.runSignature(mapOf("cats" to cats, "nums" to nums, "pairs" to pairs, "valid" to valid), outputs, key)
        val logits = outputs.getValue("logits").apply { rewind() }
        trace?.invoke("r4-logits-7-heads",R4Tensorizer.NODES,R4Tensorizer.NODES*7,FloatArray(logits.capacity()/4) {logits.getFloat(it*4)})
        val rows = gate.getValue("relations").jsonObject
        val inputScopeMatches=
            gate["input_scope"]?.jsonPrimitive?.content==
                "confirmed-current-objects-only; depth-unknown; segmentation-unknown; describe-camera-forward" &&
            input.channels.size==3 && input.channels[1].status==Availability.UNKNOWN &&
            input.channels[2].status==Availability.UNKNOWN
        val candidates = mutableListOf<R4RelationCandidate>()
        val diagnostics = mutableListOf<R4RawDiagnostic>()
        var scored = 0
        for ((name, relationIndex) in relations) {
            val rule = (rows[name] as? JsonObject) ?: buildJsonObject {}
            val enabled = rule["enabled"]?.jsonPrimitive?.booleanOrNull == true
            val selection = rule["selection"] as? JsonObject
            val provisional = provisionalPolicies[name]
            val appEnabled=R4RelationTypes.enabledInApp(name)
            val active = appEnabled && (enabled || provisional != null)
            // This projection always zeroes depth/segmentation features. Extra
            // channels do not change box-model inputs or become R4 evidence.
            val scopeMatches = if (!enabled && provisional != null) true else inputScopeMatches
            val parameters = if (enabled) rule["probability"] as? JsonObject
                else provisional?.get("probability_fit") as? JsonObject
            val scale = parameters?.get("scale")?.jsonPrimitive?.doubleOrNull
            val bias = parameters?.get("bias")?.jsonPrimitive?.doubleOrNull
            val calibrated = scale != null && bias != null && scale.isFinite() && scale > 0.0 && bias.isFinite()
            val threshold = selection?.get("threshold")?.jsonPrimitive?.doubleOrNull
                ?: provisional?.get("threshold")?.jsonPrimitive?.doubleOrNull
            val classPairs = when {
                enabled -> selectionPairs(rule)
                provisional?.get("class_pair_policy")?.jsonPrimitive?.content == "current_coco_pairs" ->
                    tensor.nodes.flatMap {a->tensor.nodes.map {b->listOf(a.cls,b.cls)}}.toSet()
                provisional != null -> provisional.getValue("allowed_class_pairs").jsonArray
                    .map { it.jsonArray.map { value -> value.jsonPrimitive.int } }.toSet()
                else -> emptySet()
            }
            val provisionalMetrics = provisional?.let(::provisionalMetrics)
            for (i in tensor.nodes.indices) for (j in tensor.nodes.indices) {
                if (i == j) continue
                val a = tensor.nodes[i]
                val b = tensor.nodes[j]
                val offset = ((i * R4Tensorizer.NODES + j) * 7 + relationIndex) * 4
                val logit = logits.getFloat(offset).toDouble()
                if (!logit.isFinite()) continue
                val probability = 1.0 / (1.0 + exp(-(if (calibrated) scale!! * logit + bias!! else logit)))
                val reason = when {
                    !appEnabled -> "Relation in der App deaktiviert"
                    !active -> "Relation gesperrt"
                    !scopeMatches -> "Tiefen-/Segmentierungskontext außerhalb der R4-Freigabe"
                    a.source != b.source -> "Quellen unterschiedlich"
                    minOf(a.confidence, b.confidence) < .5f -> "Objektkonfidenz zu niedrig"
                    maxOf(a.ageMs, b.ageMs) > 500 -> "Objektbeobachtung veraltet"
                    listOf(a.cls,b.cls) !in classPairs -> "Klassenpaar nicht freigegeben"
                    threshold == null || probability < threshold -> "Unter Relationsschwelle"
                    provisional != null && !enabled -> "Experimenteller $name-Modellkandidat"
                    else -> "Relationsgate bestanden; weitere Sprachprüfung ausstehend"
                }
                diagnostics += R4RawDiagnostic(tensor.snapshot,a.id,b.id,a.cls,b.cls,name,probability,calibrated,reason)
                if (!active || !scopeMatches || a.source != b.source || minOf(a.confidence, b.confidence) < .5f ||
                    maxOf(a.ageMs, b.ageMs) > 500 || listOf(a.cls, b.cls) !in classPairs) continue
                scored++
                if (!probability.isFinite() || threshold == null || probability < threshold) continue
                val accepted=selection ?: provisionalMetrics ?: continue
                candidates += R4RelationCandidate(tensor.snapshot, R4Tensorizer.PROJECTION_ID, a.id, b.id,
                    a.cls, b.cls, a.source, name, probability, threshold,
                    accepted.getValue("precision").jsonPrimitive.double,
                    accepted.getValue("bootstrap_lower_95").jsonPrimitive.double,
                    accepted.getValue("statements").jsonPrimitive.int,
                    accepted.getValue("images").jsonPrimitive.int,
                    if (enabled) "qualified" else provisional!!.getValue("status").jsonPrimitive.content)
            }
        }
        val safe = R4RelationTypes.normalize(candidates)
        val safeKeys=safe.map { Triple(it.subjectId,it.objectId,it.relation) }.toSet()
        val rejectedKeys=candidates.map { Triple(it.subjectId,it.objectId,it.relation) }.toSet()-safeKeys
        val all=diagnostics.map { d ->
            if (Triple(d.subjectId,d.objectId,d.relation) in rejectedKeys) d.copy(reason="Widersprüchliche Relationskandidaten") else d
        }
        val shown=all.sortedByDescending { it.score }.take(12)
        return Result(true, safe, scored, R4Tensorizer.PROJECTION_ID, tensor.snapshot, shown,all)
    }

    private fun selectionPairs(rule: JsonObject): Set<List<Int>> =
        rule.getValue("class_pairs").jsonArray.map { row -> row.jsonArray.map { it.jsonPrimitive.int } }.toSet()

    private fun provisionalMetrics(policy: JsonObject): JsonObject {
        val known=policy["known_annotation_metrics"] as? JsonObject
        if(known==null) return buildJsonObject {
            put("precision",0.0)
put("bootstrap_lower_95",0.0)
put("statements",0)
            put("images",0)
        }
        val metrics=known
            .getValue("detail_age_strata").jsonObject.values.flatMap { detail -> detail.jsonObject.values }
            .map { it.jsonObject }
        return buildJsonObject {
            put("precision",metrics.minOf { it.getValue("precision").jsonPrimitive.double })
            put("bootstrap_lower_95",metrics.minOf { it.getValue("bootstrap_lower_95").jsonPrimitive.double })
            put("statements",metrics.minOf { it.getValue("statements").jsonPrimitive.int })
            put("images",metrics.minOf { it.getValue("images").jsonPrimitive.int })
        }
    }

    override fun close() {
        interpreter?.close()
        interpreter = null
        modelBytes = null
    }
}
