package com.algorithmic_alliance.eyeaiapp.scenenarrator

import android.content.Context
import com.algorithmic_alliance.eyeaiapp.worldstate.TinyWorldInput
import kotlinx.serialization.json.*
import java.security.MessageDigest

// Deterministic speech from evidence, with optional offline R4 perception.
class LocalNarratorModel(context: Context) : AutoCloseable {
    data class Result(val text: String,
        val error: String? = null, val r4ModelExecuted: Boolean = false,
        val modelName: String = "Deterministische Beschreibung",
        val r4Diagnostics: List<R4RawDiagnostic> = emptyList(),val r4AllDiagnostics: List<R4RawDiagnostic> = emptyList())

    private val appContext = context.applicationContext
    private fun bytes(name: String) = appContext.assets.open("scene-narrator/$name").use { it.readBytes() }
    private fun json(name: String) = Json.parseToJsonElement(bytes(name).toString(Charsets.UTF_8)).jsonObject
    private val manifest = json("deterministic-manifest.json").also { m ->
        require(m.getValue("schema").jsonPrimitive.content == "android-deterministic-r4-1")
        require(m.getValue("realization").jsonPrimitive.content == "spatial-orientation-de-v4-two-styles")
        val files = m.getValue("files").jsonObject
        require(files.keys == setOf("language.json", "evidence-contract.json"))
        for ((name, hash) in files) {
            val actual = MessageDigest.getInstance("SHA-256").digest(bytes(name))
                .joinToString("") { "%02x".format(it.toInt() and 255) }
            require(actual == hash.jsonPrimitive.content) { "Packaged narration checksum mismatch: $name" }
        }
    }
    private val contract = json("evidence-contract.json").also {
        require(it.getValue("schema").jsonPrimitive.content == "shared-evidence-android-1")
        require(it.getValue("r4_projection_id").jsonPrimitive.content == R4Tensorizer.PROJECTION_ID)
    }
    private val language = LocalNarrative(json("language.json"))
    private val r4State = lazy { runCatching { LocalR4RelationModel(appContext) } }

    fun generate(input: TinyWorldInput, detail: String = "STANDARD",
        useR4: Boolean = false, trace: ((String,Int,Int,FloatArray)->Unit)? = null): Result {
        var relationModel: LocalR4RelationModel? = null
        var r4Result: LocalR4RelationModel.Result? = null
        var error: String? = null
        if (useR4) {
            try {
                relationModel = r4State.value.getOrThrow()
                r4Result = relationModel.infer(input,trace)
            } catch (e: Exception) {
                // Missing perception only removes its relation facts, technical diagnostics are never spoken.
                error = e.message ?: e.javaClass.simpleName
            }
        }
        val prepared = language.prepareDeterministic(input, detail, r4Result?.candidates.orEmpty())
        val output = language.render(prepared)
        val name = when {
            !useR4 -> "Deterministische Beschreibung"
            r4Result?.executed != true -> "Deterministische Beschreibung · R4 nicht verfügbar"
            else -> "Deterministische Beschreibung + R4 ${relationModel?.modelPrecision}"
        }
        return Result(text = output, error = error, r4ModelExecuted = r4Result?.executed == true,
            modelName = name, r4Diagnostics = r4Result?.diagnostics.orEmpty(),r4AllDiagnostics=r4Result?.allDiagnostics.orEmpty())
    }

    override fun close() {
        if (r4State.isInitialized()) r4State.value.getOrNull()?.close()
    }
}
