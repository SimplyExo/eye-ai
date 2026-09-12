package com.algorithmic_alliance.eyeaiapp.settingsparser

import android.content.Context

/**
 * Product composition root for the frozen Clean-v2 parser. It has no UI,
 * confirmation, persistence, or settings-write side effect.
 */
class LocalSettingsParser(
	private val numberNormalizer: GermanNumberNormalizer,
	private val operationPredictor: OperationPredictor,
	private val speakerPredictor: SpeakerPredictor,
	private val magnitudeParser: GermanMagnitudeParser = GermanMagnitudeParser(),
	private val unitParser: SettingUnitParser = SettingUnitParser(),
	private val assembler: SettingCommandAssembler = SettingCommandAssembler(),
	private val validator: SettingCommandValidator = SettingCommandValidator(),
	private val closeableRuntime: AutoCloseable? = null
) : AutoCloseable {
	fun parse(target: SettingTarget, text: String): SettingCommand {
		val numbers = numberNormalizer.normalize(text)
		val operation = operationPredictor.predictOperation(target, numbers.normalizedText)
		val speaker = if (target == SettingTarget.SPEAKER) {
			speakerPredictor.predictSpeaker(target, numbers.normalizedText)
		} else {
			null
		}
		val magnitude = magnitudeParser.parse(
			text = text,
			operation = operation.operation,
			numericValue = numbers.values.singleOrNull()
		)
		val unit = unitParser.parse(target, text)
		return validator.validate(
			assembler.assemble(target, text, numbers, operation, speaker, magnitude, unit)
		)
	}

	override fun close() {
		closeableRuntime?.close()
	}

	companion object {
		fun fromAssets(context: Context): LocalSettingsParser {
			val runtime = SpecializedSettingsTfliteRuntime.fromAssets(context)
			return LocalSettingsParser(
				numberNormalizer = Text2NumGermanNumberNormalizer(),
				operationPredictor = runtime,
				speakerPredictor = runtime,
				closeableRuntime = runtime
			)
		}
	}
}
