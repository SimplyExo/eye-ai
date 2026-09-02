package com.algorithmic_alliance.eyeaiapp.settingsparser

/**
 * Legacy test/debug composition root. The product path is [LocalSettingsParser]
 * from [SettingsHandler]; this class stays injectable for deterministic unit
 * tests and does not own a TFLite interpreter.
 */
object LocalSettingsParserFeatureGate {
	/** Legacy-only marker; the production parser is not gated by this object. */
	const val ENABLED = false
}

class DebugSettingsParser(
	private val numberNormalizer: GermanNumberNormalizer,
	private val operationPredictor: OperationPredictor,
	private val speakerPredictor: SpeakerPredictor? = null,
	private val magnitudeParser: GermanMagnitudeParser = GermanMagnitudeParser(),
	private val unitParser: SettingUnitParser = SettingUnitParser(),
	private val assembler: SettingCommandAssembler = SettingCommandAssembler(),
	private val validator: SettingCommandValidator = SettingCommandValidator()
) {
	fun parse(target: SettingTarget, text: String): SettingCommand {
		val numbers = numberNormalizer.normalize(text)
		val operation = operationPredictor.predictOperation(target, numbers.normalizedText)
		val speaker = if (target == SettingTarget.SPEAKER) {
			speakerPredictor?.predictSpeaker(target, numbers.normalizedText)
		} else {
			null
		}
		val numericValue = numbers.values.singleOrNull()
		val magnitude = magnitudeParser.parse(text, operation.operation, numericValue)
		val unit = unitParser.parse(target, text)
		return validator.validate(
			assembler.assemble(target, text, numbers, operation, speaker, magnitude, unit)
		)
	}
}
