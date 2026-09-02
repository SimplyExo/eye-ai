package com.algorithmic_alliance.eyeaiapp.settingsparser

import org.json.JSONArray
import org.json.JSONObject

/**
 * Translates an already validated/resolved command into the existing settings
 * confirmation JSON schema. It neither applies nor confirms a change.
 */
sealed interface LocalSettingsCommandExecution {
	data class Ready(
		val command: SettingCommand,
		val resolution: SettingResolution,
		val settingsJson: String
	) : LocalSettingsCommandExecution

	data class NotReady(
		val command: SettingCommand,
		val resolution: SettingResolution
	) : LocalSettingsCommandExecution

	data class UnsupportedAppRepresentation(
		val command: SettingCommand,
		val resolution: SettingResolution,
		val diagnostic: String
	) : LocalSettingsCommandExecution
}

class SettingsCommandExecutor(
	private val resolver: SettingsStateResolver = SettingsStateResolver()
) {
	fun execute(
		command: SettingCommand,
		current: CurrentSettingsState
	): LocalSettingsCommandExecution {
		val resolution = resolver.resolve(command, current)
		if (resolution.status != SettingParseStatus.COMPLETE) {
			return LocalSettingsCommandExecution.NotReady(command, resolution)
		}
		val change = toChangedSetting(resolution)
			?: return LocalSettingsCommandExecution.UnsupportedAppRepresentation(
				command,
				resolution,
				"ANDROID_BPS_REQUIRES_INTEGER"
			)
		return LocalSettingsCommandExecution.Ready(
			command = command,
			resolution = resolution,
			settingsJson = JSONObject().apply {
				put("settings_parameter_complete", true)
				put("changed_settings", JSONArray().put(change))
			}.toString()
		)
	}

	private fun toChangedSetting(resolution: SettingResolution): JSONObject? = when (resolution.target) {
		SettingTarget.FREQUENCY -> numericChange("frequency", resolution, requireWhole = true)
		SettingTarget.BPS -> numericChange("bps", resolution, requireWhole = true)
		SettingTarget.SPEECH_SPEED -> numericChange("tts_speed", resolution, requireWhole = false)
		SettingTarget.SPEAKER -> {
			val speaker = (resolution.requestedValue as? ResolvedSettingValue.Speaker)?.value ?: return null
			JSONObject().put("voice", if (speaker == SpeakerChoice.MALE) 1 else 0)
		}
	}

	private fun numericChange(
		key: String,
		resolution: SettingResolution,
		requireWhole: Boolean
	): JSONObject? {
		val value = (resolution.requestedValue as? ResolvedSettingValue.Numeric)?.value ?: return null
		if (!value.isFinite() || (requireWhole && value % 1.0 != 0.0)) return null
		return JSONObject().apply {
			if (requireWhole) put(key, value.toInt()) else put(key, value)
		}
	}
}
