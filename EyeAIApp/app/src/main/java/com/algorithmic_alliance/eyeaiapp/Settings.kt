package com.algorithmic_alliance.eyeaiapp

import android.content.Context
import androidx.preference.PreferenceManager

class Settings(val context: Context) {
	private var sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)

	var depthModel: String
		private set

	var showProfilingInfo: Boolean
		private set

	var spatialAudioEnabled: Boolean
		private set

	init {
		depthModel = sharedPreferences.getString(
			context.getString(R.string.depth_model_setting),
			EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
		).toString()

		showProfilingInfo = sharedPreferences.getBoolean(
			context.getString(R.string.show_profiling_info_setting),
			false
		)

		spatialAudioEnabled = sharedPreferences.getBoolean(
			context.getString(R.string.spatial_audio_enabled_setting),
			true
		)
	}
}