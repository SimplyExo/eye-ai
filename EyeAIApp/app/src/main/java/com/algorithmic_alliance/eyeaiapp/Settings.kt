package com.algorithmic_alliance.eyeaiapp

import android.app.Activity
import androidx.preference.PreferenceManager

object Settings {
	fun getDepthModel(activity: Activity): String {
		val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(activity)
		return sharedPreferences.getString(
			activity.getString(R.string.depth_model_setting),
			EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
		).toString()
	}
}