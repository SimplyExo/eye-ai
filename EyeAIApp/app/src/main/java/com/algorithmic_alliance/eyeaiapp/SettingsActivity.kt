package com.algorithmic_alliance.eyeaiapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.ListPreference
import androidx.preference.PreferenceFragmentCompat

class SettingsActivity : AppCompatActivity() {

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		supportFragmentManager
			.beginTransaction()
			.replace(android.R.id.content, SettingsFragment())
			.commit()

		supportActionBar?.setDisplayHomeAsUpEnabled(true)
	}

	override fun onSupportNavigateUp(): Boolean {
		finish()
		return true
	}

	class SettingsFragment : PreferenceFragmentCompat() {
		override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
			setPreferencesFromResource(R.xml.settings_preferences, rootKey)

			findPreference<ListPreference>(getString(R.string.depth_model_setting))?.let {
				val modelNames =
					EyeAIApp.DEPTH_MODELS.map { it.name as CharSequence }.toTypedArray()

				it.entries = modelNames
				it.entryValues = modelNames
				it.setDefaultValue(EyeAIApp.DEFAULT_DEPTH_MODEL_NAME)
			}
		}
	}
}