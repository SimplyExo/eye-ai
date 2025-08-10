package com.algorithmic_alliance.eyeaiapp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.ListPreference
import androidx.preference.Preference
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

			findPreference<ListPreference>(getString(R.string.depth_model_setting))?.let { list ->
				val modelNames =
					EyeAIApp.DEPTH_MODELS.map { it.name }.toTypedArray()

				list.entries = modelNames
				list.entryValues = modelNames
				list.setDefaultValue(EyeAIApp.DEFAULT_DEPTH_MODEL_NAME)
				if (list.value == null || list.value?.equals("") == true) {
					list.value = EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
				}
			}

			findPreference<EditTextPreference>(getString(R.string.custom_google_gen_ai_studio_endpoint_setting))?.let { endpointPreference ->
				updateCustomGoogleGenAIStudioEndpointPreferenceSummary(
					this,
					endpointPreference,
					endpointPreference.text
				)

				// formats the "Custom Google Gen AI Studio endpoint" summary
				endpointPreference.onPreferenceChangeListener =
					Preference.OnPreferenceChangeListener { preference, newValue ->
						if (preference is EditTextPreference && preference.key == getString(
								R.string.custom_google_gen_ai_studio_endpoint_setting
							) && newValue is String?
						) {
							updateCustomGoogleGenAIStudioEndpointPreferenceSummary(
								this,
								preference,
								newValue
							)
						}
						true // save the new value
					}
			}

			findPreference<Preference>(getString(R.string.version_info_settings))?.summary =
				BuildInfoHelper.getVersionInfo()
			findPreference<Preference>(getString(R.string.build_time_settings))?.summary =
				BuildInfoHelper.getFormattedBuildTime()
			findPreference<Preference>(getString(R.string.git_info_settings))?.summary =
				BuildInfoHelper.getGitInfo()
			findPreference<Preference>(getString(R.string.build_variant_settings))?.summary =
				BuildInfoHelper.getBuildVariant()
		}
	}
}

private fun updateCustomGoogleGenAIStudioEndpointPreferenceSummary(
	settingsFragment: SettingsActivity.SettingsFragment,
	preference: EditTextPreference,
	value: String?
) {
	preference.summary = if (value?.isEmpty() ?: true) {
		""
	} else {
		settingsFragment.getString(
			R.string.custom_google_gen_ai_studio_endpoint_summary,
			value
		)
	}
}