package com.algorithmic_alliance.eyeaiapp

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.preference.EditTextPreference
import androidx.preference.ListPreference
import androidx.preference.Preference
import androidx.preference.PreferenceCategory
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
		private var mediaPref: Preference? = null

		private val openDocument = registerForActivityResult(
			ActivityResultContracts.StartActivityForResult()
		) { result ->
			if (result.resultCode == Activity.RESULT_OK) {
				result.data?.data?.let { uri ->
					try {
						requireContext().contentResolver.takePersistableUriPermission(
							uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
						)
					} catch (e: SecurityException) {
						e.printStackTrace()
					}

					val path = uri.toString()
					preferenceManager.sharedPreferences
						?.edit()
						?.putString(getString(R.string.media_path_setting), path)
						?.apply()
					mediaPref?.summary = path
				}
			}
		}

		override fun onCreatePreferences(savedInstanceState: Bundle?, rootKey: String?) {
			setPreferencesFromResource(R.xml.settings_preferences, rootKey)

			val showDevelopmentSettings = BuildConfig.BUILD_VARIANT != "Production"

			findPreference<PreferenceCategory>(getString(R.string.debugging_settings_category))
				?.isVisible = showDevelopmentSettings

			findPreference<PreferenceCategory>(getString(R.string.build_info_settings_category))
				?.isVisible = showDevelopmentSettings

			// Depth Model selector
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

			// Object Detection Model selector
			findPreference<ListPreference>(getString(R.string.object_detection_model_setting))?.let { list ->
				val modelNames = EyeAIApp.YOLO_MODELS.map { it.name }.toTypedArray()

				list.entries = modelNames
				list.entryValues = modelNames
				list.setDefaultValue(EyeAIApp.DEFAULT_YOLO_MODEL_NAME)
				if (list.value !in modelNames) {
					list.value = EyeAIApp.DEFAULT_YOLO_MODEL_NAME
				}
			}

			// Custom Google Gen Ai Studio Endpoint
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

			// Media File Selector
			mediaPref = findPreference(this.getString(R.string.media_path_setting))

			// Falls schon gespeichert -> direkt anzeigen
			val savedPath = preferenceManager.sharedPreferences
				?.getString(this.getString(R.string.media_path_setting), null)
			if (savedPath != null) {
				mediaPref?.summary = savedPath
			}

			val openFilePref: Preference? = findPreference(this.getString(R.string.media_path_setting))
			openFilePref?.setOnPreferenceClickListener {
				openFile()
				true
			}

			// Build Infos
			findPreference<Preference>(getString(R.string.version_info_settings))?.summary =
				BuildInfoHelper.getVersionInfo()
			findPreference<Preference>(getString(R.string.build_time_settings))?.summary =
				BuildInfoHelper.getFormattedBuildTime()
			findPreference<Preference>(getString(R.string.git_info_settings))?.summary =
				BuildInfoHelper.getGitInfo()
			findPreference<Preference>(getString(R.string.build_variant_settings))?.summary =
				BuildInfoHelper.getBuildVariant()
		}

		private fun openFile() {
			val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
				addCategory(Intent.CATEGORY_OPENABLE)
				type = "*/*"
				putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("image/*", "video/*"))
			}
			openDocument.launch(intent)
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
