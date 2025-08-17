package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
import android.graphics.Bitmap.createBitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.View.GONE
import android.view.View.VISIBLE
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.graphics.createBitmap
import androidx.core.net.toUri
import androidx.core.view.isVisible
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.llm.LLM
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
	var cameraManager = CameraManager()

	@RequiresApi(Build.VERSION_CODES.P)
	private var permissionManager =
		PermissionManager(this, ::onCameraPermissionResult, ::onMicrophonePermissionResult)

	private var cameraPreviewView: PreviewView? = null
	private var ungrantedPermissionsNotice: LinearLayout? = null
	private var ungrantedPermissionsNoticeText: TextView? = null
	private var allowCameraPermission: Button? = null
	private var flashlightButton: FloatingActionButton? = null
	private var killTTS: FloatingActionButton? = null

	private var depthPreviewImage: ImageView? = null

	private var debugInputBitmapPreview: ImageView? = null
	private var mediaImageView: ImageView? = null

	private var performanceText: TextView? = null

	private var overlayObjectDetection: OverlayViewOD? = null
	private var overlayOcr: OverlayViewOCR? = null

	private var speechRecognitionPartialResultText: TextView? = null
	private var speechRecognitionFinalResultText: TextView? = null
	private var llmResponseText: TextView? = null
	private var lastFinalResultMillis = System.currentTimeMillis()

	private var llmThreadExecutor = Executors.newSingleThreadExecutor()

	private lateinit var textToSpeechInstance: TextToSpeechInstance

	private var lastLlmJsonResponse: String? = null

	private enum class State {
		IDLE,
		SETTINGS_MENU,
		SETTINGS_CHOICE,


		SETTINGS_ACTION,

	}

	private var currentState: State = State.IDLE

	private var mediaFrameAnalyzer: CameraFrameAnalyzer? = null
	private var mediaPlayer: MediaPlayer? = null

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		enableEdgeToEdge()
		setContentView(R.layout.activity_main)

		cameraPreviewView = findViewById(R.id.camera_view)

		depthPreviewImage = findViewById(R.id.depth_preview_image)

		mediaImageView = findViewById(R.id.media_view)

		debugInputBitmapPreview = findViewById(R.id.debug_input_bitmap)

		performanceText = findViewById(R.id.performance_text)

		overlayObjectDetection = findViewById(R.id.overlay_object_detection)
		overlayOcr = findViewById(R.id.overlay_ocr)

		ungrantedPermissionsNotice = findViewById(R.id.ungranted_permissions_notice)
		ungrantedPermissionsNoticeText = findViewById(R.id.ungranted_permissions_notice_text)

		allowCameraPermission = findViewById(R.id.allow_camera_permission_btn)
		allowCameraPermission!!.setOnClickListener { permissionManager.openAppPermissionSettings() }

		flashlightButton = findViewById(R.id.flashlight_button)
		updateFlashlightButtonTint(cameraManager.isCameraFlashlightOn())
		flashlightButton!!.setOnClickListener {
			val flashlightOn = cameraManager.toggleCameraFlashlight()
			updateFlashlightButtonTint(flashlightOn)
		}

		killTTS = findViewById(R.id.stop_tts_button)
		killTTS!!.setOnClickListener {
			textToSpeechInstance.stop()
		}

		speechRecognitionPartialResultText = findViewById(R.id.speech_recognition_partial_output)
		speechRecognitionFinalResultText = findViewById(R.id.speech_recognition_final_output)
		llmResponseText = findViewById(R.id.llm_response)

		findViewById<FloatingActionButton>(R.id.settings_button).setOnClickListener {
			startActivity(Intent(this, SettingsActivity::class.java))
			overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
		}

		updateUngrantedPermissionsNotice()

		window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

		eyeAIApp().updateSettings()

		updateSpeechRecognitionUIVisibility()

		textToSpeechInstance = TextToSpeechInstance(this, ::onTTSFinishedSpeaking)
	}

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onResume() {
		super.onResume()

		eyeAIApp().updateSettings()

		updateSpeechRecognitionUIVisibility()

		permissionManager.requestPermissions()
		updateUngrantedPermissionsNotice()

		debugInputBitmapPreview?.visibility = if (eyeAIApp().settings.showDebugInputBitmap) {
			VISIBLE
		} else {
			GONE
		}

		updateFlashlightButtonTint(cameraManager.isCameraFlashlightOn())

		llmResponseText?.apply {
			text = if (eyeAIApp().llm == null)
				getString(R.string.setup_llm_notice)
			else
				""
		}
	}

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onPause() {
		super.onPause()

		eyeAIApp().voskModel.stopListening()

		cameraManager.pauseAnalyzer()
		mediaFrameAnalyzer?.shutdown()
		mediaPlayer?.shutdown()
	}

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onDestroy() {
		super.onDestroy()

		cameraManager.shutdown()
		textToSpeechInstance.shutdown()
		mediaFrameAnalyzer?.shutdown()
		mediaPlayer?.shutdown()

		eyeAIApp().voskModel.closeService()
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun onCameraPermissionResult(isGranted: Boolean) {
		if (isGranted) {
			ungrantedPermissionsNotice!!.visibility = GONE
			initCamera()
		} else {
			ungrantedPermissionsNotice!!.visibility = VISIBLE
		}
	}

	private fun onMicrophonePermissionResult(isGranted: Boolean) {
		if (isGranted && eyeAIApp().settings.enableSpeechRecognition) {
			eyeAIApp()
				.voskModel
				.initService(
					::onPartialSpeechRecognitionResult,
					::onFinalSpeechRecognitionResult,
					::onSpeechRecognitionLoaded
				)
		} else {
			Log.w(EyeAIApp.APP_LOG_TAG, "Microphone Permission not granted!")
		}
	}



	private fun eyeAIApp(): EyeAIApp {
		return application as EyeAIApp
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun initCamera() {
		if (eyeAIApp().settings.inputSource == getString(R.string.input_is_camera)) {
			mediaImageView!!.isVisible = false
			if (permissionManager.isCameraPermissionGranted()) {
				ungrantedPermissionsNotice!!.visibility = GONE

				cameraManager.cameraFrameAnalyzer?.shutdown()
				cameraManager.cameraFrameAnalyzer =
					CameraFrameAnalyzer(
						eyeAIApp(),
						depthPreviewImage!!,
						performanceText!!,
						overlayObjectDetection!!,
						overlayOcr!!,
						debugInputBitmapPreview!!,
						mediaImageView!!
					)
				cameraManager.cameraFrameAnalyzer?.start()

				cameraManager
					.init(
						this,
						EyeAIApp.PREFERRED_CAMERA_RESOLUTION,
						cameraPreviewView
					)
			} else {
				ungrantedPermissionsNotice!!.visibility = VISIBLE
			}
		} else if (eyeAIApp().settings.inputSource == getString(R.string.input_is_media)) {
			if (eyeAIApp().settings.mediaSource!!.isNotEmpty()) {
				mediaImageView!!.isVisible = true
				ProcessCameraProvider.getInstance(this).get().unbindAll()
				overlayObjectDetection!!.reset()
				overlayOcr!!.reset()
				depthPreviewImage!!.setImageBitmap(createBitmap(256,256))

				mediaPlayer?.shutdown()
				mediaPlayer = MediaPlayer(this, eyeAIApp().settings.mediaSource!!.toUri(), mediaImageView!!)

				mediaFrameAnalyzer?.shutdown()

				mediaFrameAnalyzer =
					CameraFrameAnalyzer(
						eyeAIApp(),
						depthPreviewImage!!,
						performanceText!!,
						overlayObjectDetection!!,
						overlayOcr!!,
						debugInputBitmapPreview!!,
						mediaImageView!!
					)

				mediaFrameAnalyzer?.start()
			} else {
				val builder = AlertDialog.Builder(this)
				builder.setMessage("No media file has been selected. Please select one in the settings menu")
					.setPositiveButton("Open settings") { dialog, id ->
						startActivity(Intent(this, SettingsActivity::class.java))
						overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
					}
				builder.create().show()
			}

		} else if (eyeAIApp().settings.inputSource == getString(R.string.input_is_eyeaivision)) {
			if (eyeAIApp().settings.eyeAIVisionIP!!.isNotEmpty()) {
				// HTTP Logic
			} else {
				val builder = AlertDialog.Builder(this)
				builder.setMessage("No IP address has been entered. Please enter one in the settings menu")
					.setPositiveButton("Open settings") { dialog, id ->
						startActivity(Intent(this, SettingsActivity::class.java))
						overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
					}
				builder.create().show()
			}
		}
	}

	private fun updateSpeechRecognitionUIVisibility() {
		val visibility = if (eyeAIApp().settings.enableSpeechRecognition) {
			VISIBLE
		} else {
			GONE
		}

		speechRecognitionPartialResultText?.visibility = visibility
		speechRecognitionFinalResultText?.visibility = visibility
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun updateUngrantedPermissionsNotice() {
		val cameraGranted = permissionManager.isCameraPermissionGranted()
		val microphoneGranted = permissionManager.isMicrophonePermissionGranted()

		if (cameraGranted && microphoneGranted) {
			ungrantedPermissionsNotice?.visibility = GONE
		} else {
			ungrantedPermissionsNotice?.visibility = VISIBLE
			ungrantedPermissionsNoticeText?.text = getString(
				if (cameraGranted) {
					R.string.allow_microphone_permission_notice
				} else if (microphoneGranted) {
					R.string.allow_camera_permission_notice
				} else {
					R.string.allow_camera_and_microphone_permission_notice
				}
			)
		}
	}

	private fun updateFlashlightButtonTint(isFlashlightOn: Boolean) {
		flashlightButton?.backgroundTintList = getColorStateList(
			if (isFlashlightOn) {
				R.color.flashlight_button_on
			} else {
				R.color.flashlight_button_off
			}
		)
	}



	/*All TTS methods start here*/

	private fun onPartialSpeechRecognitionResult(partial: String) {
		CoroutineScope(Dispatchers.Main).launch {
			speechRecognitionPartialResultText?.text = partial
		}
	}

	private fun onFinalSpeechRecognitionResult(final: String) {
		if (final.isEmpty()) {
			return
		}



		CoroutineScope(Dispatchers.Main).launch {
			speechRecognitionFinalResultText?.text = final

			// minimum of 1 second pause between speech commands
			if (System.currentTimeMillis() - lastFinalResultMillis <= 1000)
				return@launch

			lastFinalResultMillis = System.currentTimeMillis()

			if (eyeAIApp().llm == null) {
				llmResponseText?.text = getString(R.string.setup_llm_notice)
			} else {
				llmResponseText?.text = getString(R.string.llm_responding_notice)

				// start after onTTSFinished speaking
				eyeAIApp().voskModel.stopListening()

				// vibrate for 100ms
				vibrate(eyeAIApp(), 100)


				withContext(llmThreadExecutor.asCoroutineDispatcher()){
					onSpeechResult(final)
				}

			}
		}
	}

	private fun onSpeechRecognitionLoaded() {
		speechRecognitionFinalResultText?.text = getString(R.string.speech_recognition_ready)
	}

	private fun onTTSFinishedSpeaking() {
		// Muss auf dem Main thread laufen
		CoroutineScope(Dispatchers.Main).launch {
			speechRecognitionFinalResultText?.apply {
				text = getString(R.string.speech_recognition_ready)
			}

			eyeAIApp().voskModel.startListening()
		}
	}


	private suspend fun onSpeechResult(final: String) {
		when (currentState) {
			State.IDLE -> handleIdle(final)
			State.SETTINGS_MENU -> handleSettingsMenu(final)
			State.SETTINGS_CHOICE -> handleSettingsChoice(final)
			State.SETTINGS_ACTION -> handleSettingsAction(final)
		}
	}

	// Handling of the IDLE state
	private suspend fun handleIdle(final: String) {
		val initialResponse = eyeAIApp().llm!!.generate(final, false)
		when {
			initialResponse.contains("texterkennung", true) -> {
				val prompt = eyeAIApp().llm!!.buildOcrPrompt(eyeAIApp().ocrModel.lastResult)
				val ocrResponse = eyeAIApp().llm!!.generate(prompt, false)
				speakAndHandleUi(ocrResponse)
			}
			initialResponse.contains("einstellungen", true) -> {
				currentState = State.SETTINGS_MENU
				val settingsResponse = eyeAIApp().llm!!.generate(LLM.SETTINGS_PROMPT, false)
				speakAndHandleUi(settingsResponse)
			}
			else -> speakAndHandleUi(initialResponse)
		}
	}

	// Handling of the settings menu
	private suspend fun handleSettingsMenu(final: String) {
		currentState = State.SETTINGS_CHOICE
		// LLM explains options
		val prompt = "Erkläre kurz die Einstellungsmöglichkeit '$final' und frage, wie die Einstellung geändert werden soll je nach Kontext"
		// TODO: Create individual responses for each adaption
		val response = eyeAIApp().llm!!.generate(prompt, false)
		speakAndHandleUi(response)
	}

	// LLM executes user command
	private suspend fun handleSettingsChoice(final: String) {


		// 1. Send a prompt to the LLM
		val prompt = "Führe die folgende Aktion aus: '$final'."

		val jsonResponse = try {
			eyeAIApp().llm!!.generate(prompt, true) //Generating a structured response
		} catch (e: Exception) {
			// Catching invalid JSONs

			textToSpeechInstance.speak("LLM hat kein valides JSON-Format geliefert!")
			currentState = State.SETTINGS_MENU // Leaving the settings
			return
		}

		// 2. Saving the last JSON-response
		lastLlmJsonResponse = jsonResponse

		// 3. Parsing JSON to create the request
		var confirmationQuestion = "Soll ich die angeforderte Änderung durchführen?" // Fallback
		try {
			val jsonObject = JSONObject(jsonResponse)
			val changedSettings = jsonObject.optJSONArray("changed_settings")
			if (changedSettings != null && changedSettings.length() > 0) {
				val firstChange = changedSettings.getJSONObject(0)
				if (firstChange.has("tts_speed")) {
					val newSpeed = firstChange.getDouble("tts_speed")
					confirmationQuestion = "Verstanden. Soll ich die Sprachgeschwindigkeit auf ${newSpeed} setzen?"
				}
				if (firstChange.has("voice"))
				{

					val voice = firstChange.getString("voice")
					confirmationQuestion = "Verstanden. Soll ich die Assistentenstimme auf ${voice} setzen?"
				}
			}
		} catch (e: Exception) {

			Log.e(EyeAIApp.APP_LOG_TAG, "JSON-Parsing failed", e)
		}

		// 4. Change state to SETTINGS_ACTION, waiting for confirmation
		currentState = State.SETTINGS_ACTION
		speakAndHandleUi(confirmationQuestion)
	}


	// Handling of settings adaption
	private suspend fun handleSettingsAction(final: String) {

		val jsonResponse = try {
			eyeAIApp().llm!!.generate("Würdest du sagen der Nutzer hat diesen Command bestätigt? Die Antwort des Nutzers war $final" +
				"Antworte bitte mit einer JSON-Antwort in approval.", true) //Generating a structured response
		} catch (e: Exception) {
			// Catching invalid JSONs

			textToSpeechInstance.speak("LLM hat kein valides JSON-Format geliefert!")
			currentState = State.SETTINGS_MENU // Leaving the settings
			return
		}


		val jsonObject = JSONObject(jsonResponse)
		val changedSettings = jsonObject.getDouble("approval")


		//Checking whether the user confirms his action
		if (changedSettings.toInt() == 1 && lastLlmJsonResponse != null) {
			try {
				// 1. Parsing the JSONObject
				val jsonObject = JSONObject(lastLlmJsonResponse!!)
				val changedSettings = jsonObject.getJSONArray("changed_settings")

				// 2. Changing the settings
				for (i in 0 until changedSettings.length()) {
					val setting = changedSettings.getJSONObject(i)
					if (setting.has("tts_speed")) {
						//Changing speed
						val newSpeed = setting.getDouble("tts_speed").toFloat()
						textToSpeechInstance.setSpeechRate(newSpeed)
						Log.d(EyeAIApp.APP_LOG_TAG, "TTS-Geschwindigkeit wird auf $newSpeed gesetzt.")
					}
					if (setting.has("voice"))
					{

						val voice = setting.getDouble("voice")
						Log.d(EyeAIApp.APP_LOG_TAG, "Stimme wird auf $voice gesetzt.")
						textToSpeechInstance.setVoice(voice)


					}
					if(setting.has("leave")){
						currentState = State.IDLE
					}

				}

				// 3. Notifying the user that the changes have been applied
				speakAndHandleUi("Die Einstellung wurde erfolgreich geändert.")

			} catch (e: Exception) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Fehler bei der Verarbeitung der JSON-Aktion.", e)
				speakAndHandleUi("Entschuldigung, beim Anwenden der Einstellung ist ein Fehler aufgetreten.")
			}
		} else {
			// Managing an exit
			speakAndHandleUi("Okay, ich habe den Vorgang abgebrochen.")
		}

		// 4. Clearing up
		lastLlmJsonResponse = null
		currentState = State.IDLE
	}

	/**
	 * Adapting the UI
	 * Starting the TTS speech
	 */
	private suspend fun speakAndHandleUi(text: String) {
		// UI-Update using the main-thread
		withContext(Dispatchers.Main) {
			llmResponseText?.text = getString(R.string.llm_response, text)
		}
		// TTS (using the worker-thread)
		textToSpeechInstance.speak(text)
	}
}
