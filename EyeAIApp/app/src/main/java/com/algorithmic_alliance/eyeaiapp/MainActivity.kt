package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
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
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.ocr.GoogleOCR
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance

class MainActivity : AppCompatActivity() {
	var cameraManager = CameraManager()

	private var permissionManager =
		PermissionManager(this, ::onCameraPermissionResult, ::onMicrophonePermissionResult)

	private var cameraPreviewView: PreviewView? = null
	private var ungrantedPermissionsNotice: LinearLayout? = null
	private var ungrantedPermissionsNoticeText: TextView? = null
	private var allowCameraPermission: Button? = null
	private var flashlightButton: FloatingActionButton? = null
	private var killLLM: FloatingActionButton? = null

	private var depthPreviewImage: ImageView? = null

	private var debugInputBitmapPreview: ImageView? = null

	private var performanceText: TextView? = null

	private var overlayObjectDetection: OverlayViewOD? = null
	private var overlayOcr: OverlayViewOCR? = null

	private var speechRecognitionPartialResultText: TextView? = null
	private var speechRecognitionFinalResultText: TextView? = null
	private var llmResponseText: TextView? = null
	private var lastFinalResultMillis = System.currentTimeMillis()

	private var llmThreadExecutor = Executors.newSingleThreadExecutor()

	private lateinit var textToSpeechInstance: TextToSpeechInstance

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		enableEdgeToEdge()
		setContentView(R.layout.activity_main)

		cameraPreviewView = findViewById(R.id.camera_view)

		depthPreviewImage = findViewById(R.id.depth_preview_image)

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

		killLLM = findViewById(R.id.stop_llm_button)
		flashlightButton!!.setOnClickListener {
			// Kill LLM
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

	override fun onPause() {
		super.onPause()

		eyeAIApp().voskModel?.stopListening()
	}

	override fun onDestroy() {
		super.onDestroy()

		cameraManager.shutdown()
		textToSpeechInstance.shutdown()


		eyeAIApp().voskModel?.closeService()
	}

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
				?.initService(
					::onPartialSpeechRecognitionResult,
					::onFinalSpeechRecognitionResult,
					::onSpeechRecognitionLoaded
				)
		} else {
			Log.w(EyeAIApp.APP_LOG_TAG, "Microphone Permission not granted!")
		}
	}

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

				// wird erst wieder durch [onTTSFinishedSpeaking] gestartet
				eyeAIApp().voskModel?.stopListening()

				// vibrate for 100ms
				vibrate(eyeAIApp(), 100)

				val llmResponse = withContext(llmThreadExecutor.asCoroutineDispatcher()) {
					if (eyeAIApp().llm!!.generate(final).contains("texterkennung", true)){
						eyeAIApp().llm!!.generate("Das ist der zuletzt erkannte Text mit den zusätzlichen Koordinaten: " + eyeAIApp().ocrModel.lastResult + " \nBitte gib nur diesen in einem Format aus, dass es für einen menschen verständlich macht, der die Daten nur hören, nicht lesen kann. Mache anhand der übergebenen x und y Koordinaten des Handybildschirms aus, wo sich der Text in der Kameraperspektive befindet. Formuliere den Text so, als würdest du einer Person erklären, wo diese den erkannten Text sieht. Ein Beispiel wäre: Der Text ... befindet sich links oben von dir aus. Sprich also bitte nicht von einem Bildschirm, sondern sprich diese Person an. Nur in diesem Fall sollst du anschließend nicht Texterkennung wiederholen!")
					}
					else{
						eyeAIApp().llm!!.generate(final)
					}
				}

				textToSpeechInstance.speak(llmResponse.toString())


				llmResponseText?.text =
					getString(R.string.llm_response, llmResponse)
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

			eyeAIApp().voskModel?.startListening()
		}
	}

	private fun eyeAIApp(): EyeAIApp {
		return application as EyeAIApp
	}

	private fun initCamera() {
		if (permissionManager.isCameraPermissionGranted()) {
			ungrantedPermissionsNotice!!.visibility = GONE

			cameraManager.cameraFrameAnalyzer?.shutdown()
			cameraManager.cameraFrameAnalyzer =
				CameraFrameAnalyzer(
					eyeAIApp(), depthPreviewImage!!, performanceText!!, overlayObjectDetection!!,
					overlayOcr!!, debugInputBitmapPreview!!
				)
			cameraManager.cameraFrameAnalyzer?.start()

			val preferredInputSize = eyeAIApp().getPreferredCameraResolution()
			if (preferredInputSize != null) {
				cameraManager
					.init(
						this,
						preferredInputSize,
						cameraPreviewView
					)
			} else {
				Log.e(EyeAIApp.APP_LOG_TAG, "COULD NOT INIT CAMERA, DEPTH MODEL NOT LOADED YET!")
			}
		} else {
			ungrantedPermissionsNotice!!.visibility = VISIBLE
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
}
