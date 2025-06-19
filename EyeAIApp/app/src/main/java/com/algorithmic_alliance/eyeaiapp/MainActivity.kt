package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.camera.view.PreviewView
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
	private var permissionManager =
		PermissionManager(this, ::onCameraPermissionResult, ::onMicrophonePermissionResult)

	private var cameraFrameAnalyzer: CameraFrameAnalyzer? = null

	private var cameraPreviewView: PreviewView? = null
	private var ungrantedPermissionsNotice: LinearLayout? = null
	private var ungrantedPermissionsNoticeText: TextView? = null
	private var allowCameraPermission: Button? = null
	private var flashlightButton: FloatingActionButton? = null

	private var depthPreviewImage: ImageView? = null

	private var performanceText: TextView? = null

	private var speechRecognitionPartialResultText: TextView? = null
	private var speechRecognitionFinalResultText: TextView? = null
	private var llmResponseText: TextView? = null
	private var lastFinalResultMillis = System.currentTimeMillis()

	private var llmThreadExecutor = Executors.newSingleThreadExecutor()

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		enableEdgeToEdge()
		setContentView(R.layout.activity_main)

		cameraPreviewView = findViewById(R.id.camera_view)

		depthPreviewImage = findViewById(R.id.depth_preview_image)

		performanceText = findViewById(R.id.performance_text)

		ungrantedPermissionsNotice = findViewById(R.id.ungranted_permissions_notice)
		ungrantedPermissionsNoticeText = findViewById(R.id.ungranted_permissions_notice_text)

		allowCameraPermission = findViewById(R.id.allow_camera_permission_btn)
		allowCameraPermission!!.setOnClickListener { permissionManager.openAppPermissionSettings() }

		flashlightButton = findViewById(R.id.flashlight_button)
		updateFlashlightButtonTint(eyeAIApp().cameraManager.isCameraFlashlightOn())
		flashlightButton!!.setOnClickListener {
			val flashlightOn = eyeAIApp().cameraManager.toggleCameraFlashlight()
			updateFlashlightButtonTint(flashlightOn)
		}

		speechRecognitionPartialResultText = findViewById(R.id.speech_recognition_partial_output)
		speechRecognitionFinalResultText = findViewById(R.id.speech_recognition_final_output)
		llmResponseText = findViewById(R.id.llm_response)

		findViewById<FloatingActionButton>(R.id.settings_button).setOnClickListener {
			startActivity(Intent(this, SettingsActivity::class.java))
			overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
		}

		cameraFrameAnalyzer =
			CameraFrameAnalyzer(eyeAIApp(), depthPreviewImage!!, performanceText!!)

		permissionManager.requestPermissions()

		eyeAIApp().onDepthModelLoadedCallback = { initCamera() }

		updateUngrantedPermissionsNotice()

		if (permissionManager.isCameraPermissionGranted())
			initCamera()

		if (permissionManager.isMicrophonePermissionGranted()) {
			eyeAIApp()
				.voskModel
				?.initService(
					::onPartialSpeechRecognitionResult,
					::onFinalSpeechRecognitionResult,
					::onSpeechRecognitionLoaded
				)
		}

		window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

		eyeAIApp().updateSettings()

		updateSpeechRecognitionUIVisibility()
	}

	override fun onResume() {
		super.onResume()

		eyeAIApp().updateSettings()

		updateSpeechRecognitionUIVisibility()

		permissionManager.requestPermissions()
		updateUngrantedPermissionsNotice()

		updateFlashlightButtonTint(eyeAIApp().cameraManager.isCameraFlashlightOn())

		val cameraPermissionGranted = permissionManager.isCameraPermissionGranted()

		if (eyeAIApp().cameraManager.cameraPreview == null && cameraPermissionGranted)
			initCamera()

		val voskModelInitialized = eyeAIApp().voskModel?.isInitialized() == true
		if (permissionManager.isMicrophonePermissionGranted() && !voskModelInitialized) {
			eyeAIApp().voskModel?.apply {
				initService(
					::onPartialSpeechRecognitionResult,
					::onFinalSpeechRecognitionResult,
					::onSpeechRecognitionLoaded
				)

				startListening()
			}
		}

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

		eyeAIApp().voskModel?.closeService()
	}

	private fun onCameraPermissionResult(isGranted: Boolean) {
		if (isGranted) {
			ungrantedPermissionsNotice!!.visibility = View.GONE
			initCamera()
		} else {
			ungrantedPermissionsNotice!!.visibility = View.VISIBLE
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
		speechRecognitionPartialResultText?.apply { text = partial }
	}

	private fun onFinalSpeechRecognitionResult(final: String) {
		if (final.isEmpty()) {
			return
		}

		speechRecognitionFinalResultText?.apply { text = final }

		// pause recognition for 500ms after final speech command to prevent mic picking up the vibration sounds
		if (System.currentTimeMillis() - lastFinalResultMillis > 1000) {

			if (eyeAIApp().llm == null) {
				llmResponseText?.apply { text = getString(R.string.setup_llm_notice) }
			} else {
				llmResponseText?.apply { text = getString(R.string.llm_responding_notice) }

				CoroutineScope(llmThreadExecutor.asCoroutineDispatcher()).launch {
					val llmResponse = eyeAIApp().llm!!.generate(final)

					withContext(Dispatchers.Main) {
						llmResponseText?.apply {
							text =
								getString(R.string.llm_response, llmResponse)
						}
					}
				}
			}

			eyeAIApp().voskModel?.stopListening()

			// vibrate for 100ms
			vibrate(eyeAIApp(), 100)

			eyeAIApp().voskModel?.startListening()

			lastFinalResultMillis = System.currentTimeMillis()
		}
	}

	private fun onSpeechRecognitionLoaded() {
		speechRecognitionFinalResultText?.apply {
			text = getString(R.string.speech_recognition_ready)
		}
	}

	private fun eyeAIApp(): EyeAIApp {
		return application as EyeAIApp
	}

	private fun initCamera() {
		if (permissionManager.isCameraPermissionGranted()) {
			ungrantedPermissionsNotice!!.visibility = View.GONE
			val preferredInputSize = eyeAIApp().depthModel?.getInputSize()
			if (preferredInputSize != null) {
				eyeAIApp()
					.cameraManager
					.init(
						this,
						preferredInputSize,
						cameraPreviewView,
						cameraFrameAnalyzer!!
					)
			}
		} else {
			ungrantedPermissionsNotice!!.visibility = View.VISIBLE
		}
	}

	private fun updateSpeechRecognitionUIVisibility() {
		val visibility = if (eyeAIApp().settings.enableSpeechRecognition) {
			View.VISIBLE
		} else {
			View.GONE
		}

		speechRecognitionPartialResultText?.visibility = visibility
		speechRecognitionFinalResultText?.visibility = visibility
	}

	private fun updateUngrantedPermissionsNotice() {
		val cameraGranted = permissionManager.isCameraPermissionGranted()
		val microphoneGranted = permissionManager.isMicrophonePermissionGranted()

		if (cameraGranted && microphoneGranted) {
			ungrantedPermissionsNotice?.visibility = View.GONE
		} else {
			ungrantedPermissionsNotice?.visibility = View.VISIBLE
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
