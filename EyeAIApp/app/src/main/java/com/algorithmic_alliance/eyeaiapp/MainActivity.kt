package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
	private var permissionManager =
		PermissionManager(this, ::onCameraPermissionResult, ::onMicrophonePermissionResult)

	private var cameraFrameAnalyzer: CameraFrameAnalyzer? = null

	private var cameraPreviewView: PreviewView? = null
	private var cameraPermissionNotice: LinearLayout? = null
	private var allowCameraPermission: Button? = null
	private var enableFlashlightCheckbox: CheckBox? = null

	private var depthPreviewImage: ImageView? = null

	private var performanceText: TextView? = null

	private var speechRecognitionPartialResultText: TextView? = null
	private var speechRecognitionFinalResultText: TextView? = null
	private var lastFinalResultMillis = System.currentTimeMillis()

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)

		enableEdgeToEdge()
		setContentView(R.layout.activity_main)

		cameraPreviewView = findViewById(R.id.camera_view)

		depthPreviewImage = findViewById(R.id.depth_preview_image)

		performanceText = findViewById(R.id.performance_text)

		cameraPermissionNotice = findViewById(R.id.camera_permission_notice)

		allowCameraPermission = findViewById(R.id.allow_camera_permission_btn)
		allowCameraPermission!!.setOnClickListener { permissionManager.openAppPermissionSettings() }

		enableFlashlightCheckbox = findViewById(R.id.enable_flashlight)
		enableFlashlightCheckbox!!.isChecked = eyeAIApp().cameraManager.isCameraFlashlightOn()
		enableFlashlightCheckbox!!.setOnClickListener {
			val flashlightOn = eyeAIApp().cameraManager.toggleCameraFlashlight()
			enableFlashlightCheckbox!!.isChecked = flashlightOn
		}

		speechRecognitionPartialResultText = findViewById(R.id.speech_recognition_partial_output)
		speechRecognitionFinalResultText = findViewById(R.id.speech_recognition_final_output)

		findViewById<FloatingActionButton>(R.id.settings_button).setOnClickListener {
			startActivity(Intent(this, SettingsActivity::class.java))
			overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
		}

		cameraFrameAnalyzer =
			CameraFrameAnalyzer(eyeAIApp(), depthPreviewImage!!, performanceText!!)

		permissionManager.requestPermissions()

		if (permissionManager.isCameraPermissionGranted()) {
			cameraPermissionNotice!!.visibility = View.GONE
			initCamera()
		}

		if (permissionManager.isMicrophonePermissionGranted())
			eyeAIApp()
				.voskModel
				.init(::onPartialSpeechRecognitionResult, ::onFinalSpeechRecognitionResult)

		window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

		eyeAIApp().updateSettings()
	}

	override fun onResume() {
		super.onResume()

		eyeAIApp().updateSettings()

		enableFlashlightCheckbox!!.isChecked = eyeAIApp().cameraManager.isCameraFlashlightOn()

		val cameraPermissionGranted = permissionManager.isCameraPermissionGranted()

		if (eyeAIApp().cameraManager.cameraPreview == null && cameraPermissionGranted) initCamera()

		eyeAIApp().voskModel.setPaused(false)

		NativeLib.enableSpatialAudio()
		NativeLib.updateSpatialAudio(
			0.5f,
			440.0f,
			0.0f,
			0.0f,
			0.0f
		)
	}

	override fun onPause() {
		super.onPause()

		eyeAIApp().voskModel.setPaused(false)

		NativeLib.disableSpatialAudio()
	}

	private fun onCameraPermissionResult(isGranted: Boolean) {
		if (isGranted) {
			cameraPermissionNotice!!.visibility = View.GONE
			initCamera()
		} else {
			cameraPermissionNotice!!.visibility = View.VISIBLE
		}
	}

	private fun onMicrophonePermissionResult(isGranted: Boolean) {
		Log.i(EyeAIApp.APP_LOG_TAG, "Microphone permission granted: $isGranted")
	}

	private fun onPartialSpeechRecognitionResult(partial: String) {
		speechRecognitionPartialResultText?.apply { text = partial }
	}

	private fun onFinalSpeechRecognitionResult(final: String) {
		speechRecognitionFinalResultText?.apply { text = final }
		val lastFinalResultDurationMillis = System.currentTimeMillis() - lastFinalResultMillis
		if (!final.isEmpty() && lastFinalResultDurationMillis > 1500) {
			lastFinalResultMillis = System.currentTimeMillis()

			// turns speech recognition model off for 500ms to not pick up vibration sounds
			CoroutineScope(Dispatchers.Main).launch {
				eyeAIApp().voskModel.setPaused(true)
				// vibrate for 100ms
				vibrate(eyeAIApp(), 100)
				delay(1000)
				eyeAIApp().voskModel.setPaused(false)
			}
		}
	}

	private fun eyeAIApp(): EyeAIApp {
		return application as EyeAIApp
	}

	private fun initCamera() {
		if (permissionManager.isCameraPermissionGranted()) {
			cameraPermissionNotice!!.visibility = View.GONE
			eyeAIApp()
				.cameraManager
				.init(
					this,
					eyeAIApp().depthModel.getInputSize(),
					cameraPreviewView,
					cameraFrameAnalyzer!!
				)
		} else {
			cameraPermissionNotice!!.visibility = View.VISIBLE
		}
	}
}
