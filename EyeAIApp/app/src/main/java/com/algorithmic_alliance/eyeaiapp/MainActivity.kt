package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
import android.graphics.Bitmap
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
import androidx.lifecycle.lifecycleScope
import com.algorithmic_alliance.eyeaiapp.NativeLib.setDepthAudioPaused
import com.algorithmic_alliance.eyeaiapp.NativeLib.setObjectAudioPaused
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import com.algorithmic_alliance.eyeaiapp.audio.AudioDeviceManager
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.flow.MutableSharedFlow

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
	private var startStopVosk: FloatingActionButton? = null

	private lateinit var eyeAIVision: EyeAIVision
	private var bitmapFlow: MutableSharedFlow<Bitmap>? = null

	private val voskStarting = AtomicBoolean(false)

	private val voskUserStart = AtomicBoolean(false)


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

	private var currentStateMachine: StateMachine? = null

	private var tcpErrorShowing = false
	private var mjpegErrorShowing = false

	private var mjpegErrorIgnored = false

	enum class State {
		IDLE,
		SETTINGS_MENU,
		SETTINGS_CHOICE,
		SETTINGS_ACTION,
	}

	private var currentState: State = State.IDLE

	private var mediaFrameAnalyzer: CameraFrameAnalyzer? = null
	private var mediaPlayer: MediaPlayer? = null

	private lateinit var audioDeviceManager : AudioDeviceManager

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

		startStopVosk = findViewById(R.id.stop_tts_button)
		startStopVosk!!.setOnClickListener {

			if (voskUserStart.get()){
				stopVoskListening()
			}
			else{
				startVoskListening()
			}

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

		textToSpeechInstance = TextToSpeechInstance(this) {
			// starting vosk if no stream is active
			CoroutineScope(Dispatchers.Main).launch {
				try {
					//Announcing that the global callback has been fired.
					Log.d(EyeAIApp.APP_LOG_TAG, "GLOBAL TTS CALLBACK: Fired.")

					if(!voskUserStart.get()){
						Log.d(EyeAIApp.APP_LOG_TAG,"User hasn't started Vosk yet - skipping")
						return@launch
					}

					//checking whether a stream is ongoing

					val isStreaming = currentStateMachine?.isStreaming() ?: false
					if (isStreaming) {
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"GLOBAL TTS CALLBACK: Streaming is active — NOT starting Vosk."
						)
						return@launch
					}

					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"GLOBAL TTS CALLBACK: TextToSpeechInstance already waited for silence -> attempting to start Vosk."
					)

					//avoiding prallel starts
					if (voskStarting.compareAndSet(false, true)) {
						try {
							eyeAIApp().voskModel.startListening()
							Log.d(
								EyeAIApp.APP_LOG_TAG,
								"GLOBAL TTS CALLBACK: Vosk startListening() invoked."
							)
						} catch (e: Exception) {
							Log.e(
								EyeAIApp.APP_LOG_TAG,
								"GLOBAL TTS CALLBACK: Failed to start Vosk.",
								e
							)
						} finally {
							voskStarting.set(false)
						}
					} else {
						Log.d(
							EyeAIApp.APP_LOG_TAG,
							"GLOBAL TTS CALLBACK: Vosk is already starting; skipping duplicate start."
						)
					}
				} catch (e: Exception) {
					Log.e(EyeAIApp.APP_LOG_TAG, "Exception in global TTS finished handler", e)
					if (voskUserStart.get() && voskStarting.compareAndSet(false, true)) {
						try {
							eyeAIApp().voskModel.startListening()
						} catch (_: Exception) {
						} finally {
							voskStarting.set(false)
						}
					}
				}
			}
		}


		CoroutineScope(Dispatchers.IO).launch {
			SpatialAudio.setup(this@MainActivity)
			SpatialAudio.start()
		}

		audioDeviceManager = AudioDeviceManager(this@MainActivity)
		audioDeviceManager.register()
	}

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onResume() {
		super.onResume()

		eyeAIApp().updateSettings()

		updateSpeechRecognitionUIVisibility()

		permissionManager.requestCameraPermission()
		if (eyeAIApp().settings.enableSpeechRecognition)
			permissionManager.requestMicrophonePermission()
		updateUngrantedPermissionsNotice()

		debugInputBitmapPreview?.visibility = if (eyeAIApp().settings.showDebugInputBitmap) {
			VISIBLE
		} else {
			GONE
		}

		updateFlashlightButtonTint(cameraManager.isCameraFlashlightOn())

		val isLLMConfigured = eyeAIApp().settings.googleAiStudioApiKey?.isEmpty() == false
		llmResponseText?.text = if (isLLMConfigured)
			""
		else
			getString(R.string.setup_llm_notice)

		// re-enabling the audio playback in accordance to the settings
		val settings = Settings.load(this@MainActivity)
		setObjectAudioPaused(!settings.objectAudioPlayback)
		setDepthAudioPaused(!settings.depthAudioPlayback)


	}

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onPause() {
		super.onPause()

		eyeAIApp().voskModel.stopListening()

		cameraManager.pauseAnalyzer()
		mediaFrameAnalyzer?.shutdown()
		mediaPlayer?.shutdown()

		// stopping audio playback
		setObjectAudioPaused(true)
		setDepthAudioPaused(true)
	}

	@RequiresApi(Build.VERSION_CODES.P)
	override fun onDestroy() {
		super.onDestroy()
		cameraManager.shutdown()
		textToSpeechInstance.shutdown()
		mediaFrameAnalyzer?.shutdown()
		mediaPlayer?.shutdown()
		SpatialAudio.destroy()
		unregisterReceiver(audioDeviceManager)

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

	@RequiresApi(Build.VERSION_CODES.P)
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
				depthPreviewImage!!.setImageBitmap(createBitmap(256, 256))

				mediaPlayer?.shutdown()
				mediaPlayer =
					MediaPlayer(this, eyeAIApp().settings.mediaSource!!.toUri(), mediaImageView!!)

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
			if (!eyeAIApp().settings.eyeAIVisionIP!!.isEmpty()) {
				bitmapFlow = MutableSharedFlow(replay = 1)

				eyeAIVision = EyeAIVision(
					ip = eyeAIApp().settings.eyeAIVisionIP.toString(),
					lifecycleScope = lifecycleScope,
					bitmapFlow = bitmapFlow,
					onSingleClick = {
						Log.i("CLICK", "SINGLE")


						if (!voskUserStart.get()) {
							startVoskListening()
						}


					},
					onDoubleClick = {
						Log.i("CLICK", "DOUBLE")

						if(voskUserStart.get()) {
							stopVoskListening()
						}
					},
					onSocketFailed = { e ->
						runOnUiThread {
							if (!tcpErrorShowing) {
								tcpErrorShowing = true
								val errorMessage = AlertDialog.Builder(this)
								errorMessage.setMessage("TCP connection to EyeAIVision (IP: ${eyeAIApp().settings.eyeAIVisionIP.toString()}) has failed: ${e.message.toString()}")
								errorMessage.setPositiveButton("Open settings") { dialog, which ->
									tcpErrorShowing = false
									startActivity(Intent(this, SettingsActivity::class.java))
									dialog.dismiss()
									overridePendingTransition(
										android.R.anim.fade_in,
										android.R.anim.fade_out
									)
								}

								errorMessage.setNegativeButton("Ignore") { dialog, which ->
									tcpErrorShowing = false
									dialog.dismiss()
								}
								errorMessage.show()
							}
						}
					},

					onMjpegError = { e->
						runOnUiThread {
							if (!mjpegErrorShowing && !mjpegErrorIgnored) {
								mjpegErrorShowing = true
								val errorMessage = AlertDialog.Builder(this)
								errorMessage.setMessage("Error while getting camera frame from EyeAIVision (IP: ${eyeAIApp().settings.eyeAIVisionIP.toString()}): ${e.message.toString()}")
								errorMessage.setPositiveButton("Open settings") { dialog, which ->
									mjpegErrorShowing = false
									dialog.dismiss()
									startActivity(Intent(this, SettingsActivity::class.java))
									overridePendingTransition(
										android.R.anim.fade_in,
										android.R.anim.fade_out
									)
								}

								errorMessage.setNegativeButton("Ignore") { dialog, which ->
									dialog.dismiss()
									mjpegErrorIgnored = true
									mjpegErrorShowing = false
								}
								errorMessage.show()
							}
						}
					}
				)

				mediaPlayer?.shutdown()
				mediaPlayer = MediaPlayer(
					context = this,
					uri = null,
					targetImageView = mediaImageView!!,
					bitmapFlow = bitmapFlow
				)

				mediaFrameAnalyzer?.shutdown()
				mediaFrameAnalyzer = CameraFrameAnalyzer(
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

	/*All TTS methods start here*/

	private fun onFinalSpeechRecognitionResult(final: String) {
		if (final.isEmpty()) {
			return
		}

		val receiveTs = System.nanoTime()
		Log.d(
			EyeAIApp.APP_LOG_TAG,
			"SR final RECEIVED at ${System.currentTimeMillis()} (ms), text='${final.take(200)}'"
		)


		CoroutineScope(Dispatchers.Main).launch {
			speechRecognitionFinalResultText?.text = final

			// minimum of 1 second pause between speech commands
			if (System.currentTimeMillis() - lastFinalResultMillis <= 1000)
				return@launch

			lastFinalResultMillis = System.currentTimeMillis()

			if (eyeAIApp().llm == null) {
				if (eyeAIApp().settings.enableSpeechRecognition) {
					llmResponseText?.text = getString(R.string.setup_llm_notice)
				}
			} else {
				llmResponseText?.text = getString(R.string.llm_responding_notice)

				//start after onTTSFinished speaking
				//Logging when Vosk is stopped.
				Log.d(EyeAIApp.APP_LOG_TAG, "Stopping Vosk to process command.")
				eyeAIApp().voskModel.stopListening()

				// vibrate for 100ms
				vibrate(eyeAIApp(), 100)

				Log.d(
					EyeAIApp.APP_LOG_TAG,
					"Dispatching to LLM worker at ${System.currentTimeMillis()} (ms); latency since SR receive = ${
						elapsedMs(receiveTs)
					} ms"
				)

				withContext(llmThreadExecutor.asCoroutineDispatcher()) {
					val workerStart = System.nanoTime()
					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"LLM worker START processing at ${System.currentTimeMillis()} (ms)"
					)
					onSpeechResult(final)
					Log.d(
						EyeAIApp.APP_LOG_TAG,
						"LLM worker FINISHED processing at ${System.currentTimeMillis()} (ms); duration=${
							elapsedMs(workerStart)
						} ms"
					)
				}

			}
		}
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun onSpeechRecognitionLoaded() {
		updateVoskStatusText()
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun updateVoskStatusText() {
		speechRecognitionFinalResultText?.text = when {
			!permissionManager.isMicrophonePermissionGranted() ->
				"Mikrofon-Berechtigung erforderlich"
			!eyeAIApp().settings.enableSpeechRecognition ->
				"Spracherkennung deaktiviert"
			voskUserStart.get() ->
				getString(R.string.speech_recognition_ready)
			else ->
				"Vosk bereit - Button klicken zum Starten"
		}
	}



	private suspend fun onSpeechResult(final: String) {
		Log.d(EyeAIApp.APP_LOG_TAG, "onSpeechResult: Creating new StateMachine for input: '$final'")

		val stateMachine = StateMachine(
			eyeAIApp(),
			textToSpeechInstance,
			lastLlmJsonResponse,
			llmResponseText,
			cameraManager.cameraFrameAnalyzer ?: mediaFrameAnalyzer
		) {

			CoroutineScope(Dispatchers.Main).launch {
				Log.d(
					EyeAIApp.APP_LOG_TAG,
					"onStreamingComplete CALLBACK: Fired, but logic is now handled by the global callback."
				)
			}
		}


		currentStateMachine = stateMachine

		val update = when (currentState) {
			State.IDLE -> stateMachine.handleIdle(final)
			State.SETTINGS_MENU -> stateMachine.handleSettingsMenu(final)
			State.SETTINGS_CHOICE -> stateMachine.handleSettingsChoice(final)
			State.SETTINGS_ACTION -> stateMachine.handleSettingsAction(final)
		}

		//Logging the state transition.
		Log.d(EyeAIApp.APP_LOG_TAG, "State transition: $currentState -> ${update.newState}")
		currentState = update.newState
		lastLlmJsonResponse = update.newJson
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun startVoskListening() {
		if (voskUserStart.get()) return // Check whether already started

		setObjectAudioPaused(true)
		setDepthAudioPaused(true)
		voskUserStart.set(true)
		eyeAIApp().voskModel.startListening()
		Log.d(EyeAIApp.APP_LOG_TAG, "User started Vosk Model")
		updateVoskStatusText()
	}

	@RequiresApi(Build.VERSION_CODES.P)
	private fun stopVoskListening() {
		if (!voskUserStart.get()) return // Check whether already stopped

		setObjectAudioPaused(false)
		setDepthAudioPaused(false)
		voskUserStart.set(false)
		eyeAIApp().voskModel.stopListening()
		Log.d(EyeAIApp.APP_LOG_TAG, "User stopped Vosk Model")
		updateVoskStatusText()
	}


	fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

}