package com.algorithmic_alliance.eyeaiapp

import android.content.Intent
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.Looper
import android.util.Log
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.ProgressBar
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import com.algorithmic_alliance.eyeaiapp.UI.EyeAIAppUI
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.llm.statemachine.StateMachine
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio.SpeechManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as UI_LOG_TAG
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import kotlinx.coroutines.flow.MutableSharedFlow


class MainActivity : AppCompatActivity() {
    //var cameraManager = CameraManager()

    @RequiresApi(Build.VERSION_CODES.P)
    private var permissionManager =
        PermissionManager(
            this,
            //::onCameraPermissionResult,
            //::onMicrophonePermissionResult
        )

    /*
    private var cameraPreviewView: PreviewView? = null
    private var ungrantedPermissionsNotice: LinearLayout? = null
    private var ungrantedPermissionsNoticeText: TextView? = null
    private var allowCameraPermission: Button? = null
    private var startStopVosk: FloatingActionButton? = null
     */
    //private lateinit var eyeAIVision: EyeAIVision
    //private var bitmapFlow: MutableSharedFlow<Bitmap>? = null

    private val voskStarting = AtomicBoolean(false)

    private val voskUserStart = AtomicBoolean(false)

    /*
    private var depthPreviewImage: ImageView? = null

    private var debugInputBitmapPreview: ImageView? = null
    private var mediaImageView: ImageView? = null

    private var performanceText: TextView? = null

    private var overlayObjectDetection: OverlayViewOD? = null
    private var overlayOcr: OverlayViewOCR? = null

    private var speechRecognitionPartialResultText: TextView? = null
    private var speechRecognitionFinalResultText: TextView? = null
    private var llmResponseText: TextView? = null
     */
    private var lastFinalResultMillis = System.currentTimeMillis()
    private var llmThreadExecutor = Executors.newSingleThreadExecutor()

    //private lateinit var textToSpeechInstance: TextToSpeechInstance

    //private var lastLlmJsonResponse: String? = null

    //private var currentStateMachine: StateMachine? = null

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

    //private var mediaFrameAnalyzer: CameraFrameAnalyzer? = null
    //private var mediaPlayer: MediaPlayer? = null

    private val viewModel: MainViewModel by viewModels()

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            EyeAIAppUI(
                onEvent = viewModel::onEvent,
                uiState = uiState,
                cameraManager = eyeAIApp().cameraManager
            )
        }

        /*

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

        startStopVosk = findViewById(R.id.stop_tts_button)
        startStopVosk!!.setOnClickListener {

            State.IDLE

            if (voskUserStart.get()){

                SpeechManager.forceStop()


                android.os.Handler(Looper.getMainLooper()).postDelayed({
                    stopVoskListening()
                }, 100)

            }
            else{
                startVoskListening()
            }

        }

        done speechRecognitionPartialResultText = findViewById(R.id.speech_recognition_partial_output)
        done speechRecognitionFinalResultText = findViewById(R.id.speech_recognition_final_output)
        done llmResponseText = findViewById(R.id.llm_response)

        findViewById<FloatingActionButton>(R.id.settings_button).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
        }



        updateUngrantedPermissionsNotice()

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        */
        eyeAIApp().updateSettings()

        //updateSpeechRecognitionUIVisibility()

        eyeAIApp().textToSpeechInstance = TextToSpeechInstance(this) {
            // starting vosk if no stream is active
            CoroutineScope(Dispatchers.Main).launch {
                try {
                    //Announcing that the global callback has been fired.
                    Log.d(EyeAIApp.APP_LOG_TAG, "GLOBAL TTS CALLBACK: Fired.")

                    if (!voskUserStart.get()) {
                        Log.d(EyeAIApp.APP_LOG_TAG, "User hasn't started Vosk yet - skipping")
                        return@launch
                    }

                    //checking whether a stream is ongoing

                    val isStreaming = eyeAIApp().currentStateMachine?.isStreaming() ?: false
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

        SpeechManager.tts = eyeAIApp().textToSpeechInstance


    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onResume() {
        super.onResume()

        eyeAIApp().updateSettings()

        CoroutineScope(Dispatchers.IO).launch {
            Log.d("Spatial Audio", "[SpatialAudio] Starting spatial audio")
            SpatialAudio.setup(this@MainActivity)
            SpatialAudio.start()
        }
        updateSpeechRecognitionUIVisibility()
        /*
        permissionManager.requestCameraPermission()
        if (eyeAIApp().settings.enableSpeechRecognition)
            permissionManager.requestMicrophonePermission()
        updateUngrantedPermissionsNotice()
         */
        viewModel.onResume()
        val isLLMConfigured = eyeAIApp().settings.googleAiStudioApiKey?.isEmpty() == false
        viewModel.updateLlmResponseText(
            if (isLLMConfigured)
                ""
            else
                getString(R.string.setup_llm_notice)
        )

        // re-enabling the audio playback in accordance to the settings
        val settings = Settings.load(this@MainActivity)
        uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
        uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onPause() {
        super.onPause()

        SpatialAudio.stop()

        eyeAIApp().voskModel.stopListening()

        eyeAIApp().cameraManager.pauseAnalyzer()
        eyeAIApp().mediaFrameAnalyzer?.shutdown()
        eyeAIApp().mediaPlayer?.shutdown()

        // stopping audio playback
        uniffi.NativeLib.setObjectAudioPaused(true)
        uniffi.NativeLib.setDepthAudioPaused(true)
    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onDestroy() {
        super.onDestroy()
        eyeAIApp().cameraManager.shutdown()
        eyeAIApp().textToSpeechInstance.shutdown()
        eyeAIApp().mediaFrameAnalyzer?.shutdown()
        eyeAIApp().mediaPlayer?.shutdown()
        SpatialAudio.stop()

        eyeAIApp().voskModel.closeService()
    }

    /*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun onCameraPermissionResult(isGranted: Boolean) {
        if (isGranted) {
            //ungrantedPermissionsNotice!!.visibility = GONE
            initCamera()
        } else {
            //ungrantedPermissionsNotice!!.visibility = VISIBLE
        }
    }
     */
    /*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun onMicrophonePermissionResult(isGranted: Boolean) {
        if (isGranted && eyeAIApp().settings.enableSpeechRecognition) {
            if (!eyeAIApp().voskModel.isListening()) {
                eyeAIApp()
                    .voskModel
                    .initService(
                        ::onPartialSpeechRecognitionResult,
                        ::onFinalSpeechRecognitionResult,
                        ::onSpeechRecognitionLoaded
                    )
            }
        } else {
            Log.w(EyeAIApp.APP_LOG_TAG, "Microphone Permission not granted!")
        }
    }


     */

    private fun eyeAIApp(): EyeAIApp {
        return application as EyeAIApp
    }
    /*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun initCamera() {
        if (eyeAIApp().settings.inputSource == getString(R.string.input_is_camera)) {
            //mediaImageView!!.isVisible = false
            if (permissionManager.isCameraPermissionGranted()) {
                //ungrantedPermissionsNotice!!.visibility = GONE

                eyeAIApp().cameraManager.cameraFrameAnalyzer?.shutdown()
                eyeAIApp().cameraManager.cameraFrameAnalyzer =
                    CameraFrameAnalyzer(
                        eyeAIApp(),
                        //depthPreviewImage!!,
                        null,
                        //performanceText!!,
                        null,
                        //overlayObjectDetection!!,
                        null,
                        //debugInputBitmapPreview!!,
                        null,
                        //mediaImageView!!,
                        null
                    )
                eyeAIApp().cameraManager.cameraFrameAnalyzer?.start()

                eyeAIApp().cameraManager
                    .init(
                        this,
                        EyeAIApp.PREFERRED_CAMERA_RESOLUTION,
                        //cameraPreviewView,
                        null
                    )
            } else {
                //ungrantedPermissionsNotice!!.visibility = VISIBLE
            }
        } else if (eyeAIApp().settings.inputSource == getString(R.string.input_is_media)) {
            if (eyeAIApp().settings.mediaSource!!.isNotEmpty()) {
                //mediaImageView!!.isVisible = true
                ProcessCameraProvider.getInstance(this).get().unbindAll()
                //overlayObjectDetection!!.reset()
                //overlayOcr!!.reset()
                //depthPreviewImage!!.setImageBitmap(createBitmap(256, 256))

                eyeAIApp().mediaPlayer?.shutdown()
                //mediaPlayer = MediaPlayer(this, eyeAIApp().settings.mediaSource!!.toUri(), mediaImageView!!)

                eyeAIApp().mediaFrameAnalyzer?.shutdown()

                eyeAIApp().mediaFrameAnalyzer =
                    CameraFrameAnalyzer(
                        eyeAIApp(),
                        //depthPreviewImage!!,
                        null,
                        //performanceText!!,
                        null,
                        //overlayObjectDetection!!,
                        null,
                        //debugInputBitmapPreview !!,
                        null,
                        //mediaImageView!!
                        null
                    )

                eyeAIApp().mediaFrameAnalyzer?.start()
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

                val connectingTCPDialog = AlertDialog.Builder(this)
                connectingTCPDialog.setMessage("Connecting to Button Server...")
                connectingTCPDialog.setView(ProgressBar(this))

                var shownConnectDialog: AlertDialog? = null

                eyeAIVision = EyeAIVision(
                    ip = eyeAIApp().settings.eyeAIVisionIP.toString(),
                    eyeAIApp().settings.jpegCompression,
                    lifecycleScope = lifecycleScope,
                    bitmapFlow = bitmapFlow,
                    onSingleClick = {
                        Log.i("CLICK", "SINGLE")

                        State.IDLE

                        if (!voskUserStart.get()) {
                            //TODO reimplement
                            //startVoskListening()
                        }


                    },
                    onDoubleClick = {
                        Log.i("CLICK", "DOUBLE")

                        State.IDLE

                        if (voskUserStart.get()) {
                            SpeechManager.forceStop()
                            //TODO reimplement
                            //stopVoskListening()
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

                    onMjpegError = { e ->
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
                    },

                    onConnectingSocket = {
                        runOnUiThread {
                            shownConnectDialog = connectingTCPDialog.show()
                        }
                    },

                    onSocketConnectionEstablished = {
                        runOnUiThread {
                            shownConnectDialog?.dismiss()
                        }
                    },

                    onConnectingHTTP = {

                    },

                    onHTTPConnectionEstablished = {

                    }
                )

                eyeAIApp().mediaPlayer?.shutdown()
                /*
                mediaPlayer = MediaPlayer(
                    context = this,
                    uri = null,
                    targetImageView = mediaImageView!!,
                    bitmapFlow = bitmapFlow
                )

                 */

                eyeAIApp().mediaFrameAnalyzer?.shutdown()
                eyeAIApp().mediaFrameAnalyzer = CameraFrameAnalyzer(
                    eyeAIApp(),
                    //depthPreviewImage!!,
                    null,
                    //performanceText!!,
                    null,
                    //overlayObjectDetection!!,
                    null,
                    //debugInputBitmapPreview!!,
                    null,
                    //mediaImageView!!,
                    null
                )
                eyeAIApp().mediaFrameAnalyzer?.start()

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

     */


    private fun updateSpeechRecognitionUIVisibility() {
        val visibility = if (eyeAIApp().settings.enableSpeechRecognition) {
            VISIBLE
        } else {
            GONE
        }

        //speechRecognitionPartialResultText?.visibility = visibility
        //speechRecognitionFinalResultText?.visibility = visibility
    }


    /*All TTS methods start here*/


    /*All TTS methods start here*/


    companion object


}