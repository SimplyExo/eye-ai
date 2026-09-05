package com.algorithmic_alliance.eyeaiapp

import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.camera.view.PreviewView
import androidx.core.os.LocaleListCompat
import com.algorithmic_alliance.eyeaiapp.UI.EyeAIAppUI
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.camera.CameraFrameAnalyzer
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.media.MediaPlayer
import com.algorithmic_alliance.eyeaiapp.audio.SpatialAudio
import com.algorithmic_alliance.eyeaiapp.connectivity.EyeAIVision
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import com.algorithmic_alliance.eyeaiapp.tts.TextToSpeechInstance
import com.example.compose.AppTheme
import kotlinx.coroutines.flow.MutableSharedFlow


class MainActivity : AppCompatActivity() {
    var cameraManager = CameraManager()

    @RequiresApi(Build.VERSION_CODES.P)
    private var permissionManager = PermissionManager(
        this,
        //::onCameraPermissionResult,
        //::onMicrophonePermissionResult
    )

    private var cameraPreviewView: PreviewView? = null
    private var ungrantedPermissionsNotice: LinearLayout? = null
    private var ungrantedPermissionsNoticeText: TextView? = null
    private var allowCameraPermission: Button? = null
    private var startStopVosk: FloatingActionButton? = null

    private lateinit var eyeAIVision: EyeAIVision
    private var bitmapFlow: MutableSharedFlow<Bitmap>? = null

    private val voskStarting = AtomicBoolean(false)

    private val voskManualRestartRequired = AtomicBoolean(false)

    @Volatile
    private var resumeSpatialAudioAfterTtsJob: Job? = null


    private var depthPreviewImage: ImageView? = null

    private var debugInputBitmapPreview: ImageView? = null
    private var mediaImageView: ImageView? = null

    private var performanceText: TextView? = null

    private var overlayObjectDetection: OverlayViewOD? = null
    private var overlayOcr: OverlayViewOCR? = null

    private var speechRecognitionPartialResultText: TextView? = null
    private var speechRecognitionFinalResultText: TextView? = null
    private var speechResponseText: TextView? = null
    private var lastFinalResultMillis = System.currentTimeMillis()
    private val speechThreadExecutor = Executors.newSingleThreadExecutor()

    private lateinit var textToSpeechInstance: TextToSpeechInstance

    private var lastDialogContext: String? = null

    private var tcpErrorShowing = false
    private var mjpegErrorShowing = false

    private var mjpegErrorIgnored = false


    enum class State {
        IDLE, SETTINGS_MENU, SETTINGS_CHOICE, SETTINGS_ACTION, SETTINGS_EXTERNAL_CONFIRMATION,
    }

    private var currentState: State = State.IDLE

    private var mediaFrameAnalyzer: CameraFrameAnalyzer? = null
    private var mediaPlayer: MediaPlayer? = null

    private val viewModel: MainViewModel by viewModels()

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        enableEdgeToEdge()
        super.onCreate(savedInstanceState)

        if (AppCompatDelegate.getApplicationLocales().isEmpty) {
            AppCompatDelegate.setApplicationLocales(
                LocaleListCompat.forLanguageTags("de")
            )
        }

        window.isNavigationBarContrastEnforced = false
        setContent {
            AppTheme() {
                EyeAIAppUI(
                    onEvent = viewModel::onEvent, viewModel = viewModel
                )
            }
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
            viewModel.setTTSSpeaking(false)
            viewModel.updateVoskStatusText()
            // Restart Vosk after the local TTS response has finished.
            CoroutineScope(Dispatchers.Main).launch {
                try {
                    //Announcing that the global callback has been fired.
                    Log.d(
                        EyeAIApp.APP_LOG_TAG, "[DecisionTrace][Vosk][TTS_FINISHED] callback=fired"
                    )

                    if (!eyeAIApp().voskUserStart.get()) {
                        val skipReason = if (voskManualRestartRequired.get()) {
                            "SETTINGS_APPLIED_BUTTON_PRESS_REQUIRED"
                        } else {
                            "LISTENING_NOT_ARMED_BY_USER"
                        }
                        Log.d(
                            EyeAIApp.APP_LOG_TAG,
                            "[DecisionTrace][Vosk][AUTO_RESTART] outcome=SKIPPED " + "reason=$skipReason"
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
                            if (eyeAIApp().voskUserStart.get()) {
                                eyeAIApp().voskModel.startListening()
                                Log.d(
                                    EyeAIApp.APP_LOG_TAG,
                                    "[DecisionTrace][Vosk][AUTO_RESTART] outcome=STARTED"
                                )
                            } else {
                                val skipReason = if (voskManualRestartRequired.get()) {
                                    "SETTINGS_APPLIED_BUTTON_PRESS_REQUIRED"
                                } else {
                                    "LISTENING_NOT_ARMED_BY_USER"
                                }
                                Log.d(
                                    EyeAIApp.APP_LOG_TAG,
                                    "[DecisionTrace][Vosk][AUTO_RESTART] outcome=SKIPPED " + "reason=$skipReason"
                                )
                            }
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
                    if (eyeAIApp().voskUserStart.get() && voskStarting.compareAndSet(false, true)) {
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

    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onResume() {
        super.onResume()
        eyeAIApp().updateSettings()

        /*
        permissionManager.requestCameraPermission()
        if (eyeAIApp().settings.enableSpeechRecognition)
            permissionManager.requestMicrophonePermission()
        updateUngrantedPermissionsNotice()
         */
        viewModel.onResume()

    }

    @RequiresApi(Build.VERSION_CODES.P)
    override fun onPause() {
        super.onPause()

        viewModel.onPause()
        SpatialAudio.stop()

        eyeAIApp().voskModel.stopListening()
        eyeAIApp().textToSpeechInstance.stop()

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
     *//*
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
    }/*
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
                                startVoskListening(trigger = "EYEAIVISION_BUTTON")
                        }


                    },
                    onDoubleClick = {
                        Log.i("CLICK", "DOUBLE")

                        State.IDLE

                        if(voskUserStart.get()) {
                            textToSpeechInstance.stop()

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

    */


    /*All TTS methods start here*/


    /*All TTS methods start here*//*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun onFinalSpeechRecognitionResult(final: String) {
        if (final.isEmpty()) {
            return
        }

        val receiveTs = System.nanoTime()
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Vosk][RECOGNIZED] originalText='${final.take(200)}' " +
                "next=STATE_MACHINE currentState=$currentState"
        )


        CoroutineScope(Dispatchers.Main).launch {
            speechRecognitionFinalResultText?.text = final

            // minimum of 1 second pause between speech commands
            if (System.currentTimeMillis() - lastFinalResultMillis <= 1000)
                return@launch

            lastFinalResultMillis = System.currentTimeMillis()

            // Pause listening while the local state machine evaluates and speaks.
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Vosk][PAUSE_FOR_PROCESSING] autoRestartAfterTts=true"
            )
            eyeAIApp().voskModel.stopListening()

            // vibrate for 100ms
            vibrate(eyeAIApp(), 100)

            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][DISPATCH] state=$currentState; latencySinceVosk=${
                    elapsedMs(receiveTs)
                }ms"
            )

            withContext(speechThreadExecutor.asCoroutineDispatcher()) {
                val workerStart = System.nanoTime()
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][WORKER] phase=START state=$currentState"
                )
                onSpeechResult(final)
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][StateMachine][WORKER] phase=FINISH duration=${
                        elapsedMs(workerStart)
                    }ms"
                )
            }
        }
    }

     *//*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun onSpeechRecognitionLoaded() {
        updateVoskStatusText()
    }

     *//*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun updateVoskStatusText() {
        runOnUiThread {
            speechRecognitionFinalResultText?.text = when {
                !permissionManager.isMicrophonePermissionGranted() ->
                    "Mikrofon-Berechtigung erforderlich"

                !eyeAIApp().settings.enableSpeechRecognition ->
                    "Spracherkennung deaktiviert"

                voskUserStart.get() ->
                    getString(R.string.speech_recognition_ready)

                voskManualRestartRequired.get() ->
                    "Einstellung geändert – Button klicken zum erneuten Zuhören"

                else ->
                    "Vosk bereit - Button klicken zum Starten"
            }
        }
    }

     */


    /*
    @RequiresApi(Build.VERSION_CODES.P)
    private suspend fun onSpeechResult(final: String) {
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][INPUT] state=$currentState originalText='$final'"
        )

        val stateMachine = StateMachine(
            eyeAIApp(),
            textToSpeechInstance,
            lastDialogContext,
            speechResponseText,
            cameraManager.cameraFrameAnalyzer ?: mediaFrameAnalyzer
        )

        val cancellationResponse = GenericCancellation.responseFor(final)
        val update = if (cancellationResponse != null) {
            Log.d(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][StateMachine][CANCEL] input matched generic cancellation before state dispatch"
            )
            stateMachine.handleCancellation()
        } else {
            when (currentState) {
                State.IDLE -> stateMachine.handleIdle(final)
                State.SETTINGS_MENU -> stateMachine.handleSettingsMenu(final)
                State.SETTINGS_CHOICE -> stateMachine.handleSettingsChoice(final)
                State.SETTINGS_ACTION -> stateMachine.handleSettingsAction(final)
                State.SETTINGS_EXTERNAL_CONFIRMATION ->
                    stateMachine.handleSettingsExternalConfirmation(final)
            }
        }

        if (update.voskRestartPolicy == VoskRestartPolicy.REQUIRE_MANUAL_RESTART) {
            Log.i(
                EyeAIApp.APP_LOG_TAG,
                "[DecisionTrace][Vosk][POLICY] source=SETTINGS_APPLIED " +
                    "policy=REQUIRE_MANUAL_RESTART"
            )
            stopVoskListening(
                trigger = "SETTINGS_APPLIED",
                requireManualRestart = true,
                spatialAudioResume = SpatialAudioResume.AFTER_TTS
            )
        }

        // Logging der state transition
        Log.d(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][StateMachine][TRANSITION] $currentState -> ${update.newState}; " +
                "voskRestartPolicy=${update.voskRestartPolicy}"
        )
        currentState = update.newState
        lastDialogContext = update.newJson
    }

     *//*
    @RequiresApi(Build.VERSION_CODES.P)
    private fun startVoskListening(trigger: String = "USER_BUTTON") {
        if (voskUserStart.get()) return // Check whether already started

        cancelPendingSpatialAudioResume()
        uniffi.NativeLib.setObjectAudioPaused(true)
        uniffi.NativeLib.setDepthAudioPaused(true)
        voskManualRestartRequired.set(false)
        voskUserStart.set(true)
        eyeAIApp().voskModel.startListening()
        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Vosk][START] trigger=$trigger outcome=LISTENING"
        )
        updateVoskStatusText()
    }

    @RequiresApi(Build.VERSION_CODES.P)
    private fun stopVoskListening(
        trigger: String = "USER_BUTTON",
        requireManualRestart: Boolean = false,
        spatialAudioResume: SpatialAudioResume = SpatialAudioResume.IMMEDIATE
    ) {
        voskManualRestartRequired.set(requireManualRestart)
        val wasArmedByUser = voskUserStart.getAndSet(false)

        if (wasArmedByUser) {
            eyeAIApp().voskModel.stopListening()
        }

        when (spatialAudioResume) {
            SpatialAudioResume.IMMEDIATE -> {
                cancelPendingSpatialAudioResume()
                restoreSpatialAudioFromSettings(trigger)
            }

            SpatialAudioResume.AFTER_TTS -> scheduleSpatialAudioResumeAfterTts(trigger)
        }

        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][Vosk][STOP] trigger=$trigger outcome=STOPPED " +
                "autoRestartArmed=false spatialAudioResume=$spatialAudioResume"
        )
        updateVoskStatusText()
    }


     *//*
    private fun scheduleSpatialAudioResumeAfterTts(trigger: String) {
        cancelPendingSpatialAudioResume()

        // Keep both spatial output channels muted until Android TTS has really
        // finished the acknowledgement sentence (including its quiet window).
        uniffi.NativeLib.setObjectAudioPaused(true)
        uniffi.NativeLib.setDepthAudioPaused(true)

        val job = lifecycleScope.launch {
            val silent = textToSpeechInstance.awaitSilence(
                quietMs = 500L,
                maxWaitMs = 30_000L
            )

            if (!silent) {
                Log.w(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED " +
                        "reason=TTS_SILENCE_TIMEOUT"
                )
                return@launch
            }

            if (voskUserStart.get() || !voskManualRestartRequired.get()) {
                Log.d(
                    EyeAIApp.APP_LOG_TAG,
                    "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=SKIPPED " +
                        "reason=LISTENING_STATE_CHANGED"
                )
                return@launch
            }

            restoreSpatialAudioFromSettings(trigger)
        }

        resumeSpatialAudioAfterTtsJob = job
        job.invokeOnCompletion {
            if (resumeSpatialAudioAfterTtsJob === job) {
                resumeSpatialAudioAfterTtsJob = null
            }
        }
    }


     *//*
    private fun restoreSpatialAudioFromSettings(trigger: String) {
        val settings = Settings.load(this)
        uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
        uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
        Log.i(
            EyeAIApp.APP_LOG_TAG,
            "[DecisionTrace][SpatialAudio][RESUME] trigger=$trigger outcome=RESTORED " +
                "objectAudioEnabled=${settings.objectAudioPlayback} " +
                "depthAudioEnabled=${settings.depthAudioPlayback}"
        )
    }

    private fun cancelPendingSpatialAudioResume() {
        resumeSpatialAudioAfterTtsJob?.cancel()
        resumeSpatialAudioAfterTtsJob = null
    }


     */

    fun elapsedMs(startNano: Long): Long = (System.nanoTime() - startNano) / 1_000_000

    private enum class SpatialAudioResume {
        IMMEDIATE, AFTER_TTS
    }

    companion object


}
