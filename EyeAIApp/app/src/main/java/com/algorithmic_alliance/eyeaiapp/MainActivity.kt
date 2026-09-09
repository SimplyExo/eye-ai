package com.algorithmic_alliance.eyeaiapp

import android.os.Build
import android.os.Bundle
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
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntimeService

/**
 * UI-only entry point. The Activity observes and commands the runtime but does
 * not own CameraX, models, audio engines, speech recognition, or TTS.
 */
class MainActivity : AppCompatActivity() {
    private val viewModel: MainViewModel by viewModels()

    @RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
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
                    onEvent = viewModel::onEvent,
                    viewModel = viewModel,
                )
            }
        }
    }

    override fun onResume() {
        super.onResume()
        viewModel.onResume()
    }

    override fun onDestroy() {
        // Only the optional surface is detached. The service-owned analysis,
        // audio and models intentionally outlive Activity recreation.
        (application as? EyeAIApp)?.runtime?.detachPreview()
        if (isFinishing && !isChangingConfigurations) {
            // Fallback for devices that finish the task without reliably
            // delivering Service.onTaskRemoved before destroying the Activity.
            EyeAIRuntimeService.stop(this)
        }
        super.onDestroy()
    }
}
