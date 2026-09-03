package com.algorithmic_alliance.eyeaiapp

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.algorithmic_alliance.eyeaiapp.UI.EyeAIAppUI
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.theme.EyeAITheme
import com.algorithmic_alliance.eyeaiapp.runtime.EyeAIRuntimeService

/**
 * UI-only entry point. The Activity observes and commands the runtime but does
 * not own CameraX, models, audio engines, speech recognition, or TTS.
 */
class MainActivity : AppCompatActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EyeAITheme {
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
