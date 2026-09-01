package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview

import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.ShimmerBox
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.UIState
import com.algorithmic_alliance.eyeaiapp.UI.rememberShimmerBrush
import com.algorithmic_alliance.eyeaiapp.data.Spacing

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomePage(
    modifier: Modifier = Modifier,
    onOpenSettings: () -> Unit,
    onEvent: (UIEvent) -> Unit,
    viewModel: MainViewModel,
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current
    Log.d(LOG_TAG, "[HomePage] Loading HomePage")
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val speechRecognitionKey = stringResource(R.string.enable_speech_recognition_setting)
    val speechRecognitionEnabled by rememberSaveable {
        mutableStateOf(
            sharedPreferences.getBoolean(
                speechRecognitionKey, true
            )
        )
    }
    val shimmerBrush = rememberShimmerBrush(backgroundColor = MaterialTheme.colorScheme.surface, contrastColor = MaterialTheme.colorScheme.onSurface)
    LaunchedEffect(Unit) {
        if (ActivityCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            if (speechRecognitionEnabled) {
                Log.d(LOG_TAG, "[HomePage] Loading Vosk model")
                onEvent(UIEvent.InitVoskService)
            } else {
                Log.d(LOG_TAG, "[HomePage] Speech Recognition disabled not loading Vosk model")
                onEvent(UIEvent.CloseVoskService)
            }
        }
        onEvent(UIEvent.UIinitCamera(null, lifecycleOwner))
        onEvent(UIEvent.UpdateVoskStatusText)
        onEvent(UIEvent.UpdateLlmStatusText)
    }


    Scaffold(modifier = Modifier.fillMaxSize(), contentWindowInsets = WindowInsets.safeDrawing, topBar = {
        TopAppBar(
            title = {
                Text("EyeAI App")
            },
            modifier = Modifier.shadow(elevation = Spacing.sm),
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
            ),
        )
    }, floatingActionButton = {
        Row(
            modifier = Modifier
                .padding(Spacing.sm)
                .fillMaxWidth(0.35f),
            horizontalArrangement = if (speechRecognitionEnabled) Arrangement.SpaceBetween else Arrangement.End
        ) {
            if (speechRecognitionEnabled) FloatingActionButton(
                onClick = {
                    onEvent(UIEvent.VoskListeningChanged)
                },
            ) {
                Icon(
                    painter = if (uiState.voskListening) {
                        painterResource(R.drawable.stop_24px)
                    } else if (uiState.llmSpeaking) {
                        painterResource(
                            R.drawable.pause_playback_24px
                        )
                    } else {
                        painterResource(R.drawable.play_arrow_24px)
                    }, contentDescription = "Start Vosk"
                )
            }
            FloatingActionButton(onClick = { onOpenSettings() }) {
                Icon(
                    painter = painterResource(R.drawable.settings_24px),
                    contentDescription = "Open Settings"
                )
            }
        }
    }, content = { paddingValues ->
        LazyVerticalGrid(modifier = Modifier.padding(paddingValues), columns = GridCells.Fixed(2)) {
            item {
                VoskStatusCard(viewModel = viewModel)
            }
            item {
                DepthStatusCard(viewModel = viewModel, shimmerBrush =shimmerBrush)
            }
            item { ObjectStatusCard(viewModel = viewModel, shimmerBrush = shimmerBrush) }
            item { VisionStatusCard(viewModel = viewModel) }
        }
    })

}

@Composable
fun ObjectStatusCard(viewModel: MainViewModel, shimmerBrush: Brush) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Objekterkennung", fontSize = 18.sp, textAlign = TextAlign.Center)
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (!sharedPreferences.getBoolean(
                        stringResource(R.string.enable_object_detection_setting), true
                    )
                ) {
                    Text(text = "Objekterkennung deaktiviert", textAlign = TextAlign.Center)
                } else if (getObjectFPS(uiState.performanceText) == -1) {
                    ShimmerBox(shimmerBrush, Modifier.fillMaxWidth(0.75f).height(Spacing.xl))
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(text = "Aktiv", textAlign = TextAlign.Center)
                        Text(
                            text = "Leistung: ${getPerformance(getObjectFPS(uiState.performanceText))}",
                            textAlign = TextAlign.Center
                        )
                    }

                }
            }
        }
    }
}

@Composable
fun VisionStatusCard(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm), horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("EyeAI-Vision", fontSize = 18.sp, textAlign = TextAlign.Center)
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                //TODO == statt != für echte App
                if (sharedPreferences.getString(
                        stringResource(R.string.input_source_setting),
                        stringResource(R.string.input_is_camera)
                    ) != stringResource(R.string.input_is_camera)
                ) {
                    Text(
                        "Keine EyeAI-Vision verbunden. Handykamera wird benutzt.",
                        textAlign = TextAlign.Center
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text("Name: EyeAI-Vision von Robert", textAlign = TextAlign.Center)
                        Text("Verbindung: Gut ", textAlign = TextAlign.Center)
                        Text("Akku: 49% ", textAlign = TextAlign.Center)
                    }
                }
            }
        }
    }
}

@Composable
fun DepthStatusCard(viewModel: MainViewModel, shimmerBrush: Brush) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm), horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Distanzmessung", fontSize = 18.sp, textAlign = TextAlign.Center)
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (getDepthFPS(uiState.performanceText) == -1) {
                    ShimmerBox(shimmerBrush, Modifier.fillMaxWidth(0.75f).height(Spacing.xl))
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(text = "Aktiv", textAlign = TextAlign.Center)
                        Text(
                            text = "Leistung: ${getPerformance(getDepthFPS(uiState.performanceText))}",
                            textAlign = TextAlign.Center
                        )
                    }

                }
            }
        }
    }
}

@Composable
fun VoskStatusCard(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm), horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Spracherkennung", fontSize = 18.sp, textAlign = TextAlign.Center)
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (uiState.llmResponseText.isEmpty()) Text(
                    modifier = Modifier.padding(Spacing.sm),
                    text = uiState.speechRecognitionFinalResultText,
                    textAlign = TextAlign.Center
                )
                if (uiState.llmResponseText.isNotEmpty()) if (uiState.llmResponseText == stringResource(
                        R.string.llm_responding_notice
                    )
                ) {
                    Text("EyeAI denkt nach...", textAlign = TextAlign.Center)
                } else if (uiState.llmResponseText != "") Text(
                    "EyeAI antwortet...", textAlign = TextAlign.Center
                )
            }
        }

    }
}

private fun getPerformance(fps: Int): String {
    return when (fps) {
        in 0..5 -> "Schlecht"
        in 5..10 -> "Ausreichend"
        in 10..20 -> "Gut"
        in 20..100 -> "Sehr Gut"
        in 100..1000 -> "Rekordverdächtig"
        in 1000..10000 -> "Rechenzentrum"
        in 10000..100000 -> "Quantencomputer"
        in 100000..1000000 -> "Außerirdisch"
        else -> "Berechne..."
    }
}

private fun getDepthFPS(text: String): Int {
    val index = text.indexOf("Depth Frame: ")
    if (index != -1 && index + 15 <= text.length) {
        var result = text.substring(index + 13, index + 15)
        if (result.endsWith(".")) result = result.dropLast(1)
        return try {
            result.toInt()
        } catch (e: NumberFormatException) {
            -1
        }
    }
    return -1
}

private fun getObjectFPS(text: String): Int {
    val index = text.indexOf("Object Frame: ")
    if (index != -1 && index + 16 <= text.length) {
        var result = text.substring(index + 14, index + 16)
        if (result.endsWith(".")) result = result.dropLast(1)
        return try {
            result.toInt()
        } catch (e: NumberFormatException) {
            -1
        }
    }
    return -1
}
