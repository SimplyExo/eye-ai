package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.pm.PackageManager
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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
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
    uiState: UIState,
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val speechRecognitionKey = stringResource(R.string.enable_speech_recognition_setting)
    val profilingInformationKey = stringResource(R.string.show_profiling_info_setting)
    val speechRecognitionEnabled by rememberPreferenceBooleanState(speechRecognitionKey, true)
    val shimmerBrush = rememberShimmerBrush(
        backgroundColor = MaterialTheme.colorScheme.surface,
        contrastColor = MaterialTheme.colorScheme.onSurface,
    )

    // Status cards need profiling values, but this is strictly a UI preference:
    // the runtime and foreground service continue independently of this effect.
    DisposableEffect(sharedPreferences, profilingInformationKey) {
        val previousValue = sharedPreferences.getBoolean(profilingInformationKey, false)
        if (!previousValue) {
            sharedPreferences.edit(commit = true) { putBoolean(profilingInformationKey, true) }
            onEvent(UIEvent.UpdateSettings)
        }
        onDispose {
            if (!previousValue) {
                sharedPreferences.edit(commit = true) { putBoolean(profilingInformationKey, false) }
                onEvent(UIEvent.UpdateSettings)
            }
        }
    }

    LaunchedEffect(speechRecognitionEnabled) {
        if (
            ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        ) {
            if (speechRecognitionEnabled) onEvent(UIEvent.InitVoskService)
            else onEvent(UIEvent.CloseVoskService)
        }
        // A null surface starts/keeps the service-owned source headlessly.
        onEvent(UIEvent.UIinitCamera(null))
        onEvent(UIEvent.UpdateVoskStatusText)
        onEvent(UIEvent.UpdateSpeechStatusText)
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = { Text("EyeAI App", style = MaterialTheme.typography.titleLarge) },
                modifier = Modifier.shadow(elevation = Spacing.sm),
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                ),
            )
        },
        floatingActionButton = {
            Row(
                modifier = Modifier
                    .padding(Spacing.sm)
                    .fillMaxWidth(0.35f),
                horizontalArrangement = if (speechRecognitionEnabled) {
                    Arrangement.SpaceBetween
                } else {
                    Arrangement.End
                },
            ) {
                if (speechRecognitionEnabled) {
                    FloatingActionButton(onClick = { onEvent(UIEvent.VoskListeningChanged) }) {
                        Icon(
                            painter = when {
                                uiState.voskListening -> painterResource(R.drawable.stop_24px)
                                uiState.ttsSpeaking -> painterResource(R.drawable.pause_playback_24px)
                                else -> painterResource(R.drawable.play_arrow_24px)
                            },
                            contentDescription = "Start or stop speech recognition",
                        )
                    }
                }
                FloatingActionButton(onClick = onOpenSettings) {
                    Icon(
                        painter = painterResource(R.drawable.settings_24px),
                        contentDescription = "Open settings",
                    )
                }
            }
        },
    ) { paddingValues ->
        LazyVerticalGrid(
            modifier = Modifier.padding(paddingValues),
            columns = GridCells.Fixed(2),
        ) {
            item { VoskStatusCard(uiState) }
            item { DepthStatusCard(uiState, shimmerBrush) }
            item { ObjectStatusCard(uiState, shimmerBrush) }
            item { VisionStatusCard() }
        }
    }
}

@Composable
fun ObjectStatusCard(uiState: UIState, shimmerBrush: Brush) {
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "Objekterkennung",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center,
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                val fps = extractFps(uiState.performanceText, "Object Frame")
                when {
                    !sharedPreferences.getBoolean(
                        stringResource(R.string.enable_object_detection_setting),
                        true,
                    ) -> Text(
                        text = "Objekterkennung deaktiviert",
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center,
                    )
                    fps == null -> ShimmerBox(
                        brush = shimmerBrush,
                        modifier = Modifier
                            .fillMaxWidth(0.75f)
                            .height(Spacing.xl),
                    )
                    else -> Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Aktiv",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                        Text(
                            text = "Leistung: ${performanceLabel(fps)}",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun VisionStatusCard() {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val visionSelected = sharedPreferences.getString(
        context.getString(R.string.input_source_setting),
        context.getString(R.string.input_is_camera),
    ) == context.getString(R.string.input_is_eyeaivision)

    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "EyeAI-Vision",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center,
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (!visionSelected) {
                    Text(
                        text = "Keine EyeAI-Vision verbunden. Handykamera wird benutzt.",
                        style = MaterialTheme.typography.bodyMedium,
                        textAlign = TextAlign.Center,
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "EyeAI-Vision ausgewählt",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                        Text(
                            text = "Verbindung wird über die Runtime hergestellt",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun DepthStatusCard(uiState: UIState, shimmerBrush: Brush) {
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "Distanzmessung",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center,
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                val fps = extractFps(uiState.performanceText, "Depth Frame")
                if (fps == null) {
                    ShimmerBox(
                        brush = shimmerBrush,
                        modifier = Modifier
                            .fillMaxWidth(0.75f)
                            .height(Spacing.xl),
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Aktiv",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                        Text(
                            text = "Leistung: ${performanceLabel(fps)}",
                            style = MaterialTheme.typography.bodyMedium,
                            textAlign = TextAlign.Center,
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun VoskStatusCard(uiState: UIState) {
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "Spracherkennung",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center,
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    text = speechRecognitionStatus(uiState),
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

private fun speechRecognitionStatus(uiState: UIState): String = when {
    uiState.ttsSpeaking -> "EyeAI antwortet"
    uiState.voskListening -> "Vosk listening"
    else -> "Vosk bereit"
}

private fun extractFps(performanceText: String, frameName: String): Float? {
    val match = Regex("${Regex.escape(frameName)}:\\s*([0-9]+(?:\\.[0-9]+)?)")
        .find(performanceText)
        ?: return null
    return match.groupValues[1].toFloatOrNull()
}

private fun performanceLabel(fps: Float): String = when {
    fps < 5f -> "Schlecht"
    fps < 10f -> "Ausreichend"
    fps < 20f -> "Gut"
    fps < 100f -> "Sehr gut"
    else -> "Sehr schnell"
}
