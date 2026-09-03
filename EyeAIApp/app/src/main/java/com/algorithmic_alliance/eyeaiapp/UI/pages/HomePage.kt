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
            item { VisionStatusCard(uiState = uiState) }
        }
    }
}

@Composable
fun ObjectStatusCard(uiState: UIState, shimmerBrush: Brush) {
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
            Text(
                "Objekterkennung",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (!sharedPreferences.getBoolean(
                        stringResource(R.string.enable_object_detection_setting), true
                    )
                ) {
                    Text(
                        text = "Objekterkennung deaktiviert",
                        textAlign = TextAlign.Center,
                        style = MaterialTheme.typography.bodyMedium
                    )
                } else if (getObjectFPS(uiState.performanceText) == -1) {
                    ShimmerBox(
                        shimmerBrush, Modifier
                            .fillMaxWidth(0.75f)
                            .height(Spacing.xl)
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Aktiv",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "Leistung: ${getPerformance(getObjectFPS(uiState.performanceText))}",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                }
            }
        }
    }
}

@Composable
fun VisionStatusCard(uiState: UIState) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "EyeAI-Vision",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center
            )
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
                        textAlign = TextAlign.Center,
                        style = MaterialTheme.typography.bodyMedium
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            "Name: EyeAI-Vision von Robert",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            "Verbindung: Gut ",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            "Akku: 49% ",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun DepthStatusCard(uiState: UIState, shimmerBrush: Brush) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "Distanzmessung",
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.titleMedium
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                if (getDepthFPS(uiState.performanceText) == -1) {
                    ShimmerBox(
                        shimmerBrush, Modifier
                            .fillMaxWidth(0.75f)
                            .height(Spacing.xl)
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Aktiv",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            text = "Leistung: ${getPerformance(getDepthFPS(uiState.performanceText))}",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
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
            .aspectRatio(4f / 3f)
    ) {
        Column(
            modifier = Modifier.padding(Spacing.sm),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                "Spracherkennung",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column() {
                    when{
                        uiState.ttsSpeaking -> {
                            Text(
                                "EyeAI antwortet...",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                        uiState.voskListening -> {
                            Text(
                                "EyeAI hört zu...",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                        else -> {
                            Text(
                                "EyeAI bereit - Button zum Starten drücken",
                                textAlign = TextAlign.Center,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
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