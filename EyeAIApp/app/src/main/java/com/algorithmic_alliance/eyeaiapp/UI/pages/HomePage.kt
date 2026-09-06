package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.PremiumFloatingActionButton
import com.algorithmic_alliance.eyeaiapp.UI.ShimmerBox
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.rememberShimmerBrush
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

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
    //Log.d(LOG_TAG, "[HomePage] Loading HomePage")
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val speechRecognitionKey = stringResource(R.string.enable_speech_recognition_setting)
    val profilingInformationKey = stringResource(R.string.show_profiling_info_setting)
    var initialShowProfilingInformationSetting = false
    val speechRecognitionEnabled by remember {
        mutableStateOf(
            sharedPreferences.getBoolean(
                speechRecognitionKey, true
            )
        )
    }
    val shimmerBrush = rememberShimmerBrush(
        backgroundColor = MaterialTheme.colorScheme.surface,
        contrastColor = MaterialTheme.colorScheme.onSurface
    )
    DisposableEffect(Unit) {
        initialShowProfilingInformationSetting =
            sharedPreferences.getBoolean(profilingInformationKey, false)
        sharedPreferences.edit(commit = true) {
            putBoolean(profilingInformationKey, true)
        }
        onEvent(UIEvent.UpdateSettings)


        if (speechRecognitionEnabled) {
            if (ActivityCompat.checkSelfPermission(
                    context, Manifest.permission.RECORD_AUDIO
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                Log.d(LOG_TAG, "[HomePage] Loading Vosk model")
                onEvent(UIEvent.InitVoskService)
            }
        } else {
            Log.d(LOG_TAG, "[HomePage] Speech Recognition disabled not loading Vosk model")
            onEvent(UIEvent.CloseVoskService)
        }

        onEvent(UIEvent.UIinitCamera(null, lifecycleOwner))
        onEvent(UIEvent.UpdateVoskStatusText)
        onEvent(UIEvent.UpdateSpeechStatusText)
        onDispose {
            sharedPreferences.edit(commit = true) {
                putBoolean(profilingInformationKey, initialShowProfilingInformationSetting)
            }
            onEvent(UIEvent.UpdateSettings)
        }
    }


    Scaffold(
        modifier = Modifier.fillMaxSize(),
        contentWindowInsets = WindowInsets.safeDrawing,
        topBar = {
            TopAppBar(
                title = {
                    Text("EyeAI App", style = MaterialTheme.typography.titleLarge)
                },
                modifier = Modifier.shadow(elevation = Spacing.sm),
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                ),
            )
        },
        floatingActionButton = {
            Row(
                modifier = Modifier
                    .padding(Spacing.sm)
                    .fillMaxWidth(0.35f),
                horizontalArrangement = if (speechRecognitionEnabled) Arrangement.SpaceBetween else Arrangement.End
            ) {
                if (speechRecognitionEnabled) PremiumFloatingActionButton(
                    shadowElevation = 8.dp,
                    onClick = {
                        onEvent(UIEvent.VoskListeningChanged)
                    },
                ) {
                    Icon(
                        painter = if (uiState.voskListening) {
                            painterResource(R.drawable.stop_24px)
                        } else if (uiState.ttsSpeaking) {
                            painterResource(
                                R.drawable.pause_playback_24px
                            )
                        } else {
                            painterResource(R.drawable.play_arrow_24px)
                        }, contentDescription = stringResource(R.string.start_vosk_button_description)
                    )
                }

                PremiumFloatingActionButton(
                    shadowElevation = 8.dp,
                    onClick = { onOpenSettings() }) {
                    Icon(
                        painter = painterResource(R.drawable.settings_24px),
                        contentDescription = stringResource(R.string.open_settings_button_description)
                    )
                }
            }
        },
        content = { paddingValues ->
            LazyVerticalGrid(
                modifier = Modifier.padding(paddingValues).fillMaxHeight(), columns = GridCells.Fixed(2)
            ) {
                item {
                    VoskStatusCard(viewModel = viewModel)
                }
                item {
                    DepthStatusCard(viewModel = viewModel, shimmerBrush = shimmerBrush)
                }
                item { ObjectStatusCard(viewModel = viewModel, shimmerBrush = shimmerBrush) }
                item { VisionStatusCard(viewModel = viewModel) }

            }

        })

}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun ObjectStatusCard(viewModel: MainViewModel, shimmerBrush: Brush) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val isDark = isSystemInDarkTheme()
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
        shape = PremiumShapes.medium,
        elevation = CardDefaults.cardElevation(AppElevation.level3),
        border = BorderStroke(if(isDark) 1.dp else 0.dp, color = Color.White.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(Spacing.xs),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                stringResource(R.string.object_detection_card_title),
                modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(R.string.object_detection_card_title_semantic)},
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
                        text = stringResource(R.string.object_detection_card_disabled_text),
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
                            text = stringResource(R.string.status_card_active_text),
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${stringResource(R.string.status_card_performance_text)}: ${getPerformance(LocalContext.current,getObjectFPS(uiState.performanceText))}",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                }
            }
        }
    }
}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun VisionStatusCard(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val isDark = isSystemInDarkTheme()
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
        shape = PremiumShapes.medium,
                elevation = CardDefaults.cardElevation(AppElevation.level3),
        border = BorderStroke(if(isDark) 1.dp else 0.dp, color = Color.White.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(Spacing.xs),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                stringResource(R.string.vision_card_title),
                modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(R.string.vision_card_title_semantic)},
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
                        stringResource(R.string.vision_card_no_vision_connected_text),
                        textAlign = TextAlign.Center,
                        style = MaterialTheme.typography.bodyMedium
                    )
                } else {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            "${stringResource(R.string.vision_card_name_text)}: EyeAI-Vision",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            "${stringResource(R.string.vision_card_connection_text)}: Good ",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Text(
                            "${stringResource(R.string.vision_card_battery_text)}: 49% ",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
        }
    }
}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun DepthStatusCard(viewModel: MainViewModel, shimmerBrush: Brush) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val isDark = isSystemInDarkTheme()
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
        shape = PremiumShapes.medium,elevation = CardDefaults.cardElevation(AppElevation.level3),
        border = BorderStroke(if(isDark) 1.dp else 0.dp, color = Color.White.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(Spacing.xs),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                stringResource(R.string.depth_estimation_card_title),
                modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(R.string.depth_estimation_card_title_semantic)},
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
                            text = stringResource(R.string.status_card_active_text),
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${stringResource(R.string.status_card_performance_text)}: ${getPerformance(LocalContext.current,getDepthFPS(uiState.performanceText))}",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                }
            }
        }
    }
}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun VoskStatusCard(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val isDark = isSystemInDarkTheme()
    val context = LocalContext.current
    Card(
        modifier = Modifier
            .padding(Spacing.sm)
            .aspectRatio(4f / 3f),
        shape = PremiumShapes.medium,elevation = CardDefaults.cardElevation(AppElevation.level3),
        border = BorderStroke(if(isDark) 1.dp else 0.dp, color = Color.White.copy(alpha = 0.2f))
    ) {
        Column(
            modifier = Modifier.padding(Spacing.xs),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                stringResource(R.string.vosk_card_title),
                modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(R.string.vosk_card_title_semantic)},
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center
            )
            HorizontalDivider()
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Column {
                    if (uiState.ttsSpeaking) {
                        Text(
                            stringResource(R.string.vosk_card_responding),
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    } else {
                        Text(
                            modifier = Modifier.padding(Spacing.sm),
                            text = uiState.speechRecognitionFinalResultText,
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }

                }
            }
        }

    }
}

private fun getPerformance(context: Context, fps: Int): String {
    return when (fps) {
        in 0..5 -> context.getString(R.string.status_card_performance_poor)
        in 5..10 -> context.getString(R.string.status_card_performance_sufficient)
        in 10..20 -> context.getString(R.string.status_card_performance_good)
        in 20..100 -> context.getString(R.string.status_card_performance_very_good)
        in 100..1000 -> context.getString(R.string.status_card_performance_record)
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
