package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Size
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.UIState
import com.algorithmic_alliance.eyeaiapp.data.Shapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import uniffi.NativeLib.UniffiDetectedObject

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DebugPage(
    modifier: Modifier = Modifier,
    onOpenSettings: () -> Unit,
    onEvent: (UIEvent) -> Unit,
    uiState: UIState,
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val speechRecognitionKey = stringResource(R.string.enable_speech_recognition_setting)
    val inputSourceKey = stringResource(R.string.input_source_setting)
    val cameraInput = stringResource(R.string.input_is_camera)
    val mediaInput = stringResource(R.string.input_is_media)
    val showDebugInputKey = stringResource(R.string.show_debug_input_bitmap_setting)
    var speechRecognitionEnabled by rememberSaveable { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        speechRecognitionEnabled = sharedPreferences.getBoolean(speechRecognitionKey, true)
        if (
            ActivityCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            if (speechRecognitionEnabled) {
                onEvent(UIEvent.InitVoskService)
            } else {
                onEvent(UIEvent.CloseVoskService)
            }
        }
        onEvent(UIEvent.UpdateVoskStatusText)
        onEvent(UIEvent.UpdateSpeechStatusText)
    }

    key(uiState.reloadDebugPageKey) {
        val selectedInput = sharedPreferences.getString(inputSourceKey, cameraInput)
        val showDebugInput = sharedPreferences.getBoolean(showDebugInputKey, false)
        val selectedMediaPath = sharedPreferences.getString(
            stringResource(R.string.media_path_setting),
            "",
        ).orEmpty()

        Scaffold(
            modifier = modifier.fillMaxSize(),
            contentWindowInsets = WindowInsets.safeDrawing,
            topBar = {
                TopAppBar(
                    title = { Text("Debug", style = MaterialTheme.typography.titleLarge) },
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
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Card(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(
                            top = Spacing.sm,
                            bottom = Spacing.sm,
                            start = Spacing.xs,
                            end = Spacing.xs,
                        ),
                    shape = Shapes.medium,
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        when {
                            showDebugInput -> DebugInputPreview(
                                bitmap = uiState.debugInputPreviewBitmap,
                                onEvent = onEvent,
                            )
                            selectedInput == cameraInput -> CameraPreview(onEvent)
                            selectedInput == mediaInput -> {
                                MediaPreview(bitmap = uiState.mediaPreviewBitmap, onEvent = onEvent)
                                if (selectedMediaPath.isBlank()) MediaSourceMissingMessage()
                            }
                        }

                        ObjectDetectionOverlay(
                            modifier = Modifier
                                .matchParentSize()
                                .padding(Spacing.sm),
                            results = uiState.detectedObjects,
                            cameraResolution = uiState.cameraResolution,
                        )
                        OCROverlay(
                            modifier = Modifier
                                .matchParentSize()
                                .padding(Spacing.sm),
                            results = uiState.ocrResults,
                            cameraResolution = uiState.cameraResolution,
                        )
                        if (sharedPreferences.getBoolean(speechRecognitionKey, true)) {
                            SpeechTranscriptOverlay(uiState)
                        }
                    }
                }
                Card(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                        .padding(bottom = Spacing.sm, start = Spacing.xs, end = Spacing.xs),
                    shape = Shapes.medium,
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        if (selectedInput == mediaInput && selectedMediaPath.isBlank()) {
                            MediaSourceMissingMessage()
                        }
                        DepthPreview(
                            bitmap = uiState.depthPreviewBitmap,
                            performanceText = uiState.performanceText,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun MediaSourceMissingMessage() {
    Column(
        modifier = Modifier
            .fillMaxHeight()
            .padding(Spacing.sm),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "Bitte in den Einstellungen eine Media-Quelle auswählen!",
            style = MaterialTheme.typography.bodyLarge,
        )
    }
}

@Composable
private fun BoxScope.SpeechTranscriptOverlay(uiState: UIState) {
    Card(
        modifier = Modifier
            .padding(bottom = Spacing.md)
            .align(Alignment.BottomCenter)
            .fillMaxWidth(0.75f)
            .heightIn(min = Spacing.xxl, max = Spacing.xxxxl),
        shape = Shapes.medium,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainerHighest.copy(alpha = 0.75f),
        ),
    ) {
        LazyColumn(
            modifier = Modifier
                .padding(Spacing.sm)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            item {
                Text(
                    text = uiState.speechRecognitionPartialResultText,
                    style = MaterialTheme.typography.bodyMedium,
                )
                Text(
                    text = uiState.speechRecognitionFinalResultText,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = uiState.speechResponseText,
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

@Composable
fun ObjectDetectionOverlay(
    modifier: Modifier = Modifier,
    results: Array<UniffiDetectedObject>,
    cameraResolution: Size,
    onOverlayCreated: (OverlayViewOD) -> Unit = {},
) {
    AndroidView(
        modifier = modifier,
        factory = { context -> OverlayViewOD(context, null).also(onOverlayCreated) },
        update = { overlayView ->
            overlayView.setCameraResolution(cameraResolution)
            overlayView.setResults(results)
        },
    )
}

@Composable
fun OCROverlay(
    modifier: Modifier = Modifier,
    results: Array<TextBoundingBox>,
    cameraResolution: Size,
    onOverlayCreated: (OverlayViewOCR) -> Unit = {},
) {
    AndroidView(
        modifier = modifier,
        factory = { context -> OverlayViewOCR(context, null).also(onOverlayCreated) },
        update = { overlayView ->
            overlayView.setCameraResolution(cameraResolution)
            overlayView.setResults(results)
        },
    )
}

@Composable
fun DebugInputPreview(
    modifier: Modifier = Modifier,
    bitmap: Bitmap?,
    onEvent: (UIEvent) -> Unit,
) {
    LaunchedEffect(Unit) {
        // No PreviewView is attached in this mode; the foreground runtime keeps processing.
        onEvent(UIEvent.UIinitCamera(previewView = null))
    }
    PreviewBitmapContainer(modifier, bitmap, "Debug input preview")
}

@Composable
fun MediaPreview(
    modifier: Modifier = Modifier,
    bitmap: Bitmap?,
    onEvent: (UIEvent) -> Unit,
) {
    LaunchedEffect(Unit) {
        // Media processing is owned by the runtime, independently of this composable.
        onEvent(UIEvent.UIinitCamera(previewView = null))
    }
    PreviewBitmapContainer(modifier, bitmap, "Media preview")
}

@Composable
private fun PreviewBitmapContainer(modifier: Modifier, bitmap: Bitmap?, description: String) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(Spacing.sm)
            .clip(Shapes.small)
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        bitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = description,
                modifier = Modifier.clip(Shapes.small),
                contentScale = ContentScale.Fit,
            )
        }
    }
}

@Composable
fun DepthPreview(
    modifier: Modifier = Modifier,
    bitmap: Bitmap?,
    performanceText: String,
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(Spacing.sm)
            .clip(Shapes.small)
            .background(Color.Black),
        contentAlignment = Alignment.Center,
    ) {
        bitmap?.let {
            Box(modifier = Modifier.aspectRatio(1f)) {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Depth preview",
                    modifier = Modifier
                        .fillMaxSize()
                        .clip(Shapes.small),
                    contentScale = ContentScale.Fit,
                )
                LazyColumn(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .padding(start = Spacing.xs, top = Spacing.xs),
                ) {
                    item {
                        Text(
                            text = performanceText,
                            style = MaterialTheme.typography.bodySmall,
                            fontSize = 8.sp,
                            lineHeight = 10.sp,
                            letterSpacing = (-0.2).sp,
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun CameraPreview(onEvent: (UIEvent) -> Unit) {
    AndroidView(
        modifier = Modifier
            .padding(Spacing.sm)
            .clip(Shapes.small),
        factory = { context ->
            PreviewView(context).apply {
                scaleType = PreviewView.ScaleType.FIT_CENTER
            }.also { previewView ->
                // The PreviewView is an optional surface only; camera processing stays headless.
                onEvent(UIEvent.UIinitCamera(previewView))
            }
        },
        onRelease = { previewView ->
            onEvent(UIEvent.UIDetachCameraPreview(previewView))
        },
    )
}

@Preview(showBackground = true, name = "DebugPagePreview")
@Composable
private fun DebugPagePreview() {
    DebugPage(
        modifier = Modifier.fillMaxSize(),
        onOpenSettings = {},
        onEvent = {},
        uiState = UIState(),
    )
}
