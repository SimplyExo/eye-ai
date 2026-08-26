package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import android.util.Size
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.IconButton
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOCR
import com.algorithmic_alliance.eyeaiapp.UI.OverlayViewOD
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.UIState
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

    var ttsEnabled by rememberSaveable() { mutableStateOf(false) }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val speechRecognitionKey = stringResource(R.string.enable_speech_recognition_setting)
    var speechRecognitionEnabled by rememberSaveable { mutableStateOf(true) }

    LaunchedEffect(Unit) {

        speechRecognitionEnabled = sharedPreferences.getBoolean(speechRecognitionKey, true)
        if (ActivityCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            if (speechRecognitionEnabled) {
                Log.d(LOG_TAG, "[DebugPage] Loading Vosk model")
                onEvent(UIEvent.InitVoskService)
            } else {
                Log.d(LOG_TAG, "[DebugPage] Speech Recognition disabled not loading Vosk model")
            }
        }
        onEvent(UIEvent.UpdateVoskStatusText)
        onEvent(UIEvent.UpdateLlmStatusText)
    }

    Scaffold(modifier = Modifier.fillMaxSize(), topBar = {
        TopAppBar(
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Debug")
                }
            },
            modifier = Modifier.shadow(elevation = 8.dp),
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
            ),
        )
    }, floatingActionButton = {
        Row(
            modifier = Modifier
                .padding(8.dp)
                .fillMaxWidth(0.35f),
            horizontalArrangement = if (speechRecognitionEnabled) Arrangement.SpaceBetween else Arrangement.End
        ) {
            if (speechRecognitionEnabled) FloatingActionButton(
                onClick = {
                    onEvent(UIEvent.VoskListeningChanged)
                    ttsEnabled = !ttsEnabled
                    Log.d(LOG_TAG, "[DebugPage] Vosk on: $ttsEnabled")
                },
            ) {
                Icon(
                    painter = if (ttsEnabled) painterResource(R.drawable.stop_24px) else painterResource(
                        R.drawable.play_arrow_24px
                    ), contentDescription = "Start Vosk"
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
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            horizontalAlignment = Alignment.CenterHorizontally,

            ) {
            item {
                Card(
                    modifier = Modifier
                        .padding(8.dp)
                        .aspectRatio(4f / 3f)
                ) {
                    Box(modifier = Modifier.fillMaxSize()) {
                        if (sharedPreferences.getString(
                                stringResource(R.string.input_source_setting), "camera"
                            ) == "camera" && !sharedPreferences.getBoolean(
                                stringResource(R.string.show_debug_input_bitmap_setting), false
                            )
                        ) CameraPreview(onEvent = onEvent)
                        if (sharedPreferences.getString(
                                stringResource(R.string.input_source_setting), "camera"
                            ) == "media" && !sharedPreferences.getBoolean(
                                stringResource(R.string.show_debug_input_bitmap_setting), false
                            )
                        ) MediaPreview(bitmap = uiState.mediaPreviewBitmap, onEvent = onEvent)
                        if (sharedPreferences.getBoolean(
                                stringResource(R.string.show_debug_input_bitmap_setting), false
                            )
                        ) DebugInputPreview(
                            bitmap = uiState.debugInputPreviewBitmap,
                            onEvent = onEvent
                        )
                        ObjectDetectionOverlay(
                            modifier = Modifier.matchParentSize().padding(8.dp),
                            uiState.detectedObjects,
                            cameraResolution = uiState.cameraResolution
                        )
                        OCROverlay(
                            modifier = Modifier.matchParentSize().padding(8.dp),
                            results = uiState.ocrResults,
                            cameraResolution = uiState.cameraResolution
                        )
                    }
                }
                Column() {
                    Text(uiState.speechRecognitionPartialResultText)
                    Text(uiState.speechRecognitionFinalResultText)
                    Text(uiState.llmResponseText)
                }
                Text(uiState.performanceText, fontSize = 10.sp)
                DepthPreview(bitmap = uiState.depthPreviewBitmap)
            }


        }
    })


}

@Composable
fun ObjectDetectionOverlay(
    modifier: Modifier = Modifier,
    results: Array<UniffiDetectedObject>,
    cameraResolution: Size,
    onOverlayCreated: (OverlayViewOD) -> Unit = {}
) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            OverlayViewOD(context, null).also { onOverlayCreated(it) }
        },
        update = { overlayView ->
            overlayView.setCameraResolution(cameraResolution)
            overlayView.setResults(results)
        }
    )
}

@Composable
fun OCROverlay(
    modifier: Modifier = Modifier,
    results: Array<TextBoundingBox>,
    cameraResolution: Size,
    onOverlayCreated: (OverlayViewOCR) -> Unit = {}
) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            OverlayViewOCR(context, null).also { onOverlayCreated(it) }
        },
        update = { overlayView ->
            overlayView.setCameraResolution(cameraResolution)
            overlayView.setResults(results)
        }
    )
}

@Composable
fun DebugInputPreview(
    modifier: Modifier = Modifier, bitmap: Bitmap?, onEvent: (UIEvent) -> Unit
) {

    val lifecycleOwner = LocalLifecycleOwner.current
    LaunchedEffect(Unit) {
        Log.d(LOG_TAG, "Loading DebugInputPreview")
        onEvent(UIEvent.UIinitCamera(previewView = null, lifecycleOwner = lifecycleOwner))
    }

    bitmap?.let {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Depth preview",
                modifier = Modifier
                    .padding(8.dp)
                    .aspectRatio(1f / 1f)
                    .clip(RoundedCornerShape(8.dp)),
                contentScale = ContentScale.Crop

            )
        }

    }
}

@Composable
fun MediaPreview(
    modifier: Modifier = Modifier, bitmap: Bitmap?, onEvent: (UIEvent) -> Unit
) {

    val lifecycleOwner = LocalLifecycleOwner.current

    LaunchedEffect(Unit) {
        Log.d(LOG_TAG, "[DebugPage.MediaPreview] Loading Media Preview")
        onEvent(UIEvent.UIinitCamera(previewView = null, lifecycleOwner = lifecycleOwner))
    }

    bitmap?.let {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Depth preview",
                modifier = Modifier
                    .padding(8.dp)
                    .aspectRatio(1f / 1f)
                    .clip(RoundedCornerShape(8.dp)),
                contentScale = ContentScale.Crop
            )
        }
    }

}

@Composable
fun DepthPreview(
    modifier: Modifier = Modifier, bitmap: Bitmap?
) {
    Card(
        modifier = Modifier
            .padding(8.dp)
            .aspectRatio(4f / 3f)
    ) {
        bitmap?.let {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Depth preview",
                    modifier = Modifier
                        .padding(8.dp)
                        .aspectRatio(1f / 1f)
                        .clip(RoundedCornerShape(8.dp)),
                    contentScale = ContentScale.Crop

                )
            }
        }
    }
}

@Composable
fun CameraPreview(onEvent: (UIEvent) -> Unit) {
    Log.d(LOG_TAG, "Loading CameraPreview")
    val lifecycleOwner = LocalLifecycleOwner.current

    AndroidView(
        modifier = Modifier
            .aspectRatio(4f / 3f)
            .padding(8.dp)
            .clip(RoundedCornerShape(8.dp)),
        factory = { context ->
            PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }.also {
                onEvent(
                    UIEvent.UIinitCamera(
                        it, lifecycleOwner
                    )
                )
            }
        })

}

@Preview(showBackground = true, name = "DebugPagePreview")
@Composable
fun DebugPagePreview() {
    DebugPage(
        modifier = Modifier.fillMaxSize(),
        onOpenSettings = {},
        onEvent = {},
        uiState = UIState()
    )
}