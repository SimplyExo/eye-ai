package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.graphics.Bitmap
import android.util.Log
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
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.IconButton
import androidx.compose.ui.Alignment
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.UIState

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DebugPage(
    modifier: Modifier = Modifier,
    onOpenSettings: () -> Unit,
    onBack: () -> Unit,
    onEvent: (UIEvent) -> Unit,
    uiState: UIState,
) {

    var ttsEnabled by rememberSaveable() { mutableStateOf(false) }
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        IconButton(onClick = { onBack() }) {
                            Icon(
                                painter = painterResource(R.drawable.arrow_back_24px),
                                contentDescription = ""
                            )
                        }
                        Text("Debug")
                    }
                },
                modifier = Modifier.shadow(elevation = 8.dp),
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                ),
            )
        },
        floatingActionButton = {
            Row(
                modifier = Modifier
                    .padding(8.dp)
                    .fillMaxWidth(0.35f),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                FloatingActionButton(
                    onClick = {
                        onEvent(UIEvent.VoskListeningChanged)
                        ttsEnabled = !ttsEnabled
                        Log.d(LOG_TAG, "[DebugPage] Vosk on: $ttsEnabled")
                    },
                ) {
                    Icon(
                        painter = if (ttsEnabled) painterResource(R.drawable.stop_24px) else painterResource(
                            R.drawable.play_arrow_24px
                        ),
                        contentDescription = "Start Vosk"
                    )
                }
                FloatingActionButton(onClick = { onOpenSettings() }) {
                    Icon(
                        painter = painterResource(R.drawable.settings_24px),
                        contentDescription = "Open Settings"
                    )
                }
            }
        },
        content = { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues),
                horizontalAlignment = Alignment.CenterHorizontally,

                ) {
                CameraPreview(onEvent = onEvent)
                Column() {
                    Text(uiState.speechRecognitionPartialResultText)
                    Text(uiState.speechRecognitionFinalResultText)
                    Text(uiState.llmResponseText)
                }
                DepthPreview(bitmap = uiState.depthPreviewBitmap)
            }
        })


}

@Composable
fun DepthPreview(
    modifier: Modifier = Modifier,
    bitmap: Bitmap?
) {
    Card(modifier = Modifier
        .padding(8.dp)
        .aspectRatio(4f / 3f)) {
        bitmap?.let {
            Row(modifier = Modifier.fillMaxWidth(),horizontalArrangement = Arrangement.Center) {
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
    val lifecycleOwner = LocalLifecycleOwner.current
    Card(modifier = Modifier.padding(8.dp)) {
        AndroidView(
            modifier = Modifier
                .aspectRatio(4f / 3f)
                .padding(8.dp)
                .clip(RoundedCornerShape(8.dp)),
            factory = { context ->
                PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }.also {
                    onEvent(
                        UIEvent.CameraPreviewReady(
                            it,
                            lifecycleOwner
                        )
                    )
                }
            }
        )
    }
}

@Preview(showBackground = true, name = "DebugPagePreview")
@Composable
fun DebugPagePreview() {
    DebugPage(
        modifier = Modifier.fillMaxSize(),
        onOpenSettings = {},
        onBack = {},
        onEvent = {},
        uiState = UIState()
    )
}