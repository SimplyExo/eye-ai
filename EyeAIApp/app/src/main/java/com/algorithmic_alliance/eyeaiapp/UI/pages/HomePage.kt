package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomePage(
    modifier: Modifier = Modifier,
    onOpenSettings: () -> Unit,
    onEvent: (UIEvent) -> Unit,
    viewModel: MainViewModel,
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
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
        onEvent(UIEvent.UpdateVoskStatusText)
        onEvent(UIEvent.UpdateLlmStatusText)
    }


    Scaffold(modifier = Modifier.fillMaxSize(), topBar = {
        TopAppBar(
            title = {
                Text("EyeAI App")
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
                    Log.d(LOG_TAG, "[DebugPage] Vosk on: ${uiState.voskListening}")
                },
            ) {
                Icon(
                    painter = if (uiState.voskListening) painterResource(R.drawable.stop_24px) else painterResource(
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
        LazyVerticalGrid(modifier = Modifier.padding(paddingValues), columns = GridCells.Fixed(2)) {
            item {
                VoskStatusCard(viewModel = viewModel)
            }
            item {
                Text("hi2")
            }
        }
    })

}

@Composable
fun VoskStatusCard(viewModel: MainViewModel) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    Card(
        modifier = Modifier
            .padding(8.dp)
            .aspectRatio(4f / 3f)
    ) {
        Column(modifier = Modifier.padding(8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text("Spracherkennung", fontSize = 18.sp)
            HorizontalDivider()
            Text(modifier = Modifier.padding(8.dp), text = uiState.speechRecognitionFinalResultText) }

    }
}