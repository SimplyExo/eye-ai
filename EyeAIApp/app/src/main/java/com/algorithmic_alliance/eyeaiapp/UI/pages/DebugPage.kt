package com.algorithmic_alliance.eyeaiapp.UI.pages

import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.IconButton
import androidx.compose.ui.Alignment

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DebugPage(modifier: Modifier = Modifier, onOpenSettings: () -> Unit, onBack: () -> Unit) {

    var ttsEnabled by rememberSaveable() { mutableStateOf(true) }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                title = {
                    Row (verticalAlignment = Alignment.CenterVertically){
                        IconButton(onClick = {onBack()}) {
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
                FloatingActionButton(onClick = { ttsEnabled = !ttsEnabled }) {
                    Icon(
                        painter = if (ttsEnabled) painterResource(R.drawable.stop_24px) else painterResource(
                            R.drawable.play_arrow_24px
                        ),
                        contentDescription = "Start Vosk"
                    )
                }
                FloatingActionButton(onClick = {onOpenSettings()}) {
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
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Image(
                    modifier = Modifier
                        .height(256.dp)
                        .width(256.dp),
                    painter = painterResource(R.drawable.ic_launcher_web),
                    contentDescription = ""
                )
                Column(){
                    Text("speech recognition partial output")
                    Text("speech recognition final output")
                    Text("llm response")
                }
                Image(
                    modifier = Modifier
                        .height(256.dp)
                        .width(256.dp),
                    painter = painterResource(R.drawable.ic_launcher_web),
                    contentDescription = ""
                )
            }
        })


}

@Preview(showBackground = true, name = "DebugPagePreview")
@Composable
fun DebugPagePreview() {
    DebugPage(modifier = Modifier.fillMaxSize(), onOpenSettings = {}, onBack = {})
}