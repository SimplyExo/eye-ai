package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.UIState
import com.algorithmic_alliance.eyeaiapp.UI.connectToDevice
import com.algorithmic_alliance.eyeaiapp.UI.rememberAudioDeviceState
import com.algorithmic_alliance.eyeaiapp.UI.rememberWifiScanState
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import kotlin.collections.emptyList


@RequiresApi(Build.VERSION_CODES.S)
@Composable
fun ConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    onExitSelection: () -> Unit,
    uiState: UIState,
    onEvent: (UIEvent) -> Unit
) {
    Log.d(LOG_TAG, "[PermissionPage] Loading ConnectionPage")

    val context = LocalContext.current

    if (ActivityCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) != PackageManager.PERMISSION_GRANTED
    ) {
        Log.d(
            LOG_TAG,
            "[ConnectionPage] Canceling WIFI-Scan due to permissions not being granted."
        )
        return
    }

    val wifiScanState = rememberWifiScanState(context, autoScanOnStart = false)
    //wifiScanState.rescan()

    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)

    //TODO connection to Backend
    val devices: List<Any> = listOf(
        mapOf(
            "name" to "Audio-Gerät",
            "type" to "audio",
            "rememberKey" to R.string.remember_audio_device,
            "remember" to sharedPreferences.getBoolean(
                stringResource(R.string.remember_audio_device),
                false
            ),
            "selectedKey" to R.string.selected_audio_device,
            "selected" to sharedPreferences.getString(
                stringResource(R.string.selected_audio_device),
                ""
            ),
        ),
        mapOf(
            "name" to "EyeAI-Vision",
            "type" to "eye-ai-vision",
            "rememberKey" to R.string.remember_eye_ai_vision,
            "remember" to sharedPreferences.getBoolean(
                stringResource(R.string.remember_eye_ai_vision),
                false
            ),
            "selectedKey" to R.string.selected_eye_ai_vision,
            "selected" to sharedPreferences.getString(
                stringResource(R.string.selected_eye_ai_vision),
                ""
            ),
        )
    )

    var currentlyDisplayedDevices by remember { mutableIntStateOf(0) }

    ChooseConnectionPage(
        onConnectionSuccessful = {
            if (currentlyDisplayedDevices < devices.size - 1)
                currentlyDisplayedDevices++
            else onConnectionSuccessful()
        },
        goBack = { if (currentlyDisplayedDevices != 0) currentlyDisplayedDevices-- else onExitSelection() },
        devicesData = devices[currentlyDisplayedDevices] as Map<Any, Any>,
        uiState = uiState,
        onEvent = onEvent
    )

}

@RequiresApi(Build.VERSION_CODES.S)
@Composable
fun ChooseConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    goBack: () -> Unit,
    devicesData: Map<Any, Any>,
    uiState: UIState,
    onEvent: (UIEvent) -> Unit
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    var shouldRememberDevice by rememberSaveable { mutableStateOf(false) }
    var selectedDevice by remember { mutableStateOf("") }
    var showConnectionFailedDialog by remember { mutableStateOf(false) }
    var showLocationDisabledDialog by remember { mutableStateOf(false) }

    val shouldRememberKey = stringResource(devicesData["rememberKey"] as Int)
    val selectedDeviceKey = stringResource(devicesData["selectedKey"] as Int)
    val deviceCategory = devicesData["name"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val wifiScanState = rememberWifiScanState(context, autoScanOnStart = false)
    val (audioDevices, refreshAudioDevices) = rememberAudioDeviceState(context)
    val devices: List<String> = when (devicesData["type"]) {
        "audio" -> audioDevices
        "eye-ai-vision" -> wifiScanState.networks
        else -> emptyList()
    }

    Log.d(LOG_TAG, "[ConnectionPage] Choosing connection for $deviceCategory")

    LaunchedEffect(devicesData["type"]) {
        if (devicesData["type"] == "eye-ai-vision" && (devicesData["remember"] == true && devicesData["selected"] != "Handykamera verwenden")) {
            val locationManager =
                context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            if (locationManager.isLocationEnabled)
                wifiScanState.rescan()
            else {
                Log.d(
                    LOG_TAG,
                    "[ChooseConnectionPage] Wifi-Scan failed. Location services are not turned on."
                )
                showLocationDisabledDialog = true
            }

        }
    }
    if (!uiState.connectionTutorialCompleted) {
        LaunchedEffect(devicesData) {
            if (devicesData["remember"] == true && (devices.contains(devicesData["selected"]) || devicesData["selected"] == "Handykamera verwenden")) {
                Log.d(
                    LOG_TAG,
                    "[ConnectionPage] Attempting to connect to remembered ${devicesData["type"]} device"
                )
                connectToDevice(
                    context,
                    devicesData["type"] as String,
                    devicesData["selected"] as String,
                    onEvent = onEvent
                )
                { success ->
                    if (success) {
                        Log.d(
                            LOG_TAG,
                            "[ConnectionPage] Connection to remembered device successful"
                        )
                        onConnectionSuccessful()
                    }

                }
            }
        }
    }

    Surface(
        modifier = Modifier
            .fillMaxSize(),
        color = MaterialTheme.colorScheme.surface
    ) {
        Column(verticalArrangement = Arrangement.Center) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    contentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            ) {
                Column {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Text(
                            "$deviceCategory auswählen",
                            modifier = Modifier.clearAndSetSemantics {
                                contentDescription = "$deviceCategory auswählen"
                            },
                            fontSize = 30.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    HorizontalDivider(
                        modifier = Modifier
                            .padding(8.dp)
                            .clearAndSetSemantics {})
                    if (devices.isNotEmpty()) {
                        LazyColumn(
                            modifier = Modifier
                                .padding(8.dp)
                                .fillMaxWidth()
                        ) {
                            items(items = devices) { item ->
                                val index = devices.indexOf(item)
                                DeviceListEntry(
                                    item as String,
                                    onSelected = {
                                        selectedDevice = item
                                    },
                                    isSelected = item == selectedDevice
                                )
                            }
                            if (devicesData["type"] == "eye-ai-vision") {
                                item {
                                    DeviceListEntry(
                                        "Handykamera verwenden",
                                        onSelected = { selectedDevice = "Handykamera verwenden" },
                                        isSelected = "Handykamera verwenden" == selectedDevice
                                    )
                                }
                            }
                        }
                    } else {
                        Row(
                            modifier = Modifier
                                .padding(8.dp)
                        ) { Text("Lade...") }
                        DeviceListEntry(
                            "Handykamera verwenden",
                            onSelected = { selectedDevice = "Handykamera verwenden" },
                            isSelected = "Handykamera verwenden" == selectedDevice
                        )


                    }
                    HorizontalDivider(
                        modifier = Modifier.padding(
                            top = 8.dp,
                            start = 8.dp,
                            end = 8.dp
                        )
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(start = 8.dp, end = 32.dp, top = 4.dp, bottom = 4.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(
                                checked = shouldRememberDevice,
                                onCheckedChange = { shouldRememberDevice = !shouldRememberDevice })
                            Text("Als Standardgerät festlegen")
                        }
                        Box(modifier = Modifier.clickable {
                            when (devicesData["type"]) {
                                "audio" -> {
                                    refreshAudioDevices()
                                }

                                "eye-ai-vision" -> {
                                    val locationManager =
                                        context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
                                    if (locationManager.isLocationEnabled)
                                        wifiScanState.rescan()
                                    else {
                                        Log.d(
                                            LOG_TAG,
                                            "[ChooseConnectionPage] Wifi-Scan failed. Location services are not turned on."
                                        )
                                        showLocationDisabledDialog = true
                                    }
                                }
                            }
                        }) {
                            Icon(
                                painter = painterResource(R.drawable.refresh_24px),
                                contentDescription = ""
                            )
                        }
                    }
                    HorizontalDivider(
                        modifier = Modifier.padding(
                            bottom = 8.dp,
                            start = 8.dp,
                            end = 8.dp
                        )
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        Button(
                            modifier = Modifier.weight(1f),
                            onClick = { goBack() }) { Text("Zurück") }
                        Button(
                            modifier = Modifier
                                .weight(1f)
                                .semantics {
                                    contentDescription =
                                        "Mit Gerät $selectedDevice verbinden"
                                }, enabled = selectedDevice != "",
                            onClick = {

                                connectToDevice(
                                    context,
                                    devicesData["type"] as String,
                                    selectedDevice,
                                    onEvent = onEvent
                                )
                                { success ->
                                    if (success) {
                                        Log.d(
                                            LOG_TAG,
                                            "[ConnectionPage] Setting SharedPreferences ShouldRememberDevice: $shouldRememberDevice"
                                        )
                                        sharedPreferences.edit(commit = true) {
                                            putBoolean(
                                                shouldRememberKey,
                                                shouldRememberDevice
                                            )
                                        }
                                        Log.d(
                                            LOG_TAG,
                                            "[ConnectionPage] Setting SharedPreferences SelectedDevice: $selectedDevice"
                                        )
                                        sharedPreferences.edit(commit = true) {
                                            putString(
                                                selectedDeviceKey,
                                                if (shouldRememberDevice) selectedDevice else ""
                                            )
                                        }
                                        shouldRememberDevice = false
                                        selectedDevice = ""
                                        onConnectionSuccessful()
                                    } else
                                        showConnectionFailedDialog = true
                                }
                            }) { Text("Verbinden", modifier = Modifier.clearAndSetSemantics {}) }
                    }
                }
            }
        }
    }

    if (showConnectionFailedDialog) {
        ErrorDialog(
            titel = "Verbindung fehlgeschlagen",
            content = "",
            onDismissed = { showConnectionFailedDialog = false })
    }

    if (showLocationDisabledDialog) {
        ErrorDialog(
            titel = "Standortdienste sind ausgeschaltet",
            content = "Da durch einen Scan der WLAN-Netzwerke in der nähe Informationen zu ihrem Standort anfallen könnten, muss laut Android-Richtlinien der Standort angeschaltet sein. Die App nutzt ihren Standort jedoch nicht.",
            onDismissed = { showLocationDisabledDialog = false }
        )
    }
}

@Composable
fun DeviceListEntry(deviceName: String, isSelected: Boolean = false, onSelected: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription =
                    "Gerät: $deviceName." + if (isSelected) "Das Gerät ist ausgewählt." else "Das Gerät ist nicht ausgewählt."
            }, verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(modifier = Modifier.semantics {
            contentDescription =
                if (!isSelected) "Gerät $deviceName auswählen?" else "Gerät $deviceName ist ausgewählt."
        }, onClick = { onSelected() }, selected = isSelected)
        Text(
            deviceName, Modifier
                .clearAndSetSemantics {})
    }
}

@Composable
fun ErrorDialog(titel: String, content: String, onDismissed: () -> Unit) {
    AlertDialog(
        onDismissRequest = { onDismissed() },
        title = { Text(titel) },
        text = { Text(content) },
        confirmButton = {
            Button(onClick = { onDismissed() }) {
                Text(
                    "Verstanden",
                    modifier = Modifier.clearAndSetSemantics {
                        contentDescription = "Verstanden. Dialog-Feld verlassen."
                    })
            }
        })

}

class WifiScanState(
    val networks: List<String>,
    val rescan: () -> Unit
)

@RequiresApi(Build.VERSION_CODES.S)
@Preview(showBackground = true, name = "ConnectionPagePreview")
@Composable
fun ConnectionPagePreview() {
    MaterialTheme {
        ConnectionPage(
            Modifier.fillMaxSize(),
            onConnectionSuccessful = {},
            onExitSelection = {},
            uiState = UIState(), onEvent = {}
        )
    }
}


