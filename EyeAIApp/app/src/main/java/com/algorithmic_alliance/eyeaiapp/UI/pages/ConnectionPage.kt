package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioManager
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.annotation.RequiresPermission
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
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
import com.algorithmic_alliance.eyeaiapp.UI.audioDeviceTypeName
import com.algorithmic_alliance.eyeaiapp.UI.connectToDevice
import com.algorithmic_alliance.eyeaiapp.UI.rememberWifiScanState
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import kotlin.collections.emptyList


@RequiresApi(Build.VERSION_CODES.Q)
@Composable
fun ConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    onExitSelection: () -> Unit
) {
    Log.d("EyeAIUI", "[PermissionPage] Loading ConnectionPage")

    val context = LocalContext.current
    val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    val availableAudioDevices = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
    val displayAudioDevices = mutableListOf<String>()
    val wifiManager = remember { context.getSystemService(Context.WIFI_SERVICE) as WifiManager }
    var scanResults by remember { mutableStateOf<List<ScanResult>>(emptyList()) }

    if (ActivityCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) != PackageManager.PERMISSION_GRANTED
    ) {
        Log.d(
            "EyeAIUI",
            "[ConnectionPage] Canceling WIFI-Scan due to permissions not being granted."
        )
        return
    }

    val wifiScanState = rememberWifiScanState(context)

    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)

    for (device in availableAudioDevices) {
        val audioDeviceType = audioDeviceTypeName(device.type)
        val audioDeviceName = device.productName
        if (audioDeviceType != "Unbekannt")
            displayAudioDevices.add("$audioDeviceName ($audioDeviceType)")
    }
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
            "devices" to displayAudioDevices
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
            "devices" to wifiScanState.networks
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
        devicesData = devices[currentlyDisplayedDevices] as Map<Any, Any>
    )

}

@RequiresApi(Build.VERSION_CODES.Q)
@Composable
fun ChooseConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    goBack: () -> Unit,
    devicesData: Map<Any, Any>
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    var shouldRememberDevice by rememberSaveable { mutableStateOf(false) }
    var selectedDevice by remember { mutableIntStateOf(0) }
    var showConnectionFailedDialog by remember { mutableStateOf(false) }

    val shouldRememberKey = stringResource(devicesData["rememberKey"] as Int)
    val selectedDeviceKey = stringResource(devicesData["selectedKey"] as Int)
    val deviceCategory = devicesData["name"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val devices = devicesData["devices"] as? List<*> ?: emptyList<String>()


    Log.d("EyeAIUI", "[ConnectionPage] Choosing connection for $deviceCategory")
    LaunchedEffect(devicesData) {
        if (devicesData["remember"] == true && devices.contains(devicesData["selected"])) {
            Log.d(
                "EyeAIUI",
                "[ConnectionPage] Attempting to connect to remembered ${devicesData["type"]} device"
            )
            connectToDevice(
                context,
                devicesData["type"] as String,
                devicesData["selected"] as String
            )
            { success ->
                if (success) {
                    Log.d("EyeAIUI", "[ConnectionPage] Connection to remembered device successful")
                    onConnectionSuccessful()
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
                                        selectedDevice = index
                                    },
                                    isSelected = index == selectedDevice
                                )
                            }
                        }
                    } else {
                        Row(
                            modifier = Modifier
                                .padding(8.dp)
                                .fillMaxWidth()
                        ) { Text("Lade...") }

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
                            .padding(4.dp), verticalAlignment = Alignment.CenterVertically
                    ) {
                        Checkbox(
                            checked = shouldRememberDevice,
                            onCheckedChange = { shouldRememberDevice = !shouldRememberDevice })
                        Text("Als Standardgerät festlegen")
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
                                        "Mit Gerät " + if (devices.isNotEmpty()) devices[selectedDevice] else "" + " verbinden"
                                }, onClick = {
                                connectToDevice(
                                    context,
                                    devicesData["type"] as String,
                                    devices[selectedDevice] as String
                                )
                                { success ->
                                    if (success) {
                                        Log.d(
                                            "EyeAIUI",
                                            "[ConnectionPage] Setting SharedPreferences ShouldRememberDevice: $shouldRememberDevice"
                                        )
                                        sharedPreferences.edit(commit = true) {
                                            putBoolean(
                                                shouldRememberKey,
                                                shouldRememberDevice
                                            )
                                        }
                                        Log.d(
                                            "EyeAIUI",
                                            "[ConnectionPage] Setting SharedPreferences SelectedDevice: ${devices[selectedDevice]}"
                                        )
                                        sharedPreferences.edit(commit = true) {
                                            putString(
                                                selectedDeviceKey,
                                                devices[selectedDevice] as String
                                            )
                                        }
                                        shouldRememberDevice = false
                                        selectedDevice = 0
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
        ConnectionFailedDialog(onDismissed = { showConnectionFailedDialog = false })
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
                .fillMaxHeight()
                .clearAndSetSemantics {})
    }
}

@Composable
fun ConnectionFailedDialog(onDismissed: () -> Unit) {
    AlertDialog(
        onDismissRequest = { onDismissed() },
        title = { Text("Verbindung fehlgeschlagen") },
        text = { Text("Die Verbindung mit dem Gerät konnte nicht hergestellt werden. Versuchen Sie es noch einmal, oder wählen Sie ein anderes Gerät aus.") },
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

data class WifiScanState(
    val networks: List<String>,
    val rescan: () -> Unit
)

@RequiresApi(Build.VERSION_CODES.Q)
@Preview(showBackground = true, name = "ConnectionPagePreview")
@Composable
fun ConnectionPagePreview() {
    MaterialTheme {
        ConnectionPage(
            Modifier.fillMaxSize(),
            onConnectionSuccessful = {},
            onExitSelection = {}
        )
    }
}


