package com.algorithmic_alliance.eyeaiapp.UI

import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
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
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.paneTitle
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import kotlin.collections.emptyList


@Composable
fun ConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
) {


    //TODO connection to Backend
    val devices: List<Map<String, Any>> = listOf(
        mapOf(
            "name" to "Audio-Gerät",
            "devices" to listOf("Wireless Earbuds", "AirPods", "Headphone Jack")
        ),
        mapOf(
            "name" to "EyeAI-Vision",
            "devices" to listOf("EyeAI-Vision 1", "EyeAI-Vision von Robert")
        )
    )

    var currentlyDisplayedDevices by remember { mutableIntStateOf(0) }



    ChooseConnectionPage(
        onConnectionSuccessful = {
            if (currentlyDisplayedDevices < devices.size - 1)
                currentlyDisplayedDevices++
            else onConnectionSuccessful()
        },
        goBack = { if (currentlyDisplayedDevices != 0) currentlyDisplayedDevices-- },
        devicesData = devices[currentlyDisplayedDevices]
    )

}

@Composable
fun ChooseConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    goBack: () -> Unit,
    devicesData: Map<String, Any>
) {

    var announcement by remember { mutableStateOf("") }

    var selectedDevice by remember { mutableIntStateOf(0) }
    var showConnectionFailedDialog by remember { mutableStateOf(false) }

    val deviceCategory = devicesData["name"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val devices = devicesData["devices"] as? List<*> ?: emptyList<String>()


    announcement =
        if (deviceCategory == "Audio-Gerät")
            "Auf dieser Seite können Sie das Audio-Gerät für die Audio-Ausgabe wählen."
        else
            "Auf dieser Seite können sie die EyeAI-Vision zum Verbinden auswählen."


    Surface(
        modifier = Modifier
            .fillMaxSize()
            .semantics { paneTitle = announcement },
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
                    LazyColumn(
                        modifier = Modifier
                            .padding(8.dp)
                            .fillMaxWidth()
                    ) {
                        items(items = devices) { item ->
                            val index = devices.indexOf(item)
                            DeviceListEntry(
                                item as String,
                                onSelected = { selectedDevice = index },
                                isSelected = index == selectedDevice
                            )
                        }
                    }
                    HorizontalDivider(modifier = Modifier.padding(8.dp))
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
                                        "Mit Gerät " + devices[selectedDevice] + " verbinden"
                                }, onClick = {
                                //TODO Verbindung mit Backend
                                //check here if connection attempt was successful
                                if (true) {
                                    selectedDevice = 0
                                    onConnectionSuccessful()
                                } else
                                    showConnectionFailedDialog = true
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

@Preview(showBackground = true, name = "ConnectionPagePreview")
@Composable
fun ConnectionPagePreview() {
    MaterialTheme {
        ConnectionPage(
            Modifier.fillMaxSize(),
            onConnectionSuccessful = {},
        )
    }
}