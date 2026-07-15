package com.algorithmic_alliance.eyeaiapp.UI

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
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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

    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
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
}

@Composable
fun ChooseConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    goBack: () -> Unit,
    devicesData: Map<String, Any>
) {


    var selectedDevice by remember { mutableIntStateOf(0) }
    var showConnectionFailedDialog by remember { mutableStateOf(false) }

    val deviceCategory = devicesData["name"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val devices = devicesData["devices"] as? List<*> ?: emptyList<String>()


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
                        fontSize = 30.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                HorizontalDivider(modifier = Modifier.padding(8.dp))
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
                    Button(modifier = Modifier.weight(1f), onClick = {
                        //TODO Verbindung mit Backend
                        //check here if connection attempt was successful
                        if (true){
                            selectedDevice = 0
                            onConnectionSuccessful()
                        } else
                            showConnectionFailedDialog = true
                    }) { Text("Verbinden") }
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
    Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        RadioButton(onClick = { onSelected() }, selected = isSelected)
        Text(deviceName, Modifier.fillMaxHeight())
    }
}

@Composable
fun ConnectionFailedDialog(onDismissed: () -> Unit) {
    AlertDialog(
        onDismissRequest = { onDismissed() },
        title = { Text("Verbindung fehlgeschlagen") },
        text = { Text("Die Verbindung mit dem Gerät konnte nicht hergestellt werden. Versuchen Sie es noch einmal, oder wählen Sie ein anderes Gerät aus.") },
        confirmButton = { Button(onClick = { onDismissed() }) { Text("Verstanden") } })

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