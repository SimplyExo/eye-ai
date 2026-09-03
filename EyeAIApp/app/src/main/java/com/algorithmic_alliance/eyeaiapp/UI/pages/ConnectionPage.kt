package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.IntentSender
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.PHONE_CAMERA_DEVICE
import com.algorithmic_alliance.eyeaiapp.UI.ShimmerBox
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.UIState
import com.algorithmic_alliance.eyeaiapp.UI.connectToDevice
import com.algorithmic_alliance.eyeaiapp.UI.rememberAudioDeviceState
import com.algorithmic_alliance.eyeaiapp.UI.rememberShimmerBrush
import com.algorithmic_alliance.eyeaiapp.UI.rememberWifiScanState
import com.algorithmic_alliance.eyeaiapp.data.Shapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import com.google.android.gms.common.api.ResolvableApiException
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.LocationSettingsRequest
import com.google.android.gms.location.Priority

private const val AUDIO_DEVICE_TYPE = "audio"
private const val VISION_DEVICE_TYPE = "eye-ai-vision"

private data class ConnectionCategory(
    val name: String,
    val type: String,
    val rememberKey: Int,
    val selectedKey: Int,
)

@RequiresApi(Build.VERSION_CODES.S)
@Composable
fun ConnectionPage(
    modifier: Modifier = Modifier,
    onConnectionSuccessful: () -> Unit,
    onExitSelection: () -> Unit,
    uiState: UIState,
    onEvent: (UIEvent) -> Unit,
) {
    BackHandler(onBack = onExitSelection)

    val context = LocalContext.current
    val preferences = PreferenceManager.getDefaultSharedPreferences(context)
    val categories = listOf(
        ConnectionCategory(
            name = "Audio-Gerät",
            type = AUDIO_DEVICE_TYPE,
            rememberKey = R.string.remember_audio_device,
            selectedKey = R.string.selected_audio_device,
        ),
        ConnectionCategory(
            name = "EyeAI-Vision",
            type = VISION_DEVICE_TYPE,
            rememberKey = R.string.remember_eye_ai_vision,
            selectedKey = R.string.selected_eye_ai_vision,
        ),
    )
    var currentCategoryIndex by rememberSaveable { mutableIntStateOf(0) }

    val category = categories[currentCategoryIndex]
    val remembered = preferences.getBoolean(context.getString(category.rememberKey), false)
    val rememberedDevice = preferences.getString(context.getString(category.selectedKey), "").orEmpty()

    ChooseConnectionPage(
        modifier = modifier,
        category = category,
        remembered = remembered,
        rememberedDevice = rememberedDevice,
        connectionTutorialCompleted = uiState.connectionTutorialCompleted,
        onConnectionSuccessful = {
            if (currentCategoryIndex < categories.lastIndex) {
                currentCategoryIndex += 1
            } else {
                onConnectionSuccessful()
            }
        },
        goBack = {
            if (currentCategoryIndex > 0) currentCategoryIndex -= 1 else onExitSelection()
        },
        onEvent = onEvent,
    )
}

@RequiresApi(Build.VERSION_CODES.S)
@Composable
private fun ChooseConnectionPage(
    modifier: Modifier,
    category: ConnectionCategory,
    remembered: Boolean,
    rememberedDevice: String,
    connectionTutorialCompleted: Boolean,
    onConnectionSuccessful: () -> Unit,
    goBack: () -> Unit,
    onEvent: (UIEvent) -> Unit,
) {
    val context = LocalContext.current
    val preferences = PreferenceManager.getDefaultSharedPreferences(context)
    val shouldRememberKey = stringResource(category.rememberKey)
    val selectedDeviceKey = stringResource(category.selectedKey)
    val hasLocationPermission = ActivityCompat.checkSelfPermission(
        context,
        Manifest.permission.ACCESS_FINE_LOCATION,
    ) == PackageManager.PERMISSION_GRANTED
    val wifiScanState = rememberWifiScanState(context, autoScanOnStart = false)
    val (audioDevices, refreshAudioDevices) = rememberAudioDeviceState(context)
    val devices = if (category.type == AUDIO_DEVICE_TYPE) audioDevices else wifiScanState.networks
    val shimmerBrush = rememberShimmerBrush(
        backgroundColor = MaterialTheme.colorScheme.primaryContainer,
        contrastColor = MaterialTheme.colorScheme.onPrimaryContainer,
    )

    var shouldRememberDevice by rememberSaveable(category.type) { mutableStateOf(remembered) }
    var selectedDevice by rememberSaveable(category.type) { mutableStateOf("") }
    var showConnectionFailedDialog by rememberSaveable(category.type) { mutableStateOf(false) }
    var showLocationDisabledDialog by rememberSaveable(category.type) { mutableStateOf(false) }
    var showLocationPermissionDialog by rememberSaveable(category.type) { mutableStateOf(false) }
    var pageLoading by rememberSaveable(category.type) {
        mutableStateOf(!connectionTutorialCompleted && remembered && rememberedDevice.isNotBlank())
    }

    fun requestVisionScan() {
        when {
            !hasLocationPermission -> showLocationPermissionDialog = true
            isLocationEnabled(context) -> wifiScanState.rescan()
            else -> showLocationDisabledDialog = true
        }
    }

    LaunchedEffect(category.type, connectionTutorialCompleted, remembered, rememberedDevice) {
        if (connectionTutorialCompleted || !remembered || rememberedDevice.isBlank()) {
            pageLoading = false
            if (category.type == VISION_DEVICE_TYPE) requestVisionScan()
            return@LaunchedEffect
        }

        val rememberedDeviceIsAvailable = when (category.type) {
            AUDIO_DEVICE_TYPE -> rememberedDevice in audioDevices
            VISION_DEVICE_TYPE -> when {
                rememberedDevice == PHONE_CAMERA_DEVICE -> true
                !hasLocationPermission -> {
                    showLocationPermissionDialog = true
                    false
                }
                !isLocationEnabled(context) -> {
                    showLocationDisabledDialog = true
                    false
                }
                else -> rememberedDevice in wifiScanState.awaitScan()
            }
            else -> false
        }

        if (!rememberedDeviceIsAvailable) {
            Log.d(LOG_TAG, "[ConnectionPage] Remembered ${category.type} device is unavailable")
            pageLoading = false
            return@LaunchedEffect
        }

        connectToDevice(
            context = context,
            deviceCategory = category.type,
            selectedDevice = rememberedDevice,
            onEvent = onEvent,
        ) { success ->
            if (success) {
                onConnectionSuccessful()
            } else {
                pageLoading = false
                showConnectionFailedDialog = true
            }
        }
    }

    Surface(modifier = modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(Spacing.md),
            contentAlignment = Alignment.Center,
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = Shapes.medium,
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    contentColor = MaterialTheme.colorScheme.onPrimaryContainer,
                ),
            ) {
                if (pageLoading) {
                    ConnectionLoadingContent(shimmerBrush)
                } else {
                    Column {
                        Text(
                            text = "${category.name} auswählen",
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(Spacing.md)
                                .clearAndSetSemantics {
                                    contentDescription = "${category.name} auswählen"
                                },
                            style = MaterialTheme.typography.headlineLarge,
                            textAlign = TextAlign.Center,
                        )
                        HorizontalDivider(
                            color = MaterialTheme.colorScheme.outline,
                            modifier = Modifier.padding(horizontal = Spacing.sm),
                        )

                        if (devices.isNotEmpty()) {
                            LazyColumn(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = Spacing.xxxxl * 3)
                                    .padding(horizontal = Spacing.sm),
                            ) {
                                items(devices, key = { it }) { device ->
                                    DeviceListEntry(
                                        deviceName = device,
                                        isSelected = selectedDevice == device,
                                        onSelected = { selectedDevice = device },
                                    )
                                }
                                if (category.type == VISION_DEVICE_TYPE) {
                                    item(key = PHONE_CAMERA_DEVICE) {
                                        DeviceListEntry(
                                            deviceName = PHONE_CAMERA_DEVICE,
                                            isSelected = selectedDevice == PHONE_CAMERA_DEVICE,
                                            onSelected = { selectedDevice = PHONE_CAMERA_DEVICE },
                                        )
                                    }
                                }
                            }
                        } else {
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(Spacing.md),
                            ) {
                                if (wifiScanState.scanning && category.type == VISION_DEVICE_TYPE) {
                                    ShimmerBox(
                                        brush = shimmerBrush,
                                        modifier = Modifier
                                            .fillMaxWidth(0.7f)
                                            .height(Spacing.md),
                                    )
                                } else {
                                    Text(
                                        text = "Keine verfügbaren Geräte gefunden.",
                                        style = MaterialTheme.typography.bodyMedium,
                                    )
                                }
                                if (category.type == VISION_DEVICE_TYPE) {
                                    DeviceListEntry(
                                        deviceName = PHONE_CAMERA_DEVICE,
                                        isSelected = selectedDevice == PHONE_CAMERA_DEVICE,
                                        onSelected = { selectedDevice = PHONE_CAMERA_DEVICE },
                                    )
                                }
                            }
                        }

                        HorizontalDivider(
                            color = MaterialTheme.colorScheme.outline,
                            modifier = Modifier.padding(horizontal = Spacing.sm),
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(
                                    start = Spacing.sm,
                                    end = Spacing.lg,
                                    top = Spacing.xs,
                                    bottom = Spacing.xs,
                                ),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Checkbox(
                                    checked = shouldRememberDevice,
                                    onCheckedChange = { shouldRememberDevice = it },
                                )
                                Text("Standardgerät", style = MaterialTheme.typography.bodyLarge)
                            }
                            Icon(
                                painter = painterResource(R.drawable.refresh_24px),
                                contentDescription = "Geräteliste aktualisieren",
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier
                                    .width(Spacing.xl)
                                    .height(Spacing.xl)
                                    .clickable {
                                        if (category.type == AUDIO_DEVICE_TYPE) {
                                            refreshAudioDevices()
                                        } else {
                                            requestVisionScan()
                                        }
                                    },
                            )
                        }
                        HorizontalDivider(
                            color = MaterialTheme.colorScheme.outline,
                            modifier = Modifier.padding(horizontal = Spacing.sm),
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(Spacing.md),
                            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
                        ) {
                            Button(modifier = Modifier.weight(1f), onClick = goBack) {
                                Text("Zurück", style = MaterialTheme.typography.labelLarge)
                            }
                            Button(
                                modifier = Modifier
                                    .weight(1f)
                                    .semantics {
                                        contentDescription = "Mit Gerät $selectedDevice verbinden"
                                    },
                                enabled = selectedDevice.isNotBlank(),
                                onClick = {
                                    connectToDevice(
                                        context = context,
                                        deviceCategory = category.type,
                                        selectedDevice = selectedDevice,
                                        onEvent = onEvent,
                                    ) { success ->
                                        if (!success) {
                                            showConnectionFailedDialog = true
                                            return@connectToDevice
                                        }
                                        preferences.edit(commit = true) {
                                            putBoolean(shouldRememberKey, shouldRememberDevice)
                                            putString(
                                                selectedDeviceKey,
                                                selectedDevice.takeIf { shouldRememberDevice }.orEmpty(),
                                            )
                                        }
                                        onConnectionSuccessful()
                                    }
                                },
                            ) {
                                Text(
                                    "Verbinden",
                                    modifier = Modifier.clearAndSetSemantics {},
                                    style = MaterialTheme.typography.labelLarge,
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    if (showConnectionFailedDialog) {
        ConnectionErrorDialog(
            title = "Verbindung fehlgeschlagen",
            message = "Das ausgewählte Gerät konnte nicht verbunden werden.",
            onDismissed = { showConnectionFailedDialog = false },
        )
    }
    if (showLocationPermissionDialog) {
        ConnectionErrorDialog(
            title = "Standortberechtigung erforderlich",
            message = "Android benötigt die Standortberechtigung, um WLAN-Geräte in der Nähe zu finden.",
            onDismissed = { showLocationPermissionDialog = false },
        )
    }
    if (showLocationDisabledDialog) {
        ActivateLocationServicesDialog(
            onDismissed = { showLocationDisabledDialog = false },
            onLocationServicesActivated = { wifiScanState.rescan() },
        )
    }
}

private fun isLocationEnabled(context: Context): Boolean {
    val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
    return locationManager?.isLocationEnabled == true
}

@Composable
private fun DeviceListEntry(
    deviceName: String,
    isSelected: Boolean,
    onSelected: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onSelected)
            .semantics {
                contentDescription = "Gerät: $deviceName. " + if (isSelected) {
                    "Das Gerät ist ausgewählt."
                } else {
                    "Das Gerät ist nicht ausgewählt."
                }
            },
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(
            selected = isSelected,
            onClick = onSelected,
        )
        Text(deviceName, modifier = Modifier.clearAndSetSemantics {})
    }
}

@Composable
private fun ConnectionLoadingContent(shimmerBrush: androidx.compose.ui.graphics.Brush) {
    Column(modifier = Modifier.padding(Spacing.md)) {
        ShimmerBox(
            brush = shimmerBrush,
            modifier = Modifier
                .fillMaxWidth(0.7f)
                .height(Spacing.xxxl)
                .align(Alignment.CenterHorizontally),
        )
        HorizontalDivider(modifier = Modifier.padding(vertical = Spacing.sm))
        repeat(3) {
            ShimmerBox(
                brush = shimmerBrush,
                modifier = Modifier
                    .fillMaxWidth(0.65f)
                    .height(Spacing.md)
                    .padding(vertical = Spacing.xs),
            )
        }
    }
}

@Composable
private fun ActivateLocationServicesDialog(
    onDismissed: () -> Unit,
    onLocationServicesActivated: () -> Unit,
) {
    val context = LocalContext.current
    val locationRequest = remember {
        LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 10_000L).build()
    }
    val locationSettingsRequest = remember(locationRequest) {
        LocationSettingsRequest.Builder().addLocationRequest(locationRequest).build()
    }
    val settingsClient = remember(context) { LocationServices.getSettingsClient(context) }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartIntentSenderForResult(),
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) onLocationServicesActivated()
        onDismissed()
    }

    fun activateLocationServices() {
        settingsClient.checkLocationSettings(locationSettingsRequest)
            .addOnSuccessListener {
                onLocationServicesActivated()
                onDismissed()
            }
            .addOnFailureListener { exception ->
                if (exception is ResolvableApiException) {
                    try {
                        launcher.launch(IntentSenderRequest.Builder(exception.resolution).build())
                    } catch (sendException: IntentSender.SendIntentException) {
                        Log.w(LOG_TAG, "[ConnectionPage] Could not open location settings", sendException)
                        onDismissed()
                    }
                } else {
                    onDismissed()
                }
            }
    }

    AlertDialog(
        onDismissRequest = onDismissed,
        title = { Text("Standortdienste sind ausgeschaltet") },
        text = {
            Text(
                "Android benötigt aktivierte Standortdienste, um WLAN-Netzwerke in der Nähe zu " +
                    "finden. EyeAI verwendet dabei nicht Ihren Standort.",
            )
        },
        confirmButton = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            ) {
                Button(modifier = Modifier.weight(1f), onClick = onDismissed) {
                    Text("Zurück")
                }
                Button(modifier = Modifier.weight(1f), onClick = ::activateLocationServices) {
                    Text("Aktivieren")
                }
            }
        },
    )
}

@Composable
private fun ConnectionErrorDialog(title: String, message: String, onDismissed: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismissed,
        title = { Text(title) },
        text = { Text(message) },
        confirmButton = {
            Button(onClick = onDismissed) {
                Text("Verstanden")
            }
        },
    )
}

@RequiresApi(Build.VERSION_CODES.S)
@Preview(showBackground = true, name = "ConnectionPagePreview")
@Composable
private fun ConnectionPagePreview() {
    MaterialTheme {
        ConnectionPage(
            modifier = Modifier.fillMaxSize(),
            onConnectionSuccessful = {},
            onExitSelection = {},
            uiState = UIState(),
            onEvent = {},
        )
    }
}
