package com.algorithmic_alliance.eyeaiapp.UI

import android.content.Context
import android.media.AudioDeviceInfo
import androidx.core.content.edit
import android.net.ConnectivityManager
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiNetworkSpecifier
import android.os.Handler
import android.os.Looper
import androidx.annotation.RequiresApi
import android.Manifest
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioManager
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresPermission
import com.algorithmic_alliance.eyeaiapp.R
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.app.ActivityCompat
import androidx.preference.PreferenceManager
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

@RequiresApi(Build.VERSION_CODES.S)
fun connectToDevice(
    context: Context,
    deviceCategory: String,
    selectedDevice: String,
    onEvent: (UIEvent) -> Unit,
    onResult: (Boolean) -> Unit,
) {
    when (deviceCategory) {
        "audio" -> {
            //TODO Backend zu OpenAL
            Log.d(
                LOG_TAG,
                "[ConnectionPageBackend.connectToDevice] Attempting to connect to audio device: '$selectedDevice'"
            )
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            val availableAudioDevices = audioManager.availableCommunicationDevices
            for (device in availableAudioDevices) {
                if (selectedDevice == "${device.productName} (${audioDeviceTypeName(device.type)})") {
                    if (audioManager.setCommunicationDevice(device)) {
                        Log.d(
                            LOG_TAG,
                            "[ConnectionPageBackend.connectToDevice] setCommunicationDevice success=true, device=${device.productName}"
                        )
                        onResult(true)
                    } else {
                        Log.d(
                            LOG_TAG,
                            "[ConnectionPageBackend.connectToDevice] setCommunicationDevice success=false, device=${device.productName}"
                        )
                        onResult(false)
                    }
                }
            }
        }

        "eye-ai-vision" -> {
            //TODO input device setting
            if (selectedDevice != "Handykamera verwenden") {
                Log.d(
                    LOG_TAG,
                    "[ConnectionPageBackend.connectToDevice] Attempting to connect to eye-ai-vision device '$selectedDevice'"
                )
                connectToWifiNetwork(
                    context, selectedDevice, "12345678",
                    onConnected = {
                        Log.d(
                            LOG_TAG,
                            "[ConnectionPageBackend.connectToDevice] Connection successful"
                        )
                        val sharedPreferences =
                            PreferenceManager.getDefaultSharedPreferences(context)
                        sharedPreferences.edit(commit = true) {
                            putString(
                                context.getString(R.string.input_source_setting),
                                "eyeaivision"
                            )
                        }
                        onEvent(UIEvent.UpdateSettings)
                        onResult(true)
                    },
                    onFailed = {
                        Log.d(LOG_TAG, "[ConnectionPageBackend.connectToDevice] Connection failed")
                        onResult(false)
                    }
                )
            } else {
                val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
                sharedPreferences.edit(commit = true) {
                    putString(context.getString(R.string.input_source_setting), "camera")
                }
                onEvent(UIEvent.UpdateSettings)
                Log.d(
                    LOG_TAG,
                    "[ConnectionPageBackend.connectToDevice] User choose phone camera over eye-ai-vison"
                )
                onResult(true)
            }

        }

        else -> onResult(false)
    }
}

@RequiresApi(Build.VERSION_CODES.Q)
fun connectToWifiNetwork(
    context: Context,
    ssid: String,
    password: String,
    onConnected: () -> Unit,
    onFailed: () -> Unit
) {
    val mainHandler = Handler(Looper.getMainLooper())

    val specifier = WifiNetworkSpecifier.Builder()
        .setSsid(ssid)
        .setWpa2Passphrase(password)
        .build()

    val request = NetworkRequest.Builder()
        .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
        .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
        .setNetworkSpecifier(specifier)
        .build()

    val connectivityManager =
        context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager

    val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            Log.d(LOG_TAG, "[ConnectionPageBackend.connectToWifiNetwork] Verbunden mit $ssid")
            connectivityManager.bindProcessToNetwork(network)
            mainHandler.post { onConnected() }
        }

        override fun onUnavailable() {
            Log.d(
                LOG_TAG,
                "[ConnectionPageBackend.connectToWifiNetwork] Verbindung zu $ssid fehlgeschlagen"
            )
            mainHandler.post { onFailed() }
        }
    }

    connectivityManager.requestNetwork(request, networkCallback)
}

data class WifiScanState(
    val networks: List<String>,
    val rescan: () -> Unit,
    val awaitScan: suspend () -> List<String>
)

@Composable
fun rememberWifiScanState(
    context: Context,
    autoScanOnStart: Boolean = true,
    setScannState: (Boolean) -> Unit
): WifiScanState {
    val wifiManager = remember { context.getSystemService(Context.WIFI_SERVICE) as WifiManager }
    var networks by remember { mutableStateOf<List<String>>(emptyList()) }
    val scope = rememberCoroutineScope()

    @RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
    suspend fun performScan(): List<String> {
        setScannState(true)
        if (ActivityCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            val filtered = scanWifiNetworks(context, wifiManager)
                .filter { it.SSID.contains("EyeAI-Vision") }
                .map { it.SSID }
            networks = filtered          // UI-Liste aktualisieren
            setScannState(false)
            return filtered               // direkt an den Aufrufer zurückgeben – keine Race Condition
        }else{
            Log.d(LOG_TAG, "[ConnectionPageBackend.rememberWifiScanState] Scan failed, missing permission")
            return emptyList()
        }

    }

    val rescan: () -> Unit = { scope.launch { performScan() } }

    LaunchedEffect(Unit) {
        if (autoScanOnStart) performScan()
    }

    return WifiScanState(networks, rescan, awaitScan = { performScan() })
}

@RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
suspend fun scanWifiNetworks(
    context: Context,
    wifiManager: WifiManager
): List<ScanResult> = suspendCancellableCoroutine { cont ->
    val receiver = object : BroadcastReceiver() {
        override fun onReceive(ctx: Context, intent: Intent) {
            try {
                context.unregisterReceiver(this)
            } catch (_: Exception) {
            }
            if (ActivityCompat.checkSelfPermission(
                    context, Manifest.permission.ACCESS_FINE_LOCATION
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                if (cont.isActive) cont.resume(emptyList())
                return
            }
            val results = wifiManager.scanResults
            Log.d(LOG_TAG, "[ConnectionPageBackend.scanWifiNetworks] Scan abgeschlossen, ${results.size} Ergebnisse")
            if (cont.isActive) cont.resume(results)
        }
    }

    val filter = IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
        context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
    } else {
        context.registerReceiver(receiver, filter)
    }

    cont.invokeOnCancellation {
        try {
            context.unregisterReceiver(receiver)
        } catch (_: Exception) {
        }
    }

    val started = wifiManager.startScan()
    if (!started) {
        Log.d(LOG_TAG, "[ConnectionPageBackend.scanWifiNetworks] Scan throttled, nutze Cache")
        try {
            context.unregisterReceiver(receiver)
        } catch (_: Exception) {
        }
        if (cont.isActive) cont.resume(wifiManager.scanResults)
    }
}

fun audioDeviceTypeName(type: Int): String = when (type) {
    AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "Eingebauter Lautsprecher"
    AudioDeviceInfo.TYPE_WIRED_HEADSET -> "Kabelgebundenes Headset (mit Mikrofon)"
    AudioDeviceInfo.TYPE_WIRED_HEADPHONES -> "Kabelgebundene Kopfhörer (Klinke)"
    AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "Bluetooth-Headset (Anruf/Sprache)"
    AudioDeviceInfo.TYPE_BLE_HEADSET -> "Bluetooth LE Kopfhörer"
    AudioDeviceInfo.TYPE_USB_HEADSET -> "USB-Kopfhörer"
    AudioDeviceInfo.TYPE_USB_DEVICE -> "USB-Audiogerät"
    AudioDeviceInfo.TYPE_HEARING_AID -> "Hörgerät"
    AudioDeviceInfo.TYPE_DOCK -> "Dockingstation"
    AudioDeviceInfo.TYPE_HDMI -> "HDMI"
    else -> "Unbekannt"
}

@RequiresApi(Build.VERSION_CODES.S)
@Composable
fun rememberAudioDeviceState(context: Context): Pair<List<String>, () -> Unit> {
    var devices by remember { mutableStateOf(getAvailableAudioDevices(context)) }
    val refresh: () -> Unit = { devices = getAvailableAudioDevices(context) }
    return devices to refresh
}

@RequiresApi(Build.VERSION_CODES.S)
fun getAvailableAudioDevices(context: Context): List<String> {
    val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    val availableAudioDevices = audioManager.availableCommunicationDevices

    val displayAudioDevices = mutableListOf<String>()
    for (device in availableAudioDevices) {
        val audioDeviceType = audioDeviceTypeName(device.type)
        val audioDeviceName = device.productName
        if (audioDeviceType != "Unbekannt")
            displayAudioDevices.add("$audioDeviceName ($audioDeviceType)")
    }
    return displayAudioDevices
}