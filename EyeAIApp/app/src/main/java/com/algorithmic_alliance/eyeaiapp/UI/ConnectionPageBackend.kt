package com.algorithmic_alliance.eyeaiapp.UI

import android.content.Context
import android.media.AudioDeviceInfo
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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.core.app.ActivityCompat

@RequiresApi(Build.VERSION_CODES.S)
fun connectToDevice(
    context: Context,
    deviceCategory: String,
    selectedDevice: String,
    onResult: (Boolean) -> Unit
) {
    when (deviceCategory) {
        "audio" -> {
            //TODO Backend zu OpenAL
            Log.d(
                LOG_TAG,
                "[ConnectionPage.connect] Attempting to connect to audio device: '$selectedDevice'"
            )
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            val availableAudioDevices = audioManager.availableCommunicationDevices
            for (device in availableAudioDevices){
                if (selectedDevice == "${device.productName} (${audioDeviceTypeName(device.type)})"){
                    if(audioManager.setCommunicationDevice(device)){
                        Log.d(LOG_TAG, "[ConnectToDevice] setCommunicationDevice success=true, device=${device.productName}")
                        onResult(true)
                    }else{
                        Log.d(LOG_TAG, "[ConnectToDevice] setCommunicationDevice success=false, device=${device.productName}")
                        onResult(false)
                    }
                }
            }
        }

        "eye-ai-vision" -> {
            //TODO input device setting
            if(selectedDevice != "Handykamera verwenden"){
                Log.d(
                    LOG_TAG,
                    "[ConnectionPage.connect] Attempting to connect to eye-ai-vision device '$selectedDevice'"
                )
                connectToWifiNetwork(
                    context, selectedDevice, "12345678",
                    onConnected = {
                        Log.d(LOG_TAG, "[ConnectionPage.connect] Connection successful")
                        onResult(true)
                    },
                    onFailed = {
                        Log.d(LOG_TAG, "[ConnectionPage.connect] Connection failed")
                        onResult(false)
                    }
                )
            } else{
                Log.d(LOG_TAG, "[ConnectionPage.connect] User choose phone camera over eye-ai-vison")
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
            Log.d(LOG_TAG, "[ConnectToWifiNetwork] Verbunden mit $ssid")
            connectivityManager.bindProcessToNetwork(network)
            mainHandler.post { onConnected() }
        }

        override fun onUnavailable() {
            Log.d(LOG_TAG, "[ConnectToWifiNetwork] Verbindung zu $ssid fehlgeschlagen")
            mainHandler.post { onFailed() }
        }
    }

    connectivityManager.requestNetwork(request, networkCallback)
}

data class WifiScanState(
    val networks: List<String>,
    val rescan: () -> Unit
)

@Composable
fun rememberWifiScanState(context: Context, autoScanOnStart: Boolean = true): WifiScanState {
    val wifiManager = remember { context.getSystemService(Context.WIFI_SERVICE) as WifiManager }
    var scanResults by remember { mutableStateOf<List<ScanResult>>(emptyList()) }

    val rescan: () -> Unit = remember(wifiManager) {
        { triggerWifiScan(context, wifiManager) { results -> scanResults = results } }
    }

    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            @RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
            override fun onReceive(ctx: Context, intent: Intent) {
                if (ActivityCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                    != PackageManager.PERMISSION_GRANTED
                ) return
                scanResults = wifiManager.scanResults
                Log.d(LOG_TAG, "[WifiScanState] Found WIFI-Networks: ${wifiManager.scanResults}")
            }
        }

        val filter = IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(receiver, filter)
        }

        if(autoScanOnStart) rescan()

        onDispose {
            context.unregisterReceiver(receiver)
        }
    }

    val networks = remember(scanResults) {
        scanResults.filter { it.SSID.contains("EyeAI-Vision") }.map { it.SSID }
    }

    return WifiScanState(networks, rescan)
}

@RequiresPermission(Manifest.permission.ACCESS_FINE_LOCATION)
fun triggerWifiScan(
    context: Context,
    wifiManager: WifiManager,
    onCachedResultsAvailable: (List<ScanResult>) -> Unit
) {
    val started = wifiManager.startScan()
    if (!started) {
        Log.d(LOG_TAG, "[WifiScanState] Scan throttled, showing cached results")
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
        onCachedResultsAvailable(wifiManager.scanResults)
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
fun getAvailableAudioDevices(context: Context): List<String>{
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