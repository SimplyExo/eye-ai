package com.algorithmic_alliance.eyeaiapp.UI

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.net.wifi.WifiNetworkSpecifier
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume
import java.util.concurrent.atomic.AtomicBoolean

private const val WIFI_SCAN_TIMEOUT_MS = 15_000L

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
            Log.d(
                LOG_TAG,
                "[ConnectionPageBackend] Connecting audio device: '$selectedDevice'",
            )
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val device = audioManager.availableCommunicationDevices.firstOrNull {
                selectedDevice == "${it.productName} (${audioDeviceTypeName(it.type)})"
            }
            if (device == null) {
                Log.d(LOG_TAG, "[ConnectionPageBackend] Selected audio device is no longer available")
                onResult(false)
                return
            }

            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            val connected = audioManager.setCommunicationDevice(device)
            Log.d(LOG_TAG, "[ConnectionPageBackend] Audio device connection success=$connected")
            onResult(connected)
        }

        "eye-ai-vision" -> {
            if (selectedDevice == PHONE_CAMERA_DEVICE) {
                setInputSource(context, R.string.input_is_camera, onEvent)
                onResult(true)
                return
            }

            Log.d(LOG_TAG, "[ConnectionPageBackend] Connecting EyeAI-Vision: '$selectedDevice'")
            connectToWifiNetwork(
                context = context,
                ssid = selectedDevice,
                password = "12345678",
                onConnected = {
                    setInputSource(context, R.string.input_is_eyeaivision, onEvent)
                    onResult(true)
                },
                onFailed = {
                    Log.d(LOG_TAG, "[ConnectionPageBackend] EyeAI-Vision connection failed")
                    onResult(false)
                },
            )
        }

        else -> onResult(false)
    }
}

private fun setInputSource(context: Context, sourceResource: Int, onEvent: (UIEvent) -> Unit) {
    PreferenceManager.getDefaultSharedPreferences(context).edit(commit = true) {
        putString(
            context.getString(R.string.input_source_setting),
            context.getString(sourceResource),
        )
    }
    // The ViewModel forwards this to the application-scoped runtime. No preview or lifecycle
    // object is handed to the connection UI.
    onEvent(UIEvent.UpdateSettings)
}

@RequiresApi(Build.VERSION_CODES.Q)
fun connectToWifiNetwork(
    context: Context,
    ssid: String,
    password: String,
    onConnected: () -> Unit,
    onFailed: () -> Unit,
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

    val completed = AtomicBoolean(false)
    fun complete(result: Boolean) {
        if (!completed.compareAndSet(false, true)) return
        mainHandler.post {
            if (result) onConnected() else onFailed()
        }
    }

    val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            Log.d(LOG_TAG, "[ConnectionPageBackend] Connected to $ssid")
            connectivityManager.bindProcessToNetwork(network)
            complete(true)
        }

        override fun onUnavailable() {
            Log.d(LOG_TAG, "[ConnectionPageBackend] Connection to $ssid unavailable")
            complete(false)
        }
    }

    try {
        connectivityManager.requestNetwork(request, networkCallback)
    } catch (exception: SecurityException) {
        Log.w(LOG_TAG, "[ConnectionPageBackend] Network request was rejected", exception)
        complete(false)
    }
}

data class WifiScanState(
    val networks: List<String>,
    val scanning: Boolean,
    val rescan: () -> Unit,
    val awaitScan: suspend () -> List<String>,
)

@Composable
fun rememberWifiScanState(
    context: Context,
    autoScanOnStart: Boolean = true,
): WifiScanState {
    val appContext = context.applicationContext
    val wifiManager = remember(appContext) {
        appContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
    }
    val scope = rememberCoroutineScope()
    var networks by remember { mutableStateOf<List<String>>(emptyList()) }
    var scanning by remember { mutableStateOf(false) }

    suspend fun performScan(): List<String> {
        if (
            ActivityCompat.checkSelfPermission(
                appContext,
                Manifest.permission.ACCESS_FINE_LOCATION,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            Log.d(LOG_TAG, "[ConnectionPageBackend] Skipping Wi-Fi scan without location permission")
            networks = emptyList()
            scanning = false
            return emptyList()
        }

        scanning = true
        val results = try {
            withTimeoutOrNull(WIFI_SCAN_TIMEOUT_MS) {
                scanWifiNetworks(appContext, wifiManager)
            } ?: wifiManager.scanResults.also {
                Log.d(LOG_TAG, "[ConnectionPageBackend] Wi-Fi scan timed out; using cached results")
            }
        } catch (exception: SecurityException) {
            Log.w(LOG_TAG, "[ConnectionPageBackend] Could not read Wi-Fi scan results", exception)
            emptyList()
        } finally {
            scanning = false
        }

        return results
            .asSequence()
            .map { it.SSID.trim() }
            .filter { it.contains("EyeAI-Vision", ignoreCase = true) }
            .distinct()
            .sorted()
            .toList()
            .also { networks = it }
    }

    val rescan: () -> Unit = { scope.launch { performScan() } }

    LaunchedEffect(autoScanOnStart) {
        if (autoScanOnStart) performScan()
    }

    return WifiScanState(
        networks = networks,
        scanning = scanning,
        rescan = rescan,
        awaitScan = { performScan() },
    )
}

private suspend fun scanWifiNetworks(
    context: Context,
    wifiManager: WifiManager,
): List<ScanResult> = suspendCancellableCoroutine { continuation ->
    lateinit var receiver: BroadcastReceiver
    fun unregisterReceiverSafely() {
        runCatching { context.unregisterReceiver(receiver) }
    }

    receiver = object : BroadcastReceiver() {
        override fun onReceive(receiverContext: Context, intent: Intent) {
            unregisterReceiverSafely()
            val results = if (
                ActivityCompat.checkSelfPermission(
                    context,
                    Manifest.permission.ACCESS_FINE_LOCATION,
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                wifiManager.scanResults
            } else {
                emptyList()
            }
            Log.d(LOG_TAG, "[ConnectionPageBackend] Wi-Fi scan completed: ${results.size} result(s)")
            if (continuation.isActive) continuation.resume(results)
        }
    }

    try {
        val filter = IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(receiver, filter)
        }
    } catch (exception: Exception) {
        Log.w(LOG_TAG, "[ConnectionPageBackend] Could not register Wi-Fi scan receiver", exception)
        if (continuation.isActive) continuation.resume(emptyList())
        return@suspendCancellableCoroutine
    }

    continuation.invokeOnCancellation { unregisterReceiverSafely() }
    val started = try {
        wifiManager.startScan()
    } catch (exception: SecurityException) {
        Log.w(LOG_TAG, "[ConnectionPageBackend] Could not start Wi-Fi scan", exception)
        false
    }
    if (!started) {
        unregisterReceiverSafely()
        val cachedResults = if (
            ActivityCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION,
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            wifiManager.scanResults
        } else {
            emptyList()
        }
        Log.d(LOG_TAG, "[ConnectionPageBackend] Wi-Fi scan throttled; using cached results")
        if (continuation.isActive) continuation.resume(cachedResults)
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
    return audioManager.availableCommunicationDevices.mapNotNull { device ->
        audioDeviceTypeName(device.type)
            .takeUnless { it == "Unbekannt" }
            ?.let { typeName -> "${device.productName} ($typeName)" }
    }
}

const val PHONE_CAMERA_DEVICE = "Handykamera verwenden"
