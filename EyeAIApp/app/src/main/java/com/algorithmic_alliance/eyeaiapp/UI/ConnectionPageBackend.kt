package com.algorithmic_alliance.eyeaiapp.UI

import android.content.Context
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
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresPermission
import com.algorithmic_alliance.eyeaiapp.R
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
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
            onResult(true)
        }

        "eye-ai-vision" -> {
            //TODO input device setting
            if (selectedDevice != context.getString(R.string.choose_camera_as_input_text)) {
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
                                context.getString(R.string.input_is_eyeaivision)
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
                    putString(context.getString(R.string.input_source_setting), context.getString(R.string.input_is_camera))
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