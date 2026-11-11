package com.algorithmic_alliance.eyeaiapp.ui

import android.Manifest
import android.content.Context
import android.os.Build

internal fun visionPermissionsForApi(api: Int): Array<String> = buildList {
	add(Manifest.permission.ACCESS_FINE_LOCATION)
	if (api >= 33) add(Manifest.permission.NEARBY_WIFI_DEVICES)
	// Android 17 separates joining WiFi from sending TCP/UDP traffic on the LAN.
	if (api >= 37) add(Manifest.permission.ACCESS_LOCAL_NETWORK)
}.toTypedArray()

internal fun hasVisionPermissions(context: Context): Boolean =
	visionPermissionsForApi(Build.VERSION.SDK_INT).all { hasPermission(context, it) }
