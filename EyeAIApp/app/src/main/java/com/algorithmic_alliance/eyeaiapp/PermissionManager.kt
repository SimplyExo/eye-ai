package com.algorithmic_alliance.eyeaiapp

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat

/** Helper class that manages all app permissions: camera and microphone for now */
class PermissionManager(
	var activity: ComponentActivity,
	onCameraPermissionResult: (isGranted: Boolean) -> Unit,
	onMicrophonePermissionResult: (isGranted: Boolean) -> Unit,
	onBluetoothPermissionResult: (isGranted: Boolean) -> Unit,
	onBluetoothConnectPermissionResult: (isGranted: Boolean) -> Unit
) {
	private val requestPermissionsLauncher =
		activity.registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
			if (permissions.containsKey(Manifest.permission.CAMERA)) {
				onCameraPermissionResult(
					permissions.getOrDefault(
						Manifest.permission.CAMERA,
						false
					)
				)
			}
			if (permissions.containsKey(Manifest.permission.RECORD_AUDIO)) {
				onMicrophonePermissionResult(
					permissions.getOrDefault(
						Manifest.permission.RECORD_AUDIO,
						false
					)
				)
			}
			if (permissions.containsKey(Manifest.permission.BLUETOOTH)) {
				onBluetoothPermissionResult(
					permissions.getOrDefault(
						Manifest.permission.BLUETOOTH,
						false
					)
				)
			}
			if (permissions.containsKey(Manifest.permission.BLUETOOTH_CONNECT)) {
				onBluetoothConnectPermissionResult(
					permissions.getOrDefault(
						Manifest.permission.BLUETOOTH_CONNECT,
						false
					)
				)
			}
		}

	fun requestCameraPermission() {
		requestPermissionsLauncher.launch(arrayOf(Manifest.permission.CAMERA))
	}

	fun requestMicrophonePermission() {
		requestPermissionsLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
	}

	fun requestBluetoothPermission() {
		requestPermissionsLauncher.launch(arrayOf(Manifest.permission.BLUETOOTH))
	}

	fun requestBluetoothConnectPermission() {
		requestPermissionsLauncher.launch(arrayOf(Manifest.permission.BLUETOOTH_CONNECT))
	}


	fun isCameraPermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) ==
			PackageManager.PERMISSION_GRANTED
	}

	fun isMicrophonePermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) ==
			PackageManager.PERMISSION_GRANTED
	}

	fun isBluetoothPermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH) ==
			PackageManager.PERMISSION_GRANTED
	}

	fun isBluetoothConnectPermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.BLUETOOTH_CONNECT) ==
			PackageManager.PERMISSION_GRANTED
	}

	fun openAppPermissionSettings() {
		val intent =
			Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
				data =
					Uri.fromParts(
						"package",
						activity.packageName,
						null
					)
				flags = Intent.FLAG_ACTIVITY_NEW_TASK
			}

		try {
			activity.startActivity(intent)
		} catch (_: ActivityNotFoundException) {
			val fallbackIntent =
				Intent(Settings.ACTION_MANAGE_APPLICATIONS_SETTINGS).apply {
					flags = Intent.FLAG_ACTIVITY_NEW_TASK
				}
			activity.startActivity(fallbackIntent)
		}
	}
}