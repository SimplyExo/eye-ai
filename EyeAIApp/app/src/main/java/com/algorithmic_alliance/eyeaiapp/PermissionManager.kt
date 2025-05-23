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
	onMicrophonePermissionResult: (isGranted: Boolean) -> Unit
) {
	private val requestPermissionsLauncher =
		activity.registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
			onCameraPermissionResult(permissions.getOrDefault(Manifest.permission.CAMERA, false))
			onMicrophonePermissionResult(
				permissions.getOrDefault(
					Manifest.permission.RECORD_AUDIO,
					false
				)
			)
		}

	fun requestPermissions() {
		requestPermissionsLauncher.launch(
			arrayOf(
				Manifest.permission.CAMERA,
				Manifest.permission.RECORD_AUDIO
			)
		)
	}

	fun isCameraPermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) ==
			PackageManager.PERMISSION_GRANTED
	}

	fun isMicrophonePermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) ==
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