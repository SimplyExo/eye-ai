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

/** Helper class that manages all app permissions, for now only camera permissions */
class PermissionManager(
	var activity: ComponentActivity,
	onCameraPermissionResult: (isGranted: Boolean) -> Unit
) {
	private val requestPermissionLauncher =
		activity.registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
			onCameraPermissionResult(isGranted)
		}

	fun requestCameraPermission() {
		requestPermissionLauncher.launch(Manifest.permission.CAMERA)
	}

	fun isCameraPermissionGranted(): Boolean {
		return ContextCompat.checkSelfPermission(activity, Manifest.permission.CAMERA) ==
			PackageManager.PERMISSION_GRANTED
	}

	fun openCameraPermissionSettings() {
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