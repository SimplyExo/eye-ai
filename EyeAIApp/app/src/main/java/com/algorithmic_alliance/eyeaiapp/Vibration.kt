package com.algorithmic_alliance.eyeaiapp

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager


fun vibrate(context: Context, milliseconds: Long) {
	/** even index: delay before, uneven index: duration
	 * patterns with only one vibration don't always work -> second fake vibration */
	val pattern = longArrayOf(0, milliseconds, 0, 0)

	if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
		val vibratorManager =
			context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
		val vibrator = vibratorManager.defaultVibrator
		vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
	} else {
		@Suppress("DEPRECATION") val vibrator =
			context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
		vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
	}
}