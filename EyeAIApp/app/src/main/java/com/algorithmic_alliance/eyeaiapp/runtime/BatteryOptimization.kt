package com.algorithmic_alliance.eyeaiapp.runtime

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.os.PowerManager
import android.provider.Settings
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp

/** User-controlled access to Android's battery-optimization settings. */
object BatteryOptimization {
    fun isExempt(context: Context): Boolean {
        val powerManager = context.getSystemService(PowerManager::class.java)
        return powerManager.isIgnoringBatteryOptimizations(context.packageName)
    }

    /**
     * Opens the system-managed list instead of silently or directly requesting
     * an exemption. The user remains in control and no exemption permission is
     * required by EyeAI.
     */
    fun openSettings(context: Context): Boolean {
        val intent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
        if (context !is Activity) intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        return try {
            context.startActivity(intent)
            true
        } catch (error: ActivityNotFoundException) {
            Log.w(
                EyeAIApp.APP_LOG_TAG,
                "Battery optimization settings are unavailable on this device",
                error,
            )
            false
        } catch (error: SecurityException) {
            Log.w(
                EyeAIApp.APP_LOG_TAG,
                "Battery optimization settings could not be opened",
                error,
            )
            false
        }
    }
}
