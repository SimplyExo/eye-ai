package com.algorithmic_alliance.eyeaiapp.runtime

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R

/** Foreground owner for the continuous EyeAI runtime and local CameraX source. */
class EyeAIRuntimeService : LifecycleService() {
    private lateinit var runtime: EyeAIRuntime
    private lateinit var wakeLock: EyeAIWakeLock

    override fun onCreate() {
        super.onCreate()
        runtime = (application as EyeAIApp).runtime
        wakeLock = EyeAIWakeLock(this)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // LifecycleService dispatches ON_START from its implementation of
        // onStartCommand. CameraX will keep service-bound use cases inactive
        // if this super call is skipped.
        super.onStartCommand(intent, flags, startId)

        if (intent?.action == ACTION_STOP) {
            stopRuntimeAndSelf()
            return START_NOT_STICKY
        }

        val serviceTypes = foregroundServiceTypes()
        if (serviceTypes == 0) {
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service has no permitted active input")
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        try {
            promoteToForeground(serviceTypes)

            // CPU continuity is needed only while the local continuous source
            // is active. FGS alone does not guarantee this after screen-off.
            if (usesCameraInput() && hasCameraPermission()) wakeLock.acquire()
            runtime.start(this)
        } catch (error: SecurityException) {
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service permission denied", error)
            wakeLock.release()
            stopSelfResult(startId)
        } catch (error: IllegalStateException) {
            // Includes a background-start rejection on Android 12+ and an
            // invalid foreground-service type. The caller must retry while
            // visible after resolving the permission/state issue.
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service could not start", error)
            wakeLock.release()
            stopSelfResult(startId)
        } catch (error: Throwable) {
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service startup failed", error)
            wakeLock.release()
            stopSelfResult(startId)
        }
        return START_NOT_STICKY
    }

    /**
     * Updates the active FGS type mask when a runtime setting changes while
     * this same service remains active (for example, microphone on/off).
     * Calling startForeground again is the Android-supported promotion/update
     * path; it does not create another service instance.
     */
    internal fun refreshForegroundTypes() {
        try {
            val serviceTypes = foregroundServiceTypes()
            if (serviceTypes == 0) {
                stopSelf()
                return
            }
            promoteToForeground(serviceTypes)
        } catch (error: SecurityException) {
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service type update denied", error)
            stopSelf()
        } catch (error: IllegalArgumentException) {
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service type update invalid", error)
            stopSelf()
        }
    }

    /**
     * A recents swipe is an explicit user stop for EyeAI. stopWithTask stays
     * false so Android delivers this callback and the runtime can release its
     * native, camera, audio and wake-lock resources before the service ends.
     */
    override fun onTaskRemoved(rootIntent: Intent?) {
        Log.i(EyeAIApp.APP_LOG_TAG, "EyeAI task removed; stopping continuous runtime")
        stopRuntimeAndSelf()
        super.onTaskRemoved(rootIntent)
    }

    override fun onDestroy() {
        try {
            runtime.stopOperation()
        } catch (error: Throwable) {
            // Cleanup must continue even if a native/audio component reports
            // an error while the service is being destroyed.
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI runtime shutdown failed", error)
        } finally {
            try {
                wakeLock.release()
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE)
                super.onDestroy()
            }
        }
    }

    /**
     * The notification action is an explicit user stop. It must not depend on
     * a start id because several idempotent UI start requests may have reached
     * this same service instance.
     */
    private fun stopRuntimeAndSelf() {
        try {
            runtime.stopOperation()
        } catch (error: Throwable) {
            Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI runtime explicit shutdown failed", error)
        } finally {
            wakeLock.release()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    private fun promoteToForeground(serviceTypes: Int) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, createNotification(), serviceTypes)
        } else {
            startForeground(NOTIFICATION_ID, createNotification())
        }
    }

    private fun foregroundServiceTypes(): Int {
        var types = 0
        if (usesCameraInput() && hasCameraPermission()) {
            types = types or if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA
            } else {
                0
            }
        }
        if (
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q &&
            (application as EyeAIApp).settings.enableSpeechRecognition &&
            hasRecordAudioPermission()
        ) {
            types = types or android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        }
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            types
        } else {
            val hasActiveInput =
                (usesCameraInput() && hasCameraPermission()) ||
                    ((application as EyeAIApp).settings.enableSpeechRecognition &&
                        hasRecordAudioPermission())
            if (hasActiveInput) 1 else 0
        }
    }

    private fun hasCameraPermission(): Boolean = ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.CAMERA,
    ) == PackageManager.PERMISSION_GRANTED

    private fun usesCameraInput(): Boolean =
        (application as EyeAIApp).settings.inputSource == getString(R.string.input_is_camera)

    private fun hasRecordAudioPermission(): Boolean = ContextCompat.checkSelfPermission(
        this,
        Manifest.permission.RECORD_AUDIO,
    ) == PackageManager.PERMISSION_GRANTED

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "EyeAI-Dauerbetrieb",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Kamera, Analyse und Audio laufen auch bei ausgeschaltetem Display."
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun createNotification(): Notification = NotificationCompat.Builder(this, CHANNEL_ID)
        .setSmallIcon(R.drawable.photo_camera_24px)
        .setContentTitle("EyeAI läuft")
        .setContentText("Kamera-Analyse und Audio sind im Dauerbetrieb aktiv")
        .setCategory(NotificationCompat.CATEGORY_SERVICE)
        .setPriority(NotificationCompat.PRIORITY_LOW)
        .setOngoing(true)
        .setOnlyAlertOnce(true)
        .addAction(
            R.drawable.stop_24px,
            "EyeAI beenden",
            stopPendingIntent(this),
        )
        .build()

    companion object {
        private const val CHANNEL_ID = "eyeai_runtime"
        private const val NOTIFICATION_ID = 4101
        private const val ACTION_STOP = "com.algorithmic_alliance.eyeaiapp.action.STOP_RUNTIME"

        /** Call only from a visible Activity/UI event. */
        fun startFromVisible(context: Context): Boolean {
            return try {
                ContextCompat.startForegroundService(
                    context,
                    Intent(context, EyeAIRuntimeService::class.java),
                )
                true
            } catch (error: SecurityException) {
                Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service permission denied", error)
                false
            } catch (error: IllegalStateException) {
                Log.e(EyeAIApp.APP_LOG_TAG, "EyeAI foreground service start rejected", error)
                false
            }
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, EyeAIRuntimeService::class.java))
        }

        private fun stopPendingIntent(context: Context): PendingIntent = PendingIntent.getService(
            context,
            4102,
            Intent(context, EyeAIRuntimeService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
    }
}
