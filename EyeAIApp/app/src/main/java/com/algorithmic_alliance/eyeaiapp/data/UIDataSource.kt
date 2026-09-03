package com.algorithmic_alliance.eyeaiapp.data

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.core.content.edit
import androidx.annotation.RequiresApi
import androidx.camera.core.impl.utils.ContextUtil.getApplication
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.BuildInfoHelper
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.Settings
import kotlin.collections.listOf

object UIDataSource {

    const val ACTION_OPEN_DEVICE_MANAGER = "open_device_manager"
    const val ACTION_OPEN_BATTERY_OPTIMIZATION = "open_battery_optimization"


    const val INFORMATION_NOT_FOUND =
        "Die Information konnte nicht geladen werden. Wir bitten um Entschuldigung."

    val ICON_NOT_FOUND = R.drawable.error_24px

    const val RETURN_SEMANTIC = "Zurück."

    const val UI_LOG_TAG = "EyeAIUI"

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    val NEEDED_PERMISSIONS = listOf<Map<String, Any>>(
        /* EXPLANATION HOW TO ADD NEW PERMISSION
        mapOf(
            "permissionName": Name of the permission (String)
            "permission": The permission
            "permissionExplanation": Explain to the user why the app needs that permission (String)
            "icon": Icon matching that permission (Int)
            "iconDescription": Describing the Icon for the semantics (String)
            "permissionDeclineSemantic": semantic for the decline button (String),
            "permissionAcceptSemantic": semantic for the accept button (String),
            "confirmPermissionDeclineExplanation": Explain the effects of declining that permission for the conformation dialog (String)
            "confirmPermissionDeclineSemantic": semantic for confirm declining button (String)
        ),
         */
        mapOf(
            "permissionName" to "Kamera",
            "permissions" to listOf(Manifest.permission.CAMERA),
            "permissionExplanation" to """Damit die KI die Umgebung analysieren kann, ist es notwendig, dass die App auf die System-Kamera zugreifen kann. 
                |Die Kamerabilder werden genutzt, um Entfernungen zu Objekten zu bestimmen und um vorhandene Objekte im Raum zu erkennen. 
                |Diese Informationen werden dann per Audio ausgegeben.""".trimMargin(),
            "icon" to R.drawable.photo_camera_24px,
            "iconDescription" to "Kamera Icon",
            "permissionDeclineSemantic" to "Zugriff auf Kamera ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf Kamera gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf die Kamera ablehnen, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf Kamera trotzdem ablehnen. Die App wird geschlossen.",
            "hardPermission" to true
        ),
        mapOf(
            "permissionName" to "Mikrofon",
            "permissions" to listOf(Manifest.permission.RECORD_AUDIO),
            "permissionExplanation" to "Um per Sprachbefehl mit der App zu interagieren, ist es notwendig, zugriff auf das System-Mikrofon zu erteilen.",
            "icon" to R.drawable.mic_24px,
            "iconDescription" to "Mikrofon Icon",
            "permissionDeclineSemantic" to "Zugriff auf Mikrofon ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf Mikrofon gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf das Mikrofon ablehnen, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf Mikrofon trotzdem ablehnen. Die App wird geschlossen.",
            "hardPermission" to false
        ),
        mapOf(
            "permissionName" to "WLAN-Netzwerke erkennen",
            "permissions" to listOf(Manifest.permission.NEARBY_WIFI_DEVICES),
            "permissionExplanation" to "Um sich mir der EyeAI-Vision zu verbinden, ist es notwendig, dass die App zugriff auf nah gelegene WLAN-Geräte hat.  ",
            "icon" to R.drawable.wifi_24px,
            "iconDescription" to "WLAN Icon",
            "permissionDeclineSemantic" to "Zugriff auf nahe WLAN-Geräte ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf nahe WLAN-Geräte gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf nahe WLAN-Geräte, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf nahe WLAN-Geräte trotzdem ablehnen. Die App wird geschlossen.",
            "hardPermission" to true
        ),
        mapOf(
            "permissionName" to "Standort",
            "permissions" to listOf(Manifest.permission.ACCESS_FINE_LOCATION),
            "permissionExplanation" to "Da bei einem Scan der WLAN-Netzwerke über diese Informationen zum Standort anfallen können, muss die Berechtigung erteilt werden." +
                    "Die App speichert oder verwertet zu keinem Zeitpunkt Standortdaten.",
            "icon" to R.drawable.location_on_24px,
            "iconDescription" to "Standort Icon",
            "permissionDeclineSemantic" to "Zugriff auf Standort ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf Standort gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf den Standort, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf den Standort trotzdem ablehnen. Die App wird geschlossen.",
            "hardPermission" to true
        )
    )

    /* EXPLANATION HOW TO ADD NEW SETTING
    Categories:
        Every setting is part of a category. A category is just a  list of its sub settings.
        To add a new category you add a new map pair to APP_SETTINGS:
            "<Category Name>" to listOf(...)
    Settings:
        Every setting is represented by a map which contains the necessary information.
        All settings have these map pairs in common:
            "title"         Title of the setting
            "description"   Description of the setting
            "settingsType"  Which type of setting it is (e.g. checkbox, text input, ...)
            "string"        String to access SharedPreferences
        Depending on the setting type, the settings need different map pairs:
            checkbox setting:
                    -               Does not need extra pairs
            select setting:
                "settingsOptions"   List of all the available options
            slider setting:
                "settingsOptions"   List of minimum, maximum value of the slider
            textInput setting:
                    -               Does not need extra pairs
            file setting:
                    -               Does not need extra pairs
            info setting:
                    -               Does not need extra pairs

        To add a new setting just add a map with all the necessary pairs for that setting
        to a category.
     */

    val APP_SETTINGS = mapOf<String, Any>(
        "General" to listOf(
            mapOf(
                "title" to "Use NPU (experimental)",
                "description" to "Only enable on device with supported Qualcomm NPU. If no supported, performance will be worse!",
                "settingsType" to "checkbox",
                "string" to R.string.enable_npu_delegate_setting,
                "default" to true
            ),
            mapOf(
                "title" to "Batterieoptimierung",
                "description" to "Optional: EyeAI in den Android-Systemeinstellungen von der Akkuoptimierung ausnehmen. Das kann den Dauerbetrieb stabilisieren, erhöht aber den Akkuverbrauch.",
                "settingsType" to "click",
                "action" to ACTION_OPEN_BATTERY_OPTIMIZATION
            )
        ),
        "Depth Estimation" to listOf(
            mapOf(
                "title" to "Depth Estimation Model",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to EyeAIApp.DEPTH_MODELS.map { it.name },     //listOf("MiDaS V2.1", "MiDaS V2.1 (quantized)"),
                "string" to R.string.depth_model_setting,
                "default" to EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
            ),
            mapOf(
                "title" to "Enable Framerate Limiter",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.enable_depth_frame_rate_limit_setting,
                "default" to true
            ),
            mapOf(
                "title" to "Framerate Limit",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120),
                "string" to R.string.max_depth_frame_rate_setting,
                "default" to Settings.DEFAULT_FRAME_RATE_LIMIT
            )
        ),
        "Speech Recognition" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Speech Recognition enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.enable_speech_recognition_setting,
                "default" to true
            )
        ),
        "Audio Playback" to listOf(
            mapOf(
                "title" to "Enable Depth Playback",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.depth_playback_setting,
                "default" to true
            ),
            mapOf(
                "title" to "Enable Object Playback",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.object_playback_setting,
                "default" to true
            ),
            mapOf(
                "title" to "Audio-Frequency",
                "description" to "Controls frequency used for depth mapping",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 100, "max" to 4000),
                "string" to R.string.audio_frequency_range_setting,
                "default" to 500
            ),
            mapOf(
                "title" to "Frequency",
                "description" to "How often per second a sound will be audible",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 10),
                "string" to R.string.audio_playback_rate_setting,
                "default" to 2
            ),
            mapOf(
                "title" to "Audio language",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("english", "german"),
                "string" to R.string.object_playback_language,
                "default" to "english"
            ),
        ),
        "Device Manager" to listOf(
            mapOf(
                "title" to "Standartgeräte ändern",
                "description" to "",
                "settingsType" to "click",
                "action" to ACTION_OPEN_DEVICE_MANAGER
            )
        ),
        "Object Detection" to listOf(
            mapOf(
                "title" to "Enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.enable_object_detection_setting,
                "default" to true
            ),
            mapOf(
                "title" to "Enable Framerate Limiter",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.enable_object_detection_frame_rate_limit_setting,
                "default" to true
            ),
            mapOf(
                "title" to "Framerate Limit",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120),
                "string" to R.string.max_object_detection_frame_rate_setting,
                "default" to Settings.DEFAULT_FRAME_RATE_LIMIT
            )
        ),
        "OCR" to listOf(
            mapOf(
                "title" to "Enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.enable_ocr_setting,
                "default" to true
            ),
        ),
        "Input Source" to listOf(
            mapOf(
                "title" to "Input Source",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("camera", "media", "eyeaivision"),
                "string" to R.string.input_source_setting,
                "default" to "camera"
            ),
            mapOf(
                "title" to "Select Media File",
                "description" to "",
                "settingsType" to "file",
                "string" to R.string.media_path_setting,
                "default" to ""
            ),
            mapOf(
                "title" to "EyeAIVisionIP",
                "description" to "",
                "settingsType" to "textInput",
                "string" to R.string.eyeaivision_ip_setting,
                "default" to ""
            ),
            mapOf(
                "title" to "JPEG Compression (only EyeAIVision)",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 64),
                "string" to R.string.jpeg_compression,
                "default" to 15
            ),
        ),
        "Developer Settings" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Show Profiling Information",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.show_profiling_info_setting,
                "default" to false
            ),
            mapOf(
                "title" to "Show Debug Input Bitmap",
                "description" to "",
                "settingsType" to "checkbox",
                "string" to R.string.show_debug_input_bitmap_setting,
                "default" to false
            ),
        ),
        "Build Info" to listOf(
            mapOf(
                "title" to "App Version",
                "description" to BuildInfoHelper.getVersionInfo(),
                "settingsType" to "Info",
            ),
            mapOf(
                "title" to "Build Time",
                "description" to BuildInfoHelper.getFormattedBuildTime(),
                "settingsType" to "Info",
            ),
            mapOf(
                "title" to "Git Information",
                "description" to BuildInfoHelper.getGitInfo(),
                "settingsType" to "Info",
            ),
            mapOf(
                "title" to "Build Variant",
                "description" to BuildInfoHelper.getBuildVariant(),
                "settingsType" to "Info",
            )
        )

    )

}
