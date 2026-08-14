package com.algorithmic_alliance.eyeaiapp.data

import android.Manifest
import android.os.Build
import androidx.annotation.RequiresApi
import com.algorithmic_alliance.eyeaiapp.BuildInfoHelper
import com.algorithmic_alliance.eyeaiapp.R
import kotlin.collections.listOf

object UIDataSource {


    const val INFORMATION_NOT_FOUND =
        "Die Information konnte nicht geladen werden. Wir bitten um Entschuldigung."

    val ICON_NOT_FOUND = R.drawable.error_24px

    const val RETURN_SEMANTIC = "Zurück."

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
            "confirmPermissionDeclineSemantic" to "Zugriff auf Kamera trotzdem ablehnen. Die App wird geschlossen."
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
            "confirmPermissionDeclineSemantic" to "Zugriff auf Mikrofon trotzdem ablehnen. Die App wird geschlossen."
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
            "confirmPermissionDeclineSemantic" to "Zugriff auf nahe WLAN-Geräte trotzdem ablehnen. Die App wird geschlossen."
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
            "confirmPermissionDeclineSemantic" to "Zugriff auf den Standort trotzdem ablehnen. Die App wird geschlossen."
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
            "function"      Here you add whatever needs to happen if the setting is changed
        Depending on the setting type, the settings need diffrent map pairs:
            checkbox setting:
                "default"           Default state of the setting (on/off)
            select setting:
                "settingsOptions"   List of all the available options (note: 1. enty is the default)
            slider setting:
                "settingsOptions"   List of minimum, maximum and default value of the slider
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
                "default" to false,
                "function" to {}
            )
        ),
        "Depth Estimation" to listOf(
            mapOf(
                "title" to "Depth Estimation Model",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("MiDaS V2.1", "MiDaSV2.1 (quantized)"),
                "function" to {}
            ),
            mapOf(
                "title" to "Enable Framerate Limiter",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to false,
                "function" to {}
            ),
            mapOf(
                "title" to "Framerate Limit",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120, "default" to 30),
                "function" to {}
            )
        ),
        "Speech Recognition" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Speech Recognition enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true,
                "function" to {}

            )
        ),
        "Audio Playback" to listOf(
            mapOf(
                "title" to "Enable Depth Playback",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true,
                "function" to {}
            ),
            mapOf(
                "title" to "Enable Object Playback",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true,
                "function" to {}
            ),
            mapOf(
                "title" to "Frequency",
                "description" to "Controls frequency used for depth mapping",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 100, "max" to 4000, "default" to 500),
                "function" to {}
            ),
            mapOf(
                "title" to "Frequency",
                "description" to "How often per second a sound will be audible",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 10, "default" to 3),
                "function" to {}
            ),
            mapOf(
                "title" to "Audio language",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("English", "Deutsch"),
                "function" to {}
            ),
        ),
        "LLM" to listOf(
            mapOf(
                "title" to "Google AI Studio API Key",
                "description" to "",
                "settingsType" to "textInput",
                "function" to {}
            ),
            mapOf(
                "title" to "Custom Google Gen AI Studio endpoint (for testing/mocking)",
                "description" to "",
                "settingsType" to "textInput",
                "function" to {}
            ),
        ),
        "Object Detection" to listOf(
            mapOf(
                "title" to "Enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true,
                "function" to {}
            ),
            mapOf(
                "title" to "Enable Framerate Limiter",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true,
                "function" to {}
            ),
            mapOf(
                "title" to "Framerate Limit",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120, "default" to 30),
                "function" to {}
            )
        ),
        "OCR" to listOf(
            mapOf(
                "title" to "Enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true,
                "function" to {}
            ),
        ),
        "Input Source" to listOf(
            mapOf(
                "title" to "Input Source",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("Kamera", "Media", "EyeAIVision"),
                "function" to {}
            ),
            mapOf(
                "title" to "Select Media File",
                "description" to "",
                "settingsType" to "file",
                "function" to {}
            ),
            mapOf(
                "title" to "EyeAIVisionIP",
                "description" to "",
                "settingsType" to "textInput",
                "function" to {}
            ),
            mapOf(
                "title" to "JPEG Compression (only EyeAIVision)",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 64, "default" to 15),
                "function" to {}
            ),
        ),
        "Developer Settings" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Show Profiling Information",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to false,
                "function" to {}
            ),
            mapOf(
                "title" to "Show Debug Input Bitmap",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to false,
                "function" to {}
            ),
        ),
        "Build Info" to listOf(
            mapOf(
                "title" to "App Version",
                "description" to BuildInfoHelper.getVersionInfo(),
                "settingsType" to "Info",
                "function" to {}
            ),
            mapOf(
                "title" to "Build Time",
                "description" to BuildInfoHelper.getFormattedBuildTime(),
                "settingsType" to "Info",
                "function" to {}
            ),
            mapOf(
                "title" to "Git Information",
                "description" to BuildInfoHelper.getGitInfo(),
                "settingsType" to "Info",
                "function" to {}
            ),
            mapOf(
                "title" to "Build Variant",
                "description" to BuildInfoHelper.getBuildVariant(),
                "settingsType" to "Info",
                "function" to {}
            )
        )

    )

}