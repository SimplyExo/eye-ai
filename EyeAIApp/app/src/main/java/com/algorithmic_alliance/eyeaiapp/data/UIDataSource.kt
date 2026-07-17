package com.algorithmic_alliance.eyeaiapp.data

import com.algorithmic_alliance.eyeaiapp.BuildInfoHelper
import com.algorithmic_alliance.eyeaiapp.R
import kotlin.collections.listOf

object UIDataSource {
    val NEEDED_PERMISSIONS = listOf<Map<String, Any>>(
        /* EXPLANATION HOW TO ADD NEW PERMISSION
        mapOf(
            "permissionName": Name of the permission (String)
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
            "permissionName" to "Mikrophon",
            "permissionExplanation" to "Um per Sprachbefehl mit der App zu interagieren, ist es notwendig, zugriff auf das System-Mikrophon zu erteilen.",
            "icon" to R.drawable.mic_24px,
            "iconDescription" to "Mikrophon Icon",
            "permissionDeclineSemantic" to "Zugriff auf Mikrophon ablehnen.",
            "permissionAcceptSemantic" to "Zugriff auf Mikrophon gestatten.",
            "confirmPermissionDeclineExplanation" to "Wenn Sie den Zugriff auf das Mikrophon ablehnen, können sie die App nicht benutzen.",
            "confirmPermissionDeclineSemantic" to "Zugriff auf Mikrophon trotzdem ablehnen. Die App wird geschlossen."
        )
        //TODO implement the rest of the permissions
    )

    const val INFORMATION_NOT_FOUND =
        "Die Information konnte nicht geladen werden. Wir bitten um Entschuldigung."

    val ICON_NOT_FOUND = R.drawable.error_24px

    const val RETURN_SEMANTIC = "Zurück."


    val APP_SETTINGS = mapOf<String, Any>(
        "General" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Use NPU (experimental)",
                "description" to "Only enable on device with supported Qualcomm NPU. If no supported, performance will be worse!",
                "settingsType" to "checkbox",
                "default" to false
            )
        ),
        "Debugging" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Show Profiling Information",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to false
            ),
            mapOf(
                "title" to "Show Debug Input Bitmap",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to false
            )
        ),
        "Depth Estimation" to listOf(
            mapOf(
                "title" to "Depth Estimation Model",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("MiDaS V2.1", "MiDaSV2.1 (quantized)")
            ),
            mapOf(
                "title" to "Enable Framerate Limiter",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to false
            ),
            mapOf(
                "title" to "Framerate Limit",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120, "default" to 30)
            )
        ),
        "Speech Recognition" to listOf<Map<String, Any>>(
            mapOf(
                "title" to "Speech Recognition enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true

            )
        ),
        "Audio Playback" to listOf(
            mapOf(
                "title" to "Enable Depth Playback",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true
            ),
            mapOf(
                "title" to "Enable Object Playback",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true
            ),
            mapOf(
                "title" to "Frequency",
                "description" to "Controls frequency used for depth mapping",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 100, "max" to 4000, "default" to 500)
            ),
            mapOf(
                "title" to "Frequency",
                "description" to "How often per second a sound will be audible",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 10, "default" to 3)
            ),
            mapOf(
                "title" to "Audio language",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("English", "Deutsch")
            ),
        ),
        "LLM" to listOf(
            mapOf(
                "title" to "Google AI Studio API Key",
                "description" to "",
                "settingsType" to "textInput",
            ),
            mapOf(
                "title" to "Custom Google Gen AI Studio endpoint (for testing/mocking)",
                "description" to "",
                "settingsType" to "textInput",
            ),
        ),
        "Object Detection" to listOf(
            mapOf(
                "title" to "Enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true
            ),
            mapOf(
                "title" to "Enable Framerate Limiter",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true
            ),
            mapOf(
                "title" to "Framerate Limit",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120, "default" to 30)
            )
        ),
        "OCR" to listOf(
            mapOf(
                "title" to "Enabled",
                "description" to "",
                "settingsType" to "checkbox",
                "default" to true
            ),
        ),
        "Input Source" to listOf(
            mapOf(
                "title" to "Input Source",
                "description" to "",
                "settingsType" to "select",
                "settingsOptions" to listOf("Kamera", "Media", "EyeAIVision")
            ),
            mapOf(
                "title" to "Select Media File",
                "description" to "",
                "settingsType" to "file",
            ),
            mapOf(
                "title" to "EyeAIVisionIP",
                "description" to "",
                "settingsType" to "textInput",
            ),
            mapOf(
                "title" to "JPEG Compression (only EyeAIVision)",
                "description" to "",
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 64, "default" to 15)
            ),
        ),
        "Build Info" to listOf(
            mapOf(
                "title" to "App Version",
                "description" to BuildInfoHelper.getVersionInfo(),
                "settingsType" to "Info"
            ),
            mapOf(
                "title" to "Build Time",
                "description" to BuildInfoHelper.getFormattedBuildTime(),
                "settingsType" to "Info"
            ),
            mapOf(
                "title" to "Git Information",
                "description" to BuildInfoHelper.getGitInfo(),
                "settingsType" to "Info"
            ),
            mapOf(
                "title" to "Build Variant",
                "description" to BuildInfoHelper.getBuildVariant(),
                "settingsType" to "Info"
            )
        )

    )

}