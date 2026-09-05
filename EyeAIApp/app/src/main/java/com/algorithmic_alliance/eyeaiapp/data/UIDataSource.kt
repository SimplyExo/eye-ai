package com.algorithmic_alliance.eyeaiapp.data

import android.Manifest
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp
import com.algorithmic_alliance.eyeaiapp.BuildInfoHelper
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.Settings
import kotlin.collections.listOf

object UIDataSource {
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
            "permissionName" to R.string.camera_permission_name,
            "permissions" to listOf(Manifest.permission.CAMERA),
            "permissionExplanation" to R.string.camera_permission_explanation_text,
            "icon" to R.drawable.photo_camera_24px,
            "iconDescription" to R.string.camera_icon_description,
            "permissionDeclineSemantic" to R.string.camera_permission_decline_semantic,
            "permissionAcceptSemantic" to R.string.camera_permission_accept_semantic,
            "confirmPermissionDeclineExplanation" to R.string.camera_confirm_decline_explanation_text,
            "confirmPermissionDeclineSemantic" to R.string.camera_confirm_permission_decline_semantic,
            "hardPermission" to true
        ),
        mapOf(
            "permissionName" to R.string.microphone_permission_name,
            "permissions" to listOf(Manifest.permission.RECORD_AUDIO),
            "permissionExplanation" to R.string.microphone_permission_explanation_text,
            "icon" to R.drawable.mic_24px,
            "iconDescription" to R.string.microphone_icon_description,
            "permissionDeclineSemantic" to R.string.microphone_permission_decline_semantic,
            "permissionAcceptSemantic" to R.string.microphone_permission_accept_semantic,
            "confirmPermissionDeclineExplanation" to R.string.microphone_confirm_permission_decline_explanation_text,
            "confirmPermissionDeclineSemantic" to R.string.microphone_confirm_permission_decline_semantic,
            "hardPermission" to false
        ),
        mapOf(
            "permissionName" to R.string.wifi_permission_name,
            "permissions" to listOf(Manifest.permission.NEARBY_WIFI_DEVICES),
            "permissionExplanation" to R.string.wifi_permission_explanation_text,
            "icon" to R.drawable.wifi_24px,
            "iconDescription" to R.string.wifi_icon_description,
            "permissionDeclineSemantic" to R.string.wifi_permission_decline_semantic,
            "permissionAcceptSemantic" to R.string.wifi_permission_accept_semantic,
            "confirmPermissionDeclineExplanation" to R.string.wifi_confirm_decline_explanation_text,
            "confirmPermissionDeclineSemantic" to R.string.wifi_confirm_permission_decline_semantic,
            "hardPermission" to false
        ),
        mapOf(
            "permissionName" to R.string.location_permission_name,
            "permissions" to listOf(Manifest.permission.ACCESS_FINE_LOCATION),
            "permissionExplanation" to R.string.location_permission_explanation_text,
            "icon" to R.drawable.location_on_24px,
            "iconDescription" to R.string.location_icon_description,
            "permissionDeclineSemantic" to R.string.location_permission_decline_semantic,
            "permissionAcceptSemantic" to R.string.location_permission_accept_semantic,
            "confirmPermissionDeclineExplanation" to R.string.location_confirm_decline_explanation_text,
            "confirmPermissionDeclineSemantic" to R.string.location_confirm_permission_decline_semantic,
            "hardPermission" to false
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

    val APP_SETTINGS = mapOf<Int, Any>(
        R.string.settings_category_general to listOf(
            mapOf(
                "title" to R.string.setting_use_npu_title,
                "description" to R.string.setting_use_npu_description,
                "settingsType" to "checkbox",
                "string" to R.string.enable_npu_delegate_setting,
                "default" to true
            )
        ),
        R.string.settings_category_depth_estimation to listOf(
            mapOf(
                "title" to R.string.setting_depth_estimation_model_title,
                "description" to R.string.setting_depth_estimation_model_description,
                "settingsType" to "select",
                "settingsOptions" to EyeAIApp.DEPTH_MODELS.map { it.name },
                "string" to R.string.depth_model_setting,
                "default" to EyeAIApp.DEFAULT_DEPTH_MODEL_NAME
            ),
            mapOf(
                "title" to R.string.setting_enable_framerate_limiter_title,
                "description" to R.string.setting_enable_depth_framerate_limiter_description,
                "settingsType" to "checkbox",
                "string" to R.string.enable_depth_frame_rate_limit_setting,
                "default" to true
            ),
            mapOf(
                "title" to R.string.setting_framerate_limit_title,
                "description" to R.string.setting_depth_framerate_limit_description,
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120),
                "string" to R.string.max_depth_frame_rate_setting,
                "default" to Settings.DEFAULT_FRAME_RATE_LIMIT
            )
        ),
        R.string.settings_category_speech_recognition to listOf(
            mapOf(
                "title" to R.string.setting_speech_recognition_enabled_title,
                "description" to R.string.setting_speech_recognition_description,
                "settingsType" to "checkbox",
                "string" to R.string.enable_speech_recognition_setting,
                "default" to true
            )
        ),
        R.string.settings_category_audio_playback to listOf(
            mapOf(
                "title" to R.string.setting_enable_depth_playback_title,
                "description" to R.string.setting_enable_depth_playback_description,
                "settingsType" to "checkbox",
                "string" to R.string.depth_playback_setting,
                "default" to true
            ),
            mapOf(
                "title" to R.string.setting_enable_object_playback_title,
                "description" to R.string.setting_enable_object_playback_description,
                "settingsType" to "checkbox",
                "string" to R.string.object_playback_setting,
                "default" to true
            ),
            mapOf(
                "title" to R.string.setting_audio_frequency_title,
                "description" to R.string.setting_audio_frequency_description,
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 100, "max" to 4000),
                "string" to R.string.audio_frequency_range_setting,
                "default" to 500
            ),
            mapOf(
                "title" to R.string.setting_frequency_title,
                "description" to R.string.setting_frequency_description,
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 10),
                "string" to R.string.audio_playback_rate_setting,
                "default" to 2
            ),
            mapOf(
                "title" to R.string.setting_audio_language_title,
                "description" to R.string.setting_audio_language_description,
                "settingsType" to "select",
                "settingsOptions" to listOf(
                    R.string.language_is_english,
                    R.string.language_is_german
                ),
                "string" to R.string.object_playback_language,
                "default" to R.string.language_is_english
            )
        ),
        R.string.settings_category_device_manager to listOf(
            mapOf(
                "title" to R.string.setting_change_default_devices_title,
                "description" to R.string.setting_change_default_devices_description,
                "settingsType" to "click"
            )
        ),
        R.string.settings_category_object_detection to listOf(
            mapOf(
                "title" to R.string.setting_enabled_title,
                "description" to R.string.setting_object_detection_description,
                "settingsType" to "checkbox",
                "string" to R.string.enable_object_detection_setting,
                "default" to true
            ),
            mapOf(
                "title" to R.string.setting_enable_framerate_limiter_title,
                "description" to R.string.setting_enable_object_framerate_limiter_description,
                "settingsType" to "checkbox",
                "string" to R.string.enable_object_detection_frame_rate_limit_setting,
                "default" to true
            ),
            mapOf(
                "title" to R.string.setting_framerate_limit_title,
                "description" to R.string.setting_object_framerate_limit_description,
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 120),
                "string" to R.string.max_object_detection_frame_rate_setting,
                "default" to Settings.DEFAULT_FRAME_RATE_LIMIT
            )
        ),
        R.string.settings_category_input_source to listOf(
            mapOf(
                "title" to R.string.setting_input_source_title,
                "description" to R.string.setting_input_source_description,
                "settingsType" to "select",
                "settingsOptions" to listOf(
                    R.string.input_is_camera,
                    R.string.input_is_media,
                    R.string.input_is_eyeaivision
                ),
                "string" to R.string.input_source_setting,
                "default" to R.string.input_is_camera
            ),
            mapOf(
                "title" to R.string.setting_select_media_file_title,
                "description" to R.string.setting_select_media_file_description,
                "settingsType" to "file",
                "string" to R.string.media_path_setting,
                "default" to ""
            ),
            mapOf(
                "title" to R.string.setting_eyeaivision_ip_title,
                "description" to R.string.setting_eyeaivision_ip_description,
                "settingsType" to "textInput",
                "string" to R.string.eyeaivision_ip_setting,
                "default" to ""
            ),
            mapOf(
                "title" to R.string.setting_jpeg_compression_title,
                "description" to R.string.setting_jpeg_compression_description,
                "settingsType" to "slider",
                "settingsOption" to mapOf("min" to 1, "max" to 64),
                "string" to R.string.jpeg_compression,
                "default" to 15
            )
        ),
        R.string.settings_category_developer to listOf(
            mapOf(
                "title" to R.string.setting_show_profiling_information_title,
                "description" to R.string.setting_show_profiling_information_description,
                "settingsType" to "checkbox",
                "string" to R.string.show_profiling_info_setting,
                "default" to false
            ),
            mapOf(
                "title" to R.string.setting_show_debug_input_bitmap_title,
                "description" to R.string.setting_show_debug_input_bitmap_description,
                "settingsType" to "checkbox",
                "string" to R.string.show_debug_input_bitmap_setting,
                "default" to false
            )
        ),
        R.string.settings_category_build_info to listOf(
            mapOf(
                "title" to R.string.setting_app_version_title,
                "description" to BuildInfoHelper.getVersionInfo(),
                "settingsType" to "Info",
            ),
            mapOf(
                "title" to R.string.setting_build_time_title,
                "description" to BuildInfoHelper.getFormattedBuildTime(),
                "settingsType" to "Info"
            ),
            mapOf(
                "title" to R.string.setting_git_information_title,
                "description" to BuildInfoHelper.getGitInfo(),
                "settingsType" to "Info"
            ),
            mapOf(
                "title" to R.string.setting_build_variant_title,
                "description" to BuildInfoHelper.getBuildVariant(),
                "settingsType" to "Info"
            )
        )
    )



}

object Spacing {
    val xs = 4.dp
    val sm = 8.dp
    val md = 16.dp
    val lg = 24.dp
    val xl = 32.dp
    val xxl = 48.dp
    val xxxl = 64.dp
    val xxxxl = 128.dp
}

object AppElevation {
    val level0 = 0.dp
    val level1 = 1.dp   // ruhende Cards
    val level2 = 3.dp   // aktive/hervorgehobene Cards
    val level3 = 6.dp   // Dialoge, wichtige Highlight-Elemente
    val level4 = 8.dp   // Navigation Drawer
    val level5 = 12.dp  // Bottom Sheets, FAB gedrückt
}

val PremiumShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),   // Chips, kleine Badges
    small = RoundedCornerShape(10.dp),       // Buttons, Textfelder
    medium = RoundedCornerShape(16.dp),      // Cards
    large = RoundedCornerShape(24.dp),       // große Container, Bottom Sheets
    extraLarge = RoundedCornerShape(32.dp)   // Hero-Elemente, Modals
)