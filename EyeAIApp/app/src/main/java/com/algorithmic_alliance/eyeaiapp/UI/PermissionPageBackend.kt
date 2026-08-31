package com.algorithmic_alliance.eyeaiapp.UI

import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

/*
If the user declines a permission, the app has to check which features need that permission and
disables these
 */
fun onPermissionDecline(
    permissionData: Map<String, Any>,
    onExitPermissionSelection: () -> Unit,
    context: Context,
    onEvent: (UIEvent) -> Unit,
    onPermissionDecline: () -> Unit
) {

    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    if (permissionData["hardPermission"] as Boolean) {
        onExitPermissionSelection()
    } else {
        when (permissionData["permissionName"]) {
            "Mikrofon" -> {
                sharedPreferences.edit(commit = true) {
                    putBoolean(context.getString(R.string.enable_speech_recognition_setting), false)
                }
                onEvent(UIEvent.UpdateSettings)
                onPermissionDecline()
            }
        }
    }
}

fun hasPermission(context: Context, permission: String): Boolean {
    return ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
}

fun checkPermissionsStatus(
    neededPermissions: List<Map<String, Any>>,
    context: Context,
    onEvent: (UIEvent) -> Unit
): List<Map<String, Any>> {
    val notGrantedPermissions = mutableListOf<Map<String, Any>>()

    for (map in neededPermissions) {
        for (permission in map["permissions"] as List<*>) {
            val hasPermission = ContextCompat.checkSelfPermission(
                context,
                permission as String
            ) == PackageManager.PERMISSION_GRANTED
            if (!hasPermission) {
                Log.d(LOG_TAG, "[PermissionPage] App does not have permission for $permission")
                notGrantedPermissions.add(map)
                continue
            } else {
                Log.d(LOG_TAG, "[PermissionPage] App already has permission for $permission")
            }
        }
    }
    return notGrantedPermissions
}