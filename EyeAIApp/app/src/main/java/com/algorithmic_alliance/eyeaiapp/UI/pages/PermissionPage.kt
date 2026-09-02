package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.Manifest
import android.app.Activity
import android.content.Context
import androidx.core.content.edit
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import androidx.annotation.RequiresApi
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.checkPermissionsStatus
import com.algorithmic_alliance.eyeaiapp.UI.onPermissionDecline
import com.algorithmic_alliance.eyeaiapp.data.Spacing

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@Composable
fun PermissionPage(
    modifier: Modifier = Modifier,
    onPermissionsGranted: () -> Unit,
    onPermissionsDeclined: () -> Unit,
    onEvent: (UIEvent) -> Unit
) {
    Log.d(LOG_TAG, "[PermissionPage] Loading PermissionPage")
    val context = LocalContext.current

    val neededPermissions = UIDataSource.NEEDED_PERMISSIONS
    val notGrantedPermissions =
        checkPermissionsStatus(neededPermissions, context, onEvent = onEvent)

    if (notGrantedPermissions.isEmpty()) {
        Log.d(LOG_TAG, "[PermissionPage] All permissions already granted. Exiting PermissionPage")
        onPermissionsGranted()
        return
    }

    var currentPermission by rememberSaveable { mutableIntStateOf(0) }

    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
        Column(modifier = Modifier.fillMaxHeight(), verticalArrangement = Arrangement.Center) {
            AskForPermission(
                modifier,
                notGrantedPermissions[currentPermission],
                onPermissionAccepted = { if (currentPermission < notGrantedPermissions.size - 1) currentPermission++ else onPermissionsGranted() },
                onExitPermissionPage = { onPermissionsDeclined() }, onEvent = onEvent
            )
        }

    }

}

@Composable
fun AskForPermission(
    modifier: Modifier = Modifier,
    permissionData: Map<String, Any>,
    onPermissionAccepted: () -> Unit,
    onExitPermissionPage: () -> Unit,
    onEvent: (UIEvent) -> Unit
) {
    val context = LocalContext.current
    Log.d(
        LOG_TAG,
        "[PermissionPage.AskForPermission] Asking for permission ${permissionData["permissions"]}"
    )
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            onPermissionAccepted()
            Log.d(
                LOG_TAG,
                "[PermissionPage.AskForPermission] Permission ${permissionData["permissions"]} granted"
            )
        } else {
            Log.d(
                LOG_TAG,
                "[PermissionPage.AskForPermission] Permission ${permissionData["permissions"]} declined"
            )
            onPermissionDecline(
                permissionData,
                onExitPermissionSelection = onExitPermissionPage,
                context,
                onEvent = onEvent,
                onPermissionDecline = onPermissionAccepted
            )

        }
    }

    var showDeclineDialog by rememberSaveable { mutableStateOf(false) }

    val permissionExplanation =
        permissionData["permissionExplanation"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val permissionIcon = permissionData["icon"] ?: UIDataSource.ICON_NOT_FOUND
    val iconDescription = permissionData["iconDescription"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val permissions = permissionData["permissions"]


    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.md),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
        )
    ) {
        Column(modifier = Modifier.padding(Spacing.md)) {
            Row(
                modifier = Modifier
                    .padding(Spacing.md)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                Icon(
                    modifier = Modifier
                        .height(Spacing.xxxxl)
                        .width(Spacing.xxxxl),
                    painter = painterResource(permissionIcon as Int),
                    contentDescription = iconDescription as String,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            Text(
                permissionExplanation as String,
                modifier = Modifier.padding(Spacing.md),
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = Spacing.md),
                horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
            ) {
                Button(
                    modifier = Modifier
                        .weight(1f)
                        .semantics {
                            contentDescription =
                                (permissionData["permissionDeclineSemantic"]
                                    ?: UIDataSource.INFORMATION_NOT_FOUND) as String
                        },
                    onClick = {
                        showDeclineDialog = !showDeclineDialog
                    }) {
                    Text("Ablehnen", modifier = Modifier.clearAndSetSemantics {}, style = MaterialTheme.typography.labelLarge)
                }
                Button(
                    modifier = Modifier
                        .weight(1f)
                        .semantics {
                            contentDescription =
                                (permissionData["permissionAcceptSemantic"]
                                    ?: UIDataSource.INFORMATION_NOT_FOUND) as String
                        },
                    onClick = {
                        for (permission in permissions as List<*>) {
                            permissionLauncher.launch(permission as String)
                        }


                    }) {
                    Text("Annehmen", modifier = Modifier.clearAndSetSemantics {}, fontSize = 16.sp)
                }

            }
        }
    }
    if (showDeclineDialog) {
        ConfirmPermissionDecline(
            modifier = modifier,
            permissionData = permissionData,
            onDialogDismissed = { showDeclineDialog = false },
            onExitPermissionSelection = onExitPermissionPage,
            onEvent = onEvent,
            onPermissionDecline = onPermissionAccepted
        )
    }

}

@Composable
fun ConfirmPermissionDecline(
    modifier: Modifier = Modifier,
    permissionData: Map<String, Any>,
    onDialogDismissed: () -> Unit,
    onExitPermissionSelection: () -> Unit,
    onEvent: (UIEvent) -> Unit,
    onPermissionDecline: () -> Unit
) {
    val context = LocalContext.current
    AlertDialog(
        onDismissRequest = { onDialogDismissed() },
        title = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                IconButton(onClick = { onDialogDismissed() }) {
                    Icon(
                        painterResource(R.drawable.arrow_back_24px),
                        contentDescription = UIDataSource.RETURN_SEMANTIC
                    )
                }
                Text("Berechtigung Ablehnen?")
            }
        },
        text = {
            Text(
                (permissionData["confirmPermissionDeclineExplanation"]
                    ?: UIDataSource.INFORMATION_NOT_FOUND) as String
            )
        },
        confirmButton = {
            Button(modifier = Modifier.semantics {
                contentDescription = (permissionData["confirmPermissionDeclineSemantic"]
                    ?: UIDataSource.INFORMATION_NOT_FOUND) as String
            }, onClick = {
                Log.d(
                    LOG_TAG,
                    "[PermissionPage.ConfirmPermissionDecline] Permission ${permissionData["permissions"]}declined"
                )
                onDialogDismissed()
                onPermissionDecline(
                    permissionData,
                    onExitPermissionSelection,
                    context = context,
                    onEvent = onEvent,
                    onPermissionDecline = onPermissionDecline
                )
            }) {
                Text("Trotzdem Ablehnen", modifier = Modifier.clearAndSetSemantics {})
            }
        }
    )
}

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@Preview(showBackground = true, name = "PermissionPage Preview")
@Composable
fun PermissionPagePreview() {
    MaterialTheme {
        PermissionPage(
            onPermissionsGranted = {},
            onPermissionsDeclined = {},
            onEvent = {})
    }
}