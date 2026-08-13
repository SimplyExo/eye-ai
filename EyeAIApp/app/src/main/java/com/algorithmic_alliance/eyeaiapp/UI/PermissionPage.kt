package com.algorithmic_alliance.eyeaiapp.UI

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
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
import android.Manifest
import android.content.pm.PackageManager
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
import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import com.algorithmic_alliance.eyeaiapp.PermissionManager

@Composable
fun PermissionPage(
    modifier: Modifier = Modifier,
    onPermissionsGranted: () -> Unit,
    onPermissionsDeclined: () -> Unit,

) {
    Log.d("EyeAIUI", "[PermissionPage] Loading PermissionPage")
    val context = LocalContext.current

    val neededPermissions = UIDataSource.NEEDED_PERMISSIONS
    val notGrantedPermissions = mutableListOf<Map<String, Any>>()

    for (map in neededPermissions) {
        val hasPermission = ContextCompat.checkSelfPermission(
            context,
            map["permission"] as String
        ) == PackageManager.PERMISSION_GRANTED
        Log.d(
            "EyeAIUI",
            "[PermissionPage] ${map["permission"]} = $hasPermission"
        )
        if(!hasPermission)
            notGrantedPermissions.add(map)
    }
    if(notGrantedPermissions.isEmpty()){
        Log.d("EyeAIUI", "[PermissionPage] All permissions already granted. Exiting PermissionPage")
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
                onPermissionDeclined = { onPermissionsDeclined() }
            )
        }

    }

}

@Composable
fun AskForPermission(
    modifier: Modifier = Modifier,
    permissionData: Map<String, Any>,
    onPermissionAccepted: () -> Unit,
    onPermissionDeclined: () -> Unit
) {
    Log.d("EyeAIUI", "[PermissionPage.AskForPermission] Asking for permission ${permissionData["permission"]}")
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            onPermissionAccepted()
            Log.d("EyeAIUI", "[PermissionPage.AskForPermission] Permission ${permissionData["permission"]} granted")
        } else{
            Log.d("EyeAIUI", "[PermissionPage.AskForPermission] Permission ${permissionData["permission"]} declined")
            onPermissionDeclined()
        }
    }

    var showDeclineDialog by rememberSaveable { mutableStateOf(false) }

    val permissionExplanation =
        permissionData["permissionExplanation"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val permissionIcon = permissionData["icon"] ?: UIDataSource.ICON_NOT_FOUND
    val iconDescription = permissionData["iconDescription"] ?: UIDataSource.INFORMATION_NOT_FOUND
    val permission = permissionData["permission"]


    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalArrangement = Arrangement.Center
            ) {
                Icon(
                    modifier = Modifier
                        .height(150.dp)
                        .width(150.dp),
                    painter = painterResource(permissionIcon as Int),
                    contentDescription = iconDescription as String,
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            Text(
                permissionExplanation as String,
                modifier = Modifier.padding(16.dp),
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                fontSize = 20.sp
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 16.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
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
                    Text("Ablehnen", modifier = Modifier.clearAndSetSemantics {}, fontSize = 16.sp)
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
                        permissionLauncher.launch(permission as String)
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
            onPermissionDeclined = {
                onPermissionDeclined()
            },
            onDialogDismissed = { showDeclineDialog = false })
    }

}

@Composable
fun ConfirmPermissionDecline(
    modifier: Modifier = Modifier,
    permissionData: Map<String, Any>,
    onPermissionDeclined: () -> Unit,
    onDialogDismissed: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = { onDialogDismissed() },
        title = {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
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
                Log.d("EyeAIUI", "[PermissionPage.ConfirmPermissionDecline] Permission ${permissionData["permission"]}declined")
                onDialogDismissed()
                onPermissionDeclined() }) {
                Text("Trotzdem Ablehnen", modifier = Modifier.clearAndSetSemantics {})
            }
        }
    )
}

@Preview(showBackground = true, name = "PermissionPage Preview")
@Composable
fun PermissionPagePreview() {
    MaterialTheme { PermissionPage(onPermissionsGranted = {}, onPermissionsDeclined = {}) }
}