package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.annotation.SuppressLint
import android.os.Build
import android.util.Log
import android.view.ViewTreeObserver
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.focusable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.isTraversalGroup
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.PremiumButton
import com.algorithmic_alliance.eyeaiapp.UI.PremiumIconButton
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.checkPermissionsStatus
import com.algorithmic_alliance.eyeaiapp.UI.onPermissionDecline
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import kotlinx.coroutines.delay
import kotlin.time.Duration.Companion.milliseconds
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@Composable
fun PermissionPage(
    modifier: Modifier = Modifier,
    onPermissionsGranted: () -> Unit,
    onPermissionsDeclined: () -> Unit,
    onEvent: (UIEvent) -> Unit
) {
    LaunchedEffect(Unit) {
        onEvent(UIEvent.OnUpdateVisionPermissionsNotGranted(false))
    }
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
            key(currentPermission) {
                AskForPermission(
                    modifier,
                    notGrantedPermissions[currentPermission],
                    onPermissionAccepted = { if (currentPermission < notGrantedPermissions.size - 1) currentPermission++ else onPermissionsGranted() },
                    onExitPermissionPage = { onPermissionsDeclined() },
                    onEvent = onEvent
                )
            }
        }

    }

}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun AskForPermission(
    modifier: Modifier = Modifier,
    permissionData: Map<String, Any>,
    onPermissionAccepted: () -> Unit,
    onExitPermissionPage: () -> Unit,
    onEvent: (UIEvent) -> Unit
) {
    val focusRequester = remember { FocusRequester() }
    val context = LocalContext.current
    val view = LocalView.current
    var hasWindowFocus by remember { mutableStateOf(view.hasWindowFocus()) }

    DisposableEffect(view) {
        val listener = ViewTreeObserver.OnWindowFocusChangeListener { hasFocus ->
            hasWindowFocus = hasFocus
        }
        view.viewTreeObserver.addOnWindowFocusChangeListener(listener)
        onDispose { view.viewTreeObserver.removeOnWindowFocusChangeListener(listener) }
    }
    val isDark = isSystemInDarkTheme()

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
        stringResource(permissionData["permissionExplanation"] as Int)
    val permissionIcon = permissionData["icon"] ?: UIDataSource.ICON_NOT_FOUND
    val iconDescription = stringResource(permissionData["iconDescription"] as Int)
    val permissions = permissionData["permissions"]


    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.md),
        shape = PremiumShapes.large,
        elevation = CardDefaults.cardElevation(AppElevation.level5),
        border = BorderStroke(
            width = if (isDark) 2.dp else 0.dp,
            color = Color.White.copy(alpha = 0.2f)
        ),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer,
            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
        )
    ) {
        Column(modifier = Modifier
            .padding(Spacing.md)
            .semantics { isTraversalGroup = true }) {
            Row(
                modifier = Modifier
                    .padding(Spacing.md)
                    .fillMaxWidth()
                    .semantics { traversalIndex = -1f },
                horizontalArrangement = Arrangement.Center
            ) {
                Icon(
                    modifier = Modifier
                        .height(Spacing.xxxxl)
                        .width(Spacing.xxxxl)
                        .focusRequester(focusRequester)
                        .focusable(),
                    painter = painterResource(permissionIcon as Int),
                    contentDescription = stringResource(permissionData["iconDescription"] as Int),
                    tint = MaterialTheme.colorScheme.onPrimaryContainer
                )
            }
            Text(
                permissionExplanation,
                modifier = Modifier
                    .padding(Spacing.md)
                    .semantics { traversalIndex = 0f },
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = Spacing.md)
                    .semantics { traversalIndex = 1f },
                horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
            ) {
                PremiumButton(
                    modifier = Modifier
                        .weight(1f),
                    shadowElevation = if (isDark) AppElevation.level4 else AppElevation.level2,
                    onClick = { showDeclineDialog = !showDeclineDialog }) {
                    Text(
                        stringResource(R.string.decline_action),
                        modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(permissionData["permissionDeclineSemantic"] as Int)},
                        style = MaterialTheme.typography.labelLarge
                    )
                }
                PremiumButton(
                    modifier = Modifier
                        .weight(1f),
                    shadowElevation = if (isDark) AppElevation.level4 else AppElevation.level2,
                    onClick = {
                        for (permission in permissions as List<*>) {
                            permissionLauncher.launch(permission as String)
                        }


                    }) {
                    Text(
                        stringResource(R.string.accept_action),
                        modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(permissionData["permissionAcceptSemantic"] as Int)},
                        style = MaterialTheme.typography.labelLarge
                    )
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

@SuppressLint("LocalContextGetResourceValueCall")
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
    val isDark = isSystemInDarkTheme()
    AlertDialog(
        onDismissRequest = { onDialogDismissed() },
        title = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                PremiumIconButton(
                    modifier = Modifier.semantics { traversalIndex = 1f },
                    onClick = { onDialogDismissed() }) {
                    Icon(
                        modifier = Modifier
                            .width(Spacing.xl)
                            .height(Spacing.xl),
                        painter = painterResource(R.drawable.arrow_back_24px),
                        contentDescription = stringResource(R.string.return_icon_description)
                    )
                }
                Text(
                    stringResource(R.string.confirm_permission_decline_title),
                    modifier = Modifier.semantics {
                        traversalIndex = -1f
                        heading()
                    },
                    style = MaterialTheme.typography.titleLarge
                )
            }
        },
        text = {
            Text(
                (stringResource(permissionData["confirmPermissionDeclineExplanation"] as Int)),
                style = MaterialTheme.typography.bodyMedium
            )
        },
        confirmButton = {
            PremiumButton(
                modifier = Modifier,
                shadowElevation = if (isDark) 8.dp else 3.dp,
                onClick = {
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
                Text(
                    stringResource(R.string.confirm_decline_action),
                    modifier = Modifier.clearAndSetSemantics{contentDescription = context.getString(permissionData["confirmPermissionDeclineSemantic"] as Int)},
                    style = MaterialTheme.typography.labelLarge
                )
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