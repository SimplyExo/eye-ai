package com.algorithmic_alliance.eyeaiapp.UI

import android.annotation.SuppressLint
import android.content.pm.ActivityInfo
import androidx.compose.foundation.layout.fillMaxSize
import androidx.core.content.edit
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.NavHost
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.preference.PreferenceManager
import kotlinx.serialization.Serializable
import android.os.Build
import android.util.Log
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.ui.platform.LocalContext
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AlertDialog
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.pages.ConnectionPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.DebugPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.HomePage
import com.algorithmic_alliance.eyeaiapp.UI.pages.PermissionPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.SettingsPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.WelcomePage
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import android.Manifest


@Serializable
object WelcomeRoute

@Serializable
object PermissionRoute

@Serializable
object HomeRoute

@Serializable
object ConnectionRoute

@Serializable
object SettingsRoute

@Serializable
object DebugRoute

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@Composable
fun EyeAIAppUI(
    viewModel: MainViewModel,
    onEvent: (UIEvent) -> Unit,
) {
    Log.d(LOG_TAG, "Starting UI")
    val navController = rememberNavController()
    val context = LocalContext.current
    val activity = LocalActivity.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val debugPageActivatedKey = stringResource(R.string.debug_page_activated)

    LaunchedEffect(Unit) {
        activity?.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

    }
    NavHost(
        navController = navController,
        startDestination = WelcomeRoute,
        modifier = Modifier.fillMaxSize(),
    ) {
        composable<WelcomeRoute> {
            WelcomePage(
                modifier = Modifier.fillMaxSize(),
                onGetStarted = {
                    navController.navigate(PermissionRoute)
                },
            )
        }
        composable<PermissionRoute> {
            PermissionPage(
                modifier = Modifier.fillMaxSize(), onPermissionsDeclined = {
                    navController.popBackStack()
                }, onPermissionsGranted = {
                    navController.navigate(ConnectionRoute)

                }, onEvent = onEvent
            )
        }
        composable<ConnectionRoute> {
            ConnectionPage(modifier = Modifier.fillMaxSize(), onConnectionSuccessful = {
                if (!sharedPreferences.getBoolean(debugPageActivatedKey, false)) {
                    navController.navigate(
                        HomeRoute
                    ) { popUpTo(WelcomeRoute) { inclusive = false } }
                } else {
                    navController.navigate(
                        DebugRoute
                    ) { popUpTo(WelcomeRoute) { inclusive = false } }
                }

            }, onExitSelection = {
                navController.navigate(WelcomeRoute) {
                    popUpTo(WelcomeRoute) {
                        inclusive = false
                    }
                }
            })
        }
        composable<HomeRoute> {
            HomePage(modifier = Modifier.fillMaxSize(), onOpenSettings = {
                navController.navigate(
                    SettingsRoute
                )
            }, onEvent = onEvent)
        }
        composable<SettingsRoute> {
            SettingsPage(modifier = Modifier.fillMaxSize(), onReturn = {
                navController.popBackStack()
            }, onOpenDebugPage = {
                navController.navigate(DebugRoute) {
                    popUpTo(WelcomeRoute) {
                        inclusive = false
                    }
                }
            }, onOpenHomePage = {
                navController.navigate(
                    HomeRoute
                ) {
                    popUpTo(WelcomeRoute) {
                        inclusive = false
                    }
                }
            }, onEvent = onEvent, viewModel = viewModel)
        }
        composable<DebugRoute> {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            DebugPage(
                modifier = Modifier.fillMaxSize(), onOpenSettings = {
                    navController.navigate(
                        SettingsRoute
                    )
                }, onEvent = onEvent, uiState = uiState
            )
        }
    }
    UIDialogs(viewModel = viewModel, onEvent = onEvent)
}

@Composable
fun UIDialogs(
    viewModel: MainViewModel,
    onEvent: (UIEvent) -> Unit,
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    if (uiState.appMissingVoskPermission) {
        AppMissingVoskPermissionDialog(onEvent = onEvent)
    }
}

@Composable
fun AppMissingVoskPermissionDialog(onEvent: (UIEvent) -> Unit) {
    val context = LocalContext.current
    val speechRecognitionEnabledKey = stringResource(R.string.enable_speech_recognition_setting)

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            onEvent(UIEvent.OnReloadSettingsPage)
            onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
        } else {
            val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
            if (sharedPreferences.getBoolean(speechRecognitionEnabledKey, true)) {
                sharedPreferences.edit(commit = true) {
                    putBoolean(speechRecognitionEnabledKey, false)
                }
                onEvent(UIEvent.UpdateSettings)
            }
            onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
            onEvent(UIEvent.OnReloadSettingsPage)
        }
    }


    AlertDialog(onDismissRequest = {
        val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
        if (sharedPreferences.getBoolean(speechRecognitionEnabledKey, true)) {
            sharedPreferences.edit(commit = true) {
                putBoolean(speechRecognitionEnabledKey, false)
            }
            onEvent(UIEvent.UpdateSettings)
        }
        onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
        onEvent(UIEvent.OnReloadSettingsPage)

    }, title = {
        Text("Fehlende Berechtigung")
    }, text = {
        Text("Um die Spracherkennung zu aktivieren, braucht die App die Berechtigung für das Mikrophon. Wollen sie die Berechtigung erteilen?")
    }, confirmButton = {
        Button(onClick = {
            launcher.launch(Manifest.permission.RECORD_AUDIO)
        }) {
            Text(
                "Berechtigung erteilen"
            )
        }
    })
}
