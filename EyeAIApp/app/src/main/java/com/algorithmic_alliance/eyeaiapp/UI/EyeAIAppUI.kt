package com.algorithmic_alliance.eyeaiapp.UI

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.os.Build
import android.util.Log
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.app.ActivityCompat
import androidx.core.content.edit
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.pages.ConnectionPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.DebugPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.HomePage
import com.algorithmic_alliance.eyeaiapp.UI.pages.PermissionPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.SettingsPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.TutorialPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.WelcomePage
import kotlinx.serialization.Serializable
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG


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

@Serializable
object TutorialRoute

@RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
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
    val tutorialCompleted =
        sharedPreferences.getBoolean(stringResource(R.string.app_tutorial_completed), false)

    LaunchedEffect(Unit) {
        activity?.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT

    }
    NavHost(
        navController = navController,
        startDestination = if (!tutorialCompleted) WelcomeRoute else PermissionRoute,
        modifier = Modifier.fillMaxSize(),
    ) {
        composable<WelcomeRoute> {
            WelcomePage(
                modifier = Modifier.fillMaxSize(),
                onGetStarted = {
                    navController.navigate(PermissionRoute){
                        popUpTo<WelcomeRoute> {
                            inclusive = true
                        }
                    }
                },
                onStartTutorial = {
                    navController.navigate(TutorialRoute)
                },
            )
        }
        composable<TutorialRoute> {
            TutorialPage(onAbortTutorial = { navController.popBackStack() }, onFinishTutorial = {
                navController.navigate(PermissionRoute) {
                    popUpTo(
                        WelcomeRoute
                    ) { inclusive = true }
                }
            })
        }
        composable<PermissionRoute> {
            PermissionPage(
                modifier = Modifier.fillMaxSize(), onPermissionsDeclined = {
                    (context as? Activity)?.finish()
                }, onPermissionsGranted = {
                    onEvent(UIEvent.OnUpdatePermissionTutorialCompleted(true))
                    Log.d(LOG_TAG, "1")
                    navController.navigate(ConnectionRoute){
                        popUpTo<PermissionRoute> {
                            inclusive = true
                        }
                    }
                }, onEvent = onEvent
            )
        }
        composable<ConnectionRoute> {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            ConnectionPage(onConnectionSuccessful = {
                onEvent(UIEvent.OnUpdateConnectionTutorialCompleted(true))
                if (uiState.actionStartedFromSettings) {
                    onEvent(UIEvent.OnUpdateActionStartedFromSettings(false))
                    navController.popBackStack()
                } else if (!sharedPreferences.getBoolean(debugPageActivatedKey, false)) {
                    navController.navigate(
                        HomeRoute
                    ) { popUpTo<ConnectionRoute> {
                        inclusive = true
                    } }
                } else {
                    navController.navigate(
                        DebugRoute
                    ) { popUpTo<ConnectionRoute> {
                        inclusive = true
                    }}
                }

            }, onExitSelection = {
                if (!uiState.actionStartedFromSettings) {
                    (context as? Activity)?.finish()
                } else {
                    onEvent(UIEvent.OnUpdateActionStartedFromSettings(false))
                    navController.popBackStack()
                }
            }, viewModel = viewModel, onEvent = onEvent)
        }
        composable<HomeRoute> {
            HomePage(modifier = Modifier.fillMaxSize(), onOpenSettings = {
                navController.navigate(
                    SettingsRoute
                )
            }, onEvent = onEvent, viewModel = viewModel)
        }
        composable<SettingsRoute> {
            SettingsPage(modifier = Modifier.fillMaxSize(), onReturn = {
                navController.popBackStack()
            }, onOpenDebugPage = {
                navController.navigate(DebugRoute) {
                    popUpTo(HomeRoute) {
                        inclusive = true
                    }
                }
            }, onOpenHomePage = {
                navController.navigate(
                    HomeRoute
                ) {
                    popUpTo(DebugRoute) {
                        inclusive = true
                    }
                }
            }, onEvent = onEvent, viewModel = viewModel, onOpenConnectionPage = {
                navController.navigate(
                    ConnectionRoute
                )
            })
        }
        composable<DebugRoute> {
            DebugPage(
                modifier = Modifier.fillMaxSize(), onOpenSettings = {
                    navController.navigate(
                        SettingsRoute
                    )
                }, onEvent = onEvent, viewModel = viewModel
            )
        }
    }
    UIDialogs(viewModel = viewModel, onEvent = onEvent, onExitApp = {
        navController.navigate(
            WelcomeRoute
        ) { popUpTo(WelcomeRoute) { inclusive = true } }
    }, onOpenSettings = { navController.navigate(SettingsRoute) })
}

@Composable
fun UIDialogs(
    viewModel: MainViewModel,
    onEvent: (UIEvent) -> Unit,
    onExitApp: () -> Unit,
    onOpenSettings: () -> Unit
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    if (uiState.appMissingVoskPermission) {
        AppMissingVoskPermissionDialog(onEvent = onEvent)
    }
    if (uiState.appMissingCameraPermission) {
        AppMissingCameraPermissionDialog(onEvent = onEvent, onExitApp = onExitApp)
    }
    if (uiState.appMissingSelectedMediaSource) {
        AppMissingSelectedMediaSourceDialog(onEvent = onEvent, onOpenSettings = onOpenSettings)
    }
}

@Composable
fun AppMissingSelectedMediaSourceDialog(onEvent: (UIEvent) -> Unit, onOpenSettings: () -> Unit) {
    AlertDialog(
        onDismissRequest = {
            onEvent(UIEvent.OnUpdateAppMissingSelectedMediaSource(false))
        },
        title = { Text("Fehlende Media-Quelle") },
        text = { Text("In den Einstellungen ist als Eingabequelle 'Media' ausgewählt. Es wurde jedoch keine Media Datei ausgewählt. Wollen Sie die Einstellungen öffnen, um eine Datei oder eine andere Eingabequelle auszuwählen?") },
        confirmButton = {
            Button(onClick = {
                onOpenSettings()
                onEvent(UIEvent.OnUpdateAppMissingSelectedMediaSource(false))
            }) {
                Text(
                    "Einstellungen öffnen"
                )
            }
        }
    )
}


@Composable
fun AppMissingCameraPermissionDialog(onEvent: (UIEvent) -> Unit, onExitApp: () -> Unit) {
    val context = LocalContext.current

    val activity = context as? Activity
    val shouldShowRationale = activity?.let {
        ActivityCompat.shouldShowRequestPermissionRationale(it, Manifest.permission.CAMERA)
    } ?: false
    if (!shouldShowRationale) {
        onEvent(UIEvent.OnUpdateAppMissingCameraPermission(false))
    }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            onEvent(UIEvent.OnUpdateAppMissingCameraPermission(false))
            onEvent(UIEvent.OnReloadDebugPage)
        } else {
            onExitApp()
            onEvent(UIEvent.OnUpdateAppMissingCameraPermission(false))
        }
    }


    AlertDialog(onDismissRequest = {
        onExitApp()
        onEvent(UIEvent.OnUpdateAppMissingCameraPermission(false))
    }, title = {
        Text("Fehlende Berechtigung")
    }, text = {
        Text("Damit die KI ihre Umgebung analysieren kann, braucht die App zugriff auf ihre Kamera. Wollen sie die Berechtigung erteilen?")
    }, confirmButton = {
        Button(onClick = {
            launcher.launch(Manifest.permission.CAMERA)
        }) {
            Text(
                "Berechtigung erteilen"
            )
        }
    })
}

@Composable
fun AppMissingVoskPermissionDialog(onEvent: (UIEvent) -> Unit) {


    val context = LocalContext.current

    val activity = context as? Activity
    val shouldShowRationale = activity?.let {
        ActivityCompat.shouldShowRequestPermissionRationale(it, Manifest.permission.RECORD_AUDIO)
    } ?: false
    if (!shouldShowRationale) {
        onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
    }
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
            onEvent(UIEvent.OnReloadSettingsPage)
            onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
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
        onEvent(UIEvent.OnReloadSettingsPage)
        onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
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
