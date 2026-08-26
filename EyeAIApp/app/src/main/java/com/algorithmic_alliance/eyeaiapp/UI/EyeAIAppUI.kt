package com.algorithmic_alliance.eyeaiapp.UI

import android.annotation.SuppressLint
import androidx.compose.foundation.layout.fillMaxSize
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
import androidx.compose.ui.platform.LocalContext
import androidx.annotation.RequiresApi
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.UI.pages.ConnectionPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.DebugPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.HomePage
import com.algorithmic_alliance.eyeaiapp.UI.pages.PermissionPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.SettingsPage
import com.algorithmic_alliance.eyeaiapp.UI.pages.WelcomePage
import com.algorithmic_alliance.eyeaiapp.camera.CameraManager


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
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)


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
                modifier = Modifier.fillMaxSize(),
                onPermissionsDeclined = {
                    navController.popBackStack()
                },
                onPermissionsGranted = {

                    navController.navigate(ConnectionRoute)

                })
        }
        composable<ConnectionRoute> {
            ConnectionPage(
                modifier = Modifier.fillMaxSize(),
                onConnectionSuccessful = {
                    navController.navigate(
                        HomeRoute
                    ) { popUpTo(WelcomeRoute) { inclusive = false } }
                },
                onExitSelection = {
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
            }, onOpenDebugPage = { navController.navigate(DebugRoute)}, onEvent = onEvent)
        }
        composable<DebugRoute> {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            DebugPage(
                modifier = Modifier.fillMaxSize(), onOpenSettings = {
                    navController.navigate(
                        SettingsRoute
                    )
                }, onBack = {
                    navController.navigate(
                        HomeRoute
                    ) { popUpTo(HomeRoute) { inclusive = false } }
                }, onEvent = onEvent, uiState = uiState
            )
        }
    }
}