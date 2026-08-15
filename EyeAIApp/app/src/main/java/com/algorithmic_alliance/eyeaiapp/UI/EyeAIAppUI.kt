package com.algorithmic_alliance.eyeaiapp.UI

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.preference.PreferenceManager
import kotlinx.serialization.Serializable
import android.os.Build
import android.util.Log
import androidx.compose.ui.platform.LocalContext
import androidx.annotation.RequiresApi
import com.algorithmic_alliance.eyeaiapp.UI.pages.ConnectionPage


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

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@Composable
fun EyeAIAppUI() {
    Log.d("EyeAIUI", "Starting UI")
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
            })
        }
        composable<SettingsRoute> {
            SettingsPage(modifier = Modifier.fillMaxSize(), onReturn = {
                navController.navigate(
                    HomeRoute
                ) { popUpTo(HomeRoute) { inclusive = false } }
            })
        }
    }
}

fun showPermissionPage(){

}

@RequiresApi(Build.VERSION_CODES.TIRAMISU)
@Preview(showBackground = true, name = "Navigation-Preview")
@Composable
fun EyeAIAppUIPreview() {
    MaterialTheme {
        EyeAIAppUI()
    }
}