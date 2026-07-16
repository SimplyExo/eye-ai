package com.algorithmic_alliance.eyeaiapp.UI

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import kotlinx.serialization.Serializable



@Serializable
object WelcomeRoute

@Serializable
object PermissionRoute

@Serializable
object HomeRoute

@Serializable object ConnectionRoute


@Composable
fun EyeAIAppUI() {
    val navController = rememberNavController()

    //TODO implement if permissions are granted
    var permissionsGranted by remember { mutableStateOf(false) }
    var devicesSelected by remember { mutableStateOf(false) }

    NavHost(
        navController = navController,
        startDestination = WelcomeRoute,
        modifier = Modifier.fillMaxSize(),
    ) {
        composable<WelcomeRoute> {
            WelcomePage(
                modifier = Modifier.fillMaxSize(),
                onGetStarted = {
                    if (!permissionsGranted)
                        navController.navigate(PermissionRoute)
                    else if(!devicesSelected)
                        navController.navigate(ConnectionRoute)
                    else
                        navController.navigate((HomeRoute))
                },

                )
        }
        composable<PermissionRoute> {
            PermissionPage(
                modifier = Modifier.fillMaxSize(),
                onPermissionsDeclined = { navController.popBackStack() },
                onPermissionsGranted = {
                    permissionsGranted = true
                    if(!devicesSelected)
                        navController.navigate(ConnectionRoute)
                    else
                        navController.navigate(HomeRoute)
                })
        }
        composable<ConnectionRoute>{
            ConnectionPage(modifier = Modifier.fillMaxSize(), onConnectionSuccessful = {navController.navigate(
                HomeRoute)})
        }
        composable<HomeRoute> { HomePage(modifier = Modifier.fillMaxSize(), onOpenSettings = {}) }

    }
}

@Preview(showBackground = true, name = "Navigation-Preview")
@Composable
fun EyeAIAppUIPreview() {
    MaterialTheme {
        EyeAIAppUI()
    }
}