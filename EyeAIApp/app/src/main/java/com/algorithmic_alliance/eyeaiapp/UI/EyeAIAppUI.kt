package com.algorithmic_alliance.eyeaiapp.UI

import android.Manifest
import android.content.Context
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.remote.creation.dsl.fillMaxSize
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
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import kotlinx.serialization.Serializable
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.compose.ui.res.stringResource
import android.view.View.GONE
import android.view.View.VISIBLE
import android.widget.Button
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.compose.ui.platform.LocalContext
import android.widget.TextView
import androidx.activity.compose.LocalActivity
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.content.res.AppCompatResources
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.content.edit
import com.algorithmic_alliance.eyeaiapp.PermissionManager
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource


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

@Preview(showBackground = true, name = "Navigation-Preview")
@Composable
fun EyeAIAppUIPreview() {
    MaterialTheme {
        EyeAIAppUI()
    }
}