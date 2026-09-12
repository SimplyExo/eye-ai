package com.algorithmic_alliance.eyeaiapp.UI

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.util.Log
import androidx.activity.compose.LocalActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.edit
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavDestination.Companion.hasRoute
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
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
import com.algorithmic_alliance.eyeaiapp.data.Spacing
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

@SuppressLint("LocalContextGetResourceValueCall")
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

    val currentBackStackEntry by navController.currentBackStackEntryAsState()
    Log.d(LOG_TAG, "!!! Current Route: ${currentBackStackEntry?.destination?.route} !!!")
	val currentBackStackEntry by navController.currentBackStackEntryAsState()

	val isOnPermissionOrOnboarding = currentBackStackEntry?.destination?.let { dest ->
		dest.hasRoute<PermissionRoute>() || dest.hasRoute<WelcomeRoute>() || dest.hasRoute<TutorialRoute>()
	} ?: true

	LaunchedEffect(currentBackStackEntry) {
		if (!isOnPermissionOrOnboarding) {
			val hasCamera = ContextCompat.checkSelfPermission(
				context, Manifest.permission.CAMERA
			) == PackageManager.PERMISSION_GRANTED

			val hasAudio = ContextCompat.checkSelfPermission(
				context, Manifest.permission.RECORD_AUDIO
			) == PackageManager.PERMISSION_GRANTED

			val hasNearbyWIFI = ContextCompat.checkSelfPermission(
				context, Manifest.permission.NEARBY_WIFI_DEVICES
			) == PackageManager.PERMISSION_GRANTED

			val hasLocation = ContextCompat.checkSelfPermission(
				context, Manifest.permission.ACCESS_FINE_LOCATION
			) == PackageManager.PERMISSION_GRANTED

			if (!hasCamera) {
				onEvent(UIEvent.OnUpdateAppMissingCameraPermission(true))
			}
			if (!hasAudio && viewModel.isSpeechRecognitionEnabled()) {
				onEvent(UIEvent.OnUpdateAppMissingVoskPermission(true))
			}

			if ((!hasNearbyWIFI || !hasLocation) && viewModel.appInputSource() == context.getString(
					R.string.input_is_eyeaivision
				)
			) {
				onEvent(UIEvent.OnUpdateAppMissingVisionPermission(true))
			}
		}
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
					navController.navigate(PermissionRoute) {
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
				navController.navigate(ConnectionRoute) {
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
					) {
						popUpTo<ConnectionRoute> {
							inclusive = true
						}
					}
				} else {
					navController.navigate(
						DebugRoute
					) {
						popUpTo<ConnectionRoute> {
							inclusive = true
						}
					}
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
		(context as? Activity)?.finish()
	}, onOpenSettings = { navController.navigate(SettingsRoute) })
}

@Composable
fun UIDialogs(
	viewModel: MainViewModel,
	onEvent: (UIEvent) -> Unit,
	onExitApp: () -> Unit,
	onOpenSettings: () -> Unit
) {
	val uiState by viewModel.uiDialogsUIState.collectAsStateWithLifecycle()
	if (uiState.appMissingVoskPermission) {
		AppMissingVoskPermissionDialog(onEvent = onEvent)
	}
	if (uiState.appMissingCameraPermission) {
		AppMissingCameraPermissionDialog(onEvent = onEvent, onExitApp = onExitApp)
	}
	if (uiState.appMissingSelectedMediaSource) {
		AppMissingSelectedMediaSourceDialog(onEvent = onEvent, onOpenSettings = onOpenSettings)
	}
	if (uiState.appMissingVisionPermission) {
		AppMissingVisionPermissionDialog(onEvent = onEvent)
	}
}

@Composable
fun AppMissingSelectedMediaSourceDialog(onEvent: (UIEvent) -> Unit, onOpenSettings: () -> Unit) {
	AlertDialog(
		onDismissRequest = {
		onEvent(UIEvent.OnUpdateAppMissingSelectedMediaSource(false))
	},
		title = { Text(stringResource(R.string.missing_media_source_alert_dialog_title)) },
		text = { Text(stringResource(R.string.missing_media_source_alert_dialog_text)) },
		confirmButton = {
			Button(onClick = {
				onOpenSettings()
				onEvent(UIEvent.OnUpdateAppMissingSelectedMediaSource(false))
			}) {
				Text(
					stringResource(R.string.button_open_settings_text)
				)
			}
		})
}


@Composable
fun AppMissingVisionPermissionDialog(onEvent: (UIEvent) -> Unit) {
	val context = LocalContext.current
	val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
	val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
	val hasRequestedKey = "has_requested_vision_permission"
	val appInputSourceKey = stringResource(R.string.input_source_setting)
	val inputIsVision = stringResource(R.string.input_is_eyeaivision)
	val inputIsCamera = stringResource(R.string.input_is_camera)

	// 1. Array der benötigten Berechtigungen (Versionsabhängig)
	val permissionsToRequest = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
		arrayOf(
			Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.NEARBY_WIFI_DEVICES
		)
	} else {
		arrayOf(
			Manifest.permission.ACCESS_FINE_LOCATION
		)
	}

	fun disableVisionSetting() {
		if (sharedPreferences.getString(appInputSourceKey, inputIsCamera) == inputIsVision) {
			sharedPreferences.edit(commit = true) {
				putString(appInputSourceKey, inputIsCamera)
			}
			onEvent(UIEvent.UpdateSettings)
		}
	}

	DisposableEffect(lifecycleOwner) {
		val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
			if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
				// 2. Prüfen, ob ALLE Berechtigungen aus dem Array gewährt wurden
				val allGranted = permissionsToRequest.all {
					ContextCompat.checkSelfPermission(
						context, it
					) == PackageManager.PERMISSION_GRANTED
				}

				if (allGranted) {
					onEvent(UIEvent.OnUpdateAppMissingVisionPermission(false))
					onEvent(UIEvent.OnReloadDebugPage)
				}
			}
		}
		lifecycleOwner.lifecycle.addObserver(observer)
		onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
	}

	// 3. RequestMultiplePermissions anstatt RequestPermission verwenden
	val launcher = rememberLauncherForActivityResult(
		ActivityResultContracts.RequestMultiplePermissions()
	) { permissionsMap ->
		// permissionsMap enthält ein Mapping z.B. ["android.permission.ACCESS_FINE_LOCATION" -> true]
		// Wir prüfen, ob alle angefragten Berechtigungen auf 'true' stehen
		val allGranted = permissionsMap.values.all { it }

		if (allGranted) {
			onEvent(UIEvent.OnUpdateAppMissingVisionPermission(false))
			onEvent(UIEvent.OnReloadDebugPage)
		} else {
			disableVisionSetting()
			onEvent(UIEvent.OnUpdateAppMissingVisionPermission(false))
		}
	}

	AlertDialog(onDismissRequest = {
		disableVisionSetting()
		onEvent(UIEvent.OnUpdateAppMissingVisionPermission(false))
	}, title = {
		Row(
			modifier = Modifier.fillMaxWidth(),
			verticalAlignment = Alignment.CenterVertically,
		) {
			PremiumIconButton(modifier = Modifier.semantics { traversalIndex = 1f }, onClick = {
				disableVisionSetting()
				onEvent(UIEvent.OnUpdateAppMissingVisionPermission(false))
			}) {
				Icon(
					modifier = Modifier
						.width(Spacing.xl)
						.height(Spacing.xl),
					painter = painterResource(R.drawable.arrow_back_24px),
					contentDescription = stringResource(R.string.return_icon_description)
				)
			}
			Text(
				stringResource(R.string.missing_permission_alert_dialog_title_text), // Passe den String an, falls nötig
				modifier = Modifier.semantics {
					traversalIndex = -1f
					heading()
				},
			)
		}
	}, text = {
		Text(stringResource(R.string.missing_permission_alert_dialog_vision_text))
	}, confirmButton = {
		Button(onClick = {
			val activity = context as? Activity

			// 4. Prüfen, ob für IRGENDEINE der Berechtigungen noch ein System-Pop-up angezeigt werden darf
			val shouldShowRationale = activity?.let { act ->
				permissionsToRequest.any { permission ->
					ActivityCompat.shouldShowRequestPermissionRationale(act, permission)
				}
			} ?: false

			val hasRequestedBefore = sharedPreferences.getBoolean(hasRequestedKey, false)

			if (hasRequestedBefore && !shouldShowRationale) {
				// Fall A: Android blockiert das Pop-up für alle fehlenden Berechtigungen
				val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
					data = Uri.fromParts("package", context.packageName, null)
				}
				context.startActivity(intent)
			} else {
				// Fall B: System-Pop-up(s) abrufen
				sharedPreferences.edit { putBoolean(hasRequestedKey, true) }
				launcher.launch(permissionsToRequest)
			}
		}) {
			Text(stringResource(R.string.missing_permission_alert_dialog_grant_permission_text))
		}
	})
}

@Composable
fun AppMissingCameraPermissionDialog(onEvent: (UIEvent) -> Unit, onExitApp: () -> Unit) {
	val context = LocalContext.current
	val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
	val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
	val hasRequestedKey = "has_requested_camera_permission"

	DisposableEffect(lifecycleOwner) {
		val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
			if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
				val isGranted = ContextCompat.checkSelfPermission(
					context, Manifest.permission.CAMERA
				) == PackageManager.PERMISSION_GRANTED

				if (isGranted) {
					onEvent(UIEvent.OnUpdateAppMissingCameraPermission(false))
					onEvent(UIEvent.OnReloadDebugPage)
				}
			}
		}
		lifecycleOwner.lifecycle.addObserver(observer)
		onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
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
		Row(
			modifier = Modifier.fillMaxWidth(),
			verticalAlignment = Alignment.CenterVertically,
		) {
			PremiumIconButton(modifier = Modifier.semantics { traversalIndex = 1f }, onClick = {
				onExitApp()
				onEvent(UIEvent.OnUpdateAppMissingCameraPermission(false))
			}) {
				Icon(
					modifier = Modifier
						.width(Spacing.xl)
						.height(Spacing.xl),
					painter = painterResource(R.drawable.arrow_back_24px),
					contentDescription = stringResource(R.string.return_icon_description)
				)
			}
			Text(
				stringResource(R.string.missing_permission_alert_dialog_title_text),
				modifier = Modifier.semantics {
					traversalIndex = -1f
					heading()
				},
			)
		}
	}, text = {
		Text(stringResource(R.string.missing_permission_alert_dialog_camera_text))
	}, confirmButton = {
		Button(onClick = {
			val activity = context as? Activity
			val shouldShowRationale = activity?.let {
				ActivityCompat.shouldShowRequestPermissionRationale(
					it, Manifest.permission.CAMERA
				)
			} ?: false

			val hasRequestedBefore = sharedPreferences.getBoolean(hasRequestedKey, false)

			if (hasRequestedBefore && !shouldShowRationale) {
				val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
					data = Uri.fromParts("package", context.packageName, null)
				}
				context.startActivity(intent)
			} else {
				sharedPreferences.edit { putBoolean(hasRequestedKey, true) }
				launcher.launch(Manifest.permission.CAMERA)
			}
		}) {
			Text(stringResource(R.string.missing_permission_alert_dialog_grant_permission_text))
		}
	})
}

@Composable
fun AppMissingVoskPermissionDialog(onEvent: (UIEvent) -> Unit) {
	val context = LocalContext.current
	val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
	val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
	val hasRequestedKey = "has_requested_record_audio_permission"
	val speechRecognitionEnabledKey = stringResource(R.string.enable_speech_recognition_setting)

	fun disableSpeechSetting() {
		if (sharedPreferences.getBoolean(speechRecognitionEnabledKey, true)) {
			sharedPreferences.edit(commit = true) {
				putBoolean(speechRecognitionEnabledKey, false)
			}
			onEvent(UIEvent.UpdateSettings)
		}
	}

	DisposableEffect(lifecycleOwner) {
		val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
			if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
				val isGranted = ContextCompat.checkSelfPermission(
					context, Manifest.permission.RECORD_AUDIO
				) == PackageManager.PERMISSION_GRANTED

				if (isGranted) {
					onEvent(UIEvent.OnReloadSettingsPage)
					onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
				}
			}
		}
		lifecycleOwner.lifecycle.addObserver(observer)
		onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
	}

	val launcher = rememberLauncherForActivityResult(
		ActivityResultContracts.RequestPermission()
	) { isGranted ->
		if (isGranted) {
			onEvent(UIEvent.OnReloadSettingsPage)
			onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
		} else {
			disableSpeechSetting()
			onEvent(UIEvent.OnReloadSettingsPage)
			onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
		}
	}

	AlertDialog(onDismissRequest = {
		disableSpeechSetting()
		onEvent(UIEvent.OnReloadSettingsPage)
		onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
	}, title = {
		Row(
			modifier = Modifier.fillMaxWidth(),
			verticalAlignment = Alignment.CenterVertically,
		) {
			PremiumIconButton(modifier = Modifier.semantics { traversalIndex = 1f }, onClick = {
				disableSpeechSetting()
				onEvent(UIEvent.OnReloadSettingsPage)
				onEvent(UIEvent.OnUpdateAppMissingVoskPermission(false))
			}) {
				Icon(
					modifier = Modifier
						.width(Spacing.xl)
						.height(Spacing.xl),
					painter = painterResource(R.drawable.arrow_back_24px),
					contentDescription = stringResource(R.string.return_icon_description)
				)
			}
			Text(
				stringResource(R.string.missing_permission_alert_dialog_title_text),
				modifier = Modifier.semantics {
					traversalIndex = -1f
					heading()
				},
			)
		}
	}, text = {
		Text(stringResource(R.string.missing_permission_alert_dialog_vosk_text))
	}, confirmButton = {
		Button(onClick = {
			val activity = context as? Activity
			val shouldShowRationale = activity?.let {
				ActivityCompat.shouldShowRequestPermissionRationale(
					it, Manifest.permission.RECORD_AUDIO
				)
			} ?: false

			val hasRequestedBefore = sharedPreferences.getBoolean(hasRequestedKey, false)

			if (hasRequestedBefore && !shouldShowRationale) {
				val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
					data = Uri.fromParts("package", context.packageName, null)
				}
				context.startActivity(intent)
			} else {
				sharedPreferences.edit { putBoolean(hasRequestedKey, true) }
				launcher.launch(Manifest.permission.RECORD_AUDIO)
			}
		}) {
			Text(
				stringResource(R.string.missing_permission_alert_dialog_grant_permission_text)
			)
		}
	})
}
