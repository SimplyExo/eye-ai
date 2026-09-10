package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.content.IntentSender
import android.location.LocationManager
import android.os.Build
import android.util.Log
import android.view.ViewTreeObserver
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.focusable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
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
import androidx.compose.ui.semantics.isTraversalGroup
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.edit
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.PremiumButton
import com.algorithmic_alliance.eyeaiapp.UI.PremiumIconButton
import com.algorithmic_alliance.eyeaiapp.UI.ShimmerBox
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.connectToDevice
import com.algorithmic_alliance.eyeaiapp.UI.rememberWifiScanState
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.google.android.gms.common.api.ResolvableApiException
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.LocationSettingsRequest
import com.google.android.gms.location.Priority
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

private const val AUDIO_DEVICE_TYPE = "audio"
private const val VISION_DEVICE_TYPE = "eye-ai-vision"


private data class ConnectionCategory(
	val name: String,
	val nameSemantic: String,
	val type: String,
	val rememberKey: Int,
	val selectedKey: Int,
)

@RequiresApi(Build.VERSION_CODES.S)
@Composable
fun ConnectionPage(
	onConnectionSuccessful: () -> Unit,
	onExitSelection: () -> Unit,
	viewModel: MainViewModel,
	onEvent: (UIEvent) -> Unit
) {
	val uiState by viewModel.connectionPageUIState.collectAsStateWithLifecycle()
	BackHandler {
		onExitSelection()
	}

	Log.d(LOG_TAG, "[ConnectionPage] Loading ConnectionPage")

	val context = LocalContext.current

	val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)

	val categories = listOf(
		ConnectionCategory(
			name = stringResource(R.string.device_eyeaivision_name),
			nameSemantic = stringResource(R.string.choose_vision_name_semantic),
			type = VISION_DEVICE_TYPE,
			rememberKey = R.string.remember_eye_ai_vision,
			selectedKey = R.string.selected_eye_ai_vision,
		),
		ConnectionCategory(
			name = stringResource(R.string.device_audio_name),
			nameSemantic = stringResource(R.string.choose_audio_device_name_semantic),
			type = AUDIO_DEVICE_TYPE,
			rememberKey = R.string.remember_audio_device,
			selectedKey = R.string.selected_audio_device,
		),
	)

	var currentlyDisplayedDevices by remember { mutableIntStateOf(0) }
	var startAutoConnect by remember { mutableStateOf(true) }
	val shouldRememberAudioDeviceKey = stringResource(R.string.remember_audio_device)
	val selectedAudioDeviceKey = stringResource(R.string.selected_audio_device)
	val shouldRememberVisionDeviceKey = stringResource(R.string.remember_eye_ai_vision)
	val selectedVisionDeviceKey = stringResource(R.string.selected_eye_ai_vision)
	val inputSourceSettingKey = stringResource(R.string.input_source_setting)
	val inputIsCameraKey = stringResource(R.string.input_is_camera)
	val chooseCameraAsInput = stringResource(R.string.choose_camera_as_input_text)

	LaunchedEffect(Unit) {
		if (uiState.visionPermissionsNotGranted) {
			sharedPreferences.edit(commit = true) {
				putBoolean(
					shouldRememberVisionDeviceKey, false
				)
				putString(
					selectedVisionDeviceKey, chooseCameraAsInput
				)
				putString(inputSourceSettingKey, inputIsCameraKey)
			}
			onEvent(UIEvent.UpdateSettings)
			onConnectionSuccessful()
		}
	}

	key(currentlyDisplayedDevices) {
		ChooseConnectionPage(
			onConnectionSuccessful = { visionDevice ->
			if (visionDevice == chooseCameraAsInput) {
				sharedPreferences.edit(commit = true) {
					putBoolean(shouldRememberAudioDeviceKey, false)
					putString(selectedAudioDeviceKey, "")
				}
				onConnectionSuccessful()
			} else if (currentlyDisplayedDevices < categories.size - 1) {
				currentlyDisplayedDevices++
				startAutoConnect = true
			} else onConnectionSuccessful()
		},
			goBack = {
				if (currentlyDisplayedDevices != 0) {
					currentlyDisplayedDevices--
					startAutoConnect = false
				} else onExitSelection()
			},
			devicesData = categories[currentlyDisplayedDevices],
			viewModel = viewModel,
			onEvent = onEvent,
			startAutoConnect = startAutoConnect
		)
	}

}

@SuppressLint("LocalContextGetResourceValueCall")
@RequiresApi(Build.VERSION_CODES.S)
@Composable
private fun ChooseConnectionPage(
	onConnectionSuccessful: (String) -> Unit,
	goBack: () -> Unit,
	devicesData: ConnectionCategory,
	viewModel: MainViewModel,
	onEvent: (UIEvent) -> Unit,
	startAutoConnect: Boolean = true
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
	val uiState by viewModel.chooseConnectionPageUIState.collectAsStateWithLifecycle()
	val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
	var shouldRememberDevice by rememberSaveable { mutableStateOf(false) }
	var selectedDevice by remember { mutableStateOf("") }
	var showConnectionFailedDialog by remember { mutableStateOf(false) }
	var showLocationDisabledDialog by remember { mutableStateOf(false) }
	var scanningForDevices by rememberSaveable { mutableStateOf(false) }

	val shouldRememberKey = stringResource(devicesData.rememberKey)
	val selectedDeviceKey = stringResource(devicesData.selectedKey)
	val rememberedDevice = sharedPreferences.getString(
		selectedDeviceKey, ""
	)
	val deviceCategory = devicesData.name
	val wifiScanState = rememberWifiScanState(
		autoScanOnStart = false, setScannState = { bool -> scanningForDevices = bool })
	val devices: List<String> = remember(devicesData.type, wifiScanState.networks) {
		when (devicesData.type) {
			AUDIO_DEVICE_TYPE -> listOf(
				context.getString(R.string.choose_eyeaivision_as_audio_text),
				context.getString(R.string.choose_system_as_audio_text)
			)

			VISION_DEVICE_TYPE -> wifiScanState.networks
			else -> emptyList()
		}
	}
	val deviceType = devicesData.type
	var pageLoading by rememberSaveable { mutableStateOf(true) }
	val chooseCameraAsInput = stringResource(R.string.choose_camera_as_input_text)

	Log.d(LOG_TAG, "[ConnectionPage] Choosing connection for $deviceCategory")

	LaunchedEffect(deviceType) {
		val locationManager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager

		if (!startAutoConnect) {
			if (deviceType == VISION_DEVICE_TYPE && locationManager.isLocationEnabled) wifiScanState.rescan()
			else if (deviceType == VISION_DEVICE_TYPE && !locationManager.isLocationEnabled) {
				showLocationDisabledDialog = true
			}
			return@LaunchedEffect
		}
		//If opened from the settings, the auto-connection is not needed, but in case of the
		//eye-ai-vision a Wi-Fi scan is started for the user
		if (uiState.connectionTutorialCompleted) {
			Log.d(
				LOG_TAG,
				"[ConnectionPage:LaunchedEffect] ConnectionTutorial completed, not automatic connection: Exiting LaunchedEffect"
			)
			if (deviceType == VISION_DEVICE_TYPE && locationManager.isLocationEnabled) wifiScanState.rescan()
			else if (deviceType == VISION_DEVICE_TYPE && !locationManager.isLocationEnabled) {
				showLocationDisabledDialog = true
			}
			pageLoading = false
			return@LaunchedEffect
		}

		//If the user does not want to automatically connect, exit
		//in case of the eye-ai-vision a Wi-Fi scan is started for the user
		if (!sharedPreferences.getBoolean(shouldRememberKey, false)) {
			Log.d(
				LOG_TAG,
				"[ConnectionPage:LaunchedEffect] User does not want automatic connection: Exiting LaunchedEffect"
			)
			pageLoading = false
			if (deviceType == VISION_DEVICE_TYPE && locationManager.isLocationEnabled) wifiScanState.rescan()
			else if (deviceType == VISION_DEVICE_TYPE && !locationManager.isLocationEnabled) {
				showLocationDisabledDialog = true
			}
			return@LaunchedEffect
		}

		// WIFI-Scan only necessary for eye-ai-vision, but not if user automatically connects to
		// phone camera
		var scanNetworks: List<String>? = null
		if (deviceType == VISION_DEVICE_TYPE && rememberedDevice != chooseCameraAsInput) {
			Log.d(LOG_TAG, "[ConnectionPage:LaunchedEffect] WIFI-Scan necessary, starting")
			if (locationManager.isLocationEnabled) scanNetworks = wifiScanState.awaitScan()
			else {
				Log.d(
					LOG_TAG,
					"[ConnectionPage:LaunchedEffect] Wifi-Scan failed. Location services are not turned on."
				)
				showLocationDisabledDialog = true
			}
		} else {
			Log.d(LOG_TAG, "[ConnectionPage:LaunchedEffect] WIFI-Scan not necessary, not starting")
		}
		val availableDevices = when (deviceType) {
			VISION_DEVICE_TYPE -> scanNetworks ?: emptyList()
			AUDIO_DEVICE_TYPE -> listOf(
				context.getString(R.string.choose_eyeaivision_as_audio_text),
				context.getString(R.string.choose_system_as_audio_text)
			)

			else -> emptyList()
		}

		//If the device the user wants to automatically connect to is not available, exit
		//Because phone camera is handled outside the devices list, it needs extra checking
		if (!availableDevices.contains(rememberedDevice) && rememberedDevice != chooseCameraAsInput) {
			Log.d(
				LOG_TAG,
				"[ConnectionPage:LaunchedEffect] Device for automatic connection not available: Exiting LaunchedEffect"
			)
			pageLoading = false
			return@LaunchedEffect
		}
		Log.d(
			LOG_TAG,
			"[ConnectionPage:LaunchedEffect] Attempting to connect to remembered ${devicesData.type} device"
		)
		connectToDevice(
			context, devicesData.type, rememberedDevice as String, onEvent = onEvent
		) { success ->
			if (success) {
				Log.d(
					LOG_TAG,
					"[ConnectionPage:LaunchedEffect] Connection to remembered device successful"
				)
				onConnectionSuccessful(rememberedDevice)
			} else {
				pageLoading = false
				Log.d(
					LOG_TAG,
					"[ConnectionPage:LaunchedEffect] Connection to remembered device not successful"
				)
			}
		}

	}


	Surface(
		modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface
	) {
		Column(verticalArrangement = Arrangement.Center) {
			Card(
				modifier = Modifier
					.fillMaxWidth()
					.padding(Spacing.md),
				shape = PremiumShapes.large,
				elevation = CardDefaults.cardElevation(AppElevation.level5),
				border = BorderStroke(
					width = if (isDark) 2.dp else 0.dp, color = Color.White.copy(alpha = 0.2f)
				),
				colors = CardDefaults.cardColors(
					containerColor = MaterialTheme.colorScheme.primaryContainer,
					contentColor = MaterialTheme.colorScheme.onPrimaryContainer
				)
			) {
				if (pageLoading) {
					LoadingPage()
				} else {
					Column(modifier = Modifier.semantics { isTraversalGroup = true }) {
						Row(
							modifier = Modifier
								.fillMaxWidth()
								.padding(Spacing.md)
								.semantics { traversalIndex = -1f },
							horizontalArrangement = Arrangement.Center
						) {
							Text(
								deviceCategory,
								modifier = Modifier
									.focusRequester(focusRequester)
									.focusable()
									.clearAndSetSemantics {
										contentDescription = devicesData.nameSemantic
									},
								style = MaterialTheme.typography.headlineLarge,
								textAlign = TextAlign.Center
							)
						}
						HorizontalDivider(
							color = MaterialTheme.colorScheme.outline,
							modifier = Modifier
								.padding(Spacing.sm)
								.clearAndSetSemantics {})
						if (devices.isNotEmpty()) {
							LazyColumn(
								modifier = Modifier
									.padding(Spacing.sm)
									.fillMaxWidth()
									.semantics { traversalIndex = 0f }) {
								items(items = devices) { item ->
									DeviceListEntry(
										item, onSelected = {
											selectedDevice =
												if (selectedDevice != item) item else ""
										}, isSelected = item == selectedDevice
									)
								}
								if (devicesData.type == VISION_DEVICE_TYPE) {
									item {
										DeviceListEntry(
											stringResource(R.string.choose_camera_as_input_text),
											onSelected = {
												selectedDevice =
													if (selectedDevice != chooseCameraAsInput) chooseCameraAsInput else ""
											},
											isSelected = chooseCameraAsInput == selectedDevice
										)
									}
								}
							}
						} else {
							Row(
								modifier = Modifier
									.fillMaxWidth()
									.padding(
										start = Spacing.md,
										top = Spacing.sm,
										bottom = Spacing.sm,
										end = Spacing.sm
									)
							) {
								if (scanningForDevices) {
									ShimmerBox(
										modifier = Modifier
											.height(Spacing.md)
											.fillMaxWidth(0.6f),
										backgroundColor = MaterialTheme.colorScheme.primaryContainer,
										contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
									)
								} else Text(
									stringResource(R.string.no_available_devices_text),
									style = MaterialTheme.typography.bodyMedium
								)
							}
							DeviceListEntry(
								chooseCameraAsInput, onSelected = {
									selectedDevice =
										if (selectedDevice != chooseCameraAsInput) chooseCameraAsInput else ""
								}, isSelected = chooseCameraAsInput == selectedDevice
							)


						}
						HorizontalDivider(
							color = MaterialTheme.colorScheme.outline, modifier = Modifier.padding(
								top = Spacing.sm, start = Spacing.sm, end = Spacing.sm
							)
						)
						Row(
							modifier = Modifier
								.fillMaxWidth()
								.padding(
									start = Spacing.sm,
									end = Spacing.lg,
									top = Spacing.xs,
									bottom = Spacing.xs
								)
								.semantics { traversalIndex = 0f },
							verticalAlignment = Alignment.CenterVertically,
							horizontalArrangement = Arrangement.SpaceBetween
						) {
							Row(verticalAlignment = Alignment.CenterVertically) {
								Checkbox(modifier = Modifier.semantics {
									contentDescription =
										context.getString(R.string.set_device_as_default_semantic)
								}, checked = shouldRememberDevice, onCheckedChange = {
									shouldRememberDevice = !shouldRememberDevice
								})
								Text(
									stringResource(R.string.standard_device_text),
									modifier = Modifier.clearAndSetSemantics {},
									style = MaterialTheme.typography.bodyLarge
								)
							}
							if (deviceType == VISION_DEVICE_TYPE) {
								val locationManager = remember(context) {
									context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
								}
								val onRescanClick = remember(wifiScanState) {
									{
										if (locationManager.isLocationEnabled) {
											wifiScanState.rescan()
										} else {
											showLocationDisabledDialog = true
										}
									}
								}
								PremiumIconButton(
									onClick = onRescanClick
								) {
									Icon(
										modifier = Modifier
											.heightIn(Spacing.xl)
											.width(Spacing.xl),
										painter = painterResource(R.drawable.refresh_24px),
										contentDescription = stringResource(R.string.refresh_icon_description),
										tint = MaterialTheme.colorScheme.onSurfaceVariant
									)
								}
							}
						}
						HorizontalDivider(
							color = MaterialTheme.colorScheme.outline, modifier = Modifier.padding(
								bottom = Spacing.sm, start = Spacing.sm, end = Spacing.sm
							)
						)
						Row(
							modifier = Modifier
								.fillMaxWidth()
								.padding(Spacing.md)
								.semantics { traversalIndex = 1f },
							horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
						) {
							PremiumButton(
								modifier = Modifier.weight(1f),
								shadowElevation = if (isDark) AppElevation.level4 else AppElevation.level2,
								onClick = { goBack() }) {
								Text(
									stringResource(R.string.return_text),
									style = MaterialTheme.typography.labelLarge
								)
							}
							PremiumButton(
								modifier = Modifier.weight(1f),
								enabled = selectedDevice != "",
								shadowElevation = if (isDark) AppElevation.level4 else AppElevation.level2,
								onClick = {
									connectToDevice(
										context, devicesData.type, selectedDevice, onEvent = onEvent
									) { success ->
										if (success) {
											Log.d(
												LOG_TAG,
												"[ConnectionPage] Setting SharedPreferences ShouldRememberDevice: $shouldRememberDevice"
											)
											sharedPreferences.edit(commit = true) {
												putBoolean(
													shouldRememberKey, shouldRememberDevice
												)
											}
											Log.d(
												LOG_TAG,
												"[ConnectionPage] Setting SharedPreferences SelectedDevice: $selectedDevice"
											)
											sharedPreferences.edit(commit = true) {
												putString(
													selectedDeviceKey,
													if (shouldRememberDevice) selectedDevice else ""
												)
											}
											onConnectionSuccessful(selectedDevice)
											shouldRememberDevice = false
											selectedDevice = ""
										} else showConnectionFailedDialog = true
									}
								}) {
								Text(
									stringResource(R.string.connect_text),
									modifier = Modifier.clearAndSetSemantics {
										contentDescription =
											context.getString(R.string.connect_to_device_semantic)
									},
									style = MaterialTheme.typography.labelLarge
								)
							}
						}
					}
				}
			}
		}
	}

	if (showConnectionFailedDialog) {
		ErrorDialog(titel = {
			Text(
				stringResource(R.string.connection_failed_title),
				style = MaterialTheme.typography.titleLarge
			)
		}, content = {
			Text(
				stringResource(R.string.connection_failed_text),
				style = MaterialTheme.typography.bodyMedium
			)
		}, onDismissed = { showConnectionFailedDialog = false })
	}

	if (showLocationDisabledDialog) {
		ActivateLocationServicesDialog(
			onDismissed = { showLocationDisabledDialog = false },
			onGranted = {
				showLocationDisabledDialog = false
				wifiScanState.rescan()
			})
	}
}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun DeviceListEntry(deviceName: String, isSelected: Boolean = false, onSelected: () -> Unit) {
	val context = LocalContext.current
	Row(
		modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically
	) {
		RadioButton(modifier = Modifier.semantics {
			contentDescription = context.getString(R.string.radio_button_semantic) + deviceName
		}, onClick = { onSelected() }, selected = isSelected)
		Text(
			deviceName, Modifier.clearAndSetSemantics {})
	}
}

@Composable
fun ActivateLocationServicesDialog(onDismissed: () -> Unit, onGranted: () -> Unit) {
	val context = LocalContext.current

	val locationRequest = remember {
		LocationRequest.Builder(
			Priority.PRIORITY_HIGH_ACCURACY, 10_000L
		).build()
	}

	val locationSettingsRequest = remember {
		LocationSettingsRequest.Builder().addLocationRequest(locationRequest).build()
	}

	val settingsClient = remember {
		LocationServices.getSettingsClient(context)
	}

	val launcher = rememberLauncherForActivityResult(
		contract = ActivityResultContracts.StartIntentSenderForResult()
	) { result ->
		if (result.resultCode == Activity.RESULT_OK) {
			// Nutzer hat die Standortdienste aktiviert
			onGranted()
		}
	}

	fun activateLocationServices() {
		settingsClient.checkLocationSettings(locationSettingsRequest).addOnSuccessListener {
			// Standort ist bereits aktiviert
			onGranted()
		}.addOnFailureListener { exception ->
			if (exception is ResolvableApiException) {
				try {
					val intentSenderRequest = IntentSenderRequest.Builder(
						exception.resolution
					).build()

					launcher.launch(intentSenderRequest)
				} catch (_: IntentSender.SendIntentException) {
					// Systemdialog konnte nicht geöffnet werden
				}
			}
		}
	}

	AlertDialog(
		onDismissRequest = { onDismissed() },
		title = { Text(stringResource(R.string.location_disabled_title)) },
		text = { Text(stringResource(R.string.location_disabled_text)) },
		confirmButton = {
			Row(
				modifier = Modifier.fillMaxWidth(),
				horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
			) {
				PremiumButton(modifier = Modifier.weight(1f), onClick = { onDismissed() }) {
					Text(stringResource(R.string.return_text))
				}
				PremiumButton(
					modifier = Modifier.weight(1f), onClick = { activateLocationServices() }) {
					Text(stringResource(R.string.activate_text))
				}
			}

		})
}

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun ErrorDialog(
	titel: @Composable () -> Unit, content: @Composable () -> Unit, onDismissed: () -> Unit
) {
	val context = LocalContext.current
	AlertDialog(
		onDismissRequest = { onDismissed() },
		title = titel,
		text = content,
		confirmButton = {
			PremiumButton(onClick = { onDismissed() }) {
				Text(
					stringResource(R.string.understood_button_text),
					modifier = Modifier.clearAndSetSemantics {
						contentDescription = context.getString(R.string.understood_button_semantic)
					})
			}
		})

}

@Composable
fun LoadingPage() {
	Column(verticalArrangement = Arrangement.Center) {
		Card(
			modifier = Modifier
				.fillMaxWidth()
				.padding(Spacing.md),
			colors = CardDefaults.cardColors(
				containerColor = MaterialTheme.colorScheme.primaryContainer,
				contentColor = MaterialTheme.colorScheme.onPrimaryContainer
			)
		) {
			Column {
				Row(
					modifier = Modifier
						.fillMaxWidth()
						.padding(Spacing.md),
					horizontalArrangement = Arrangement.Center
				) {
					ShimmerBox(
						modifier = Modifier
							.fillMaxWidth(0.7f)
							.height(Spacing.xxxl),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
				}
				HorizontalDivider(
					modifier = Modifier
						.padding(Spacing.sm)
						.clearAndSetSemantics {})
				Row(
					modifier = Modifier
						.fillMaxWidth()
						.padding(
							start = Spacing.sm,
							top = Spacing.sm,
							bottom = Spacing.sm,
							end = Spacing.sm
						)
				) {
					ShimmerBox(
						modifier = Modifier
							.fillMaxWidth(0.6f)
							.height(Spacing.md),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
				}
				Row(
					modifier = Modifier
						.fillMaxWidth()
						.padding(
							start = Spacing.sm,
							top = Spacing.sm,
							bottom = Spacing.sm,
							end = Spacing.sm
						)
				) {
					ShimmerBox(
						modifier = Modifier
							.fillMaxWidth(0.6f)
							.height(Spacing.md),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
				}
				Row(
					modifier = Modifier
						.fillMaxWidth()
						.padding(
							start = Spacing.sm,
							top = Spacing.sm,
							bottom = Spacing.sm,
							end = Spacing.sm
						)
				) {
					ShimmerBox(
						modifier = Modifier
							.fillMaxWidth(0.6f)
							.height(Spacing.md),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
				}
				HorizontalDivider(
					modifier = Modifier.padding(
						top = Spacing.sm, start = Spacing.sm, end = Spacing.sm
					)
				)
				Row(
					modifier = Modifier
						.fillMaxWidth()
						.padding(
							start = Spacing.md,
							end = Spacing.xl,
							top = Spacing.md,
							bottom = Spacing.md
						),
					verticalAlignment = Alignment.CenterVertically,
					horizontalArrangement = Arrangement.SpaceBetween
				) {
					ShimmerBox(
						modifier = Modifier
							.fillMaxWidth(0.7f)
							.height(Spacing.xl),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
				}
				HorizontalDivider(
					modifier = Modifier.padding(
						bottom = Spacing.sm, start = Spacing.sm, end = Spacing.sm
					)
				)
				Row(
					modifier = Modifier
						.fillMaxWidth()
						.padding(Spacing.md),
					horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
				) {
					ShimmerBox(
						modifier = Modifier
							.weight(1f)
							.height(Spacing.xxl),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
					ShimmerBox(
						modifier = Modifier
							.weight(1f)
							.height(Spacing.xxl),
						backgroundColor = MaterialTheme.colorScheme.primaryContainer,
						contrastColor = MaterialTheme.colorScheme.onPrimaryContainer
					)
				}
			}
		}
	}
}



