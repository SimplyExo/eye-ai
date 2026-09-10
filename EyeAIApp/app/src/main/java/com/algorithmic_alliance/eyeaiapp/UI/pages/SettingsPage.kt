package com.algorithmic_alliance.eyeaiapp.UI.pages


import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.compose.foundation.clickable
import androidx.compose.material3.Checkbox
import androidx.core.content.edit
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.runtime.key
import androidx.compose.foundation.lazy.items
import com.algorithmic_alliance.eyeaiapp.R
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.font.FontWeight
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.UI.MainViewModel
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.UI.hasPermission
import com.algorithmic_alliance.eyeaiapp.data.SelectOption
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import com.algorithmic_alliance.eyeaiapp.runtime.BatteryOptimization
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsPage(
    viewModel: MainViewModel,
    modifier: Modifier = Modifier,
    onReturn: () -> Unit,
    onOpenDebugPage: () -> Unit,
    onOpenHomePage: () -> Unit,
    onEvent: (UIEvent) -> Unit,
    onOpenConnectionPage: () -> Unit
) {

    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    val settingsData = UIDataSource.APP_SETTINGS
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val debugPageActivatedKey = stringResource(R.string.debug_page_activated)
    val debugPageActivated = sharedPreferences.getBoolean(debugPageActivatedKey, false)

    DisposableEffect(Unit) {
        onEvent(UIEvent.OnOpenSettings)
        onEvent(UIEvent.OnUpdateSettingsOpened(true))
        onDispose {
            if (!viewModel.uiState.value.actionStartedFromSettings) {
                onEvent(UIEvent.OnReturnFromSettings)
                onEvent(UIEvent.OnUpdateSettingsOpened(false))
            }
        }
    }

    Scaffold(modifier = Modifier.fillMaxSize(), topBar = {
        TopAppBar(
            modifier = Modifier.shadow(elevation = Spacing.sm),
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
                titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
            ),
            title = {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = { onReturn() }) {
                        Icon(
                            painter = painterResource(R.drawable.arrow_back_24px),
                            contentDescription = "Zurück"
                        )
                    }
                    Text("Einstellungen", style = MaterialTheme.typography.titleLarge)
                }
            },
        )
    }, content = { innerPadding ->
        Column(
            modifier = Modifier.padding(innerPadding),
        ) {
            key(uiState.reloadSettingsPageKey) {
                LazyColumn(modifier = modifier.padding(vertical = Spacing.xs)) {
                    items(
                        items = settingsData.entries.toList(),
                        key = { entry -> entry.key }) { entry ->
                        if(entry.key != "Developer Settings" || (entry.key == "Developer Settings" && sharedPreferences.getBoolean(stringResource(R.string.debug_page_activated), false)))
                            SettingsCategoryCard(
                                categorySettings = entry.value as List<Any>,
                                category = entry.key,
                                onEvent = onEvent,
                                onOpenConnectionPage = onOpenConnectionPage
                            )
                    }
                    item {
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(Spacing.sm)
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(Spacing.md)
                                    .clickable {
                                        if (!debugPageActivated) {
                                            onOpenDebugPage()
                                            sharedPreferences.edit(commit = true) {
                                                putBoolean(debugPageActivatedKey, true)
                                            }
                                        } else {
                                            onOpenHomePage()
                                            sharedPreferences.edit(commit = true) {
                                                putBoolean(debugPageActivatedKey, false)
                                            }
                                        }

                                    }, horizontalArrangement = Arrangement.Center
                            ) {
                                Text(
                                    if (!debugPageActivated) "DebugPage aktivieren" else "DebugPage deaktivieren",
                                    style = MaterialTheme.typography.titleMedium,
                                )
                            }
                        }
                    }
                }
            }

        }
    })

}

@Composable
fun SettingsCategoryCard(
    modifier: Modifier = Modifier,
    categorySettings: List<Any>,
    category: String,
    onEvent: (UIEvent) -> Unit,
    onOpenConnectionPage: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.sm)
    ) {
        Column(modifier = Modifier.padding(Spacing.md)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                Text(
                    category,
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            HorizontalDivider()
            for (item in categorySettings) {

                val settingData = item as Map<String, Any>

                when (settingData.getValue("settingsType")) {
                    "checkbox" -> CheckBoxSetting(
                        modifier = Modifier, settingData = settingData, onEvent = onEvent
                    )

                    "select" -> SelectSetting(
                        modifier = Modifier, settingData = settingData, onEvent = onEvent
                    )

                    "slider" -> SliderSetting(
                        modifier = Modifier, settingData = settingData, onEvent = onEvent
                    )

                    "textInput" -> TextInputSetting(
                        modifier = Modifier, settingData = settingData, onEvent = onEvent
                    )

                    "file" -> FileSetting(
                        modifier = Modifier, settingData = settingData, onEvent = onEvent
                    )

                    "Info" -> InfoSetting(
                        modifier = Modifier, settingData = settingData,
                    )

                    "click" -> ClickSetting(
                        settingData = settingData,
                        onEvent = onEvent,
                        onOpenConnectionPage = onOpenConnectionPage
                    )
                }
            }

        }
    }
}

@Composable
fun ClickSetting(
    settingData: Map<String, Any>, onEvent: (UIEvent) -> Unit, onOpenConnectionPage: () -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val action = settingData["action"] as? String
    var batteryOptimizationExempt by remember(action) {
        mutableStateOf(
            action == UIDataSource.ACTION_OPEN_BATTERY_OPTIMIZATION &&
                BatteryOptimization.isExempt(context)
        )
    }

    DisposableEffect(lifecycleOwner, action) {
        if (action != UIDataSource.ACTION_OPEN_BATTERY_OPTIMIZATION) {
            onDispose { }
        } else {
            val observer = LifecycleEventObserver { _, event ->
                if (event == Lifecycle.Event.ON_RESUME) {
                    batteryOptimizationExempt = BatteryOptimization.isExempt(context)
                }
            }
            lifecycleOwner.lifecycle.addObserver(observer)
            onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
        }
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                settingData.getValue("title") as String,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            if (settingData.getValue("description") as String != "")
                Text(
                    settingData.getValue("description") as String,
                    style = MaterialTheme.typography.bodySmall,
                )
            if (action == UIDataSource.ACTION_OPEN_DEVICE_MANAGER) {
                val sharedPreferences =
                    PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
                val standardAudioDevice =
                    sharedPreferences.getString(stringResource(R.string.selected_audio_device), "")
                val standardVisionDevice =
                    sharedPreferences.getString(stringResource(R.string.selected_eye_ai_vision), "")
                Text(
                    "Audiogerät: ${if (standardAudioDevice != "") standardAudioDevice else "   -"}",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "Vision: ${if (standardVisionDevice != "") standardVisionDevice else "   -"}",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            if (action == UIDataSource.ACTION_OPEN_BATTERY_OPTIMIZATION) {
                Text(
                    if (batteryOptimizationExempt) {
                        "Status: von der Batterieoptimierung ausgenommen"
                    } else {
                        "Status: Batterieoptimierung aktiv"
                    },
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Box {
            IconButton(onClick = {
                when (action) {
                    UIDataSource.ACTION_OPEN_DEVICE_MANAGER -> {
                        onEvent(UIEvent.OnUpdateActionStartedFromSettings(true))
                        onOpenConnectionPage()
                    }
                    UIDataSource.ACTION_OPEN_BATTERY_OPTIMIZATION -> {
                        BatteryOptimization.openSettings(context)
                    }
                }
            }) {
                Icon(
                    painter = painterResource(R.drawable.change_circle_24px),
                    contentDescription = ""
                )
            }


        }
    }
}

@Composable
fun CheckBoxSetting(
    modifier: Modifier = Modifier, settingData: Map<String, Any>, onEvent: (UIEvent) -> Unit
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val settingKey = stringResource(settingData["string"] as Int)

    var checked by rememberSaveable {
        mutableStateOf(
            sharedPreferences.getBoolean(
                settingKey, settingData["default"] as Boolean
            )
        )
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                settingData.getValue("title") as String,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            if (settingData.getValue("description") as String != "") {
                Text(
                    settingData.getValue("description") as String,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Checkbox(checked = checked, onCheckedChange = { isChecked ->

            if (settingData["string"] == R.string.enable_speech_recognition_setting && !hasPermission(
                    context, Manifest.permission.RECORD_AUDIO
                )
            ) {
                onEvent(UIEvent.OnUpdateAppMissingVoskPermission(true))
                return@Checkbox
            }

            Log.d(
                LOG_TAG, "[SettingsPage.CheckBoxSetting] Setting $settingKey changed to $isChecked"
            )
            checked = isChecked
            sharedPreferences.edit(commit = true) {
                putBoolean(
                    settingKey, isChecked
                )
            }
            onEvent(UIEvent.UpdateSettings)

        })
    }
}

@Composable
fun SelectSetting(
    modifier: Modifier = Modifier, settingData: Map<String, Any>, onEvent: (UIEvent) -> Unit
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val settingKey = stringResource(settingData["string"] as Int)

    var dropDownEnabled by rememberSaveable { mutableStateOf(false) }
    var currentlySelected by rememberSaveable {
        mutableStateOf(
            (sharedPreferences.getString(
                settingKey, settingData["default"] as String?
            ))
        )
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                settingData.getValue("title") as String,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            if (settingData.getValue("description") as String != "") {
                Text(
                    settingData.getValue("description") as String,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Box {
            Row(
                modifier = Modifier.clickable {
                    dropDownEnabled = true
                }, verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    displayOption(context, settingData, currentlySelected.orEmpty()),
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                )
                Icon(
                    painter = painterResource(R.drawable.arrow_drop_down_24px),
                    contentDescription = ""
                )
            }
            DropdownMenu(
                expanded = dropDownEnabled, onDismissRequest = { dropDownEnabled = false }) {
                    for (item in (settingData.getValue("settingsOptions") as List<Any>)) {
                    DropdownMenuItem(text = {
                        Text(
                            displayOption(context, settingData, optionValue(item)),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }, onClick = {
                        currentlySelected = optionValue(item)
                        Log.d(
                            LOG_TAG,
                            "[SettingsPage.SelectSetting] Changed setting $settingKey to $currentlySelected"
                        )
                        sharedPreferences.edit(commit = true) {
                            putString(settingKey, currentlySelected)
                        }
                        onEvent(UIEvent.UpdateSettings)
                        dropDownEnabled = false
                    })
                }
            }
        }
    }
}

@Composable
fun SliderSetting(
    modifier: Modifier = Modifier, settingData: Map<String, Any>, onEvent: (UIEvent) -> Unit
) {
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val settingKey = stringResource(settingData["string"] as Int)
    val settingsOptions = settingData.getValue("settingsOption") as Map<Any, Any>
    val min = (settingsOptions.getValue("min") as Number).toFloat()
    val max = (settingsOptions.getValue("max") as Number).toFloat()

    val settingRes = settingData["string"] as Int
    val isDepthFrameRate = settingRes == R.string.max_depth_frame_rate_setting
    val isObjectFrameRate = settingRes == R.string.max_object_detection_frame_rate_setting
    val depthLimiterEnabled by rememberPreferenceBooleanState(
        key = stringResource(R.string.enable_depth_frame_rate_limit_setting),
        default = true,
    )
    val objectLimiterEnabled by rememberPreferenceBooleanState(
        key = stringResource(R.string.enable_object_detection_frame_rate_limit_setting),
        default = true,
    )
    val visible = when {
        isDepthFrameRate -> depthLimiterEnabled
        isObjectFrameRate -> objectLimiterEnabled
        else -> true
    }

    var currentValue by rememberSaveable {
        mutableIntStateOf(
            sharedPreferences.getInt(
                settingKey, settingData["default"] as Int
            ).coerceIn(min.toInt(), max.toInt())
        )
    }

    AnimatedVisibility(
        visible = visible,
        enter = fadeIn() + expandVertically(),
        exit = fadeOut() + shrinkVertically(),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(Spacing.sm),
            horizontalArrangement = Arrangement.spacedBy(Spacing.md),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(Spacing.md),
                ) {
                    Text(
                        settingData.getValue("title") as String,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        "$currentValue",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold,
                    )
                }
                if (settingData.getValue("description") as String != "") {
                    Text(
                        settingData.getValue("description") as String,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                Slider(
                    value = currentValue.toFloat(),
                    onValueChange = { value ->
                        currentValue = if (settingData.getValue("title") == "Audio-Frequency") {
                            (value / 10.0).roundToInt() * 10
                        } else {
                            value.roundToInt()
                        }
                        Log.d(
                            LOG_TAG,
                            "[SettingsPage.SliderSetting] Changed setting $settingKey to $currentValue",
                        )
                        sharedPreferences.edit(commit = true) {
                            putInt(settingKey, currentValue)
                        }
                        onEvent(UIEvent.UpdateSettings)
                    },
                    valueRange = min..max,
                )
            }
        }
    }
}

@Composable
fun TextInputSetting(
    modifier: Modifier = Modifier, settingData: Map<String, Any>, onEvent: (UIEvent) -> Unit
) {
    var showTextFieldDialog by rememberSaveable { mutableStateOf(false) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                settingData.getValue("title") as String,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            if (settingData.getValue("description") as String != "") {
                Text(
                    settingData.getValue("description") as String,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Box {
            IconButton(onClick = { showTextFieldDialog = true }) {
                Icon(
                    painter = painterResource(R.drawable.ink_pen_24px), contentDescription = ""
                )
            }


        }

    }

    if (showTextFieldDialog) {
        TextFieldDialog(
            onDismiss = { showTextFieldDialog = false },
            settingName = settingData.getValue("title") as String,
            settingData = settingData,
            onEvent = onEvent
        )
    }
}

@Composable
fun TextFieldDialog(
    modifier: Modifier = Modifier,
    onDismiss: () -> Unit,
    settingName: String,
    settingData: Map<String, Any>,
    onEvent: (UIEvent) -> Unit
) {
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val settingKey = stringResource(settingData["string"] as Int)
    var text by rememberSaveable {
        mutableStateOf(
            sharedPreferences.getString(
                settingKey, settingData["default"]?.toString()
            )
        )
    }
    AlertDialog(onDismissRequest = { onDismiss() }, title = {
        Row(
            modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(onClick = { onDismiss() }) {
                Icon(
                    painterResource(R.drawable.arrow_back_24px),
                    contentDescription = UIDataSource.RETURN_SEMANTIC
                )
            }
            Text(settingName)
        }
    }, text = {
        OutlinedTextField(
            value = text ?: "",
            onValueChange = { text = it },
            label = { Text("Eingeben...") })
    }, confirmButton = {
        Button(onClick = {
            onDismiss()
            Log.d(LOG_TAG, "[SettingsPage.TextFieldSetting] Changed setting $settingKey to $text")
            sharedPreferences.edit(commit = true) {
                putString(settingKey, text)
            }
            onEvent(UIEvent.UpdateSettings)
        }) {
            Text("Fertig", modifier = Modifier.clearAndSetSemantics {})
        }
    })
}

@Composable
fun InfoSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {

    Row(modifier = Modifier.padding(Spacing.md)) {
        Column {
            Text(
                settingData.getValue("title") as String,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            if (settingData.getValue("description") as String != "") {
                Text(
                    settingData.getValue("description") as String,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }


}

@Composable
fun FileSetting(
    modifier: Modifier = Modifier, settingData: Map<String, Any>, onEvent: (UIEvent) -> Unit
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val settingKey = stringResource(settingData["string"] as Int)
    var selectedFileUri by rememberSaveable { mutableStateOf<Uri?>(null) }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri == null) {
            return@rememberLauncherForActivityResult
        }

        try {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        } catch (e: SecurityException) {
            Log.e(LOG_TAG, "Could not persist URI permission", e)
        }

        selectedFileUri = uri

        sharedPreferences.edit {
            putString(settingKey, uri.toString())
        }
        Log.d(LOG_TAG, "[SettingsPage.FileSetting] Changed setting $settingKey to $uri")
        onEvent(UIEvent.UpdateSettings)
    }


    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                settingData.getValue("title") as String,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Medium,
            )
            if (settingData.getValue("description") as String != "") {
                Text(
                    settingData.getValue("description") as String,
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        IconButton(onClick = {
            filePickerLauncher.launch(
                arrayOf("image/*", "video/*")
            )
        }) {
            Icon(
                painter = painterResource(R.drawable.upload_file_24px), contentDescription = ""
            )
        }
    }
}

private fun optionValue(option: Any): String = when (option) {
    is SelectOption -> option.value
    else -> option.toString()
}

private fun displayOption(context: Context, settingData: Map<String, Any>, value: String): String {
    val options = settingData["settingsOptions"] as? List<*> ?: return value
    val option = options.firstOrNull { candidate ->
        candidate != null && optionValue(candidate) == value
    }
    return if (option is SelectOption) context.getString(option.labelRes) else value
}

@Composable
fun rememberPreferenceBooleanState(
    key: String,
    default: Boolean,
): androidx.compose.runtime.State<Boolean> {
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(LocalContext.current)
    val state = remember(key) { mutableStateOf(sharedPreferences.getBoolean(key, default)) }

    DisposableEffect(sharedPreferences, key, default) {
        val listener = SharedPreferences.OnSharedPreferenceChangeListener { preferences, changedKey ->
            if (changedKey == key) state.value = preferences.getBoolean(key, default)
        }
        sharedPreferences.registerOnSharedPreferenceChangeListener(listener)
        onDispose { sharedPreferences.unregisterOnSharedPreferenceChangeListener(listener) }
    }
    return state
}
