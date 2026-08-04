package com.algorithmic_alliance.eyeaiapp.UI


import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.material3.Checkbox
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.lazy.items
import com.algorithmic_alliance.eyeaiapp.R
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.text.input.rememberTextFieldState
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
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.remote.core.operations.layout.modifiers.ShapeType.getString
import androidx.compose.remote.creation.toFloat
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.preference.Preference
import com.algorithmic_alliance.eyeaiapp.BuildInfoHelper
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource
import kotlin.math.exp
import kotlin.math.roundToInt

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsPage(modifier: Modifier = Modifier, onReturn: () -> Unit) {

    val settingsData = UIDataSource.APP_SETTINGS
    var developerSettingsEnabled by rememberSaveable { mutableStateOf(false) }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            TopAppBar(
                modifier = Modifier.shadow(elevation = 8.dp),
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
                        Text("Einstellungen")
                    }
                },
            )
        },
        content = { innerPadding ->
            Column(
                modifier = Modifier.padding(innerPadding),
            ) {
                LazyColumn(modifier = modifier.padding(vertical = 4.dp)) {
                    items(
                        items = settingsData.entries.toList(),
                        key = { entry -> entry.key }) { entry ->
                        if ((entry.key != "Developer Settings") || (entry.key == "Developer Settings" && developerSettingsEnabled))
                            SettingsCategoryCard(
                                categorySettings = entry.value as List<Any>,
                                category = entry.key
                            )
                        else
                            Card(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(8.dp)
                                    .clickable { developerSettingsEnabled = true }
                                    .clearAndSetSemantics {}
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(16.dp),
                                    horizontalArrangement = Arrangement.Center
                                ) {
                                    Text(
                                        "Enable Developer Settings",
                                        fontSize = 22.sp,
                                        fontWeight = FontWeight.Bold
                                    )
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
    category: String
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
                Text(
                    category,
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            HorizontalDivider()
            for (item in categorySettings) {

                val settingData = item as Map<String, Any>

                when (settingData.getValue("settingsType")) {
                    "checkbox" -> CheckBoxSetting(modifier = Modifier, settingData = settingData)
                    "select" -> SelectSetting(modifier = Modifier, settingData = settingData)
                    "slider" -> SliderSetting(modifier = Modifier, settingData = settingData)
                    "textInput" -> TextInputSetting(modifier = Modifier, settingData = settingData)
                    "file" -> FileSetting(modifier = Modifier, settingData = settingData)
                    "Info" -> InfoSetting(modifier = Modifier, settingData = settingData)
                }
            }

        }
    }
}

@Composable
fun CheckBoxSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {

    var checked by rememberSaveable { mutableStateOf(settingData.getValue("default") as Boolean) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(settingData.getValue("title") as String, fontSize = 18.sp)
            if (settingData.getValue("description") as String != "") Text(settingData.getValue("description") as String)
        }
        Checkbox(checked = checked, onCheckedChange = { isChecked -> checked = isChecked })
    }
}

@Composable
fun SelectSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {

    var dropDownEnabled by rememberSaveable { mutableStateOf(false) }
    var currentlySelected by rememberSaveable { mutableStateOf((settingData.getValue("settingsOptions") as List<Any>)[0]) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(settingData.getValue("title") as String, fontSize = 18.sp)
            if (settingData.getValue("description") as String != "") Text(settingData.getValue("description") as String)
        }
        Box {
            Row(
                modifier = Modifier.clickable {
                    dropDownEnabled = true
                },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(currentlySelected as String)
                Icon(
                    painter = painterResource(R.drawable.arrow_drop_down_24px),
                    contentDescription = ""
                )
            }
            DropdownMenu(
                expanded = dropDownEnabled,
                onDismissRequest = { dropDownEnabled = false }) {
                for (item in (settingData.getValue("settingsOptions") as List<Any>)) {
                    DropdownMenuItem(text = { Text(item as String) }, onClick = {
                        currentlySelected = item as String
                        dropDownEnabled = false
                    })
                }
            }
        }
    }
}

@Composable
fun SliderSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {

    val settingsOptions = settingData.getValue("settingsOption") as Map<Any, Any>

    val min = (settingsOptions.getValue("min") as Number).toFloat()
    val max = (settingsOptions.getValue("max") as Number).toFloat()
    val default = (settingsOptions.getValue("default") as Number).toFloat()

    var currentValue by rememberSaveable { mutableFloatStateOf(default) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Text(
                    settingData.getValue("title") as String,
                    fontSize = 18.sp,
                    modifier = Modifier.weight(1f)
                )
                Text("$currentValue")
            }
            if (settingData.getValue("description") as String != "") Text(settingData.getValue("description") as String)
            Slider(
                value = currentValue,
                onValueChange = { currentValue = (it.roundToInt() as Number).toFloat() },
                valueRange = min..max
            )
        }

    }
}

@Composable
fun TextInputSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {

    var showTextFieldDialog by rememberSaveable { mutableStateOf(false) }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(settingData.getValue("title") as String, fontSize = 18.sp)
            if (settingData.getValue("description") as String != "") Text(settingData.getValue("description") as String)
        }
        Box {

            IconButton(onClick = { showTextFieldDialog = true }) {
                Icon(
                    painter = painterResource(R.drawable.ink_pen_24px),
                    contentDescription = ""
                )
            }


        }

    }

    if (showTextFieldDialog) {
        TextFieldDialog(
            onDismiss = { showTextFieldDialog = false },
            settingName = settingData.getValue("title") as String
        )
    }
}

@Composable
fun TextFieldDialog(modifier: Modifier = Modifier, onDismiss: () -> Unit, settingName: String) {
    var text by rememberSaveable { mutableStateOf("") }
    AlertDialog(
        onDismissRequest = { onDismiss() },
        title = {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(onClick = { onDismiss() }) {
                    Icon(
                        painterResource(R.drawable.arrow_back_24px),
                        contentDescription = UIDataSource.RETURN_SEMANTIC
                    )
                }
                Text(settingName)
            }
        },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                label = { Text("Eingeben...") }
            )
        },
        confirmButton = {
            Button(onClick = {
                onDismiss()
                //TODO backend connection
            }) {
                Text("Fertig", modifier = Modifier.clearAndSetSemantics {})
            }
        }
    )
}

@Composable
fun InfoSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {

    Row(modifier = Modifier.padding(8.dp)) {
        Column {
            Text(
                settingData.getValue("title") as String,
                fontSize = 18.sp,
            )
            if (settingData.getValue("description") as String != "") Text(settingData.getValue("description") as String)
        }
    }


}

@Composable
fun FileSetting(modifier: Modifier = Modifier, settingData: Map<String, Any>) {
    var selectedFileUri by rememberSaveable { mutableStateOf<Uri?>(null) }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        selectedFileUri = uri
    }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(settingData.getValue("title") as String, fontSize = 18.sp)
            if (settingData.getValue("description") as String != "") Text(settingData.getValue("description") as String)
        }
        IconButton(onClick = { filePickerLauncher.launch("*/*") }) {
            Icon(
                painter = painterResource(R.drawable.upload_file_24px),
                contentDescription = ""
            )
        }
    }
}


@Preview(showBackground = true, name = "SettingsPage Preview")
@Composable
fun SettingsPagePreview() {
    SettingsPage(onReturn = {})
}
