package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.annotation.SuppressLint
import android.util.Log
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.isTraversalGroup
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.PremiumButton
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun WelcomePage(
    modifier: Modifier = Modifier,
    onGetStarted: () -> Unit,
    onStartTutorial: () -> Unit,
    onEvent: (UIEvent) -> Unit
) {
    val context = LocalContext.current
    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
    val isDark = isSystemInDarkTheme()
    Log.d(LOG_TAG, "[WelcomePage] Loading WelcomePage")

    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
        Column(modifier = Modifier.fillMaxHeight(), verticalArrangement = Arrangement.Center) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(Spacing.md),
                shape = PremiumShapes.large,
                elevation = CardDefaults.cardElevation(AppElevation.level5),
                border = BorderStroke(
                    width = if (isDark) 2.dp else 0.dp,
                    color = Color.White.copy(alpha = 0.2f)
                ),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    contentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            ) {
                Column(
                    modifier = Modifier
                        .padding(Spacing.md)
                        .semantics { isTraversalGroup = true }
                ) {

                    Row(
                        modifier = Modifier
                            .padding(Spacing.md)
                            .fillMaxWidth()
                            .semantics { traversalIndex = -1f },
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Image(
                            painter = painterResource(R.drawable.ic_launcher_web),
                            contentDescription = stringResource(R.string.app_logo_description)
                        )
                    }
                    Text(
                        "Möchten Sie eine interaktive Einführung in die App bekommen?",
                        modifier = Modifier
                            .padding(Spacing.md)
                            .semantics { traversalIndex = 0f },
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        style = MaterialTheme.typography.bodyLarge,
                        fontWeight = FontWeight.Medium
                    )
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = Spacing.md)
                            .semantics { traversalIndex = 1f },
                        horizontalArrangement = Arrangement.spacedBy(Spacing.sm)
                    ) {
                        PremiumButton(
                            modifier = Modifier
                                .weight(1f),
                            shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                            onClick = {
                                sharedPreferences.edit(commit = true) {
                                    putBoolean(
                                        context.getString(R.string.app_tutorial_completed),
                                        true
                                    )
                                }
                                onGetStarted()
                            }) {
                            Text(
                                stringResource(R.string.reject_tutorial_text),
                                style = MaterialTheme.typography.labelLarge
                            )
                        }
                        PremiumButton(
                            modifier = Modifier
                                .weight(1f),
                            shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                            onClick = { onStartTutorial() }) {
                            Text(
                                stringResource(R.string.accept_tutorial_text),
                                style = MaterialTheme.typography.labelLarge
                            )
                        }
                    }
                }
            }
        }
    }

}

@Preview(showBackground = true, name = "WelcomePage Preview")
@Composable
fun Preview() {
    MaterialTheme {
        WelcomePage(
            Modifier.fillMaxSize(),
            onGetStarted = {},
            onEvent = {},
            onStartTutorial = {})
    }
}
