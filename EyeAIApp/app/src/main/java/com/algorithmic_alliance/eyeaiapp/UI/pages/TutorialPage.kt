package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.annotation.SuppressLint
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.isTraversalGroup
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.PremiumButton
import com.algorithmic_alliance.eyeaiapp.UI.onPermissionDecline
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import uniffi.NativeLib.UniffiDetectedObject

enum class TutorialStage {
    ObjectTutorial,
    DepthTutorial,
    NLPTutorial,
}


val SAMPLE_OBJECTS = arrayOf(
    UniffiDetectedObject(
        x1 = 0.6430664f,
        y1 = 0.5527344f,
        x2 = 0.9995117f,
        y2 = 0.99609375f,
        cx = 0.82128906f,
        cy = 0.77441406f,
        w = 0.3564453f,
        h = 0.44335938f,
        cnf = 0.65f,
        cls = 0,
        clsName = "person",
        trackingId = 1,
    ),
    UniffiDetectedObject(
        x1 = 0.09442021f,
        y1 = 0.4609375f,
        x2 = 0.35382193f,
        y2 = 0.9941406f,
        cx = 0.22412108f,
        cy = 0.72753906f,
        w = 0.25940174f,
        h = 0.5332031f,
        cnf = 0.68f,
        cls = 56,
        clsName = "chair",
        trackingId = 2,
    )
)

@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun TutorialPage(onFinishTutorial: () -> Unit, onAbortTutorial: () -> Unit) {

    val context = LocalContext.current
    var currentTutorialStage by rememberSaveable { mutableStateOf(TutorialStage.DepthTutorial) }
    key(currentTutorialStage) {
        Column(modifier = Modifier.fillMaxHeight(), verticalArrangement = Arrangement.Center) {
            when (currentTutorialStage) {
                TutorialStage.DepthTutorial -> {
                    DepthTutorial(onBack = { onAbortTutorial() }, onGoOn = {
                        currentTutorialStage =
                            TutorialStage.ObjectTutorial
                    })
                }

                TutorialStage.ObjectTutorial -> ObjectTutorial(onBack = {
                    currentTutorialStage =
                        TutorialStage.DepthTutorial
                }, onGoOn = {
                    currentTutorialStage =
                        TutorialStage.NLPTutorial
                })

                TutorialStage.NLPTutorial -> NLPTutorial(onBack = {
                    currentTutorialStage = TutorialStage.ObjectTutorial
                }, onGoOn = {
                    val sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context)
                    sharedPreferences.edit(commit = true) {
                        putBoolean(context.getString(R.string.app_tutorial_completed), true)
                    }
                    onFinishTutorial()
                })
            }
        }
    }
}

@Composable
fun ObjectTutorial(onBack: () -> Unit, onGoOn: () -> Unit) {
    val isDark = isSystemInDarkTheme()
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
            Text(
                "Objekt Tutorial",
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
                        onBack()
                    }) {
                    Text(
                        stringResource(R.string.return_text),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
                PremiumButton(
                    modifier = Modifier
                        .weight(1f),
                    shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                    onClick = { onGoOn()}) {
                    Text(
                        stringResource(R.string.understood_button_text),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            }
        }
    }
}

@Composable
fun DepthTutorial(onBack: () -> Unit, onGoOn: () -> Unit) {
    val isDark = isSystemInDarkTheme()
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
            Text(
                "Depth Tutorial",
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
                        onBack()
                    }) {
                    Text(
                        stringResource(R.string.return_text),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
                PremiumButton(
                    modifier = Modifier
                        .weight(1f),
                    shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                    onClick = { onGoOn()}) {
                    Text(
                        stringResource(R.string.understood_button_text),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            }
        }
    }
}

@Composable
fun NLPTutorial(onBack: () -> Unit, onGoOn: () -> Unit) {
    val isDark = isSystemInDarkTheme()
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
            Text(
                "NLP Tutorial",
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
                        onBack()
                    }) {
                    Text(
                        stringResource(R.string.return_text),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
                PremiumButton(
                    modifier = Modifier
                        .weight(1f),
                    shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                    onClick = { onGoOn()}) {
                    Text(
                        stringResource(R.string.understood_button_text),
                        style = MaterialTheme.typography.labelLarge
                    )
                }
            }
        }
    }
}