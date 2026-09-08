package com.algorithmic_alliance.eyeaiapp.UI.pages

import android.annotation.SuppressLint
import android.media.MediaPlayer
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.focusable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
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
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.isTraversalGroup
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.traversalIndex
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.edit
import androidx.preference.PreferenceManager
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.PremiumButton
import com.algorithmic_alliance.eyeaiapp.UI.PremiumIconButton
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.PremiumShapes
import com.algorithmic_alliance.eyeaiapp.data.Spacing
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG

enum class TutorialStage {
    ObjectTutorial,
    DepthTutorial,
    NLPTutorial,
}


@RequiresApi(Build.VERSION_CODES.UPSIDE_DOWN_CAKE)
@SuppressLint("LocalContextGetResourceValueCall")
@Composable
fun TutorialPage(
    onFinishTutorial: () -> Unit,
    onAbortTutorial: () -> Unit,
) {
    val context = LocalContext.current
    Log.d(LOG_TAG, "[TutorialPage] Loading TutorialPage")

    var currentTutorialStage by rememberSaveable { mutableStateOf(TutorialStage.DepthTutorial) }
    key(currentTutorialStage) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.surface) {
            Column(modifier = Modifier.fillMaxHeight(), verticalArrangement = Arrangement.Center) {
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

                    when (currentTutorialStage) {
                        TutorialStage.DepthTutorial -> {
                            Tutorial(
                                onBack = { onAbortTutorial() },
                                onGoOn = {
                                    currentTutorialStage =
                                        TutorialStage.ObjectTutorial
                                },
                                title = stringResource(R.string.depth_tutorial_title_text),
                                audioRes = R.raw.depth_tutorial
                            )
                        }

                        TutorialStage.ObjectTutorial -> Tutorial(
                            onBack = {
                                currentTutorialStage =
                                    TutorialStage.DepthTutorial
                            },
                            onGoOn = {
                                currentTutorialStage =
                                    TutorialStage.NLPTutorial
                            },
                            title = stringResource(R.string.object_tutorial_title_text),
                            audioRes = R.raw.object_tutorial,
                        )

                        TutorialStage.NLPTutorial -> Tutorial(
                            onBack = {
                                currentTutorialStage = TutorialStage.ObjectTutorial
                            },
                            onGoOn = {
                                val sharedPreferences =
                                    PreferenceManager.getDefaultSharedPreferences(context)
                                sharedPreferences.edit(commit = true) {
                                    putBoolean(
                                        context.getString(R.string.app_tutorial_completed),
                                        true
                                    )
                                }
                                onFinishTutorial()
                            },
                            title = stringResource(R.string.nlp_tutorial_title_text),
                            audioRes = R.raw.nlp_tutorial
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun Tutorial(
    onBack: () -> Unit,
    onGoOn: () -> Unit,
    title: String,
    audioRes: Int
) {
    val context = LocalContext.current
    val focusRequester = remember { FocusRequester() }
    val mediaPlayer = remember(audioRes) {
        MediaPlayer.create(context, audioRes)
    }
    LaunchedEffect(mediaPlayer) {
        focusRequester.requestFocus()
        mediaPlayer.seekTo(0)
    }

    var isPlaying by remember { mutableStateOf(false) }
    DisposableEffect(mediaPlayer) {
        mediaPlayer.setOnCompletionListener { isPlaying = false }

        onDispose {
            mediaPlayer.setOnCompletionListener(null)
            mediaPlayer.pause()
            mediaPlayer.seekTo(0)
        }
    }
    val isDark = isSystemInDarkTheme()
    Column(
        modifier = Modifier
            .padding(Spacing.md)
            .semantics { isTraversalGroup = true }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .semantics { traversalIndex = -1f },
            horizontalArrangement = Arrangement.Center
        ) {
            Text(
                title,
                modifier = Modifier
                    .padding(Spacing.sm)
                    .focusRequester(focusRequester)
                    .focusable(),
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Medium,
                textAlign = TextAlign.Center
            )
        }
        HorizontalDivider(
            color = MaterialTheme.colorScheme.outline,
            modifier = Modifier
                .padding(Spacing.sm, bottom = Spacing.lg, top = Spacing.sm, end = Spacing.sm)
                .clearAndSetSemantics {})
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .semantics { traversalIndex = 0f },
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                PremiumIconButton(
                    modifier = Modifier
                        .width(Spacing.xxl)
                        .height(Spacing.xxl),
                    onClick = {
                        mediaPlayer.seekTo(0)
                        if (!mediaPlayer.isPlaying) {
                            mediaPlayer.start()
                            isPlaying = true
                        }
                    },
                    shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                ) {
                    Icon(
                        modifier = Modifier
                            .width(Spacing.xl)
                            .height(Spacing.xl),
                        painter = painterResource(R.drawable.refresh_24px),
                        contentDescription = stringResource(R.string.restart_tutorial_audio_button_semantic)
                    )
                }
                Text(
                    modifier = Modifier
                        .padding(Spacing.sm)
                        .clearAndSetSemantics {},
                    text = stringResource(R.string.restart_tutorial_button_text),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                PremiumIconButton(
                    modifier = Modifier
                        .width(Spacing.xxl)
                        .height(Spacing.xxl),
                    onClick = {
                        if (!isPlaying)
                            mediaPlayer.start()
                        else
                            mediaPlayer.pause()
                        isPlaying = !isPlaying
                    },
                    shadowElevation = if (isDark) AppElevation.level5 else AppElevation.level3,
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary
                ) {
                    Icon(
                        modifier = Modifier
                            .width(Spacing.xl)
                            .height(Spacing.xl),
                        painter = if (isPlaying) painterResource(R.drawable.pause_playback_24px) else painterResource(
                            R.drawable.play_arrow_24px
                        ),
                        contentDescription = stringResource(R.string.start_pause_tutorial_audio_button_semantic)
                    )
                }
                Text(
                    modifier = Modifier
                        .padding(Spacing.sm)
                        .clearAndSetSemantics {},
                    text = if (isPlaying) stringResource(R.string.pause_tutorial_audio_button_text) else stringResource(
                        R.string.start_tutorial_audio_button_text
                    ),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }
        HorizontalDivider(
            color = MaterialTheme.colorScheme.outline,
            modifier = Modifier
                .padding(Spacing.sm, bottom = Spacing.sm, top = Spacing.lg, end = Spacing.sm)
                .clearAndSetSemantics {})
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
                onClick = { onGoOn() }) {
                Text(
                    stringResource(R.string.understood_button_text),
                    style = MaterialTheme.typography.labelLarge
                )
            }
        }

    }
}
