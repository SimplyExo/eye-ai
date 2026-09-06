package com.algorithmic_alliance.eyeaiapp.UI.pages

import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.UI.PremiumButton
import com.algorithmic_alliance.eyeaiapp.UI.UIEvent
import com.algorithmic_alliance.eyeaiapp.data.AppElevation
import com.algorithmic_alliance.eyeaiapp.data.Spacing

@Composable
fun WelcomePage(
    modifier: Modifier = Modifier,
    onGetStarted: () -> Unit,
    onEvent: (UIEvent) -> Unit
) {
    val isDark = isSystemInDarkTheme()

    DisposableEffect(Unit) {
        onEvent(UIEvent.OnOpenSettings)
        onDispose {
            onEvent(UIEvent.OnReturnFromSettings)
        }
    }

    Surface(
        modifier = modifier,
        color = MaterialTheme.colorScheme.surface
    ) {
        Column(
            modifier = Modifier.padding(Spacing.lg), verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(R.drawable.ic_launcher_web),
                contentDescription = stringResource(R.string.app_logo_description)
            )
            PremiumButton(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(Spacing.lg),
                shadowElevation = if(isDark) AppElevation.level5 else AppElevation.level3,
                onClick = { onGetStarted() }) {
                Text(
                    stringResource(R.string.start_app_button_text),
                    style = MaterialTheme.typography.labelLarge
                )
            }
        }
    }

}

@Preview(showBackground = true, name = "WelcomePage Preview")
@Composable
fun Preview() {
    MaterialTheme { WelcomePage(Modifier.fillMaxSize(), onGetStarted = {}, onEvent = {}) }
}
