package com.algorithmic_alliance.eyeaiapp.UI.pages

import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.data.UIDataSource.UI_LOG_TAG as LOG_TAG
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import com.algorithmic_alliance.eyeaiapp.data.Spacing

@Composable
fun WelcomePage(modifier: Modifier = Modifier, onGetStarted: () -> Unit) {
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
                contentDescription = "App-Logo"
            )
            Button(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(Spacing.lg)
                    .semantics { contentDescription = "Startet die Eye-Ai Anwendung" },
                onClick = {

                    onGetStarted()
                },

                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary,
                )
            ) {
                Text(
                    "App Starten",
                    modifier = Modifier.clearAndSetSemantics {},
                    style = MaterialTheme.typography.labelLarge,
                )
            }
        }
    }

}

@Preview(showBackground = true, name = "WelcomePage Preview")
@Composable
fun Preview() {
    MaterialTheme { WelcomePage(Modifier.fillMaxSize(), onGetStarted = {}) }
}
