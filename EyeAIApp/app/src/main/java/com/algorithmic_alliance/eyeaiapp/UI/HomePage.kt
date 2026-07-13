package com.algorithmic_alliance.eyeaiapp.UI

import android.view.Surface
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

@Composable
fun HomePage(modifier: Modifier = Modifier){
    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
        Text("HomePage")
    }
}