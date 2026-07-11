package com.algorithmic_alliance.eyeaiapp.UI

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@Composable
fun WelcomePage(modifier: Modifier = Modifier){
    Surface(modifier = modifier, color = MaterialTheme.colorScheme.background) { Text(text = "EyeAI", modifier = modifier.padding(24.dp))}

}

@Preview(showBackground = true, name = "EyeAI-Preview")
@Composable
fun EyeAIPreview(){
    MaterialTheme {WelcomePage(modifier = Modifier.fillMaxSize()) }

}