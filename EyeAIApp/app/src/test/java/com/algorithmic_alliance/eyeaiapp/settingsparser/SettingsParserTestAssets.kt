package com.algorithmic_alliance.eyeaiapp.settingsparser

import java.nio.file.Files
import java.nio.file.Path

/** Locates the checked-in APK assets from either the root or app Gradle cwd. */
internal fun settingsParserAssetDirectory(): Path = listOf(
	Path.of("src/main/assets/nlp-v2/settings-parser"),
	Path.of("app/src/main/assets/nlp-v2/settings-parser"),
	Path.of("EyeAIApp/app/src/main/assets/nlp-v2/settings-parser")
).firstOrNull(Files::isDirectory) ?: error("Could not locate Clean-v2 Settings Parser assets")
