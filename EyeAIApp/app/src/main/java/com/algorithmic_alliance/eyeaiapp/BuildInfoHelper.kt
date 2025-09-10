package com.algorithmic_alliance.eyeaiapp

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object BuildInfoHelper {
	fun getFormattedBuildTime(): String {
		return try {
			val timestamp: Long = BuildConfig.BUILD_TIME.toLong()
			val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
			dateFormat.format(Date(timestamp))
		} catch (_: Exception) {
			"Unknown build time"
		}
	}

	fun getVersionInfo(): String {
		return "${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})"
	}

	fun getGitInfo(): String {
		return "Repo: https://github.com/SimplyExo/eye-ai\nBranch: ${BuildConfig.GIT_BRANCH}\nTag: ${BuildConfig.GIT_TAG}\nCommit: ${BuildConfig.GIT_COMMIT}"
	}

	fun getBuildVariant(): String {
		return BuildConfig.BUILD_VARIANT
	}
}