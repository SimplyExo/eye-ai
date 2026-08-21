package com.algorithmic_alliance.eyeaiapp.confirmation

import java.nio.file.Files
import java.nio.file.Path

object ConfirmationModelTestFixture {
	fun load(): ConfirmationModel =
		Files.newInputStream(modelAsset()).use(ConfirmationModel::load)

	private fun modelAsset(): Path {
		val candidates = listOf(
			Path.of("src/main/assets", ConfirmationModel.ASSET_PATH),
			Path.of("app/src/main/assets", ConfirmationModel.ASSET_PATH)
		)
		return candidates.firstOrNull(Files::isRegularFile)
			?: error("Cannot locate confirmation model asset from ${Path.of("").toAbsolutePath()}")
	}
}
