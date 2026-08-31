package com.algorithmic_alliance.eyeaiapp.settingsparser

/** Optional future UniFFI/JNI boundary; production currently uses the pinned Kotlin port. */
interface NativeGermanNumberNormalizerBridge {
	fun normalizeGermanNumbers(text: String): NumberNormalizationResult
}

class NativeBridgeGermanNumberNormalizer(
	private val bridge: NativeGermanNumberNormalizerBridge
) : GermanNumberNormalizer {
	override fun normalize(text: String): NumberNormalizationResult =
		bridge.normalizeGermanNumbers(text)
}
