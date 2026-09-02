package com.algorithmic_alliance.eyeaiapp.tts

import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TtsVoiceSelectorTest {
	@Test
	fun availableFemaleVoiceIsSelectedFromTheBackendCatalog() {
		val female = voice("de-de-female", "female")
		val male = voice("de-de-male", "male")

		val result = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.FEMALE,
			voices = listOf(male, female)
		)

		assertTrue(result.requestedSpeakerAvailable)
		assertEquals(female.name, result.voice?.name)
	}

	@Test
	fun availableMaleVoiceIsSelectedFromTheBackendCatalog() {
		val female = voice("de-de-female", "female")
		val male = voice("de-de-male", "male")

		val result = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.MALE,
			voices = listOf(female, male)
		)

		assertTrue(result.requestedSpeakerAvailable)
		assertEquals(male.name, result.voice?.name)
	}

	@Test
	fun twoOpaqueGermanVoicesStillAllowFemaleAndMaleVariation() {
		val first = voice("de-de-x-dea-local")
		val second = voice("de-de-x-deb-local")

		val female = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.FEMALE,
			voices = listOf(second, first)
		)
		val male = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.MALE,
			voices = listOf(second, first)
		)

		assertTrue(female.requestedSpeakerAvailable)
		assertTrue(male.requestedSpeakerAvailable)
		assertEquals(first.name, female.voice?.name)
		assertEquals(second.name, male.voice?.name)
	}

	@Test
	fun prefersDifferentVoiceFamiliesOverTwoVariants() {
		val voices = listOf(
			voice("de-de-x-dea-local"),
			voice("de-de-x-dea-embedded"),
			voice("de-de-x-deb-local")
		)

		val female = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.FEMALE,
			voices = voices
		)
		val male = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.MALE,
			voices = voices
		)

		assertTrue(female.requestedSpeakerAvailable)
		assertTrue(male.requestedSpeakerAvailable)
		assertTrue(female.voice?.name?.contains("dea") == true)
		assertEquals("de-de-x-deb-local", male.voice?.name)
	}

	@Test
	fun unavailableRequestedSpeakerDoesNotPretendThatAnotherVoiceMatches() {
		val neutral = voice("de-de-neutral")

		val result = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.FEMALE,
			voices = listOf(neutral),
			currentVoiceName = neutral.name
		)

		assertFalse(result.requestedSpeakerAvailable)
		assertEquals(neutral.name, result.voice?.name)
	}

	@Test
	fun safeFallbackIgnoresVoicesMarkedAsNotInstalled() {
		val unavailableFemale = voice("de-de-female", "female", "notInstalled")
		val safeDefault = voice("de-de-neutral")

		val result = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.FEMALE,
			voices = listOf(unavailableFemale, safeDefault),
			defaultVoiceName = safeDefault.name
		)

		assertFalse(result.requestedSpeakerAvailable)
		assertEquals(safeDefault.name, result.voice?.name)
	}

	@Test
	fun selectedVoiceKeepsTheSpeakPathUsableAfterAValidSwitch() {
		val female = voice("de-de-female", "female")
		val male = voice("de-de-male", "male")
		val backend = FakeTtsBackend(listOf(female, male))
		val selection = TtsVoiceSelector.select(
			requestedSpeaker = TtsVoiceSelector.MALE,
			voices = backend.voices
		)

		val selected = requireNotNull(selection.voice)
		assertTrue(selection.requestedSpeakerAvailable)
		assertTrue(backend.setVoice(selected))
		backend.speak("Die Stimme ist aktiv.")

		assertEquals(selected.name, backend.activeVoice?.name)
		assertEquals("Die Stimme ist aktiv.", backend.lastSpokenText)
		assertNotNull(backend.activeVoice)
	}

	private fun voice(name: String, vararg features: String) = TtsVoiceDescriptor(
		name = name,
		locale = Locale.GERMANY,
		features = features.toSet()
	)

	private class FakeTtsBackend(val voices: List<TtsVoiceDescriptor>) {
		var activeVoice: TtsVoiceDescriptor? = null
			private set
		var lastSpokenText: String? = null
			private set

		fun setVoice(voice: TtsVoiceDescriptor): Boolean {
			if (voices.none { it.name == voice.name }) return false
			activeVoice = voice
			return true
		}

		fun speak(text: String) {
			check(activeVoice != null) { "TTS output requires an active catalog voice" }
			lastSpokenText = text
		}
	}
}
