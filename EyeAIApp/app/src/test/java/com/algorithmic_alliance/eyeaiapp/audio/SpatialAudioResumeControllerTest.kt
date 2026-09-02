package com.algorithmic_alliance.eyeaiapp.audio

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SpatialAudioResumeControllerTest {
	@Test
	fun `spatial audio stays paused until TTS is silent`() = runBlocking {
		val silence = CompletableDeferred<Boolean>()
		val events = mutableListOf<String>()
		val outcome = CompletableDeferred<SpatialAudioResumeOutcome>()
		val controller = controller(
			awaitTtsSilence = { silence.await() },
			events = events,
			onOutcome = { outcome.complete(it) }
		)

		controller.schedule("SETTINGS_APPLIED")
		assertEquals(listOf("pause"), events)
		assertTrue(controller.isPending())

		silence.complete(true)
		assertEquals(SpatialAudioResumeOutcome.RESTORED, outcome.await())
		assertEquals(listOf("pause", "restore:SETTINGS_APPLIED"), events)
	}

	@Test
	fun `spatial audio is not restored when TTS silence times out`() = runBlocking {
		val events = mutableListOf<String>()
		val outcome = CompletableDeferred<SpatialAudioResumeOutcome>()
		val controller = controller(
			awaitTtsSilence = { false },
			events = events,
			onOutcome = { outcome.complete(it) }
		)

		controller.schedule("SETTINGS_APPLIED")

		assertEquals(SpatialAudioResumeOutcome.TTS_SILENCE_TIMEOUT, outcome.await())
		assertEquals(listOf("pause"), events)
	}

	@Test
	fun `spatial audio is not restored after listening starts again`() = runBlocking {
		val events = mutableListOf<String>()
		val outcome = CompletableDeferred<SpatialAudioResumeOutcome>()
		val controller = controller(
			awaitTtsSilence = { true },
			events = events,
			isListening = { true },
			onOutcome = { outcome.complete(it) }
		)

		controller.schedule("SETTINGS_APPLIED")

		assertEquals(SpatialAudioResumeOutcome.LISTENING_STATE_CHANGED, outcome.await())
		assertEquals(listOf("pause"), events)
	}

	@Test
	fun `cancelling a pending resume keeps spatial audio paused`() = runBlocking {
		val silence = CompletableDeferred<Boolean>()
		val events = mutableListOf<String>()
		val controller = controller(
			awaitTtsSilence = { silence.await() },
			events = events
		)

		controller.schedule("SETTINGS_APPLIED")
		controller.cancel()
		silence.complete(true)
		delay(10)

		assertFalse(controller.isPending())
		assertEquals(listOf("pause"), events)
	}

	private fun CoroutineScope.controller(
		awaitTtsSilence: suspend () -> Boolean,
		events: MutableList<String>,
		isListening: () -> Boolean = { false },
		onOutcome: (SpatialAudioResumeOutcome) -> Unit = {}
	) = SpatialAudioResumeController(
		scope = this,
		pauseSpatialAudio = { events += "pause" },
		restoreSpatialAudio = { events += "restore:$it" },
		awaitTtsSilence = awaitTtsSilence,
		isListening = isListening,
		onOutcome = { _, outcome -> onOutcome(outcome) }
	)
}
