package com.algorithmic_alliance.eyeaiapp.audio

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicReference

enum class SpatialAudioResumeOutcome {
	RESTORED,
	TTS_SILENCE_TIMEOUT,
	LISTENING_STATE_CHANGED
}

/**
 * Keeps spatial output muted while a TTS response is still being played.
 *
 * The controller is deliberately independent of Android and the native audio
 * implementation so it can be covered by local unit tests.
 */
class SpatialAudioResumeController(
	private val scope: CoroutineScope,
	private val pauseSpatialAudio: () -> Unit,
	private val restoreSpatialAudio: (trigger: String) -> Unit,
	private val awaitTtsSilence: suspend () -> Boolean,
	private val isListening: () -> Boolean,
	private val onOutcome: (trigger: String, outcome: SpatialAudioResumeOutcome) -> Unit
) {
	private val pendingResume = AtomicReference<Job?>(null)

	@Synchronized
	fun schedule(trigger: String) {
		cancel()
		pauseSpatialAudio()

		val job = scope.launch(start = CoroutineStart.LAZY) {
			val outcome = when {
				!awaitTtsSilence() -> SpatialAudioResumeOutcome.TTS_SILENCE_TIMEOUT
				isListening() -> SpatialAudioResumeOutcome.LISTENING_STATE_CHANGED
				else -> {
					restoreSpatialAudio(trigger)
					SpatialAudioResumeOutcome.RESTORED
				}
			}
			onOutcome(trigger, outcome)
		}

		pendingResume.set(job)
		job.invokeOnCompletion {
			pendingResume.compareAndSet(job, null)
		}
		job.start()
	}

	@Synchronized
	fun cancel() {
		pendingResume.getAndSet(null)?.cancel()
	}

	fun isPending(): Boolean = pendingResume.get()?.isActive == true
}
