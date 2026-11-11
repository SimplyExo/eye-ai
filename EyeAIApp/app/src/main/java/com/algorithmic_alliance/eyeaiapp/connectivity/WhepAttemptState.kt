package com.algorithmic_alliance.eyeaiapp.connectivity

/** Signaling is serialized; video sinks may read the current attempt concurrently. */
internal class WhepAttemptState {
	@Volatile private var generation = 0L
	@Volatile private var active = false
	private var offerSent = false

	fun begin(): Long {
		generation++
		offerSent = false
		active = true
		return generation
	}

	fun isCurrent(attempt: Long) = active && generation == attempt

	fun invalidate() {
		active = false
	}

	fun claimOffer(attempt: Long, gatheringComplete: Boolean, hasLocalDescription: Boolean): Boolean {
		if (!isCurrent(attempt) || !gatheringComplete || !hasLocalDescription || offerSent) return false
		offerSent = true
		return true
	}
}
