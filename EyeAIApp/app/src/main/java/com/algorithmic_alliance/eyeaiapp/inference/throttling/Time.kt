package com.algorithmic_alliance.eyeaiapp.inference.throttling

import android.os.SystemClock

fun interface MonotonicClock {
	fun nowNanos(): Long
}

object AnalysisClock : MonotonicClock {
	override fun nowNanos(): Long = SystemClock.elapsedRealtimeNanos()
}
