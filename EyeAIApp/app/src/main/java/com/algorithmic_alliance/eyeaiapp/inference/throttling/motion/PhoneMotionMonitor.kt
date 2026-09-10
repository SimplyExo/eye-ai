package com.algorithmic_alliance.eyeaiapp.inference.throttling.motion

import android.content.Context
import android.hardware.SensorManager
import com.algorithmic_alliance.eyeaiapp.inference.throttling.AnalysisClock
import com.algorithmic_alliance.eyeaiapp.inference.throttling.MonotonicClock

class PhoneMotionMonitor(
	private val sensorSource: PhoneMotionSensorSource,
	val config: PhoneMotionMonitorConfig = PhoneMotionMonitorConfig(),
	private val clock: MonotonicClock = AnalysisClock,
) : PhoneMotionSensorCallbacks {
	constructor(
		context: Context,
		config: PhoneMotionMonitorConfig = PhoneMotionMonitorConfig(),
		clock: MonotonicClock = AnalysisClock,
		samplingPeriodUs: Int = SensorManager.SENSOR_DELAY_GAME,
	) : this(
		sensorSource = AndroidPhoneMotionSensorSource(context, samplingPeriodUs),
		config = config,
		clock = clock,
	)

	private val scoreLogic = PhoneMotionScoreLogic(config)
	private var running = false
	private var registered = false
	private var nextSessionGeneration = 0L
	private var activeSession: SessionCallbacks? = null
	private var registeredSession: SessionCallbacks? = null

	val hasUsableSensor: Boolean
		get() = synchronized(this) {
			sensorSource.hasGyroscope || sensorSource.hasLinearAcceleration
		}

	val isRunning: Boolean
		get() = synchronized(this) { running }

	@Synchronized
	fun start(): Boolean {
		if (running) return registered
		scoreLogic.reset()
		if (!hasUsableSensor) return false

		val session = newSession()
		running = true
		registered = true
		activeSession = session
		registeredSession = session
		return try {
			val didRegister = sensorSource.register(session)
			if (!didRegister) {
				invalidateSession(session)
				unregisterQuietly(session)
				scoreLogic.reset()
			}
			didRegister
		} catch (error: Throwable) {
			invalidateSession(session)
			unregisterQuietly(session)
			scoreLogic.reset()
			throw error
		}
	}

	@Synchronized
	fun stop() {
		val session = activeSession
		invalidateSession(session)
		try {
			if (session != null) sensorSource.unregister(session)
		} finally {
			scoreLogic.reset()
		}
	}

	@Synchronized
	fun reset() {
		if (running) {
			val oldSession = activeSession
			invalidateSession(oldSession)
			try {
				if (oldSession != null) sensorSource.unregister(oldSession)
			} finally {
				scoreLogic.reset()
			}
			start()
			return
		}
		scoreLogic.reset()
	}

	@Synchronized
	fun score(atNanos: Long): Double? {
		if (!running || !registered) return null
		return scoreLogic.score(atNanos)
	}

	fun score(): Double? = synchronized(this) { score(clock.nowNanos()) }

	val phoneMotionScore: Double?
		get() = score()

	@Synchronized
	override fun onGyroscopeSample(x: Float, y: Float, z: Float, timestampNanos: Long) {
		acceptGyroscopeSample(null, x, y, z, timestampNanos)
	}

	@Synchronized
	override fun onLinearAccelerationSample(x: Float, y: Float, z: Float, timestampNanos: Long) {
		acceptLinearAccelerationSample(null, x, y, z, timestampNanos)
	}

	private fun newSession(): SessionCallbacks {
		nextSessionGeneration += 1L
		return SessionCallbacks(nextSessionGeneration)
	}

	private fun invalidateSession(session: SessionCallbacks?) {
		if (session == null || activeSession === session) {
			running = false
			registered = false
			activeSession = null
			registeredSession = null
		}
	}

	private fun unregisterQuietly(session: SessionCallbacks) {
		try {
			sensorSource.unregister(session)
		} catch (_: Throwable) {}
	}

	@Synchronized
	private fun acceptGyroscopeSample(
		session: SessionCallbacks?,
		x: Float,
		y: Float,
		z: Float,
		timestampNanos: Long,
	) {
		if (!running || !registered ||
			(session != null && !isCurrentSession(session))
		) {
			return
		}
		scoreLogic.onGyroscopeSample(x, y, z, timestampNanos)
	}

	@Synchronized
	private fun acceptLinearAccelerationSample(
		session: SessionCallbacks?,
		x: Float,
		y: Float,
		z: Float,
		timestampNanos: Long,
	) {
		if (!running || !registered ||
			(session != null && !isCurrentSession(session))
		) {
			return
		}
		scoreLogic.onLinearAccelerationSample(x, y, z, timestampNanos)
	}

	private fun isCurrentSession(session: SessionCallbacks): Boolean =
		activeSession === session && registeredSession?.generation == session.generation

	private inner class SessionCallbacks(
		val generation: Long,
	) : PhoneMotionSensorCallbacks {
		override fun onGyroscopeSample(x: Float, y: Float, z: Float, timestampNanos: Long) {
			acceptGyroscopeSample(this, x, y, z, timestampNanos)
		}

		override fun onLinearAccelerationSample(x: Float, y: Float, z: Float, timestampNanos: Long) {
			acceptLinearAccelerationSample(this, x, y, z, timestampNanos)
		}
	}
}
