package com.algorithmic_alliance.eyeaiapp.inference

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.SystemClock

/** Minimal source boundary for Android now and a future remote-IMU adapter later. */
interface PhoneMotionSensorCallbacks {
	fun onGyroscopeSample(x: Float, y: Float, z: Float, timestampNanos: Long)
	fun onLinearAccelerationSample(x: Float, y: Float, z: Float, timestampNanos: Long)
}

/** Sensor registration boundary kept independent from [SensorManager]. */
interface PhoneMotionSensorSource {
	val hasGyroscope: Boolean
	val hasLinearAcceleration: Boolean

	fun register(callbacks: PhoneMotionSensorCallbacks): Boolean

	fun unregister(callbacks: PhoneMotionSensorCallbacks)
}

/** Production time source sharing SensorEvent's elapsed-realtime-nanoseconds time base. */
object SystemPhoneMotionClock : PhoneMotionClock {
	override fun nowNanos(): Long = SystemClock.elapsedRealtimeNanos()
}

/** Android [SensorManager] adapter; it owns registration and never exposes SensorEvents upstream. */
class AndroidPhoneMotionSensorSource(
	context: Context,
	private val samplingPeriodUs: Int = SensorManager.SENSOR_DELAY_GAME,
) : PhoneMotionSensorSource {
	private val sensorManager =
		context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
	private val gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
	private val linearAcceleration =
		sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)

	private var registered = false
	private var callbacksValue: PhoneMotionSensorCallbacks? = null
	private var registeredListener: SensorEventListener? = null

	override val hasGyroscope: Boolean
		get() = gyroscope != null

	override val hasLinearAcceleration: Boolean
		get() = linearAcceleration != null

	@Synchronized
	@Suppress("ReturnCount")
	override fun register(callbacks: PhoneMotionSensorCallbacks): Boolean {
		if (registered) return callbacksValue === callbacks
		val manager = sensorManager ?: return false
		if (gyroscope == null && linearAcceleration == null) return false

		val listener = createSensorEventListener(callbacks)
		callbacksValue = callbacks
		registeredListener = listener
		var registeredAny = false
		try {
			gyroscope?.let {
				registeredAny = manager.registerListener(listener, it, samplingPeriodUs) ||
					registeredAny
			}
			linearAcceleration?.let {
				registeredAny = manager.registerListener(listener, it, samplingPeriodUs) ||
					registeredAny
			}
		} catch (error: Throwable) {
			try {
				manager.unregisterListener(listener)
			} finally {
				callbacksValue = null
				registeredListener = null
				registered = false
			}
			throw error
		}

		registered = registeredAny
		if (!registeredAny) {
			try {
				manager.unregisterListener(listener)
			} finally {
				callbacksValue = null
				registeredListener = null
			}
		}
		return registeredAny
	}

	@Synchronized
	override fun unregister(callbacks: PhoneMotionSensorCallbacks) {
		if (callbacksValue !== callbacks) return
		val listener = registeredListener
		try {
			if (registered) listener?.let { sensorManager?.unregisterListener(it) }
		} finally {
			registered = false
			callbacksValue = null
			registeredListener = null
		}
	}

	private fun createSensorEventListener(callbacks: PhoneMotionSensorCallbacks): SensorEventListener =
		object : SensorEventListener {
			override fun onSensorChanged(event: SensorEvent) {
				if (event.values.size < 3) return
				when (event.sensor.type) {
					Sensor.TYPE_GYROSCOPE -> callbacks.onGyroscopeSample(
						event.values[0],
						event.values[1],
						event.values[2],
						event.timestamp,
					)
					Sensor.TYPE_LINEAR_ACCELERATION -> callbacks.onLinearAccelerationSample(
						event.values[0],
						event.values[1],
						event.values[2],
						event.timestamp,
					)
				}
			}

			override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
		}
}

/**
 * Optional phone activity hint. It makes no camera-axis or pose assumption and never decides an
 * inference mode. Callers poll [score] when they need the current signal.
 */
class PhoneMotionMonitor(
	private val sensorSource: PhoneMotionSensorSource,
	val config: PhoneMotionMonitorConfig = PhoneMotionMonitorConfig(),
	private val clock: PhoneMotionClock = SystemPhoneMotionClock,
) : PhoneMotionSensorCallbacks {
	constructor(
		context: Context,
		config: PhoneMotionMonitorConfig = PhoneMotionMonitorConfig(),
		clock: PhoneMotionClock = SystemPhoneMotionClock,
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

	/** Idempotently registers available sensors and resets stale scoring state. */
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

	/** Idempotently unregisters sensors and drops all current motion state. */
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

	/** Starts a new callback generation when active, so callbacks from the old session stay inert. */
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

	/** Returns a fresh score in 0.0..1.0, or null when stopped, missing, or stale. */
	@Synchronized
	fun score(atNanos: Long): Double? {
		if (!running || !registered) return null
		return scoreLogic.score(atNanos)
	}

	/** Reads the polling clock only after taking the monitor lock. */
	fun score(): Double? = synchronized(this) { score(clock.nowNanos()) }

	/** Readable alias for future scheduler integration; this class itself performs no integration. */
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
		} catch (_: Throwable) {
			// Preserve the registration failure while leaving this monitor stopped.
		}
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
