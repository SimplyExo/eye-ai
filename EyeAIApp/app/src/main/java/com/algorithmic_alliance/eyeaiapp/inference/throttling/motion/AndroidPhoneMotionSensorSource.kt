package com.algorithmic_alliance.eyeaiapp.inference.throttling.motion

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

interface PhoneMotionSensorCallbacks {
	fun onGyroscopeSample(x: Float, y: Float, z: Float, timestampNanos: Long)
	fun onLinearAccelerationSample(x: Float, y: Float, z: Float, timestampNanos: Long)
}

interface PhoneMotionSensorSource {
	val hasGyroscope: Boolean
	val hasLinearAcceleration: Boolean
	fun register(callbacks: PhoneMotionSensorCallbacks): Boolean
	fun unregister(callbacks: PhoneMotionSensorCallbacks)
}

class AndroidPhoneMotionSensorSource(
	context: Context,
	private val samplingPeriodUs: Int = SensorManager.SENSOR_DELAY_GAME,
) : PhoneMotionSensorSource {
	private val sensorManager =
		context.applicationContext.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
	private val gyroscope = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
	private val linearAcceleration = sensorManager?.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
	private var registered = false
	private var callbacksValue: PhoneMotionSensorCallbacks? = null
	private var listener: SensorEventListener? = null

	override val hasGyroscope: Boolean get() = gyroscope != null
	override val hasLinearAcceleration: Boolean get() = linearAcceleration != null

	@Synchronized
	@Suppress("ReturnCount")
	override fun register(callbacks: PhoneMotionSensorCallbacks): Boolean {
		if (registered) return callbacksValue === callbacks
		val manager = sensorManager ?: return false
		if (gyroscope == null && linearAcceleration == null) return false

		val nextListener = listener(callbacks)
		callbacksValue = callbacks
		listener = nextListener
		return try {
			val gyroRegistered = gyroscope?.let {
				manager.registerListener(nextListener, it, samplingPeriodUs)
			} ?: false
			val accelerationRegistered = linearAcceleration?.let {
				manager.registerListener(nextListener, it, samplingPeriodUs)
			} ?: false
			registered = gyroRegistered || accelerationRegistered
			if (!registered) clearRegistration(manager, nextListener)
			registered
		} catch (error: Throwable) {
			clearRegistration(manager, nextListener)
			throw error
		}
	}

	@Synchronized
	override fun unregister(callbacks: PhoneMotionSensorCallbacks) {
		if (callbacksValue !== callbacks) return
		listener?.let { clearRegistration(sensorManager, it) }
	}

	private fun clearRegistration(manager: SensorManager?, current: SensorEventListener) {
		try {
			manager?.unregisterListener(current)
		} finally {
			registered = false
			callbacksValue = null
			listener = null
		}
	}

	private fun listener(callbacks: PhoneMotionSensorCallbacks) = object : SensorEventListener {
		override fun onSensorChanged(event: SensorEvent) {
			if (event.values.size < 3) return
			when (event.sensor.type) {
				Sensor.TYPE_GYROSCOPE -> callbacks.onGyroscopeSample(
					event.values[0], event.values[1], event.values[2], event.timestamp,
				)
				Sensor.TYPE_LINEAR_ACCELERATION -> callbacks.onLinearAccelerationSample(
					event.values[0], event.values[1], event.values[2], event.timestamp,
				)
			}
		}

		override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
	}
}
