package android.hardware

class Sensor(val type: Int) {
    companion object { const val TYPE_GYROSCOPE = 4; const val TYPE_LINEAR_ACCELERATION = 10 }
}
class SensorEvent(val sensor: Sensor, val values: FloatArray, val timestamp: Long)
interface SensorEventListener {
    fun onSensorChanged(event: SensorEvent)
    fun onAccuracyChanged(sensor: Sensor?, accuracy: Int)
}
class SensorManager {
    fun getDefaultSensor(type: Int): Sensor? = error("SensorManager requires device tests")
    fun registerListener(listener: SensorEventListener, sensor: Sensor, delay: Int): Boolean =
        error("SensorManager requires device tests")
    fun unregisterListener(listener: SensorEventListener): Unit = error("SensorManager requires device tests")
    companion object { const val SENSOR_DELAY_GAME = 1 }
}
