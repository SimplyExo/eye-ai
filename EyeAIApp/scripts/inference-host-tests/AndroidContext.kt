package android.content

open class Context {
    val applicationContext: Context get() = this
    fun getSystemService(name: String): Any? = error("SensorManager not simulated on host")
    companion object { const val SENSOR_SERVICE = "sensor" }
}
