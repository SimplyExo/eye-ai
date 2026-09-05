package android.util

data class Size(val width: Int, val height: Int)
object Log {
    fun d(tag: String, message: String): Int = 0
    fun e(tag: String, message: String, error: Throwable): Int = 0
}
