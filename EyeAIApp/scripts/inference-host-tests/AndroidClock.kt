package android.os

object SystemClock {
    fun elapsedRealtimeNanos(): Long = System.nanoTime()
}
