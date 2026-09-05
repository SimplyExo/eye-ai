package uniffi.NativeLib

// Mirrors the record fields in native_lib/src/lib.rs. No backend work is simulated.
data class UniffiDetectedObject(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val cx: Float, val cy: Float, val w: Float, val h: Float,
    val cnf: Float, val cls: Int, val clsName: String, val trackingId: Int,
)
