// Host-only test double. Never included in Android source sets or APKs.
package android.graphics

class Rect
class Bitmap private constructor(val width: Int, val height: Int) {
    enum class Config { ARGB_8888 }
    private val pixels = IntArray(width * height)
    var isRecycled = false
        private set
    fun getPixel(x: Int, y: Int): Int {
        check(!isRecycled)
        return pixels[y * width + x]
    }
    fun setPixel(x: Int, y: Int, color: Int) { pixels[y * width + x] = color }
    fun recycle() { isRecycled = true }
    companion object {
        fun createBitmap(width: Int, height: Int, config: Config) = Bitmap(width, height)
    }
}
