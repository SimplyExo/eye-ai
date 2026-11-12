package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import java.util.ArrayDeque

/**
 * Kleiner Pool gleich großer ARGB_8888-Bitmaps für Kamera-Frames.
 *
 * [acquire] liefert ein freies Bitmap passender Größe oder legt ein neues an.
 * [release] wird über [AnalysisFrame]'s onReleased-Callback aufgerufen, sobald der letzte
 * Verbraucher den Frame abgegeben hat. Die Bitmaps werden bewusst nie recycelt, weil Aufrufer
 * sie unter Umständen noch kurz halten. Überzählige Bitmaps übernimmt der GC.
 *
 * Thread-sicher: acquire läuft auf dem Kamera-Thread, release auf den Analyse-Threads.
 */
class FrameBitmapPool(private val maxPooled: Int = 6) {
	private val lock = Any()
	private val free = ArrayDeque<Bitmap>()

	fun acquire(width: Int, height: Int): Bitmap {
		synchronized(lock) {
			val iterator = free.iterator()
			while (iterator.hasNext()) {
				val bitmap = iterator.next()
				if (bitmap.width == width && bitmap.height == height && !bitmap.isRecycled) {
					iterator.remove()
					return bitmap
				}
			}
			// Nichts Passendes frei. Übrig gebliebene Bitmaps anderer Größe
			// (z. B. nach einem Auflösungs- oder Rotationswechsel) werden verworfen.
			free.clear()
		}
		return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
	}

	fun release(bitmap: Bitmap) {
		synchronized(lock) {
			if (!bitmap.isRecycled && free.size < maxPooled) {
				free.addLast(bitmap)
			}
		}
	}
}