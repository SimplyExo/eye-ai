package com.algorithmic_alliance.eyeaiapp.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.media.Image
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import java.nio.ByteBuffer

/**
 * CameraX-only acquisition adapter. It converts and closes ImageProxy here;
 * all actual analysis happens in the source-neutral [FrameAnalyzer].
 *
 * Pro Frame entsteht höchstens noch ein Bitmap, und das kommt aus einem Pool.
 * Es wird über den onReleased-Callback des [AnalysisFrame] zurückgegeben, sobald alle
 * Verbraucher fertig sind. Vorher wurden pro Frame zwei volle Bitmaps angelegt
 * (Konvertierung plus gedrehte Kopie), und beide blieben bis zum nächsten GC liegen.
 */
class CameraXFrameAdapter(
	private val frameAnalyzer: FrameAnalyzer,
	private val cameraCalibration: CameraCalibration? = null,
) : ImageAnalysis.Analyzer {
	private val pool = FrameBitmapPool()

	// Die folgenden Arbeitsobjekte werden nur vom Analyzer-Thread benutzt (cameraExecutor).
	private var scratchBitmap: Bitmap? = null
	private var scratchBytes: ByteArray? = null
	private val rotationMatrix = Matrix()
	private val canvas = Canvas()

	// SRC statt SRC_OVER: das Ziel-Bitmap wird komplett überschrieben, egal was vorher darin war
	private val copyPaint = Paint().apply {
		xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
	}

	@OptIn(ExperimentalGetImage::class)
	override fun analyze(image: ImageProxy) {
		var bitmap: Bitmap? = null
		try {
			val mediaImage = image.image ?: return
			val rotationDegrees = image.imageInfo.rotationDegrees
			val converted = convert(mediaImage, rotationDegrees)
			bitmap = converted
			frameAnalyzer.recordSourceFrame(image.imageInfo.timestamp)
			val frame = AnalysisFrame(
				bitmap = converted,
				width = converted.width,
				height = converted.height,
				rotationDegrees = rotationDegrees,
				timestampNanos = image.imageInfo.timestamp,
				cameraIntrinsics = cameraCalibration?.forFrame(
					widthPx = converted.width,
					heightPx = converted.height,
					rotationDegrees = rotationDegrees,
                ),
                timeOrigin = com.algorithmic_alliance.eyeaiapp.worldstate.TimeOrigin.CAMERA_TIMESTAMP,
                onReleased = { pool.release(converted) },
			)
			// Ab hier gehört das Bitmap dem Frame und kommt über onReleased zurück in den Pool
			bitmap = null
			frameAnalyzer.submitFrame(frame)
		} catch (error: Throwable) {
			bitmap?.let(pool::release)
			Log.e(EyeAIApp.APP_LOG_TAG, "CameraX frame conversion failed", error)
		} finally {
			// The bitmap is detached from the ImageProxy before this point.
			image.close()
		}
	}

	private fun convert(image: Image, rotationDegrees: Int): Bitmap {
		require(image.format == PixelFormat.RGBA_8888) {
			"Unsupported image format: ${image.format}. Expected RGBA_8888"
		}
		val rotation = ((rotationDegrees % 360) + 360) % 360
		require(rotation % 90 == 0) { "Unsupported rotation: $rotationDegrees" }

		val width = image.width
		val height = image.height

		// Ohne Rotation direkt ins Pool-Bitmap kopieren, kein Zwischenschritt
		if (rotation == 0) {
			val target = pool.acquire(width, height)
			try {
				copyPixels(image, target)
			} catch (error: Throwable) {
				pool.release(target)
				throw error
			}
			return target
		}

		// Mit Rotation: einmal in ein wiederverwendetes Zwischen-Bitmap kopieren
		// und von dort gedreht ins Pool-Bitmap zeichnen
		val scratch = scratchBitmap(width, height)
		copyPixels(image, scratch)

		val swap = rotation == 90 || rotation == 270
		val target = pool.acquire(if (swap) height else width, if (swap) width else height)

		rotationMatrix.setRotate(rotation.toFloat())
		when (rotation) {
			90 -> rotationMatrix.postTranslate(height.toFloat(), 0f)
			180 -> rotationMatrix.postTranslate(width.toFloat(), height.toFloat())
			270 -> rotationMatrix.postTranslate(0f, width.toFloat())
		}
        try {
            canvas.setBitmap(target)
            canvas.drawBitmap(scratch, rotationMatrix, copyPaint)
            return target
        } catch (error: Throwable) {
            pool.release(target)
            throw error
        } finally {
            canvas.setBitmap(null)
        }
	}

	private fun copyPixels(image: Image, target: Bitmap) {
		val plane = image.planes[0]
		val buffer = plane.buffer
		val pixelStride = plane.pixelStride
        require(pixelStride == 4) { "RGBA pixel stride must be four bytes" }
		val rowStride = plane.rowStride
		val width = image.width
		val height = image.height

		buffer.rewind()

		// Ohne Zeilen-Padding kann der Puffer direkt kopiert werden
		if (pixelStride == 4 && rowStride == width * 4) {
			target.copyPixelsFromBuffer(buffer)
		} else {
			// Mit Padding zeilenweise umkopieren, das Hilfsarray wird wiederverwendet
			val size = width * height * 4
			val bytes = scratchBytes?.takeIf { it.size == size } ?: ByteArray(size).also {
				scratchBytes = it
			}
			for (row in 0 until height) {
				buffer.position(row * rowStride)
				buffer.get(bytes, row * width * 4, width * 4)
			}
			target.copyPixelsFromBuffer(ByteBuffer.wrap(bytes))
		}
	}

	private fun scratchBitmap(width: Int, height: Int): Bitmap {
		val existing = scratchBitmap
		if (existing != null && existing.width == width && existing.height == height) {
			return existing
		}
		// Das Zwischen-Bitmap verlässt diese Klasse nie, daher ist recycle() hier sicher
		existing?.recycle()
		return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
			scratchBitmap = it
		}
	}
}
