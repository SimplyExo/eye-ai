package com.algorithmic_alliance.eyeaiapp.UI

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.util.Log
import android.util.Size
import android.view.View
import androidx.core.content.ContextCompat
import com.algorithmic_alliance.eyeaiapp.R
import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox
import androidx.core.graphics.withScale
import kotlin.math.abs

class OverlayViewOCR(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

	private var results = arrayOf<TextBoundingBox>()
	private var cameraResolution = Size(720, 1280)
	private var boxPaint = Paint()
	private var textBackgroundPaint = Paint()
	private var textPaint = Paint()

	init {
		initPaints()
	}

	fun clear() {
		textPaint.reset()
		textBackgroundPaint.reset()
		boxPaint.reset()
		invalidate()
		initPaints()
	}

	fun reset() {
		results = emptyArray()
		invalidate()
	}

	private fun initPaints() {
		textBackgroundPaint.color = Color.BLACK
		textBackgroundPaint.style = Paint.Style.FILL
		textBackgroundPaint.textSize = 50f

		textPaint.color = ContextCompat.getColor(context!!, R.color.black)
		textPaint.style = Paint.Style.FILL
		textPaint.textSize = 50f

		boxPaint.color = ContextCompat.getColor(context!!, R.color.white)
		boxPaint.strokeWidth = 8F
		boxPaint.style = Paint.Style.FILL_AND_STROKE
	}

	override fun draw(canvas: Canvas) {
		super.draw(canvas)

		val viewAspectRatio = width.toFloat() / height.toFloat()
		val cameraAspectRatio = cameraResolution.width.toFloat() / cameraResolution.height.toFloat()

		val cameraPreviewImageSize = if (viewAspectRatio > cameraAspectRatio) {
			Size(
				(height.toFloat() * cameraAspectRatio).toInt(),
				height
			)
		} else {
			Size(
				width,
				(width.toFloat() / cameraAspectRatio).toInt()
			)
		}

		val xOffset = if (cameraPreviewImageSize.width < width) {
			(width - cameraPreviewImageSize.width) / 2
		} else {
			0
		}
		val yOffset = if (cameraPreviewImageSize.height < height) {
			(height - cameraPreviewImageSize.height) / 2
		} else {
			0
		}

		results.forEach {
			val left = it.x1 * cameraPreviewImageSize.width + xOffset
			val top = it.y1 * cameraPreviewImageSize.height + yOffset
			val right = it.x2 * cameraPreviewImageSize.width + xOffset
			val bottom = it.y2 * cameraPreviewImageSize.height + yOffset

			// Zeilen
			val lines = it.text.split('\n')
			val charHeight = (bottom - top) / lines.size.toFloat()

			// Rechteck zeichnen
			canvas.drawRect(left, top, right, bottom, boxPaint)

			// Schreiben der einzelnen Lines ins Overlay
			for ((i, line) in lines.withIndex()) {
				val boxWidth = right - left
				textPaint.textSize = charHeight
				val textWidth = textPaint.measureText(line)

				val textWidthBoxWidthRatio = textWidth / boxWidth

				if (textWidthBoxWidthRatio > 1 ||
					(textWidthBoxWidthRatio < 0.8f && lines.size == 1)) {
					val scaleX = boxWidth / textWidth
					canvas.withScale(scaleX, 1f, left, 0f) {
						drawText(line, left, top + charHeight * (i + 1), textPaint)
					}
				} else {
					canvas.drawText(line, left, top + charHeight * (i + 1), textPaint)
				}
			}
		}
	}

	fun setCameraResolution(newCameraResolution: Size) {
		val changed = cameraResolution != newCameraResolution
		cameraResolution = newCameraResolution
		if (changed)
			invalidate()
	}

	fun setResults(boundingBoxes: Array<TextBoundingBox>) {
		results = boundingBoxes
		invalidate()
	}
}