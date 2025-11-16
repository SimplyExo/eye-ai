package com.algorithmic_alliance.eyeaiapp.UI

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.util.Size
import android.view.View
import androidx.core.content.ContextCompat
import com.algorithmic_alliance.eyeaiapp.R
import uniffi.NativeLib.UniffiDetectedObject

class OverlayViewOD(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

	private var results = arrayOf<UniffiDetectedObject>()
	private var cameraResolution = Size(720, 1280)
	private var boxPaint = Paint()
	private var textBackgroundPaint = Paint()
	private var textPaint = Paint()

	private var bounds = Rect()

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

		textPaint.color = Color.WHITE
		textPaint.style = Paint.Style.FILL
		textPaint.textSize = 50f

		boxPaint.color = ContextCompat.getColor(context!!, R.color.purple_500)
		boxPaint.strokeWidth = 8F
		boxPaint.style = Paint.Style.STROKE
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

			canvas.drawRect(left, top, right, bottom, boxPaint)
			val drawableText = "${it.clsName} - ${it.trackingId}"

			textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
			val textWidth = bounds.width()
			val textHeight = bounds.height()
			canvas.drawRect(
				left,
				top,
				left + textWidth + BOUNDING_RECT_TEXT_PADDING,
				top + textHeight + BOUNDING_RECT_TEXT_PADDING,
				textBackgroundPaint
			)
			canvas.drawText(drawableText, left, top + bounds.height(), textPaint)
		}
	}

	fun setResults(boundingBoxes: Array<UniffiDetectedObject>) {
		val changed = !results.contentEquals(boundingBoxes)
		results = boundingBoxes
		if (changed)
			invalidate()
	}

	fun setCameraResolution(newCameraResolution: Size) {
		val changed = cameraResolution != newCameraResolution
		cameraResolution = newCameraResolution

		if (changed)
			invalidate()
	}

	companion object {
		private const val BOUNDING_RECT_TEXT_PADDING = 8
	}
}