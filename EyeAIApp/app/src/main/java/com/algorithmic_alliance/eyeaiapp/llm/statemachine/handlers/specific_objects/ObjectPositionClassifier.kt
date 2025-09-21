package com.algorithmic_alliance.eyeaiapp.llm.statemachine.handlers.specific_objects

import android.util.Log

class ObjectPositionClassifier(
	private val screenWidth: Float = 640f,
	private val screenHeight: Float = 640f
) {

	data class ObjectData(
		val label: String,
		val height: Float,
		val width: Float,
		val x: Float,
		val y: Float,
		val distance: Float
	)

	fun generatePositionDescription(obj: ObjectData): String {

		val absoluteX = obj.x * screenWidth
		val absoluteY = obj.y * screenHeight

		val horizontalPosition = classifyHorizontalPosition(absoluteX)
		val verticalPosition = classifyVerticalPosition(absoluteY)
		val distanceDescription = formatDistance(obj.distance)

		val positionText = buildPositionText(horizontalPosition, verticalPosition)
		var description = "Das Objekt ${obj.label} befindet sich $positionText vor Ihnen und ist etwa $distanceDescription von Ihnen entfernt."

		// Size analysis
		val sizeAnalysis = analyzeObjectSizeAndDistance(obj)
		if (sizeAnalysis.isNotEmpty()) {
			description += " $sizeAnalysis"
		}

		// Distance warnings
		val proximityNote = getProximityNote(obj)
		if (proximityNote.isNotEmpty()) {
			description += " $proximityNote"
		}

		return description
	}

	private fun classifyHorizontalPosition(x: Float): String {
		val centerX = screenWidth / 2
		val threshold = screenWidth * 0.15f

		return when {
			x < centerX - threshold -> "links"
			x > centerX + threshold -> "rechts"
			else -> "mittig"
		}
	}

	private fun classifyVerticalPosition(y: Float): String {
		val centerY = screenHeight / 2
		val threshold = screenHeight * 0.15f

		return when {
			y < centerY - threshold -> "oben"
			y > centerY + threshold -> "unten"
			else -> "mittig"
		}
	}

	private fun buildPositionText(horizontal: String, vertical: String): String {
		return when {
			horizontal == "mittig" && vertical == "mittig" -> "direkt"
			horizontal == "mittig" -> vertical
			vertical == "mittig" -> horizontal
			else -> "$vertical $horizontal"
		}
	}

	private fun formatDistance(distance: Float): String {
		return when {
			distance < 1.0f -> "${String.format("%.1f", distance)} Meter"
			distance < 10.0f -> "${String.format("%.1f", distance)} Meter"
			else -> "${distance.toInt()} Meter"
		}
	}

	private fun analyzeObjectSizeAndDistance(obj: ObjectData): String {
		val screenArea = screenWidth * screenHeight
		val objectArea = obj.width * obj.height
		val screenPercentage = (objectArea / screenArea) * 100



		return when {
			screenPercentage > 25 -> {
				if (obj.distance > 3.0f) {
					"Das Objekt erscheint groß für diese Entfernung."
				} else {
					"Das Objekt nimmt einen großen Teil des Bildschirms ein."
				}
			}
			screenPercentage > 10 -> {
				if (obj.distance < 1.0f) {
					"Das Objekt erscheint aufgrund der geringen Entfernung groß."
				} else {
					"Das Objekt ist gut sichtbar."
				}
			}
			screenPercentage > 2 -> {
				if (obj.distance > 10.0f) {
					"Das Objekt ist trotz der Entfernung gut erkennbar."
				} else {
					"Das Objekt hat eine mittlere Größe im Bild."
				}
			}
			else -> {
				if (obj.distance > 20.0f) {
					"Das Objekt erscheint sehr klein aufgrund der großen Entfernung."
				} else {
					"Das Objekt ist klein im Bild."
				}
			}
		}
	}

	private fun getProximityNote(obj: ObjectData): String {
		return when {
			obj.distance < 0.5f -> "Das Objekt befindet sich sehr nah bei Ihnen."
			obj.distance < 1.0f -> "Das Objekt ist in Ihrer unmittelbaren Nähe."
			obj.distance > 50.0f -> "Das Objekt ist weit entfernt."
			obj.distance > 20.0f -> "Das Objekt befindet sich in größerer Entfernung."
			else -> ""
		}
	}

	// Detailed analysis
	fun getDetailedSizeAnalysis(obj: ObjectData): Map<String, Any> {
		val screenArea = screenWidth * screenHeight
		val objectArea = obj.width * obj.height
		val screenPercentage = (objectArea / screenArea) * 100

		return mapOf(
			"screenPercentage" to String.format("%.1f", screenPercentage),
			"pixelArea" to objectArea.toInt(),
			"aspectRatio" to String.format("%.2f", obj.width / obj.height),
			"distanceCategory" to when {
				obj.distance < 1.0f -> "nah"
				obj.distance < 5.0f -> "relativ nah"
				obj.distance < 15.0f -> "mittlere Entfernung"
				obj.distance < 30.0f -> "relativ fern"
				else -> "fern"
			}
		)
	}
}