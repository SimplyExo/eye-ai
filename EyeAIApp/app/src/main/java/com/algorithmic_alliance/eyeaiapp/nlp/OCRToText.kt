package com.algorithmic_alliance.eyeaiapp.nlp

import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox

class OCRToText(
	private val screenWidth: Float = 1.0f,
	private val screenHeight: Float = 1.0f
) {

	data class TextBlockInfo(
		val text: String,
		val position: String,
		val size: String,
		val linePosition: String
	)

	fun generateReadableText(textBoxes: List<TextBoundingBox>): String {
		if (textBoxes.isEmpty()) return ""

		// Groups lines by y-pos
		val lines = groupIntoLines(textBoxes)

		// Sorts lines from left to right
		val sortedLines = lines.map { line ->
			line.sortedBy { it.x1 }
		}

		// Analyze text layout
		val textLayoutInfo = analyzeTextLayout(sortedLines)

		// Generates natural text with position information
		return generatePositionalDescription(sortedLines, textLayoutInfo)
	}

	private fun analyzeTextLayout(lines: List<List<TextBoundingBox>>): Map<String, Any> {
		val allBoxes = lines.flatten()
		val minY = allBoxes.minOfOrNull { it.y1 } ?: 0f
		val maxY = allBoxes.maxOfOrNull { it.y2 } ?: 0f
		val minX = allBoxes.minOfOrNull { it.x1 } ?: 0f
		val maxX = allBoxes.maxOfOrNull { it.x2 } ?: 0f

		return mapOf(
			"textRegion" to classifyTextRegion(minY, maxY, minX, maxX),
			"lineCount" to lines.size,
			"totalTextArea" to calculateTotalTextArea(allBoxes),
			"verticalSpread" to (maxY - minY),
			"horizontalSpread" to (maxX - minX)
		)
	}

	private fun classifyTextRegion(minY: Float, maxY: Float, minX: Float, maxX: Float): String {
		val centerY = (minY + maxY) / 2
		val centerX = (minX + maxX) / 2

		val verticalPos = classifyVerticalPosition(centerY)
		val horizontalPos = classifyHorizontalPosition(centerX)

		return buildPositionText(horizontalPos, verticalPos)
	}

	private fun classifyVerticalPosition(y: Float): String {
		val centerY = screenHeight / 2
		val threshold = screenHeight * 0.2f

		return when {
			y < centerY - threshold -> "oben"
			y > centerY + threshold -> "unten"
			else -> "mittig"
		}
	}

	private fun classifyHorizontalPosition(x: Float): String {
		val centerX = screenWidth / 2
		val threshold = screenWidth * 0.2f

		return when {
			x < centerX - threshold -> "links"
			x > centerX + threshold -> "rechts"
			else -> "mittig"
		}
	}

	private fun buildPositionText(horizontal: String, vertical: String): String {
		return when {
			horizontal == "mittig" && vertical == "mittig" -> "in der Mitte"
			horizontal == "mittig" -> vertical
			vertical == "mittig" -> horizontal
			else -> "$vertical $horizontal"
		}
	}

	private fun calculateTotalTextArea(textBoxes: List<TextBoundingBox>): Float {
		return textBoxes.sumOf { (it.width * it.height).toDouble() }.toFloat()
	}

	private fun generatePositionalDescription(
		lines: List<List<TextBoundingBox>>,
		layoutInfo: Map<String, Any>
	): String {
		val result = StringBuilder()

		// Add overall position context for text with multiple lines
		if (lines.size > 1) {
			val textRegion = layoutInfo["textRegion"] as String
			val lineCount = layoutInfo["lineCount"] as Int
			result.append("$textRegion befinden sich $lineCount Textzeilen. ")
		}

		lines.forEachIndexed { lineIndex, line ->
			// Line position for single line text
			if (lines.size > 1) {
				val linePosition = when (lineIndex) {
					0 -> "In der ersten Zeile"
					lines.size - 1 -> "In der letzten Zeile"
					else -> "In der ${lineIndex + 1}. Zeile"
				}
				result.append("$linePosition ")
			}

			// Generate description for the current line with enhanced positioning
			val lineDescription = generateEnhancedLineDescription(line, lines.size == 1)
			result.append(lineDescription)

			// Punctuation
			if (!lineDescription.endsWith(".")) {
				result.append(".")
			}

			// Add spacing between lines
			if (lineIndex < lines.size - 1) {
				result.append(" ")
			}
		}

		return result.toString()
	}

	private fun generateEnhancedLineDescription(line: List<TextBoundingBox>, isSingleLine: Boolean): String {
		if (line.isEmpty()) return ""

		if (line.size == 1) {
			val textBox = line.first()
			val position = if (isSingleLine) {
				// For single text blocks detailed positioning description
				val detailed = getDetailedPosition(textBox)
				detailed
			} else {
				// Simpler positionin for multiline text
				"befindet sich der Text"
			}
			return "$position \"${textBox.text}\""
		}

		val result = StringBuilder()

		line.forEachIndexed { index, textBox ->
			when (index) {
				0 -> {
					// First element
					val position = if (isSingleLine) {
						getDetailedPosition(textBox)
					} else {
						getLineStartPosition(textBox.x1)
					}
					result.append("$position der Text \"${textBox.text}\"")
				}

				line.size - 1 -> {
					// Last element
					val relativePosition = getRelativePosition(line[index - 1], textBox)
					result.append(" und $relativePosition der Text \"${textBox.text}\"")
				}

				else -> {
					// Middle elements
					val relativePosition = getRelativePosition(line[index - 1], textBox)
					result.append(", anschließend $relativePosition der Text \"${textBox.text}\"")
				}
			}
		}

		return result.toString()
	}

	private fun getDetailedPosition(textBox: TextBoundingBox): String {
		val centerX = (textBox.x1 + textBox.x2) / 2
		val centerY = (textBox.y1 + textBox.y2) / 2

		val horizontalPos = classifyHorizontalPosition(centerX)
		val verticalPos = classifyVerticalPosition(centerY)
		val positionText = buildPositionText(horizontalPos, verticalPos)

		// Add size context similar to ObjectPositionClassifier
		val sizeInfo = getTextSizeInfo(textBox)

		return if (positionText == "in der Mitte") {
			"In der Mitte des Sichtfeldes befindet sich$sizeInfo"
		} else {
			"$positionText im Sichtfeldes befindet sich$sizeInfo"
		}
	}

	private fun getTextSizeInfo(textBox: TextBoundingBox): String {
		val textArea = textBox.width * textBox.height
		val screenPercentage = (textArea / (screenWidth * screenHeight)) * 100

		return when {
			screenPercentage > 15 -> " der große Text"
			screenPercentage > 5 -> " der gut sichtbare Text"
			screenPercentage > 1 -> " der Text"
			else -> " der kleine Text"
		}
	}

	private fun getLineStartPosition(x: Float): String {
		return when {
			x < 0.25f -> "Links befindet sich"
			x < 0.5f -> "Links der Mitte befindet sich"
			x < 0.75f -> "Rechts der Mitte befindet sich"
			else -> "Rechts befindet sich"
		}
	}

	private fun getRelativePosition(previous: TextBoundingBox, current: TextBoundingBox): String {
		val distance = current.x1 - previous.x2

		return when {
			distance < 0.05f -> "direkt rechts davon befindet sich"
			distance < 0.15f -> "rechts davon befindet sich"
			distance < 0.3f -> "weiter rechts befindet sich"
			else -> "ganz rechts befindet sich"
		}
	}

	private fun groupIntoLines(textBoxes: List<TextBoundingBox>): List<List<TextBoundingBox>> {
		val tolerance = 0.02f
		val sortedByY = textBoxes.sortedBy { it.y1 }
		val lines = mutableListOf<MutableList<TextBoundingBox>>()

		for (textBox in sortedByY) {
			val existingLine = lines.find { line ->
				val lineY = line.first().y1
				kotlin.math.abs(textBox.y1 - lineY) <= tolerance
			}

			if (existingLine != null) {
				existingLine.add(textBox)
			} else {
				lines.add(mutableListOf(textBox))
			}
		}

		return lines
	}

	// Debugging/Analysis
	fun getDetailedTextAnalysis(textBoxes: List<TextBoundingBox>): Map<String, String> {
		val lines = groupIntoLines(textBoxes)
		val layoutInfo = analyzeTextLayout(lines)

		return mapOf(
			"textRegion" to (layoutInfo["textRegion"] as String),
			"lineCount" to "${layoutInfo["lineCount"]}",
			"totalArea" to String.format("%.2f%%", (layoutInfo["totalTextArea"] as Float) * 100),
			"verticalSpread" to String.format("%.2f", layoutInfo["verticalSpread"] as Float),
			"horizontalSpread" to String.format("%.2f", layoutInfo["horizontalSpread"] as Float)
		)
	}
}
