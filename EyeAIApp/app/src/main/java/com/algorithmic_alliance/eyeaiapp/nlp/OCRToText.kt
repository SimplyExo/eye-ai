package com.algorithmic_alliance.eyeaiapp.nlp

import com.algorithmic_alliance.eyeaiapp.ocr.TextBoundingBox

class OCRToText {

	fun generateReadableText(textBoxes: List<TextBoundingBox>): String {
		if (textBoxes.isEmpty()) return ""

		// Groups boxes by y-position
		val lines = groupIntoLines(textBoxes)

		// Sorts lines from left to right
		val sortedLines = lines.map { line ->
			line.sortedBy { it.x1 }
		}

		// Generates natural text
		return generateDescription(sortedLines)
	}

	private fun groupIntoLines(textBoxes: List<TextBoundingBox>): List<List<TextBoundingBox>> {
		val tolerance = 0.02f // including a 2% tolerance between lines
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

	private fun generateDescription(lines: List<List<TextBoundingBox>>): String {
		val result = StringBuilder()

		lines.forEachIndexed { lineIndex, line ->
			// Signalises next line
			if (lineIndex > 0) {
				result.append("In der nächsten Zeile ")
			}

			// Generates descrition for the current line
			val lineDescription = generateLineDescription(line)
			result.append(lineDescription)

			// Punctuation
			if (!lineDescription.endsWith(".")) {
				result.append(".")
			}

			// Blankspaces
			if (lineIndex < lines.size - 1) {
				result.append(" ")
			}
		}

		return result.toString()
	}

	private fun generateLineDescription(line: List<TextBoundingBox>): String {
		if (line.isEmpty()) return ""
		if (line.size == 1) {
			return "befindet sich der Text \"${line.first().text}\""
		}

		val result = StringBuilder()

		line.forEachIndexed { index, textBox ->
			when (index) {
				0 -> {
					// First line
					val position = getAbsolutePosition(textBox.x1)
					result.append("$position befindet sich der Text \"${textBox.text}\"")
				}
				line.size - 1 -> {
					// Last line
					val relativePosition = getRelativePosition(line[index - 1], textBox)
					result.append(" und $relativePosition der Text \"${textBox.text}\"")
				}
				else -> {
					// Lines between first and last
					val relativePosition = getRelativePosition(line[index - 1], textBox)
					result.append(", anschließend $relativePosition der Text \"${textBox.text}\"")
				}
			}
		}

		return result.toString()
	}

	private fun getAbsolutePosition(x: Float): String {
		return when {
			x < 0.25f -> "Links"
			x < 0.5f -> "Links der Mitte"
			x < 0.75f -> "Rechts der Mitte"
			else -> "Rechts"
		}
	}

	private fun getRelativePosition(previous: TextBoundingBox, current: TextBoundingBox): String {
		val distance = current.x1 - previous.x2

		return when {
			distance < 0.1f -> "direkt rechts davon befindet sich"
			distance < 0.3f -> "rechts davon befindet sich"
			distance < 0.6f -> "weiter rechts befindet sich"
			else -> "ganz rechts befindet sich"
		}
}

}