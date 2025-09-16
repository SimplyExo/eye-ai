package com.algorithmic_alliance.eyeaiapp.llm.google_ai_studio

import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import java.io.BufferedReader

class StreamProcessor(private val parser: StreamParser) {

	fun processStream(
		reader: BufferedReader,
		onComplete: () -> Unit,
		onError: (Exception) -> Unit,
		hasCalledComplete: () -> Boolean
	) {
		val eventBuffer = StringBuilder()
		val rawBuffer = StringBuilder()

		try {
			reader.forEachLine { rawLine ->
				if (hasCalledComplete()) return@forEachLine

				Log.d(EyeAIApp.APP_LOG_TAG, "Raw stream line: $rawLine")
				val line = rawLine.trimEnd('\r', '\n')

				when {
					line.startsWith("data:") -> processDataLine(line, eventBuffer, rawBuffer)
					line.isBlank() -> processBlankLine(eventBuffer, rawBuffer)
					else -> processOtherLine(line, rawBuffer)
				}

				if (line.startsWith("data:") && line.substringAfter("data:").trimStart().let {
						it == "[DONE]" || it == "\"[DONE]\""
					}) {
					Log.d(EyeAIApp.APP_LOG_TAG, "Stream signalled DONE")
					processRemainingBuffer(eventBuffer, rawBuffer)
					onComplete()
					return@forEachLine
				}
			}

			// Process any remaining data
			processRemainingBuffer(eventBuffer, rawBuffer)
			val remainder = rawBuffer.toString().trim()
			if (remainder.isNotEmpty() && remainder.startsWith("[") && remainder.endsWith("]")) {
				try {
					Log.d(EyeAIApp.APP_LOG_TAG, "EOF: parsing final array remainder (len=${remainder.length})")
					parser.parseStreamChunk(remainder)
				} catch (e: Exception) {
					Log.w(EyeAIApp.APP_LOG_TAG, "EOF parse of remaining array failed", e)
				}
			}

			Log.d(EyeAIApp.APP_LOG_TAG, "Stream finished naturally")
			onComplete()

		} catch (e: Exception) {
			Log.w(EyeAIApp.APP_LOG_TAG, "Exception during stream reading", e)
			onError(e)
		} finally {
			reader.close()
		}
	}

	private fun processDataLine(line: String, eventBuffer: StringBuilder, rawBuffer: StringBuilder) {
		val payload = line.substringAfter("data:").trimStart()
		eventBuffer.append(payload)

		if (eventBuffer.isNotEmpty()) {
			rawBuffer.append(eventBuffer.toString())
			eventBuffer.clear()
			extractObjectsFromRawBuffer(rawBuffer)
		}
	}

	private fun processBlankLine(eventBuffer: StringBuilder, rawBuffer: StringBuilder) {
		if (eventBuffer.isNotEmpty()) {
			rawBuffer.append(eventBuffer.toString())
			eventBuffer.clear()
			extractObjectsFromRawBuffer(rawBuffer)
		}
	}

	private fun processOtherLine(line: String, rawBuffer: StringBuilder) {
		val trimmed = line.trim()
		if (trimmed in listOf("[", ",", "]")) {
			rawBuffer.append(trimmed)
			extractObjectsFromRawBuffer(rawBuffer)
		} else {
			rawBuffer.append(line)
			extractObjectsFromRawBuffer(rawBuffer)
		}
	}

	private fun processRemainingBuffer(eventBuffer: StringBuilder, rawBuffer: StringBuilder) {
		if (eventBuffer.isNotEmpty()) {
			rawBuffer.append(eventBuffer.toString())
			eventBuffer.clear()
		}
		extractObjectsFromRawBuffer(rawBuffer)
	}

	private fun extractObjectsFromRawBuffer(rawBuffer: StringBuilder) {
		var buf = rawBuffer.toString()
		var idx = 0

		while (true) {
			val start = buf.indexOf('{', idx)
			if (start == -1) break

			var braceCount = 0
			var end = -1
			var i = start

			while (i < buf.length) {
				when (buf[i]) {
					'{' -> braceCount++
					'}' -> {
						braceCount--
						if (braceCount == 0) {
							end = i
							break
						}
					}
				}
				i++
			}

			if (end == -1) break

			val jsonObject = buf.substring(start, end + 1)
			try {
				Log.d(EyeAIApp.APP_LOG_TAG, "Found complete JSON object in buffer (len=${jsonObject.length}), passing to parser.")
				parser.parseStreamChunk(jsonObject)
			} catch (e: Exception) {
				Log.w(EyeAIApp.APP_LOG_TAG, "parseStreamChunk threw for extracted object; continuing.", e)
			}

			buf = buf.substring(end + 1)
			idx = 0
		}

		rawBuffer.setLength(0)
		rawBuffer.append(buf)
	}
}
