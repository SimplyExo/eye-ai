package com.algorithmic_alliance.eyeaiapp.audio

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/** The network wire format is headerless 48 kHz, stereo, signed 16-bit little endian. */
class PcmMixer {
	private class Clip(val samples: ShortArray, val done: () -> Unit, var position: Int = 0)
	private val buses = mutableMapOf<String, ArrayDeque<Clip>>()
	private val pausedBuses = mutableSetOf<String>()

	@Synchronized
	fun enqueue(bus: String, samples: ShortArray, done: () -> Unit = {}) {
		buses.getOrPut(bus) { ArrayDeque() }.addLast(Clip(samples, done))
	}

	@Synchronized
	fun remove(bus: String) { buses.remove(bus); pausedBuses.remove(bus) }

	@Synchronized
	fun pause(bus: String) { if (bus in buses) pausedBuses.add(bus) }

	@Synchronized
	fun resume(bus: String) { pausedBuses.remove(bus) }

	@Synchronized
	fun clear(): List<() -> Unit> {
		val callbacks = buses.values.flatMap { queue -> queue.map { it.done } }
		buses.clear()
		pausedBuses.clear()
		return callbacks
	}

	@Synchronized
	fun contains(bus: String): Boolean = bus !in pausedBuses && buses[bus]?.isNotEmpty() == true

	/** Call completions outside the mixer lock, after the frame has been handed off. */
	@Synchronized
	fun mix(spatial: ByteArray): Pair<ByteArray, List<() -> Unit>> {
		require(spatial.size % 4 == 0)
		val sum = IntArray(spatial.size / 2) { i ->
			((spatial[i * 2].toInt() and 255) or (spatial[i * 2 + 1].toInt() shl 8)).toShort().toInt()
		}
		val completed = mutableListOf<() -> Unit>()
		val iterator = buses.entries.iterator()
		while (iterator.hasNext()) {
			val (bus, queue) = iterator.next()
			if (bus in pausedBuses) continue
			var offset = 0
			while (queue.isNotEmpty() && offset < sum.size) {
				val clip = queue.first()
				val count = minOf(sum.size - offset, clip.samples.size - clip.position)
				repeat(count) { sum[offset++] += clip.samples[clip.position++].toInt() }
				if (clip.position == clip.samples.size) {
					queue.removeFirst()
					completed += clip.done
				}
			}
			if (queue.isEmpty()) iterator.remove()
		}
		val output = ByteArray(spatial.size)
		sum.forEachIndexed { i, value ->
			val sample = value.coerceIn(-32768, 32767)
			output[i * 2] = sample.toByte()
			output[i * 2 + 1] = (sample shr 8).toByte()
		}
		return output to completed
	}

	companion object {
		/** Decode the PCM WAV assets and Android TTS synthesis files, including extra RIFF chunks. */
		fun decodeWave(bytes: ByteArray): ShortArray {
			val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
			require(bytes.size >= 12 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
				String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE") { "Expected a RIFF WAV file" }
			var channels = 0
			var rate = 0
			var bits = 0
			var format = 0
			var dataOffset = -1
			var dataSize = 0
			var offset = 12
			while (offset <= bytes.size - 8) {
				val size = buffer.getInt(offset + 4)
				require(size >= 0 && size <= bytes.size - offset - 8) { "Truncated WAV chunk" }
				when (String(bytes, offset, 4, Charsets.US_ASCII)) {
					"fmt " -> {
						require(size >= 16)
						format = buffer.getShort(offset + 8).toInt()
						channels = buffer.getShort(offset + 10).toInt()
						rate = buffer.getInt(offset + 12)
						bits = buffer.getShort(offset + 22).toInt()
					}
					"data" -> { dataOffset = offset + 8; dataSize = size }
				}
				offset += 8 + size + (size and 1)
			}
			require(channels in 1..2 && rate in 8000..192000 && dataOffset >= 0)
			require((format == 1 && bits in listOf(8, 16, 24, 32)) || (format == 3 && bits == 32)) {
				"Unsupported WAV encoding: format=$format, bits=$bits"
			}
			val stride = bits / 8
			val inputFrames = dataSize / (stride * channels)
			if (inputFrames == 0) return ShortArray(0)
			fun sample(frame: Int, channel: Int): Double {
				val index = dataOffset + (frame * channels + channel.coerceAtMost(channels - 1)) * stride
				return when {
					format == 3 -> buffer.getFloat(index).toDouble().coerceIn(-1.0, 1.0) * 32767
					bits == 8 -> ((bytes[index].toInt() and 255) - 128) * 256.0
					bits == 16 -> buffer.getShort(index).toDouble()
					bits == 24 -> (((bytes[index].toInt() and 255) or
						((bytes[index + 1].toInt() and 255) shl 8) or (bytes[index + 2].toInt() shl 16)) / 256.0)
					else -> buffer.getInt(index) / 65536.0
				}
			}
			val frames = (inputFrames.toLong() * 48000 / rate).toInt()
			return ShortArray(frames * 2) { index ->
				val position = (index / 2).toDouble() * rate / 48000
				val first = position.toInt().coerceAtMost(inputFrames - 1)
				val second = (first + 1).coerceAtMost(inputFrames - 1)
				val fraction = position - first
				(sample(first, index % 2) * (1 - fraction) + sample(second, index % 2) * fraction)
					.roundToInt().coerceIn(-32768, 32767).toShort()
			}
		}
	}
}
