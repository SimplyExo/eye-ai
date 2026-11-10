package com.algorithmic_alliance.eyeaiapp.audio

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PcmMixerTest {
	private fun pcm(vararg samples: Int): ByteArray = ByteBuffer.allocate(samples.size * 2)
		.order(ByteOrder.LITTLE_ENDIAN).apply { samples.forEach { putShort(it.toShort()) } }.array()

	@Test fun `mix preserves stereo and saturates overlapping spatial speech and signals`() {
		val mixer = PcmMixer()
		mixer.enqueue("tts", shortArrayOf(10000, -10000, 1000, 2000))
		mixer.enqueue("signal", shortArrayOf(10000, -10000, 3000, 4000))
		val (result, _) = mixer.mix(pcm(20000, -20000, 5000, 6000))
		assertArrayEquals(pcm(32767, -32768, 9000, 12000), result)
	}

	@Test fun `speech queue completes only after playback and flush removes queued audio`() {
		val mixer = PcmMixer()
		var completed = 0
		mixer.enqueue("tts", shortArrayOf(10, 20, 30, 40)) { completed++ }
		mixer.enqueue("tts", shortArrayOf(50, 60)) { completed++ }
		val (first, callbacks) = mixer.mix(ByteArray(4))
		assertArrayEquals(pcm(10, 20), first)
		assertTrue(callbacks.isEmpty())
		val (second, done) = mixer.mix(ByteArray(4))
		assertArrayEquals(pcm(30, 40), second)
		assertEquals(0, completed)
		done.forEach { it() }
		assertEquals(1, completed)
		mixer.remove("tts")
		assertArrayEquals(ByteArray(4), mixer.mix(ByteArray(4)).first)
		assertEquals(1, completed)
	}

	private fun wave(rate: Int, channels: Int, data: ByteArray): ByteArray {
		// Odd-sized metadata chunk verifies RIFF padding and a nonstandard data offset.
		return ByteBuffer.allocate(54 + data.size).order(ByteOrder.LITTLE_ENDIAN).apply {
			put("RIFF".toByteArray()); putInt(capacity() - 8); put("WAVE".toByteArray())
			put("JUNK".toByteArray()); putInt(1); put(42.toByte()); put(0.toByte())
			put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(channels.toShort())
			putInt(rate); putInt(rate * channels * 2); putShort((channels * 2).toShort()); putShort(16)
			put("data".toByteArray()); putInt(data.size); put(data)
		}.array()
	}

	@Test fun `resamples mono TTS to 48k stereo and skips padded metadata`() {
		val output = PcmMixer.decodeWave(wave(24000, 1, pcm(0, 1000)))
		assertArrayEquals(shortArrayOf(0, 0, 500, 500, 1000, 1000, 1000, 1000), output)
	}

	@Test fun `pausing a tutorial preserves its position while spatial audio continues`() {
		val mixer = PcmMixer()
		mixer.enqueue("tutorial", shortArrayOf(10, 20, 30, 40))
		assertArrayEquals(pcm(10, 20), mixer.mix(ByteArray(4)).first)
		mixer.pause("tutorial")
		assertArrayEquals(pcm(1, 2), mixer.mix(pcm(1, 2)).first)
		mixer.resume("tutorial")
		assertArrayEquals(pcm(30, 40), mixer.mix(ByteArray(4)).first)
	}

	@Test fun `48k stereo is preserved exactly including negative samples`() {
		val output = PcmMixer.decodeWave(wave(48000, 2, pcm(-32768, 32767, -1, 0)))
		assertArrayEquals(shortArrayOf(-32768, 32767, -1, 0), output)
	}

	@Test(expected = IllegalArgumentException::class)
	fun `truncated WAV is rejected`() {
		PcmMixer.decodeWave(wave(48000, 2, pcm(1, 2)).dropLast(1).toByteArray())
	}

	@Test fun `all bundled signal and tutorial WAV files decode to stereo PCM`() {
		val raw = File("src/main/res/raw").takeIf { it.isDirectory }
			?: File("app/src/main/res/raw")
		val files = raw.listFiles { file -> file.extension == "wav" }!!
		assertEquals(5, files.size)
		for (file in files) {
			val samples = PcmMixer.decodeWave(file.readBytes())
			assertTrue("${file.name} has audio", samples.any { it != 0.toShort() })
			assertEquals("${file.name} has complete stereo frames", 0, samples.size % 2)
		}
	}
}
