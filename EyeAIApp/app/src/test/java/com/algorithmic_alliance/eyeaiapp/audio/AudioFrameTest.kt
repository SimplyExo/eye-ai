package com.algorithmic_alliance.eyeaiapp.audio

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class AudioFrameTest {
    @Test
    fun `audio frame takes ownership-safe copy of PCM data`() {
        val source = byteArrayOf(1, 2, 3, 4)
        val frame = AudioFrame(source, sampleRateHz = 48_000, channelCount = 1, timestampNanos = 7L)

        source[0] = 99

        assertArrayEquals(byteArrayOf(1, 2, 3, 4), frame.pcm16)
        assertEquals(48_000, frame.sampleRateHz)
        assertEquals(1, frame.channelCount)
        assertEquals(7L, frame.timestampNanos)
    }
}
