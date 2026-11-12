package com.algorithmic_alliance.eyeaiapp.audio

import org.junit.Assert.*
import org.junit.Test
import java.io.DataInputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class PcmNetworkStreamTest {
	@Test(timeout = 5000)
	fun `TCP receives exact PCM bytes without WAV headers and closes on stop`() {
		ServerSocket(0, 1, InetAddress.getLoopbackAddress()).use { server ->
			server.soTimeout = 3000
			val ready = CountDownLatch(1)
			PcmNetworkStream("127.0.0.1", server.localPort, onConnected = { ready.countDown() }).use { stream ->
				stream.start()
				server.accept().use { receiver ->
					receiver.soTimeout = 2000
					assertTrue(ready.await(2, TimeUnit.SECONDS))
					val expected = byteArrayOf(0, -128, -1, 127, -1, -1, 0, 0)
					stream.offer(expected)
					val actual = ByteArray(expected.size)
					DataInputStream(receiver.getInputStream()).readFully(actual)
					assertArrayEquals(expected, actual)
					stream.close()
					assertEquals(-1, receiver.getInputStream().read())
				}
			}
		}
	}

	@Test(timeout = 8000)
	fun `connection is restored after receiver disconnects and continues with current audio`() {
		ServerSocket(0, 2, InetAddress.getLoopbackAddress()).use { server ->
			server.soTimeout = 5000
			val firstReady = CountDownLatch(1)
			val secondReady = CountDownLatch(1)
			val connections = AtomicInteger()
			PcmNetworkStream("127.0.0.1", server.localPort, onConnected = {
				if (connections.incrementAndGet() == 1) firstReady.countDown() else secondReady.countDown()
			}).use { stream ->
				stream.start()
				server.accept().use { first ->
					assertTrue(firstReady.await(2, TimeUnit.SECONDS))
					first.setSoLinger(true, 0) // RST makes the next write detect the disconnect.
				}
				stream.offer(ByteArray(1920))
				server.accept().use { second ->
					second.soTimeout = 2000
					assertTrue(secondReady.await(2, TimeUnit.SECONDS))
					val currentAudio = byteArrayOf(1, 2, 3, 4)
					stream.offer(currentAudio)
					val actual = ByteArray(4)
					DataInputStream(second.getInputStream()).readFully(actual)
					assertArrayEquals(currentAudio, actual)
				}
			}
		}
	}

	@Test(expected = IllegalArgumentException::class)
	fun `partial stereo frames are rejected`() {
		PcmNetworkStream("127.0.0.1").use { it.offer(byteArrayOf(1, 2, 3)) }
	}
}
