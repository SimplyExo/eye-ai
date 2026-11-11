package com.algorithmic_alliance.eyeaiapp.audio

import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Sends live PCM without headers. Congestion drops old frames rather than delaying playback. */
class PcmNetworkStream(
	private val host: String,
	private val port: Int = 4041,
	private val onConnected: () -> Unit = {},
	private val onError: (Exception) -> Unit = {},
) : AutoCloseable {
	private val frames = ArrayBlockingQueue<ByteArray>(5)
	@Volatile private var running = true
	@Volatile private var socket: Socket? = null
	@Volatile private var writeStarted = 0L
	private val watchdog = Executors.newSingleThreadScheduledExecutor()
	private val thread = Thread({ transmit() }, "EyeAI audio TCP")

	fun start() {
		watchdog.scheduleAtFixedRate({
			val since = writeStarted
			if (since != 0L && System.nanoTime() - since > TimeUnit.SECONDS.toNanos(1)) {
				try { socket?.close() } catch (_: Exception) { }
			}
		}, 250, 250, TimeUnit.MILLISECONDS)
		thread.start()
	}

	fun offer(data: ByteArray) {
		require(data.size % 4 == 0) { "PCM must contain complete 16-bit stereo frames" }
		if (!running) return
		if (!frames.offer(data)) { frames.poll(); frames.offer(data) }
	}

	override fun close() {
		running = false
		try { socket?.close() } catch (_: Exception) { }
		thread.interrupt()
		watchdog.shutdownNow()
		frames.clear()
	}

	private fun transmit() {
		while (running) {
			try {
				Socket().use { connection ->
					socket = connection // close() can interrupt both connect and write
					if (!running) return
					connection.tcpNoDelay = true
					connection.connect(InetSocketAddress(host, port), 2000)
					frames.clear()
					onConnected()
					val output = connection.getOutputStream()
					while (running) {
						val data = frames.take()
						writeStarted = System.nanoTime()
						output.write(data)
						writeStarted = 0L
					}
				}
			} catch (_: InterruptedException) {
				return
			} catch (error: Exception) {
				if (running) onError(error)
			} finally { writeStarted = 0L; socket = null }
			if (running) try { Thread.sleep(1000) } catch (_: InterruptedException) { return }
		}
	}
}
