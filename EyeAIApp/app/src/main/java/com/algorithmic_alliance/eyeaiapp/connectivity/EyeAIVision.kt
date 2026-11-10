package com.algorithmic_alliance.eyeaiapp.connectivity

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.net.InetSocketAddress
import java.net.Socket
import java.net.UnknownHostException
import java.util.concurrent.Executors


open class EyeAIVision(
	app: Context,
	private val ip: String,
	private val onSingleClick: () -> Unit,
	private val onDoubleClick: () -> Unit,
	private val onConnectingSocket: () -> Unit,
	private val onSocketConnectionEstablished: () -> Unit,
	private val onSocketFailed: (Exception) -> Unit,
	private val onWebrtcFrame: (Bitmap) -> Unit
) {
	private lateinit var touchSocket: Socket
	private var audioSocket: Socket? = null
	private var webRtcClientValue: WebRtcClient? = null
	private val socketExecutor = Executors.newSingleThreadExecutor()
	private val socketThread: CoroutineScope =
		CoroutineScope(socketExecutor.asCoroutineDispatcher())

	val whepUrl = "https://$ip:8889/cam/whep"

	init {
		// Touch Button Client starten
		socketThread.launch {
			try {
				onConnectingSocket()
				touchSocket = Socket(ip, 3333)
				val reader = BufferedReader(InputStreamReader(touchSocket.inputStream))
				onSocketConnectionEstablished()

				while (isActive) {
					val value = reader.read()

					if (value == -1) break
					val char = value.toChar()

					if (char == '1') {
						onSingleClick()
					} else if (char == '2') {
						onDoubleClick()
					}
				}
			} catch (e: IOException) {
				onSocketFailed(e)
			} catch (e: UnknownHostException) {
				onSocketFailed(e)
			}
		}

		// initialize webrtc
		Log.e(EyeAIApp.APP_LOG_TAG, "!!! Starting EyeAIVision source: ip=$ip, url=$whepUrl !!!")
		Log.e(EyeAIApp.APP_LOG_TAG, "!!! Initializing WebRtcClient for WHEP: $whepUrl !!!")
		webRtcClientValue = WebRtcClient(app) { bitmap ->
			Log.v(EyeAIApp.APP_LOG_TAG, "EyeAIVision received WebRTC frame, invoking callback")
			onWebrtcFrame(bitmap)
		}.also {
			Log.e(EyeAIApp.APP_LOG_TAG, "!!! Starting WebRtcClient !!!")
			it.start(whepUrl)
		}
	}

	/**
	 * Connects to the network microphone on port 4040.
	 * Must be called from a background thread (e.g. Dispatchers.IO).
	 * Returns the InputStream of the audio socket, or null if connection failed.
	 */
	@Synchronized
	fun startAudioStream(): InputStream? {
		stopAudioStream()
		var lastException: Exception? = null
		for (attempt in 1..3) {
			try {
				Log.i(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVision] Connecting to network microphone on $ip:4040 (attempt $attempt/3)..."
				)
				val socket = Socket()
				socket.connect(InetSocketAddress(ip, 4040), 3000)
				audioSocket = socket
				Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Network microphone connected on $ip:4040")
				return LoggingAudioInputStream(socket.inputStream)
			} catch (e: Exception) {
				lastException = e
				Log.w(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVision] Attempt $attempt to connect to network mic on port 4040 failed: ${e.message}"
				)
				if (attempt < 3) {
					try {
						Thread.sleep(150)
					} catch (_: InterruptedException) {
						Thread.currentThread().interrupt()
						break
					}
				}
			}
		}
		Log.e(
			EyeAIApp.APP_LOG_TAG,
			"[EyeAIVision] Failed to connect to network microphone on port 4040 after 3 attempts: ${lastException?.message}",
			lastException
		)
		stopAudioStream()
		return null
	}

	/**
	 * Stops and closes the network microphone connection on port 4040.
	 */
	@Synchronized
	fun stopAudioStream() {
		try {
			audioSocket?.close()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Error closing audio socket: ${e.message}")
		} finally {
			audioSocket = null
		}
	}

	/** Stops the existing external source without touching the common analyzer/models. */
	fun close() {
		stopAudioStream()
		if (::touchSocket.isInitialized) {
			try {
				touchSocket.close()
			} catch (_: IOException) {
			}
		}
		socketThread.cancel()
		socketExecutor.shutdownNow()

		// stop webrtc
		webRtcClientValue?.stop()
		webRtcClientValue = null
	}
}

private class LoggingAudioInputStream(private val delegate: InputStream) : InputStream() {
	private var totalBytes: Long = 0
	private var lastLogTime: Long = System.currentTimeMillis()
	private var bytesSinceLastLog: Long = 0

	override fun read(): Int {
		val b = delegate.read()
		if (b != -1) {
			totalBytes++
			bytesSinceLastLog++
			checkLog()
		} else {
			Log.w(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudio] Stream EOF reached after $totalBytes bytes")
		}
		return b
	}

	override fun read(b: ByteArray, off: Int, len: Int): Int {
		val bytesRead = delegate.read(b, off, len)
		if (bytesRead > 0) {
			totalBytes += bytesRead
			bytesSinceLastLog += bytesRead
			val rms = calculatePcm16Rms(b, off, bytesRead)
			val now = System.currentTimeMillis()
			if (now - lastLogTime >= 1000L) {
				Log.d(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVisionAudio] Received $bytesRead bytes PCM from port 4040 (rate: ${bytesSinceLastLog} bytes/sec, total: $totalBytes bytes, RMS amplitude: $rms)"
				)
				lastLogTime = now
				bytesSinceLastLog = 0
			}
		} else if (bytesRead == -1) {
			Log.w(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudio] Stream EOF reached after $totalBytes bytes")
		}
		return bytesRead
	}

	override fun close() {
		Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudio] Audio stream closed. Total received: $totalBytes bytes")
		delegate.close()
	}

	private fun checkLog() {
		val now = System.currentTimeMillis()
		if (now - lastLogTime >= 1000L) {
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[EyeAIVisionAudio] Received byte stream from port 4040 (rate: ${bytesSinceLastLog} bytes/sec, total: $totalBytes bytes)"
			)
			lastLogTime = now
			bytesSinceLastLog = 0
		}
	}

	private fun calculatePcm16Rms(buffer: ByteArray, offset: Int, length: Int): Int {
		var sum = 0.0
		var count = 0
		var i = offset
		while (i + 1 < offset + length) {
			val sample = ((buffer[i + 1].toInt() shl 8) or (buffer[i].toInt() and 0xFF)).toShort()
			sum += sample.toDouble() * sample.toDouble()
			count++
			i += 2
		}
		return if (count > 0) kotlin.math.sqrt(sum / count).toInt() else 0
	}
}
