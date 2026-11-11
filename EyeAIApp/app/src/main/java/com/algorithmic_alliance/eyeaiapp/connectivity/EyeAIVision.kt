package com.algorithmic_alliance.eyeaiapp.connectivity

import android.content.Context
import android.database.ContentObserver
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.projection.MediaProjection
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.HttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.net.UnknownHostException
import java.util.concurrent.Executors
import kotlin.math.roundToInt

data class EyeAIVisionVolumeStatus(
	val volume: Int = 0,
	val muted: Boolean = false,
	val card: String = "",
	val control: String = "",
	val success: Boolean = false
)

open class EyeAIVision(
	app: Context,
	private val ip: String,
	private val onSingleClick: () -> Unit,
	private val onDoubleClick: () -> Unit,
	private val onConnectingSocket: () -> Unit,
	private val onSocketConnectionEstablished: () -> Unit,
	private val onSocketFailed: (Exception) -> Unit,
	private val onStreamError: (String) -> Unit = {},
	private val onWebrtcFrame: (Bitmap) -> Unit
) {
	private val context: Context = app.applicationContext
	private lateinit var touchSocket: Socket
	private var audioSocket: Socket? = null
	private var audioOutputSocket: Socket? = null
	private var audioOutputStream: OutputStream? = null
	private var isAudioCaptureRunning: Boolean = false
	private var webRtcClientValue: WebRtcClient? = null
	private val socketExecutor = Executors.newSingleThreadExecutor()
	private val socketThread: CoroutineScope =
		CoroutineScope(socketExecutor.asCoroutineDispatcher())

	private var volumeObserver: VolumeObserver? = null
	private var lastSyncedVolume: Int = -1

	private val httpClient: OkHttpClient = OkHttpClient.Builder()
		.hostnameVerifier { hostname, _ ->
			hostname == ip || hostname == "192.168.4.1" || hostname == "127.0.0.1" || hostname == "localhost"
		}
		.build()

	val whepUrl = "http://$ip:8889/cam/whep"

	init {
		ensureWiFiNetworkBound()

		// Touch Button Client starten mit Retry-Logik
		socketThread.launch {
			ensureWiFiNetworkBound()
			onConnectingSocket()
			var connectedSocket: Socket? = null
			var lastException: Exception? = null
			for (attempt in 1..5) {
				try {
					Log.i(
						EyeAIApp.APP_LOG_TAG,
						"[EyeAIVision] Connecting to touch socket on $ip:3333 (attempt $attempt/5)..."
					)
					val socket = Socket()
					socket.connect(InetSocketAddress(ip, 3333), 3000)
					connectedSocket = socket
					Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Touch socket connected on $ip:3333")
					break
				} catch (e: Exception) {
					lastException = e
					Log.w(
						EyeAIApp.APP_LOG_TAG,
						"[EyeAIVision] Attempt $attempt to connect to touch socket on port 3333 failed: ${e.message}"
					)
					if (attempt < 5) {
						try {
							Thread.sleep(500)
						} catch (_: InterruptedException) {
							Thread.currentThread().interrupt()
							break
						}
					}
				}
			}

			if (connectedSocket == null) {
				Log.e(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVision] Failed to connect to touch socket on port 3333 after 5 attempts: ${lastException?.message}",
					lastException
				)
				onSocketFailed(lastException ?: IOException("Failed to connect to $ip:3333"))
				return@launch
			}

			touchSocket = connectedSocket
			try {
				val reader = BufferedReader(InputStreamReader(touchSocket.inputStream))
				onSocketConnectionEstablished()
				syncVolumeWithAndroid()

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
		webRtcClientValue = WebRtcClient(app, onStreamError = onStreamError) { bitmap ->
			Log.v(EyeAIApp.APP_LOG_TAG, "EyeAIVision received WebRTC frame, invoking callback")
			onWebrtcFrame(bitmap)
		}.also {
			Log.e(EyeAIApp.APP_LOG_TAG, "!!! Starting WebRtcClient !!!")
			it.start(whepUrl)
		}

		startVolumeObserver()
		syncVolumeWithAndroid()
	}

	private fun ensureWiFiNetworkBound() {
		try {
			val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager ?: return
			// Keep the exact network selected by the connection page. Android may
			// report a different default WiFi network when using concurrent WiFi.
			val boundNetwork = cm.boundNetworkForProcess
			if (boundNetwork != null && cm.getNetworkCapabilities(boundNetwork)
					?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true) return
			val activeNetwork = cm.activeNetwork
			if (activeNetwork != null) {
				val caps = cm.getNetworkCapabilities(activeNetwork)
				if (caps != null && caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
					cm.bindProcessToNetwork(activeNetwork)
					Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Bound process to active WiFi network")
					return
				}
			}
			@Suppress("DEPRECATION")
			for (network in cm.allNetworks) {
				val caps = cm.getNetworkCapabilities(network) ?: continue
				if (caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
					cm.bindProcessToNetwork(network)
					Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Bound process to WiFi network: $network")
					return
				}
			}

			val request = NetworkRequest.Builder()
				.addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
				.removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
				.build()

			cm.requestNetwork(request, object : ConnectivityManager.NetworkCallback() {
				override fun onAvailable(network: Network) {
					try {
						cm.bindProcessToNetwork(network)
						Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Bound process to WiFi network via request: $network")
					} catch (e: Exception) {
						Log.w(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Error binding process to network in callback: ${e.message}")
					}
				}
			})
		} catch (e: Exception) {
			Log.w(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Failed to bind process to WiFi network: ${e.message}")
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
		ensureWiFiNetworkBound()
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

	/**
	 * Connects to the network speaker on port 4041.
	 * Must be called from a background thread (e.g. Dispatchers.IO).
	 * Returns the OutputStream of the audio socket, or null if connection failed.
	 */
	@Synchronized
	fun startAudioOutputStream(): OutputStream? {
		stopAudioOutputStream()
		ensureWiFiNetworkBound()
		var lastException: Exception? = null
		for (attempt in 1..3) {
			try {
				Log.i(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVision] Connecting to network speaker on $ip:4041 (attempt $attempt/3)..."
				)
				val socket = Socket()
				socket.connect(InetSocketAddress(ip, 4041), 3000)
				audioOutputSocket = socket
				val stream = LoggingAudioOutputStream(socket.outputStream)
				audioOutputStream = stream
				Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Network speaker connected on $ip:4041")
				return stream
			} catch (e: Exception) {
				lastException = e
				Log.w(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVision] Attempt $attempt to connect to network speaker on port 4041 failed: ${e.message}"
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
			"[EyeAIVision] Failed to connect to network speaker on port 4041 after 3 attempts: ${lastException?.message}",
			lastException
		)
		stopAudioOutputStream()
		return null
	}

	/**
	 * Stops and closes the network speaker output stream on port 4041.
	 */
	@Synchronized
	fun stopAudioOutputStream() {
		try {
			audioOutputStream?.close()
			audioOutputSocket?.close()
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Error closing audio output socket: ${e.message}")
		} finally {
			audioOutputStream = null
			audioOutputSocket = null
		}
	}

	/**
	 * Writes raw PCM audio data (16-bit LE, 48kHz, stereo) directly to port 4041 output stream.
	 */
	@Synchronized
	fun writeAudioOutputData(data: ByteArray, offset: Int = 0, length: Int = data.size): Boolean {
		val stream = audioOutputStream ?: return false
		return try {
			stream.write(data, offset, length)
			stream.flush()
			true
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVision] Error writing audio output data to port 4041: ${e.message}")
			false
		}
	}

	/**
	 * Returns true if port 4041 audio output stream is currently connected and active.
	 */
	@Synchronized
	fun isAudioOutputConnected(): Boolean {
		return audioOutputStream != null && audioOutputSocket?.isConnected == true
	}

	/**
	 * Starts capturing app audio playback on Android 10+ (API 29+) and streams it as 48kHz Stereo PCM (s16le) to port 4041.
	 */
	fun startAudioPlaybackCapture(mediaProjection: MediaProjection? = null): Boolean {
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
			Log.w(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] AudioPlaybackCapture requires Android 10 (API 29)+")
			return false
		}
		if (isAudioCaptureRunning) return true
		isAudioCaptureRunning = true

		socketThread.launch(Dispatchers.IO) {
			val outputStream = startAudioOutputStream()
			if (outputStream == null) {
				Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] Failed to open port 4041 stream for audio capture")
				isAudioCaptureRunning = false
				return@launch
			}

			val sampleRate = 48000
			val channelConfig = AudioFormat.CHANNEL_IN_STEREO
			val audioFormat = AudioFormat.ENCODING_PCM_16BIT
			val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
			val bufferSize = maxOf(minBufferSize, 4096)

			var audioRecord: AudioRecord? = null
			try {
				val recordBuilder = AudioRecord.Builder()
					.setAudioFormat(
						AudioFormat.Builder()
							.setEncoding(audioFormat)
							.setSampleRate(sampleRate)
							.setChannelMask(channelConfig)
							.build()
					)
					.setBufferSizeInBytes(bufferSize)

				if (mediaProjection != null) {
					val config = AudioPlaybackCaptureConfiguration.Builder(mediaProjection)
						.addMatchingUsage(AudioAttributes.USAGE_MEDIA)
						.addMatchingUsage(AudioAttributes.USAGE_GAME)
						.addMatchingUsage(AudioAttributes.USAGE_UNKNOWN)
						.addMatchingUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
						.addMatchingUsage(AudioAttributes.USAGE_ASSISTANT)
						.build()
					recordBuilder.setAudioPlaybackCaptureConfig(config)
				}

				try {
					audioRecord = recordBuilder.build()
				} catch (e: SecurityException) {
					Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] Permission denied for AudioRecord: ${e.message}")
					stopAudioOutputStream()
					isAudioCaptureRunning = false
					return@launch
				}

				audioRecord.startRecording()
				Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] Audio playback capture started (48kHz Stereo s16le)")

				val buffer = ByteArray(bufferSize)
				while (isActive && isAudioCaptureRunning && audioRecord.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
					val bytesRead = audioRecord.read(buffer, 0, buffer.size)
					if (bytesRead > 0) {
						outputStream.write(buffer, 0, bytesRead)
						outputStream.flush()
					} else if (bytesRead < 0) {
						Log.w(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] AudioRecord read error: $bytesRead")
						break
					}
				}
			} catch (e: Exception) {
				Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] Exception in audio capture loop", e)
			} finally {
				try {
					audioRecord?.stop()
					audioRecord?.release()
				} catch (_: Exception) {}
				stopAudioOutputStream()
				isAudioCaptureRunning = false
				Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] Audio playback capture stopped")
			}
		}
		return true
	}

	/**
	 * Stops capturing app audio playback and closes port 4041 stream.
	 */
	fun stopAudioPlaybackCapture() {
		isAudioCaptureRunning = false
		stopAudioOutputStream()
	}

	/**
	 * Fetches the volume status from EyeAIVision HTTP API (GET /api/volume).
	 */
	fun fetchVolumeStatus(): EyeAIVisionVolumeStatus? {
		ensureWiFiNetworkBound()
		val url = "http://$ip:8080/api/volume"
		val request = Request.Builder().url(url).get().build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use {
				if (!it.isSuccessful) return null
				val bodyString = it.body.string()
				parseVolumeStatus(bodyString)
			}
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to fetch volume status from $url: ${e.message}", e)
			null
		}
	}

	/**
	 * Sets the EyeAIVision volume percentage via HTTP API (POST /api/volume).
	 */
	fun setVolume(volume: Int): Boolean {
		ensureWiFiNetworkBound()
		val clampedVolume = volume.coerceIn(0, 100)
		val url = "http://$ip:8080/api/volume"
		val jsonBody = JSONObject().put("volume", clampedVolume).toString()
		val requestBody = jsonBody.toRequestBody("application/json".toMediaType())
		val request = Request.Builder().url(url).post(requestBody).build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use { it.isSuccessful }
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to set volume to $clampedVolume at $url: ${e.message}", e)
			false
		}
	}

	/**
	 * Increases the volume on EyeAIVision (POST /api/volume/up).
	 */
	fun volumeUp(step: Int? = null): Boolean {
		ensureWiFiNetworkBound()
		val urlBuilder = HttpUrl.Builder()
			.scheme("http")
			.host(ip)
			.port(8080)
			.addPathSegment("api")
			.addPathSegment("volume")
			.addPathSegment("up")
		if (step != null) {
			urlBuilder.addQueryParameter("step", step.toString())
		}
		val emptyBody = "".toRequestBody("application/json".toMediaType())
		val request = Request.Builder().url(urlBuilder.build()).post(emptyBody).build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use { it.isSuccessful }
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to increase volume: ${e.message}", e)
			false
		}
	}

	/**
	 * Decreases the volume on EyeAIVision (POST /api/volume/down).
	 */
	fun volumeDown(step: Int? = null): Boolean {
		ensureWiFiNetworkBound()
		val urlBuilder = HttpUrl.Builder()
			.scheme("http")
			.host(ip)
			.port(8080)
			.addPathSegment("api")
			.addPathSegment("volume")
			.addPathSegment("down")
		if (step != null) {
			urlBuilder.addQueryParameter("step", step.toString())
		}
		val emptyBody = "".toRequestBody("application/json".toMediaType())
		val request = Request.Builder().url(urlBuilder.build()).post(emptyBody).build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use { it.isSuccessful }
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to decrease volume: ${e.message}", e)
			false
		}
	}

	/**
	 * Mutes or unmutes the audio output on EyeAIVision (POST /api/volume/mute or /api/volume/unmute).
	 */
	fun setMute(mute: Boolean): Boolean {
		ensureWiFiNetworkBound()
		val path = if (mute) "mute" else "unmute"
		val url = "http://$ip:8080/api/volume/$path"
		val emptyBody = "".toRequestBody("application/json".toMediaType())
		val request = Request.Builder().url(url).post(emptyBody).build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use { it.isSuccessful }
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to set mute=$mute: ${e.message}", e)
			false
		}
	}

	/**
	 * Calculates current Android STREAM_MUSIC volume percentage and syncs it to EyeAIVision.
	 */
	fun syncVolumeWithAndroid() {
		val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return
		val maxVol = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
		val currentVol = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
		val percent = if (maxVol > 0) ((currentVol.toFloat() / maxVol.toFloat()) * 100).roundToInt() else 0
		if (percent != lastSyncedVolume) {
			lastSyncedVolume = percent
			socketThread.launch(Dispatchers.IO) {
				setVolume(percent)
			}
		}
	}

	private fun startVolumeObserver() {
		try {
			val handler = Handler(Looper.getMainLooper())
			val observer = VolumeObserver(handler)
			context.contentResolver.registerContentObserver(
				Settings.System.CONTENT_URI,
				true,
				observer
			)
			volumeObserver = observer
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to register volume ContentObserver: ${e.message}")
		}
	}

	private fun stopVolumeObserver() {
		try {
			volumeObserver?.let {
				context.contentResolver.unregisterContentObserver(it)
			}
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to unregister volume ContentObserver: ${e.message}")
		} finally {
			volumeObserver = null
		}
	}

	private fun parseVolumeStatus(jsonString: String): EyeAIVisionVolumeStatus {
		return try {
			val json = JSONObject(jsonString)
			EyeAIVisionVolumeStatus(
				volume = json.optInt("volume", 0),
				muted = json.optBoolean("muted", false),
				card = json.optString("card", ""),
				control = json.optString("control", ""),
				success = json.optBoolean("success", false)
			)
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Error parsing status JSON: ${e.message}")
			EyeAIVisionVolumeStatus()
		}
	}

	private inner class VolumeObserver(handler: Handler) : ContentObserver(handler) {
		override fun onChange(selfChange: Boolean) {
			super.onChange(selfChange)
			syncVolumeWithAndroid()
		}
	}

	/** Stops the existing external source without touching the common analyzer/models. */
	fun close() {
		stopAudioPlaybackCapture()
		stopVolumeObserver()
		stopAudioStream()
		stopAudioOutputStream()
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

private class LoggingAudioOutputStream(private val delegate: OutputStream) : OutputStream() {
	private var totalBytes: Long = 0
	private var lastLogTime: Long = System.currentTimeMillis()
	private var bytesSinceLastLog: Long = 0

	override fun write(b: Int) {
		delegate.write(b)
		totalBytes++
		bytesSinceLastLog++
		checkLog()
	}

	override fun write(b: ByteArray, off: Int, len: Int) {
		delegate.write(b, off, len)
		if (len > 0) {
			totalBytes += len
			bytesSinceLastLog += len
			val now = System.currentTimeMillis()
			if (now - lastLogTime >= 1000L) {
				Log.d(
					EyeAIApp.APP_LOG_TAG,
					"[EyeAIVisionAudioOutput] Sent $len bytes PCM to port 4041 (rate: ${bytesSinceLastLog} bytes/sec, total: $totalBytes bytes)"
				)
				lastLogTime = now
				bytesSinceLastLog = 0
			}
		}
	}

	override fun close() {
		Log.i(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionAudioOutput] Audio output stream closed. Total sent: $totalBytes bytes")
		delegate.close()
	}

	private fun checkLog() {
		val now = System.currentTimeMillis()
		if (now - lastLogTime >= 1000L) {
			Log.d(
				EyeAIApp.APP_LOG_TAG,
				"[EyeAIVisionAudioOutput] Sent byte stream to port 4041 (rate: ${bytesSinceLastLog} bytes/sec, total: $totalBytes bytes)"
			)
			lastLogTime = now
			bytesSinceLastLog = 0
		}
	}
}
