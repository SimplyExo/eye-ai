package com.algorithmic_alliance.eyeaiapp.connectivity

import android.content.Context
import android.database.ContentObserver
import android.graphics.Bitmap
import android.media.AudioManager
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
	private val onWebrtcFrame: (Bitmap) -> Unit
) {
	private val context: Context = app.applicationContext
	private lateinit var touchSocket: Socket
	private var audioSocket: Socket? = null
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

	val whepUrl = "https://$ip:8889/cam/whep"

	init {
		// Touch Button Client starten
		socketThread.launch {
			try {
				onConnectingSocket()
				touchSocket = Socket(ip, 3333)
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
		webRtcClientValue = WebRtcClient(app) { bitmap ->
			Log.v(EyeAIApp.APP_LOG_TAG, "EyeAIVision received WebRTC frame, invoking callback")
			onWebrtcFrame(bitmap)
		}.also {
			Log.e(EyeAIApp.APP_LOG_TAG, "!!! Starting WebRtcClient !!!")
			it.start(whepUrl)
		}

		startVolumeObserver()
		syncVolumeWithAndroid()
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

	/**
	 * Fetches the volume status from EyeAIVision HTTP API (GET /api/volume).
	 */
	fun fetchVolumeStatus(): EyeAIVisionVolumeStatus? {
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
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to fetch volume status: ${e.message}")
			null
		}
	}

	/**
	 * Sets the EyeAIVision volume percentage via HTTP API (POST /api/volume).
	 */
	fun setVolume(volume: Int): Boolean {
		val clampedVolume = volume.coerceIn(0, 100)
		val url = "http://$ip:8080/api/volume"
		val jsonBody = JSONObject().put("volume", clampedVolume).toString()
		val requestBody = jsonBody.toRequestBody("application/json".toMediaType())
		val request = Request.Builder().url(url).post(requestBody).build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use { it.isSuccessful }
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to set volume to $clampedVolume: ${e.message}")
			false
		}
	}

	/**
	 * Increases the volume on EyeAIVision (POST /api/volume/up).
	 */
	fun volumeUp(step: Int? = null): Boolean {
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
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to increase volume: ${e.message}")
			false
		}
	}

	/**
	 * Decreases the volume on EyeAIVision (POST /api/volume/down).
	 */
	fun volumeDown(step: Int? = null): Boolean {
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
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to decrease volume: ${e.message}")
			false
		}
	}

	/**
	 * Mutes or unmutes the audio output on EyeAIVision (POST /api/volume/mute or /api/volume/unmute).
	 */
	fun setMute(mute: Boolean): Boolean {
		val path = if (mute) "mute" else "unmute"
		val url = "http://$ip:8080/api/volume/$path"
		val emptyBody = "".toRequestBody("application/json".toMediaType())
		val request = Request.Builder().url(url).post(emptyBody).build()
		return try {
			val response = httpClient.newCall(request).execute()
			response.use { it.isSuccessful }
		} catch (e: Exception) {
			Log.e(EyeAIApp.APP_LOG_TAG, "[EyeAIVisionVolume] Failed to set mute=$mute: ${e.message}")
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
		stopVolumeObserver()
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
