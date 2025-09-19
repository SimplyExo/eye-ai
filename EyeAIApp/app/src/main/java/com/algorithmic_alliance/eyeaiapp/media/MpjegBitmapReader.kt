package com.algorithmic_alliance.eyeaiapp.media

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import kotlinx.coroutines.*
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLSocketFactory
import javax.net.ssl.TrustManager
import javax.net.ssl.X509TrustManager

/**
 * MJPEG Reader
 *
 * @param ip to stream the MJPEG input-source from
 * @param onFrame callback for every bitmap
 * @param deliverOnMainThread if true, on Frame is running on the Main-Thread
 * @param parentScope determines whether a parentScope is used or not, here it is useful as we want the garbage-collector to clean up all unused bitmaps, hence we use a lifecycleScope
 */
class MjpegBitmapReader(
	private val ip: String,
	private val onFrame: (Bitmap) -> Unit,
	private val deliverOnMainThread: Boolean = false,
	parentScope: CoroutineScope? = null,
	private val onMjpegError: (Exception) -> Unit
) {

	private val trustAllCertificates: Boolean = true

	private val scope: CoroutineScope = parentScope ?: CoroutineScope(Dispatchers.IO + SupervisorJob())
	private var job: Job? = null

	private val urlObj = URL("http://$ip/cam0")
	private val connRaw = urlObj.openConnection() ?: throw IllegalStateException("Cannot open connection")

	@Volatile
	private var connection = (connRaw as? HttpURLConnection) ?: throw IllegalStateException("Not an HTTP connection")

	fun start() {
		if (job?.isActive == true) return
		job = scope.launch {
			while (isActive) {
				try {
					fetchLoop()
				} catch (ce: CancellationException) {
					onMjpegError(ce)
					throw ce
				} catch (e: Exception) {
					onMjpegError(e)
					e.printStackTrace()
					delay(1000)
				}
			}
		}
	}

	fun stop() {
		job?.cancel()
		disconnect()
	}

	private fun disconnect() {
		try {
			connection.disconnect()
		} catch (_: Throwable) { }
		//connection = null
	}

	private suspend fun fetchLoop() = withContext(Dispatchers.IO) {
		if (trustAllCertificates && connection is HttpsURLConnection) {
			val (sf, hv) = createInsecureSsl()
			(connection as HttpsURLConnection).sslSocketFactory = sf
			(connection as HttpsURLConnection).hostnameVerifier = hv
		}

		connection.requestMethod = "GET"
		connection.connectTimeout = 5000
		connection.readTimeout = 0
		connection.doInput = true
		connection.useCaches = false

		try {
			connection.connect()
			connection.inputStream.use { input ->
				parseStream(input)
			}
		} finally {
			disconnect()
		}
	}

	private suspend fun parseStream(input: InputStream) = withContext(Dispatchers.IO) {
		val soi = byteArrayOf(0xFF.toByte(), 0xD8.toByte())
		val eoi = byteArrayOf(0xFF.toByte(), 0xD9.toByte())
		val tmp = ByteArray(8 * 1024)
		val buffer = ByteArrayOutputStream()

		while (isActive) {
			ensureActive()

			val read = try {
				input.read(tmp)
			} catch (_: Throwable) {

				break
			}
			if (read <= 0) break
			buffer.write(tmp, 0, read)
			val data = buffer.toByteArray()

			val start = indexOf(data, soi, 0)
			val end = if (start >= 0) indexOf(data, eoi, start + 2) else -1

			if (start >= 0 && end >= 0) {
				val frameBytes = data.copyOfRange(start, end + 2)

				val bmp = try {
					BitmapFactory.decodeByteArray(frameBytes, 0, frameBytes.size)
				} catch (t: Throwable) {
					t.printStackTrace()
					null
				}

				bmp?.let { bitmap ->
					// turn bitmap 180°
					val matrix = Matrix()
					matrix.postRotate(180f)
					val rotatedBitmap = try {
						Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
					} catch (t: Throwable) {
						t.printStackTrace()
						bitmap // fallback if roation fails
					}

					if (deliverOnMainThread) {
						withContext(Dispatchers.Main) { onFrame(rotatedBitmap) }
					} else {
						onFrame(rotatedBitmap)
					}
				}


				val remaining = if (end + 2 < data.size) data.copyOfRange(end + 2, data.size) else ByteArray(0)
				buffer.reset()
				if (remaining.isNotEmpty()) buffer.write(remaining)
			} else {

				if (data.size > 2_000_000) {
					val keep = data.copyOfRange(data.size - 150_000, data.size)
					buffer.reset()
					buffer.write(keep)
				}
			}
		}
	}


	private fun indexOf(data: ByteArray, pattern: ByteArray, from: Int): Int {
		if (pattern.isEmpty()) return -1
		outer@ for (i in from..data.size - pattern.size) {
			for (j in pattern.indices) {
				if (data[i + j] != pattern[j]) continue@outer
			}
			return i
		}
		return -1
	}

	@SuppressLint("TrustAllX509TrustManager")
	private fun createInsecureSsl(): Pair<SSLSocketFactory, HostnameVerifier> {
		val trustAll = arrayOf<TrustManager>(@SuppressLint("CustomX509TrustManager")
		object : X509TrustManager {
			override fun checkClientTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}

			override fun checkServerTrusted(chain: Array<java.security.cert.X509Certificate>, authType: String) {}
			override fun getAcceptedIssuers(): Array<java.security.cert.X509Certificate> = arrayOf()
		})

		val sslContext = SSLContext.getInstance("TLS")
		sslContext.init(null, trustAll, java.security.SecureRandom())
		val sf = sslContext.socketFactory

		val hv = HostnameVerifier { _, _ -> true }

		return Pair(sf, hv)
	}
}
