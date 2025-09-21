package com.algorithmic_alliance.eyeaiapp.connectivity

import android.graphics.Bitmap
import androidx.lifecycle.LifecycleCoroutineScope
import androidx.lifecycle.lifecycleScope
import com.algorithmic_alliance.eyeaiapp.media.MjpegBitmapReader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.launch
import java.io.*
import java.net.Socket
import java.net.SocketException
import java.net.UnknownHostException
import java.util.concurrent.Executors


open class EyeAIVision(
	private val ip: String,
	private val compression: Int,
	private val lifecycleScope: LifecycleCoroutineScope,
	private val bitmapFlow: MutableSharedFlow<Bitmap>?,
	private val onSingleClick: () -> Unit,
	private val onDoubleClick: () -> Unit,
	private val onConnectingSocket: () -> Unit,
	private val onSocketConnectionEstablished: () -> Unit,
	private val onSocketFailed: (Exception) -> Unit,
	private val onMjpegError: (Exception) -> Unit,
	private val onConnectingHTTP: () -> Unit,
	private val onHTTPConnectionEstablished: () -> Unit
) {
	private lateinit var touchSocket: Socket
	private var mjpegBitmapReader: MjpegBitmapReader? = null

	private val socketThread: CoroutineScope =
		CoroutineScope(Executors.newSingleThreadExecutor().asCoroutineDispatcher())

	init {
		setCompression(compression)

		// Touch Button Client starten
		socketThread.launch {
			try {
				onConnectingSocket()
				touchSocket = Socket(ip, 3333)
				val reader = BufferedReader(InputStreamReader(touchSocket.inputStream))
				onSocketConnectionEstablished()

				while (true) {
					val char = reader.read().toChar()

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

		// Video Steam starten
		mjpegBitmapReader = MjpegBitmapReader(
			ip = ip,
			onFrame = { bitmap ->
				bitmapFlow?.tryEmit(bitmap)
			},
			deliverOnMainThread = false,
			parentScope = lifecycleScope,
			onMjpegError = { e ->
				onMjpegError(e)
			},

		)

		mjpegBitmapReader?.start()
	}

	public fun setCompression(value: Int) {

	}
}