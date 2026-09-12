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
import java.io.InputStreamReader
import java.net.Socket
import java.net.UnknownHostException
import java.util.concurrent.Executors


open class EyeAIVision(
	private val app: Context,
	private val ip: String,
	private val onSingleClick: () -> Unit,
	private val onDoubleClick: () -> Unit,
	private val onConnectingSocket: () -> Unit,
	private val onSocketConnectionEstablished: () -> Unit,
	private val onSocketFailed: (Exception) -> Unit,
	private val onWebrtcFrame: (Bitmap) -> Unit
) {
	private lateinit var touchSocket: Socket
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

	/** Stops the existing external source without touching the common analyzer/models. */
	fun close() {
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
