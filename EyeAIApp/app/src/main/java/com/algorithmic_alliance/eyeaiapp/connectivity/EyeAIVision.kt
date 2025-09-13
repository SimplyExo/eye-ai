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
import java.util.concurrent.Executors


open class EyeAIVision(
	private val ip: String,
	private val onSingleClick: () -> Unit,
	private val onDoubleClick: () -> Unit,
	private val lifecycleScope: LifecycleCoroutineScope,
	private val bitmapFlow: MutableSharedFlow<Bitmap>?
) {
	private lateinit var touchSocket: Socket
	private var mjpegBitmapReader: MjpegBitmapReader? = null

	private val socketThread: CoroutineScope =
		CoroutineScope(Executors.newSingleThreadExecutor().asCoroutineDispatcher())

	init {
		// Touch Button
		socketThread.launch {
			touchSocket = Socket(ip, 3333)
			val reader = BufferedReader(InputStreamReader(touchSocket.inputStream))

			while (true) {
				val char = reader.read().toChar()

				if (char == '1') {
					onSingleClick()
				} else if (char == '2') {
					onDoubleClick()
				}
			}
		}

		mjpegBitmapReader = MjpegBitmapReader(
			ip = ip,
			onFrame = { bitmap ->
				bitmapFlow?.tryEmit(bitmap)
			},
			deliverOnMainThread = false,
			parentScope = lifecycleScope
		)

		mjpegBitmapReader?.start()
	}
}