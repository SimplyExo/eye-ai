package com.algorithmic_alliance.eyeaiapp.media

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.util.Log
import android.widget.ImageView
import androidx.annotation.RequiresApi
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

@RequiresApi(Build.VERSION_CODES.P)
class MediaPlayer(
	private val context: Context,
	private val uri: Uri? = null,
	private val updateTargetImageView: (Bitmap) -> Unit,
	private val bitmapFlow: Flow<Bitmap>? = null,
	/** Optional source adapter callback into the common EyeAI FrameAnalyzer. */
	private val onFrame: ((Bitmap) -> Unit)? = null,
) {

	private var retriever: MediaMetadataRetriever? = null


	private val executor = Executors.newSingleThreadExecutor()
	private val scope: CoroutineScope = CoroutineScope(executor.asCoroutineDispatcher())

	init {
		// uri or bitmap via MJPEGStream ist needed
		if (uri == null && bitmapFlow == null) {
			throw IllegalArgumentException("Either uri or bitmapFlow must be provided")
		}


		scope.launch {
			try {
				// if bitmap is available, use it, else use uri
				bitmapFlow?.let { collectBitmapFlow(it) } ?: handleUriSource()
			} catch (t: Throwable) {
				Log.e("MediaPlayer", "Error in MediaPlayer main loop", t)
			}
		}
	}

	private suspend fun collectBitmapFlow(flow: Flow<Bitmap>) {

		flow.collect { bitmap ->
			onFrame?.invoke(bitmap)

			withContext(Dispatchers.Main) {
				try {
					updateTargetImageView(bitmap)
				} catch (t: Throwable) {
					Log.e("MediaPlayer", "Error setting bitmap to ImageView", t)
				}
			}
		}
	}

	private suspend fun handleUriSource() {
		uri?.let { u ->
			val type = context.contentResolver.getType(u)
			if (type == null) {
				Log.w("MediaPlayer", "Unknown content type for URI: $u")
				return
			}

			if (type.startsWith("image/")) {
				context.contentResolver.openInputStream(u)?.use { input ->
					val options = BitmapFactory.Options().apply {
						inPreferredConfig = Bitmap.Config.ARGB_8888
					}

					val bmp = BitmapFactory.decodeStream(input, null, options)
					bmp?.let { onFrame?.invoke(it) }
					withContext(Dispatchers.Main) {
						bmp?.let{
							updateTargetImageView(bmp)
						}
					}
				}
			} else if (type.startsWith("video/")) {
				retriever = MediaMetadataRetriever()
				retriever!!.setDataSource(context, u)

				var index = 0
				while (true) {

					try {
						val frame = retriever!!.getFrameAtIndex(index)?.toARGB8888()
						frame?.let { onFrame?.invoke(it) }
						withContext(Dispatchers.Main) {
							frame?.let{
								updateTargetImageView(frame)
							}
						}
						index++

						delay(1000L / 30L)
					} catch (_: IllegalArgumentException) {

						index = 0
					}
				}
			} else {
				Log.w("MediaPlayer", "content type not supported: $type")
			}
		}
	}

	fun shutdown() {
		try {
			scope.cancel()
		} finally {
			executor.shutdownNow()
			retriever?.release()
			retriever = null
		}
	}

	private fun Bitmap?.toARGB8888(): Bitmap? {
		return this?.let { bitmap ->
			if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
			else bitmap.copy(Bitmap.Config.ARGB_8888, true)
		}
	}
}
