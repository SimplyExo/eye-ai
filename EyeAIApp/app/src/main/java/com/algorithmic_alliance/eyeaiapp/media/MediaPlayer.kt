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
import androidx.lifecycle.compose.dropUnlessResumed
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

@RequiresApi(Build.VERSION_CODES.P)
class MediaPlayer(
	private var context: Context,
	private var uri: Uri,
	private var targetImageView: ImageView
	) {

	private var retriever: MediaMetadataRetriever? = null

	private var executor = Executors.newSingleThreadExecutor()

	private val scope: CoroutineScope =
		CoroutineScope(executor.asCoroutineDispatcher())

	init {
		scope.launch {
			val type = context.contentResolver.getType(uri)

			if (type!!.startsWith("image/")) {
				context.contentResolver.openInputStream(uri)?.use { input ->
					val options = BitmapFactory.Options().apply {
						inPreferredConfig = Bitmap.Config.ARGB_8888
					}

					withContext(Dispatchers.Main) {
						targetImageView.setImageBitmap(
							BitmapFactory.decodeStream(
								input,
								null,
								options
							)
						)
					}
				}
			} else if (type.startsWith("video/")) {
				retriever = MediaMetadataRetriever()
				retriever!!.setDataSource(context, uri)

				var index = 0
				while (true) {
					try {
						val frame = retriever!!.getFrameAtIndex(index)?.toARGB8888()

						withContext(Dispatchers.Main) {
							targetImageView.setImageBitmap(frame)
						}
						index++

						delay(1000/30)
					} catch (e: IllegalArgumentException) {
						index = 0
					}
				}
			}
		}
	}

	fun shutdown() {
		scope.cancel()
		executor.shutdownNow()
		retriever?.release()
	}

		fun Bitmap?.toARGB8888(): Bitmap? {
		return this?.let { bitmap ->
			if (bitmap.config == Bitmap.Config.ARGB_8888) bitmap
			else bitmap.copy(Bitmap.Config.ARGB_8888, true)
		}
	}
}