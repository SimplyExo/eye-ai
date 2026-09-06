package com.algorithmic_alliance.eyeaiapp.audio

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.inference.AnalysisClock
import com.algorithmic_alliance.eyeaiapp.camera.AnalysisGeneration
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.io.InputStream
import java.io.ByteArrayOutputStream


object SpatialAudio {
	private lateinit var eyeAIApp: EyeAIApp
	private var configuredLanguage: String? = null
	private val lock = Any()
	private val contentLock = Any()
	private var depthPaused = true
	private var objectPaused = true
	private var frequency = 500f
	private var incidence = 4
	private val lifecycle = SpatialAudioLifecycle(
		backend = object : SpatialAudioSessionBackend {
			override fun begin(): ULong {
				// start holds lock; configure before any worker can create an engine.
				val id = uniffi.NativeLib.beginSpatialAudioSession()
				try {
					uniffi.NativeLib.setDepthAudioPaused(id, depthPaused)
					uniffi.NativeLib.setObjectAudioPaused(id, objectPaused)
					uniffi.NativeLib.setAudioSettings(id, frequency, incidence)
				} catch (error: Throwable) {
					// No engine exists yet, so this cleanup cannot wait on device work.
					uniffi.NativeLib.destroySpatialAudio(id)
					throw error
				}
				return id
			}
			override fun invalidate(session: ULong) = uniffi.NativeLib.invalidateSpatialAudioSession(session)
			override fun create(session: ULong) = uniffi.NativeLib.createSpatialAudio(session)
			override fun destroy(session: ULong) = uniffi.NativeLib.destroySpatialAudio(session)
		},
		updates = { session -> updateLoop(session) },
		onError = { error -> Log.e(EyeAIApp.APP_LOG_TAG, "Spatial audio session failed", error) },
	)

	fun start() {
		synchronized(lock) {
			if (!::eyeAIApp.isInitialized) return
			lifecycle.start()
		}
	}

	private suspend fun CoroutineScope.updateLoop(session: ULong) {
		// The native API requires a 256x256 map even to clear object positions.
		// This all-far sentinel is never published as a measured depth result.
		val noDepth = NativeLib.NativeFloatBuffer(256 * 256).also {
			for (index in 0 until 256 * 256) it.floatBuffer.put(index, 1_000f)
		}
		var previousGeneration: AnalysisGeneration? = null

		while (isActive) {
			val results = eyeAIApp.aiData.analysisResults.get()
			val now = AnalysisClock.nowNanos()
			val depth = results.freshDepth(now)?.takeIf {
				it.width == 256 && it.height == 256 && it.prediction.floatBuffer.capacity() == 256 * 256
			}
			val objects = if (depth != null) results.alignedObjects(now) else emptyList()
			if (previousGeneration != null && previousGeneration != results.generation) {
				// A stream/content boundary must stop audio from the old generation.
				// Ordinary empty snapshots only clear pending announcements in native code.
				uniffi.NativeLib.invalidateObjectAudioPlayback(session)
			}
			previousGeneration = results.generation
			uniffi.NativeLib.sendAiDataForSpatialAudio(
				session, (depth?.prediction ?: noDepth).asUniffiWrapper(), objects,
			)
			delay(50)
		}
	}

	fun setup(context: Context) {
		val app = context.applicationContext as EyeAIApp
		val settings = app.settings
		val session = synchronized(lock) {
			eyeAIApp = app
			depthPaused = !settings.depthAudioPlayback
			objectPaused = !settings.objectAudioPlayback
			frequency = settings.depthAudioFrequency.toFloat()
			incidence = settings.depthAudioClickIncidence
			currentSessionId()
		}
		synchronized(contentLock) {
			if (configuredLanguage != settings.objectAudioPlaybackLanguage) {
				loadAudioDataFiles(app, settings.objectAudioPlaybackLanguage)
				configuredLanguage = settings.objectAudioPlaybackLanguage
			}
		}

		// A setup delayed by asset I/O keeps the session it captured before I/O.
		if (session != null) {
			uniffi.NativeLib.setDepthAudioPaused(session, !settings.depthAudioPlayback)
			uniffi.NativeLib.setObjectAudioPaused(session, !settings.objectAudioPlayback)
			uniffi.NativeLib.setAudioSettings(session, settings.depthAudioFrequency.toFloat(), settings.depthAudioClickIncidence)
		}
	}

	fun stop() {
		lifecycle.stop()
	}

	fun currentSessionId(): ULong? = lifecycle.currentSessionId()

	fun setDepthAudioPaused(paused: Boolean) {
		val session = synchronized(lock) { depthPaused = paused; currentSessionId() }
		session?.let { uniffi.NativeLib.setDepthAudioPaused(it, paused) }
	}

	fun setObjectAudioPaused(paused: Boolean) {
		val session = synchronized(lock) { objectPaused = paused; currentSessionId() }
		session?.let { uniffi.NativeLib.setObjectAudioPaused(it, paused) }
	}

	/** Delayed TTS/resume callbacks carry the session captured when scheduled. */
	fun restore(session: ULong?, objectPaused: Boolean, depthPaused: Boolean) {
		if (session == null) return
		synchronized(lock) {
			if (currentSessionId() != session) return
			this.objectPaused = objectPaused
			this.depthPaused = depthPaused
		}
		uniffi.NativeLib.setObjectAudioPaused(session, objectPaused)
		uniffi.NativeLib.setDepthAudioPaused(session, depthPaused)
	}

	fun setAudioSettings(newFrequency: Float, newIncidence: Int) {
		val session = synchronized(lock) {
			frequency = newFrequency
			incidence = newIncidence
			currentSessionId()
		}
		session?.let { uniffi.NativeLib.setAudioSettings(it, newFrequency, newIncidence) }
	}

	fun loadFileFromAssets(context: Context, fileName: String): ByteArray? {
		try {
			val assetManager = context.assets

			val inputStream: InputStream = assetManager.open(fileName)

			val byteArrayOutputStream = ByteArrayOutputStream()

			val buffer = ByteArray(1024)
			var bytesRead: Int

			while (inputStream.read(buffer).also { bytesRead = it } != -1) {
				byteArrayOutputStream.write(buffer, 0, bytesRead)
			}

			inputStream.close()
			return byteArrayOutputStream.toByteArray()

		} catch (e: Exception) {
			e.printStackTrace()
			return null
		}
	}

	fun loadTextFileFromAssets(context: Context, fileName: String): String? {
		try {
			val assetManager = context.assets

			val inputStream: InputStream = assetManager.open(fileName)

			val byteArrayOutputStream = ByteArrayOutputStream()

			val buffer = ByteArray(1024)
			var bytesRead: Int

			while (inputStream.read(buffer).also { bytesRead = it } != -1) {
				byteArrayOutputStream.write(buffer, 0, bytesRead)
			}

			inputStream.close()
			return byteArrayOutputStream.toString()

		} catch (e: Exception) {
			e.printStackTrace()
			return null
		}
	}

	fun loadAudioDataFiles(context: Context, language: String?) {
		Log.d("Spatial Audio", "[LoadAudioDataFiles] Loading files...")
		var fileNameWav: String
		var fileNameJson: String
		when (language) {
			"english" -> {
				Log.d("SpatialAudio", "[SpatialAudio] Selected english language")
				fileNameWav = "coco_labels_english.wav"
				fileNameJson = "coco_labels_data_english.json"
			}

			"german" -> {
				Log.d("SpatialAudio", "[SpatialAudio] Selected german language")
				fileNameWav = "coco_labels_german.wav"
				fileNameJson = "coco_labels_data_german.json"
			}

			else -> {
				Log.d("SpatialAudio", "[SpatialAudio] Selected no language, loading default")
				fileNameWav = "coco_labels_english.wav"
				fileNameJson = "coco_labels_data_english.json"
			}
		}

		val cocoLabelsAudio = loadFileFromAssets(context, fileNameWav)
		val cocoLabelsData = loadTextFileFromAssets(context, fileNameJson)



		if (cocoLabelsData != null && cocoLabelsAudio != null) {
			uniffi.NativeLib.setupAudioContent(cocoLabelsAudio, cocoLabelsData)
			Log.d(
				"SpatialAudio",
				"[SpatialAudio] Loaded coco data from $fileNameWav and $fileNameJson"
			)
		} else {
			Log.e("SpatialAudio", "[SpatialAudio] Could not load coco data")
		}
	}
}
