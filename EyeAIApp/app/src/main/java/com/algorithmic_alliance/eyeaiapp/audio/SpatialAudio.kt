package com.algorithmic_alliance.eyeaiapp.audio

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import uniffi.NativeLib.UniffiDetectedObject
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.time.Duration.Companion.milliseconds


object SpatialAudio {
	private lateinit var executor: ExecutorService
	private lateinit var scope: CoroutineScope
	private lateinit var eyeAIApp: EyeAIApp
	private var spatialAudioJob: Job? = null
	private var configuredLanguage: String? = null
	private val lock = Any()

	private const val LOST_AFTER_MISSED_FRAMES = 5

	// Zuletzt gesehene Version jedes getrackten Objekts
	private val objectsInSight =
		mutableMapOf<Int, UniffiDetectedObject>()

	// Anzahl aufeinanderfolgender Frames, in denen das Objekt fehlte
	private val missedFrames =
		mutableMapOf<Int, Int>()

	fun start() {
		synchronized(lock) {
			if (!::scope.isInitialized || spatialAudioJob?.isActive == true) return

			spatialAudioJob = scope.launch {
				uniffi.NativeLib.createSpatialAudio()

				while (isActive) {
					val depthData = eyeAIApp.aiData.depthEstimationData.get()
					val objectData = if(eyeAIApp.settings.objectAudioPlaybackMode == eyeAIApp.getString(R.string.object_playback_is_object_watch)){
						getObjectsObjectWatchMode(
							eyeAIApp.aiData.detectedObjects.get() ?: emptyArray()
						)
					} else {
						eyeAIApp.aiData.detectedObjects.get() ?: emptyArray()
					}
					val segmentationData = eyeAIApp.aiData.segmentationOutput.get()
					val segmentationClassImportances =
						eyeAIApp.aiData.segmentationClassImportances.get()
					if (depthData != null) {
						uniffi.NativeLib.sendAiDataForSpatialAudio(
							depthData.asUniffiWrapper(),
							objectData.toList(),
							segmentationData?.asUniffiWrapper(),
							segmentationClassImportances ?: emptyList()
						)
					}
					delay(50.milliseconds)
				}
			}
		}
	}




	fun setup(context: Context) {
		val app = context.applicationContext as EyeAIApp
		val settings = app.settings
		synchronized(lock) {
			eyeAIApp = app
			if (configuredLanguage == null || configuredLanguage != settings.objectAudioPlaybackLanguage) {
				loadAudioDataFiles(app, settings.objectAudioPlaybackLanguage, settings.objectAudioPlaybackMode)
				configuredLanguage = settings.objectAudioPlaybackLanguage
			}
		}

		uniffi.NativeLib.setDepthAudioPaused(!settings.depthAudioPlayback)
		uniffi.NativeLib.setObjectAudioPaused(!settings.objectAudioPlayback)
		uniffi.NativeLib.setAudioSettings(
			settings.depthAudioFrequency.toFloat(),
			settings.depthAudioClickIncidence,
		)

		synchronized(lock) {
			if (!::executor.isInitialized || executor.isShutdown) {
				executor = Executors.newSingleThreadExecutor()
				scope = CoroutineScope(executor.asCoroutineDispatcher())
			}
		}
	}

	fun stop() {
		configuredLanguage = null
		synchronized(lock) {
			spatialAudioJob?.cancel()
			spatialAudioJob = null
			if (::scope.isInitialized) scope.cancel()
			if (::executor.isInitialized) executor.shutdown()
		}
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

	fun loadAudioDataFiles(context: Context, language: String?, playbackMode: String?) {
		Log.d("Spatial Audio", "[LoadAudioDataFiles] Loading files...")
		var fileNameWav: String
		var fileNameJson: String
		when (language) {
			eyeAIApp.getString(R.string.language_is_english) -> {
				Log.d("SpatialAudio", "[SpatialAudio] Selected english language")
				fileNameWav = "coco_labels_english.wav"
				fileNameJson = "coco_labels_data_english.json"
			}

			eyeAIApp.getString(R.string.language_is_german) -> {
				Log.d("SpatialAudio", "[SpatialAudio] Selected german language")
				if(playbackMode == eyeAIApp.getString(R.string.object_playback_is_object_watch)){
					fileNameWav = "coco_labels_german_object_watch.wav"
					fileNameJson = "coco_labels_data_german_object_watch.json"
				} else{
					fileNameWav = "coco_labels_german.wav"
					fileNameJson = "coco_labels_data_german.json"
				}

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

	fun getObjectsObjectWatchMode(
		detectedObjects: Array<UniffiDetectedObject>
	): Array<UniffiDetectedObject> {
		synchronized(lock) {
			val currentObjects = detectedObjects.associateBy {
				it.trackingId
			}

			val newIds = currentObjects.keys - objectsInSight.keys
			val missingIds = objectsInSight.keys - currentObjects.keys

			val changedObjects = mutableListOf<UniffiDetectedObject>()

			// Neue Objekte melden.
			for (trackingId in newIds) {
				Log.d(
					"SpatialAudio",
					"New Object in sight, tracking id: $trackingId"
				)
				currentObjects[trackingId]?.let {
					changedObjects += it.copy(clsName = "new ${it.clsName}")
				}
			}

			// Aktuell sichtbare Objekte: Zähler zurücksetzen, letzten Stand merken.
			for ((trackingId, obj) in currentObjects) {
				missedFrames.remove(trackingId)
				objectsInSight[trackingId] = obj
			}

			// Fehlende Objekte: erst nach N aufeinanderfolgenden Frames als verloren melden.
			for (trackingId in missingIds) {
				val missed = (missedFrames[trackingId] ?: 0) + 1

				if (missed >= LOST_AFTER_MISSED_FRAMES) {
					Log.d(
						"SpatialAudio",
						"Lost sight of object, tracking id: $trackingId"
					)
					objectsInSight.remove(trackingId)?.let {
						changedObjects += it.copy(clsName = "lost ${it.clsName}")
					}
					missedFrames.remove(trackingId)
				} else {
					missedFrames[trackingId] = missed
				}
			}

			return changedObjects.toTypedArray()
		}
	}
}
