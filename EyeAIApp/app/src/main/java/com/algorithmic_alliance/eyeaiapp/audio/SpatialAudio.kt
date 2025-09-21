package com.algorithmic_alliance.eyeaiapp.audio

import android.content.Context
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import com.algorithmic_alliance.eyeaiapp.Settings
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import kotlinx.coroutines.*
import java.io.InputStream
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService


object SpatialAudio {
	private lateinit var  executor: ExecutorService
	private lateinit var scope: CoroutineScope
	private lateinit var eyeAIApp: EyeAIApp;

	fun start() {
		if (!::scope.isInitialized) return
		scope.launch {
			while (isActive) {
				if(NativeLib.getProcessingStatus()){
					val data = eyeAIApp.aiData.depthEstimationData.get()
					if(data != null){
						NativeLib.sendAIData(data)
					}
					delay(50)
				}
			}
		}
	}

	fun setup(context: Context){
		eyeAIApp = context.applicationContext as EyeAIApp
		val settings = Settings.load(context)
		loadAudioDataFiles(context, settings.objectAudioPlaybackLanguage)

		if (!::executor.isInitialized || executor.isShutdown) {
			executor = Executors.newSingleThreadExecutor()
			scope = CoroutineScope(executor.asCoroutineDispatcher())
		}
	}

	fun destroy() {
		if (::scope.isInitialized) scope.cancel()
		if (::executor.isInitialized) executor.shutdown()
		NativeLib.destroySpatialAudio()
		NativeLib.destroySpatialAudio()
	}

	fun loadFileFromAssets(context: Context, fileName: String): ByteArray?{
		try {
			val assetManager = context.assets

			val inputStream: InputStream = assetManager.open(fileName)

			val byteArrayOutputStream = ByteArrayOutputStream()

			val buffer = ByteArray(1024)
			var bytesRead: Int

			while(inputStream.read(buffer).also{bytesRead = it} != -1){
				byteArrayOutputStream.write(buffer, 0, bytesRead)
			}

			inputStream.close()
			return byteArrayOutputStream.toByteArray()

		} catch (e: Exception){
			e.printStackTrace()
			return null
		}
	}
	fun loadAudioDataFiles(context: Context, language: String?){
		Log.d("Spatial Audio", "[LoadAudioDataFiles] Loading files...")
		var fileNameWav: String
		var fileNameJson: String
		when(language){
			"english" -> {
				Log.d("SpatialAudio", "[SpatialAudio] Selected english language")
				fileNameWav = "coco_labels_english.wav"
				fileNameJson = "coco_labels_data_english.json"
			}
			"german" ->{
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
		val cocoLabelsData = loadFileFromAssets(context, fileNameJson)



		if(cocoLabelsData != null  && cocoLabelsAudio != null){
			NativeLib.setupAudioSettings(cocoLabelsAudio, cocoLabelsData)
			Log.d("SpatialAudio",
				"[SpatialAudio] Loaded coco data from $fileNameWav and $fileNameJson"
			)
		}else{
			Log.e("SpatialAudio", "[SpatialAudio] Could not load coco data")
		}
	}
}