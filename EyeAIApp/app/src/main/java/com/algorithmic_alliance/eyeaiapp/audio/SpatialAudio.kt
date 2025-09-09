package com.algorithmic_alliance.eyeaiapp.audio

import android.content.Context
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import kotlinx.coroutines.*
import java.io.InputStream
import java.io.ByteArrayOutputStream


object SpatialAudio {
	private var executor = Executors.newSingleThreadExecutor()
	private var scope = CoroutineScope(executor.asCoroutineDispatcher())
	private lateinit var eyeAIApp: EyeAIApp;

	fun start() {
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
		NativeLib.setAudioSettings(8, 150.0f)

		val cocoLabelsAudio = loadFileFromAssets(context, "coco_labels.wav")
		val cocoLabelsData = loadFileFromAssets(context, "coco_labels_data.json")

		if(cocoLabelsData != null  && cocoLabelsAudio != null){
			NativeLib.setupAudioSettings(cocoLabelsAudio, cocoLabelsData)
		}
	}

	fun destroy() {
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
}