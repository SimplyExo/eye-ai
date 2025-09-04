package com.algorithmic_alliance.eyeaiapp.audio

import android.content.Context
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import com.algorithmic_alliance.eyeaiapp.NativeLib
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import java.util.logging.Logger
import kotlinx.coroutines.*


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
						NativeLib.sendDepthEstimationData(data)
					}
					delay(50)
				}
			}
		}
	}

	fun setup(context: Context){
		eyeAIApp = context.applicationContext as EyeAIApp
		NativeLib.setAudioSettings(8, 150.0f)

	}

	fun destroy() {
		NativeLib.destroySpatialAudio()
	}

}