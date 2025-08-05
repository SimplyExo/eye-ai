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


object SpacialAudio {
	private var logger = Logger.getLogger("SpacialAudio")
	private var executor = Executors.newSingleThreadExecutor()
	private var scope = CoroutineScope(executor.asCoroutineDispatcher())
	private lateinit var eyeAIApp: EyeAIApp;

	fun start() {
		scope.launch {
			while (isActive) {
				val data = eyeAIApp.aiData.depthEstimationData.get()
				if(data != null){
					NativeLib.sendDepthEstimationData(data)
				}
			}
		}
	}

	fun setup(context: Context){
		eyeAIApp = context.applicationContext as EyeAIApp
		NativeLib.setupAudioDevice()
	}

	fun destroy() {
		NativeLib.destroyAudioDevice()
	}

}