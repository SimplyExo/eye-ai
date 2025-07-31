package com.algorithmic_alliance.eyeaiapp.audio

import com.algorithmic_alliance.eyeaiapp.NativeLib
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.concurrent.Executors


object SpacialAudio {

	private var executor = Executors.newSingleThreadExecutor()
	private var scope = CoroutineScope(executor.asCoroutineDispatcher())

	fun start() {
		scope.launch {
			while (isActive) {
				//playSound(200.0f, 1.0f)
			}
		}
	}

	fun setup(){
		NativeLib.setupAudioDevice()
	}
	/*
	fun playSound(frequency: Float, duration: Float) {
		NativeLib.playSound(frequency, duration)
	}
	 */

	fun destroy() {
		NativeLib.destroyAudioDevice()
	}
}