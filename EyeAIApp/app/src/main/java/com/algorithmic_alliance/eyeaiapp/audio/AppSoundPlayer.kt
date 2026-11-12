package com.algorithmic_alliance.eyeaiapp.audio

import android.content.Context
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.UUID

/** Routes bundled WAV signals/tutorials through the same mixer as speech and spatial audio. */
class AppSoundPlayer(private val context: Context, private val resource: Int) {
	private val bus = "sound-${UUID.randomUUID()}"
	private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
	private val main = Handler(Looper.getMainLooper())
	private var local: MediaPlayer? = null
	@Volatile private var localPlayRequested = false
	@Volatile private var localSeekPending = false
	private var completion: ((AppSoundPlayer) -> Unit)? = null
	@Volatile private var generation = 0
	@Volatile private var pending = false
	@Volatile private var paused = true
	val isPlaying: Boolean get() = (pending && !paused) ||
		(localPlayRequested && localSeekPending) || local?.isPlaying == true

	fun setOnCompletionListener(listener: ((AppSoundPlayer) -> Unit)?) { completion = listener }

	fun start() {
		val destination = AppAudioOutput.destination()
		if (destination == null) {
			val player = local ?: MediaPlayer.create(context, resource).also {
				local = it
				it.setOnCompletionListener {
					localPlayRequested = false
					completion?.invoke(this)
				}
				it.setOnSeekCompleteListener { ready ->
					if (local === ready) {
						localSeekPending = false
						if (localPlayRequested) ready.start()
					}
				}
			}
			localPlayRequested = true
			// MediaPlayer seeks asynchronously. Starting before completion can replay
			// the cue when the pending seek subsequently jumps back to the beginning.
			if (!localSeekPending) player.start()
			return
		}
		localPlayRequested = false
		local?.pause()
		paused = false
		if (pending) {
			AppAudioOutput.resume(bus)
			return
		}
		val token = ++generation
		pending = true
		scope.launch {
			try {
				val samples = context.resources.openRawResource(resource).use { PcmMixer.decodeWave(it.readBytes()) }
				main.post {
					if (generation == token) {
						destination(bus, samples) {
							if (generation == token) { pending = false; completion?.invoke(this@AppSoundPlayer) }
						}
						if (paused) AppAudioOutput.pause(bus)
					}
				}
			} catch (error: Exception) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Could not decode app sound", error)
				main.post {
					if (generation == token) { pending = false; completion?.invoke(this@AppSoundPlayer) }
				}
			}
		}
	}

	fun pause() {
		paused = true
		localPlayRequested = false
		AppAudioOutput.pause(bus)
		local?.pause()
	}

	fun seekTo(position: Int) {
		require(position == 0) { "App sounds support restarting from the beginning" }
		pause()
		++generation
		pending = false
		AppAudioOutput.remove(bus)
		local?.let {
			localSeekPending = true
			it.seekTo(0)
		}
	}

	fun release() {
		seekTo(0)
		scope.cancel()
		local?.release()
		local = null
		localSeekPending = false
		completion = null
	}

	companion object {
		fun create(context: Context, resource: Int) = AppSoundPlayer(context, resource)
	}
}
