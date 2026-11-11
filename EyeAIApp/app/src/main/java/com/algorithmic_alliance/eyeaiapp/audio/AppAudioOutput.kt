package com.algorithmic_alliance.eyeaiapp.audio

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.algorithmic_alliance.eyeaiapp.EyeAIApp
import java.util.concurrent.locks.LockSupport

/** Owns the single mix and TCP connection used while EyeAIVision is selected. */
object AppAudioOutput {
	private val main = Handler(Looper.getMainLooper())
	@Volatile private var session: Session? = null
	val isNetworkOutput: Boolean get() = session != null

	@Synchronized
	fun start(host: String) {
		stop()
		uniffi.NativeLib.setNetworkAudioOutput(true)
		session = Session(host).also { it.start() }
	}

	@Synchronized
	fun stop() {
		val previous = session
		session = null
		previous?.stop()
		uniffi.NativeLib.setNetworkAudioOutput(false)
	}

	/** Returns a session-bound enqueue operation; late synthesis never enters a later session. */
	fun destination(): ((String, ShortArray, () -> Unit) -> Unit)? {
		val target = session ?: return null
		return { bus, samples, done ->
			if (session === target) target.mixer.enqueue(bus, samples) { main.post { done() } }
			else main.post { done() }
		}
	}

	fun remove(bus: String) { session?.mixer?.remove(bus) }
	fun pause(bus: String) { session?.mixer?.pause(bus) }
	fun resume(bus: String) { session?.mixer?.resume(bus) }
	fun isPlaying(bus: String): Boolean = session?.mixer?.contains(bus) == true

	private class Session(private val host: String) {
		val mixer = PcmMixer()
		@Volatile private var running = true
		private val stream = PcmNetworkStream(host,
			onConnected = { Log.i(EyeAIApp.APP_LOG_TAG, "App audio connected to $host:4041 (48kHz stereo s16le)") },
			onError = { Log.w(EyeAIApp.APP_LOG_TAG, "App audio disconnected; retrying", it) },
		)
		private val renderThread = Thread({ render() }, "EyeAI audio mixer")

		fun start() {
			stream.start()
			renderThread.start()
		}

		fun stop() {
			running = false
			stream.close()
			renderThread.interrupt()
			try { renderThread.join(250) } catch (_: InterruptedException) { Thread.currentThread().interrupt() }
			mixer.clear().forEach { it() }
		}

		private fun render() {
			var deadline = System.nanoTime()
			try {
				while (running) {
					val (data, completions) = mixer.mix(uniffi.NativeLib.renderNetworkAudio(480u))
					stream.offer(data)
					completions.forEach { it() }
					deadline += 10_000_000L
					val now = System.nanoTime()
					if (deadline < now) deadline = now // never send a catch-up burst
					LockSupport.parkNanos(deadline - now)
				}
			} catch (error: Exception) {
				Log.e(EyeAIApp.APP_LOG_TAG, "Network audio mixer failed", error)
			}
		}

	}
}
